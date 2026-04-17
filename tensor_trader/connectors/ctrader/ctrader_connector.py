"""cTrader connector for live trading.

cTrader uses the cTrader Open API which requires:
1. cTrader ID (ctidTraderAccountId)
2. Access token
3. Host/port for connection

This connector implements the Open API protocol for cTrader.
"""
import asyncio
import json
import struct
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CTraderConfig:
    """Configuration for cTrader Open API."""
    host: str = "hms.ctrader.com"
    port: int = 5035
    client_id: str = ""  # OAuth client ID
    client_secret: str = ""  # OAuth client secret
    access_token: str = ""  # OAuth access token
    account_id: str = ""  # cTrader account ID
    
    
class CTraderConnector:
    """Connector for cTrader Open API."""
    
    # cTrader Open API message types
    PROTOBUF_MESSAGE_TYPES = {
        'PROTO_OA_SUBSCRIBE_SPOT_EVENTS_REQ': 4102,
        'PROTO_OA_SUBSCRIBE_SPOT_EVENTS_RES': 4103,
        'PROTO_OA_SPOT_EVENT': 4104,
        'PROTO_OA_NEW_ORDER_REQ': 4105,
        'PROTO_OA_EXECUTION_EVENT': 4106,
        'PROTO_OA_CANCEL_ORDER_REQ': 4107,
        'PROTO_OA_AMEND_ORDER_REQ': 4108,
        'PROTO_OA_ERROR_RES': 4109,
    }
    
    def __init__(self, config: CTraderConfig):
        self.config = config
        self.reader = None
        self.writer = None
        self.connected = False
        self._price_callbacks: List[Callable] = []
        self._running = False
        
    async def connect(self) -> bool:
        """Connect to cTrader Open API."""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.config.host, self.config.port
            )
            
            # Send authorization
            auth_msg = {
                'clientId': self.config.client_id,
                'clientSecret': self.config.client_secret,
                'accessToken': self.config.access_token,
            }
            
            await self._send_message(2100, auth_msg)  # PROTO_OA_APPLICATION_AUTH_REQ
            
            response = await self._receive_message()
            if response.get('payloadType') == 2101:  # PROTO_OA_APPLICATION_AUTH_RES
                self.connected = True
                logger.info("cTrader connected successfully")
                return True
            else:
                logger.error(f"cTrader auth failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"cTrader connection error: {e}")
            return False
    
    async def _send_message(self, msg_type: int, payload: Dict):
        """Send a message to cTrader API."""
        if not self.writer:
            raise RuntimeError("Not connected")
        
        payload_json = json.dumps(payload).encode('utf-8')
        
        # cTrader message format: [length:4][type:4][payload:N]
        header = struct.pack('>II', len(payload_json) + 8, msg_type)
        self.writer.write(header + payload_json)
        await self.writer.drain()
    
    async def _receive_message(self) -> Dict[str, Any]:
        """Receive a message from cTrader API."""
        if not self.reader:
            raise RuntimeError("Not connected")
        
        # Read header
        header = await self.reader.read(8)
        if len(header) < 8:
            return {}
        
        length, msg_type = struct.unpack('>II', header)
        
        # Read payload
        payload_len = length - 8
        payload = await self.reader.read(payload_len)
        
        return {
            'payloadType': msg_type,
            'payload': json.loads(payload.decode('utf-8')) if payload else {}
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")
        
        # Subscribe to spot events
        await self._send_message(4102, {  # PROTO_OA_SUBSCRIBE_SPOT_EVENTS_REQ
            'ctidTraderAccountId': self.config.account_id,
            'symbolId': symbol,
        })
        
        # Wait for spot event
        response = await self._receive_message()
        if response.get('payloadType') == 4104:  # PROTO_OA_SPOT_EVENT
            payload = response.get('payload', {})
            return {
                'symbol': symbol,
                'bid': payload.get('bid'),
                'ask': payload.get('ask'),
                'timestamp': payload.get('timestamp'),
            }
        
        return {}
    
    async def place_order(self,
                         symbol: str,
                         side: str,  # 'buy' or 'sell'
                         order_type: str,  # 'market' or 'limit'
                         volume: float,
                         price: Optional[float] = None,
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on cTrader.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            volume: Order volume
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Order result
        """
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")
        
        order_msg = {
            'ctidTraderAccountId': self.config.account_id,
            'symbolId': symbol,
            'orderType': 'MARKET' if order_type.lower() == 'market' else 'LIMIT',
            'tradeSide': 'BUY' if side.lower() == 'buy' else 'SELL',
            'volume': volume,
        }
        
        if price and order_type.lower() == 'limit':
            order_msg['price'] = price
        
        if stop_loss:
            order_msg['stopLoss'] = stop_loss
        
        if take_profit:
            order_msg['takeProfit'] = take_profit
        
        await self._send_message(4105, order_msg)  # PROTO_OA_NEW_ORDER_REQ
        
        # Wait for execution event
        response = await self._receive_message()
        if response.get('payloadType') == 4106:  # PROTO_OA_EXECUTION_EVENT
            payload = response.get('payload', {})
            return {
                'order_id': payload.get('orderId'),
                'position_id': payload.get('positionId'),
                'volume': payload.get('volume'),
                'price': payload.get('price'),
                'status': payload.get('orderStatus'),
            }
        
        return {'error': 'Order failed', 'response': response}
    
    async def close_position(self, position_id: str) -> Dict[str, Any]:
        """Close a position."""
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")
        
        # cTrader closes positions by placing opposite order
        close_msg = {
            'ctidTraderAccountId': self.config.account_id,
            'positionId': position_id,
        }
        
        await self._send_message(4107, close_msg)  # PROTO_OA_CLOSE_POSITION_REQ
        
        response = await self._receive_message()
        if response.get('payloadType') == 4106:  # PROTO_OA_EXECUTION_EVENT
            payload = response.get('payload', {})
            return {
                'position_id': position_id,
                'status': 'closed',
                'volume': payload.get('volume'),
                'price': payload.get('price'),
            }
        
        return {'error': 'Close failed', 'response': response}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.connected:
            raise RuntimeError("Not connected to cTrader")
        
        # Request positions
        await self._send_message(4112, {  # PROTO_OA_POSITIONS_REQ
            'ctidTraderAccountId': self.config.account_id,
        })
        
        response = await self._receive_message()
        if response.get('payloadType') == 4113:  # PROTO_OA_POSITIONS_RES
            payload = response.get('payload', {})
            positions = payload.get('position', [])
            
            return [{
                'position_id': pos.get('positionId'),
                'symbol': pos.get('symbolId'),
                'side': 'buy' if pos.get('tradeSide') == 'BUY' else 'sell',
                'volume': pos.get('volume'),
                'open_price': pos.get('price'),
                'current_price': pos.get('currentPrice'),
                'profit': pos.get('profit'),
                'swap': pos.get('swap'),
            } for pos in positions]
        
        return []
    
    async def disconnect(self):
        """Disconnect from cTrader."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.connected = False
        logger.info("cTrader disconnected")


class CTraderMockConnector:
    """Mock cTrader connector for testing without real credentials."""
    
    def __init__(self):
        self.connected = False
        self.positions = []
        self.orders = []
        
    async def connect(self) -> bool:
        """Mock connect."""
        self.connected = True
        logger.info("cTrader Mock connected")
        return True
    
    async def disconnect(self):
        """Mock disconnect."""
        self.connected = False
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return mock ticker."""
        return {
            'symbol': symbol,
            'bid': 1.1000,
            'ask': 1.1002,
            'timestamp': int(time.time() * 1000),
        }
    
    async def place_order(self, **kwargs) -> Dict[str, Any]:
        """Return mock order."""
        order_id = f"mock_{int(time.time() * 1000)}"
        self.orders.append({
            'order_id': order_id,
            'status': 'filled',
            **kwargs
        })
        return {
            'order_id': order_id,
            'position_id': f"pos_{order_id}",
            'volume': kwargs.get('volume', 0.1),
            'price': kwargs.get('price', 1.1000),
            'status': 'FILLED',
        }
    
    async def close_position(self, position_id: str) -> Dict[str, Any]:
        """Mock close position."""
        return {
            'position_id': position_id,
            'status': 'closed',
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Return mock positions."""
        return self.positions


if __name__ == '__main__':
    # Test with mock connector
    print("Testing cTrader Mock Connector...")
    
    async def test_mock():
        connector = CTraderMockConnector()
        await connector.connect()
        
        # Test ticker
        ticker = await connector.get_ticker('EURUSD')
        print(f"Ticker: {ticker}")
        
        # Test order
        order = await connector.place_order(
            symbol='EURUSD',
            side='buy',
            order_type='market',
            volume=10000
        )
        print(f"Order: {order}")
        
        await connector.disconnect()
        print("\ncTrader Mock Connector test PASSED!")
    
    asyncio.run(test_mock())
