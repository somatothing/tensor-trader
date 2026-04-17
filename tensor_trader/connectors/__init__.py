"""Exchange connectors for Tensor Trader."""
from .base_connector import (
    BaseExchangeConnector, 
    Order, 
    OrderSide, 
    OrderType,
    Position,
    AccountBalance,
    Ticker,
    OHLCV,
    ConnectorFactory
)
from .bitget.bitget_connector import BitgetConnector, BitgetMockConnector
from .hyperliquid.hyperliquid_connector import HyperliquidConnector, HyperliquidMockConnector
from .mt5.mt5_connector import MT5Connector, MT5MockConnector
from .ctrader.ctrader_connector import CTraderConnector, CTraderMockConnector

__all__ = [
    'BaseExchangeConnector',
    'Order',
    'OrderSide',
    'OrderType',
    'Position',
    'AccountBalance',
    'Ticker',
    'OHLCV',
    'ConnectorFactory',
    'BitgetConnector',
    'BitgetMockConnector',
    'HyperliquidConnector',
    'HyperliquidMockConnector',
    'MT5Connector',
    'MT5MockConnector',
    'CTraderConnector',
    'CTraderMockConnector',
]

# Register connectors with factory
ConnectorFactory.register('bitget', BitgetConnector)
ConnectorFactory.register('hyperliquid', HyperliquidConnector)
ConnectorFactory.register('mt5', MT5Connector)
ConnectorFactory.register('ctrader', CTraderConnector)
