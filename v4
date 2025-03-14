
from freqtrade.strategy import (
    IStrategy,
    Trade, 
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import (IStrategy, informative)
from pandas import DataFrame, Series
import talib.abstract as ta
import math
import pandas_ta as pta
# from finta import TA as fta
import logging
from logging import FATAL

class v4(IStrategy):
    """
    DCA 交易策略：
    - 初始仓位：账户余额 40%
    - 每下跌 5% 加仓 20%，最多加仓 3 次
    - 如果总亏损 ≥ 40%，触发止损
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"30": 0.3}
    stoploss = -0.40
    trailing_stop = False

    timeframe = '15m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 100
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    position_adjustment_Enabled = True

    max_dca_entries = 3  # 允许最多 3 次加仓
    dynamic_safety_order_ratio = -0.05  # 每次加仓 5%
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算 RSI 指标，并合并 1 小时级别 RSI 进行趋势对比。
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)

     #   dataframe['date'] = pd.to_datetime(dataframe['date'], utc=True)
     #   informative_tf['date'] = pd.to_datetime(informative_tf['date'], utc=True)

     #   dataframe = dataframe.merge(
     #      informative_tf[['date', 'rsi']].rename(columns={'rsi': f"rsi_{self.informative_timeframe}"}),
     #       on='date',
     #       how='left'
     #   )
     #   dataframe.set_index('date', inplace=True)

        dataframe['rsi_less_1h'] = dataframe['rsi'] < dataframe['rsi_1h']

        return dataframe

    def custom_stake_amount(self, pair: str, current_rate: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        计算入场金额：
        - 初次入场 40%
        - 每次加仓 20%
        """
        total_balance = self.wallets.get_total_stake_amount()

        if entry_tag == 'buy_signal_rsi_special_dca':
            return total_balance * 0.2  # 加仓金额 20%
        else:
            return total_balance * 0.4  # 初次入场金额 40%

    def custom_entry(self, pair: str, current_rate: float, entry_tag: str, side: str, **kwargs) -> bool:
        """
        处理入场逻辑：
        - RSI 正常低位信号：入场
        - DCA 订单：
          - 必须跌 5% 以上才加仓
          - 不能超过最大 DCA 次数
        """
        if entry_tag == 'buy_signal_rsi_normal':
            return True

        elif entry_tag == 'buy_signal_rsi_special_dca':
            trades = self.strategy.get_open_trades_for_pair(pair)
            if len(trades) >= self.max_dca_entries:
                return False  # 达到最大加仓次数，不再加仓

            last_trade = trades[-1] if trades else None
            if last_trade and last_trade.is_open:
                price_drop = (last_trade.open_rate - current_rate) / last_trade.open_rate
                return price_drop >= 0.05  # 只有当跌幅 ≥ 5% 时才允许加仓

        return False

    def custom_stoploss(self, pair: str, trade, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        计算动态止损：
        - 如果总亏损 >= 40%，触发止损
        """
        avg_entry_price = trade.open_rate
        loss_ratio = (avg_entry_price - current_rate) / avg_entry_price

        if loss_ratio >= 0.4:
            return 0.01  # 立即止损
        return 1  # 继续持仓

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入信号：
        - RSI < 13 且 15 分钟 RSI < 1 小时 RSI => 正常买入
        - RSI < 9 => 触发 DCA 加仓
        """
        dataframe.loc[
            (dataframe['rsi'] <= 20) & (dataframe['rsi_less_1h'] == True) & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']
        ] = (1, 'buy_signal_rsi_normal')

        dataframe.loc[
            (dataframe['rsi'] < 9) & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']
        ] = (1, 'buy_signal_rsi_special_dca')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成卖出信号：
        - 盈利超过 10%
        - RSI > 75，且 15 分钟 RSI 高于 1 小时 RSI
        """
        dataframe['profit'] = dataframe['close'] / dataframe['open'] - 1

        dataframe.loc[
            (dataframe['rsi'] > 85) & (dataframe['rsi_1h'] > 75) & (dataframe['rsi_less_1h'] == False) & (dataframe['volume'] > 0),
            ['exit_long', 'exit_tag']
        ] = (1, 'exit_signal_rsi')

        return dataframe
