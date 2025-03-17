from freqtrade.strategy import (
    IStrategy, 
    Order,
    PairLocks,
    informative,  # @informative decorator
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,  
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import informative
from pandas import DataFrame, Series
import math
import pandas_ta as pta
import logging
from freqtrade.persistence import Trade

class v5(IStrategy):
    """
    DCA 交易策略：
    - 对于 buy_signal_rsi_normal 信号：
         初始入场仓位为 20%，后续加仓依次为 30%、20%、30%
    - 对于 buy_signal_rsi_special_dca 信号：
         使用原来的规则：初始入场 40%，后续加仓依次为 30%、20%、10%
    - 如果总亏损 ≥ 40%，触发止损
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"240": 0.02}
    stoploss = -0.16  # 止损
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    timeframe = '15m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 100
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True
    position_adjustment_Enabled = True

    max_dca_entries = 4  # 初始入场 + 最多 4次加仓

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=6)
        dataframe['rsi_less_1h'] = dataframe['rsi'] < dataframe['rsi_1h']
        return dataframe

    def custom_stake_amount(self, pair: str, current_rate: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        根据账户总余额和触发信号计算仓位：
        对于 buy_signal_rsi_normal 信号：
            - 初始入场：20%
            - 第一次加仓：30%
            - 第二次加仓：20%
            - 第三次加仓：30%
        对于 buy_signal_rsi_special_dca 信号：
            - 初始入场：40%
            - 第一次加仓：30%
            - 第二次加仓：20%
            - 第三次加仓：10%
        """
        total_balance = self.wallets.get_available_stake_amount()
        # 使用现有方法获取该交易对所有的开仓交易
        trades = [trade for trade in Trade.get_open_trades() if trade.pair == pair and trade.is_open]
        num_entries = len(trades) if trades else 0

        if entry_tag == 'buy_normal':
            if num_entries == 0:
                return total_balance * 0.30   # 初始入场 20%
            elif num_entries == 1:
                return total_balance * 0.20   # 第一次加仓 30%
            elif num_entries == 2:
                return min(total_balance * 0.4, total_balance)
            elif num_entries == 3:
                return min(total_balance * 0.1, total_balance)   # 第三次加仓 30%
            else:
                return 0
        else:
            return 0
        
    def custom_entry(self, pair: str, current_rate: float, entry_tag: str, side: str, **kwargs) -> bool:
        """
        入场逻辑：
        - 若无开仓且触发 buy_signal_rsi_normal 信号，则直接入场；
        - 若已有开仓，则根据触发信号判断价格跌幅：
            - buy_signal_rsi_normal 信号要求最新入仓价下跌 ≥ 5%
            - buy_signal_rsi_special_dca 信号要求最新入仓价下跌 ≥ 10%
        - 同时不能超过最大加仓次数。
        """
        trades = self.strategy.get_open_trades_for_pair(pair)
        if not trades:
            return entry_tag in ['buy_normal']  # 允许两种信号触发初始入场
        else:
            if len(trades) >= self.max_dca_entries:
                return False  # 达到最大加仓次数，不再加仓
        """
        if entry_tag == 'buy_dca':
            # 从最后一笔交易中获取信号触发时的价格
            signal_price = trades[-1].custom_data.get('signal_price', None)
            if signal_price:
                price_drop = (signal_price - current_rate) / signal_price
                return price_drop >= 0.1  # 当前价格从信号触发价格下跌 ≥ 10%
        """        
        # 处理普通信号（可选，保留原有逻辑）
        if entry_tag == 'buy_normal':
           last_trade = trades[-1]
           if last_trade and last_trade.is_open:
               price_drop = (last_trade.open_rate - current_rate) / last_trade.open_rate
               return price_drop >= 0.05  # 从最新入仓价下跌 ≥ 3%
                
    def after_enter(self, pair: str, trade: Trade, **kwargs):
        if trade.enter_tag == 'buy_dca':
            trade.custom_data['signal_price'] = trade.open_rate  # 记录信号触发时的价格

    def custom_stoploss(self, pair: str, trade, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        动态止损：若总亏损 ≥ 40%，则立即止损
        """
        avg_entry_price = trade.open_rate
        loss_ratio = (avg_entry_price - current_rate) / avg_entry_price

        if loss_ratio >= 0.4:
            return 0.01  # 立即止损
        return 1  # 继续持仓

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成买入信号：
        - 当 RSI <= 20 且 15 分钟 RSI < 1 小时 RSI 时，触发 buy_signal_rsi_normal 信号
        - 当 RSI < 9 时，触发 buy_signal_rsi_special_dca 信号
        """
        dataframe.loc[
            (dataframe['rsi'] <= 16) & (dataframe['rsi_1h'] <= 25) & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']
        ] = (1, 'buy_normal')
        """
        conditions = (dataframe['rsi'] < 6) & (dataframe['volume'] > 0)
        dataframe.loc[conditions, 'enter_long'] = 1
        dataframe.loc[conditions, 'enter_tag'] = 'buy_dca'
        dataframe.loc[conditions, 'signal_price'] = dataframe.loc[conditions, 'close']
        """
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        生成卖出信号：例如当盈利超过 10% 或 RSI 超过设定阈值时触发
        """
        dataframe['profit'] = dataframe['close'] / dataframe['open'] - 1

        dataframe.loc[
            (dataframe['rsi'] > 70) & (dataframe['rsi_1h'] > 50) & (dataframe['profit'] >= 0.04),
            ['exit_long', 'exit_tag']
        ] = (1, 'exit_rsi')

        return dataframe
