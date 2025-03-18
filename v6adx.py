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
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import informative
from pandas import DataFrame, Series
import math
import pandas_ta as pta
import logging
from freqtrade.persistence import Trade

class v6adx(IStrategy):
    minimal_roi = {"0": 0.08}  # 使用自定义止盈
    stoploss = -0.3  # 止损
    timeframe = "15m"  # 15分钟线
    informative_timeframe = '1h'
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    # 自定义参数
    initial_position_size = DecimalParameter(0.05, 0.5, default=0.20, decimals=2)  # 初始仓位20%
    position_increment = DecimalParameter(0.01, 0.2, default=0.05, decimals=2)  # 加仓增量5%
    max_dca_entries = 4
    rsi_period = 6  # RSI周期改为6
    adx_period = 14
    adx_threshold = 20  # 15分钟线降低阈值到20
    sigLen = 14      # ADX 平滑周期
    diLen = 14       # DI 计算周期
    hlRange = 20     # 水平范围阈值
    hlTrend = 35
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 计算 RSI (6周期)
        dataframe['rsi_1h'] = pta.rsi(dataframe['close'], length=6)
        
        # 计算 True Range
        dataframe['tr'] = pta.true_range(dataframe['high'], dataframe['low'], dataframe['close'])
        
        # 计算 DM（Directional Movement）
        dataframe['high_diff'] = dataframe['high'].diff()
        dataframe['low_diff'] = -dataframe['low'].diff()
        dataframe['plus_dm'] = dataframe['high_diff'].clip(lower=0)
        dataframe['minus_dm'] = dataframe['low_diff'].clip(lower=0)
        
        # 计算平滑移动平均（SMA），周期为 self.diLen
        dataframe['sma_plus_dm'] = pta.sma(dataframe['plus_dm'], length=self.diLen)
        dataframe['sma_minus_dm'] = pta.sma(dataframe['minus_dm'], length=self.diLen)
        dataframe['sma_tr'] = pta.sma(dataframe['tr'], length=self.diLen)
        
        # 计算 DI+ 和 DI-（百分比表示）
        dataframe['plus_di'] = 100 * dataframe['sma_plus_dm'] / dataframe['sma_tr']
        dataframe['minus_di'] = 100 * dataframe['sma_minus_dm'] / dataframe['sma_tr']
        
        # 计算 DX 与 ADX
        dataframe['dx'] = 100 * (abs(dataframe['plus_di'] - dataframe['minus_di']) /
                                 (dataframe['plus_di'] + dataframe['minus_di']))
        dataframe['adx'] = pta.sma(dataframe['dx'], length=self.sigLen)
        
        # 计算趋势信号，并确保类型为 bool
        dataframe['hlRange'] = (dataframe['adx'] <= self.hlRange).astype(bool)  # ADX 低于阈值视为水平行情
        dataframe['diUp'] = (dataframe['plus_di'] >= dataframe['minus_di']).astype(bool)
        dataframe['diDn'] = (dataframe['minus_di'] > dataframe['plus_di']).astype(bool)
        # 对于 sigUp，当 adx 上升时置 True；用 shift 的 fill_value 参数保证布尔类型
        dataframe['sigUp'] = (dataframe['adx'] > dataframe['adx'].shift(1, fill_value=dataframe['adx'].iloc[0])).astype(bool)
        
        # 定义一个辅助函数用于判断两个序列的交叉
        def cross(s1: Series, s2: Series) -> Series:
            return ((s1 > s2) & (s1.shift(1, fill_value=s1.iloc[0]) < s2.shift(1, fill_value=s2.iloc[0]))).astype(bool)
        
        # 生成入场信号（入场信号条件可根据需要调整）
        dataframe['entryLong'] = (
            (
                (~dataframe['hlRange']) &
                dataframe['diUp'] &
                dataframe['sigUp'] &
                (~dataframe['diUp'].shift(1, fill_value=False))
            ) |
            (
                (~dataframe['hlRange']) &
                dataframe['diUp'] &
                dataframe['sigUp'] &
                (dataframe['adx'] > self.hlRange) &
                (dataframe['hlRange'].shift(1, fill_value=False))
            )
        )
        dataframe['entryShort'] = (
            (
                (~dataframe['hlRange']) &
                dataframe['diDn'] &
                dataframe['sigUp'] &
                (~dataframe['diDn'].shift(1, fill_value=False))
            ) |
            (
                (~dataframe['hlRange']) &
                dataframe['diDn'] &
                dataframe['sigUp'] &
                (dataframe['adx'] > self.hlRange) &
                (dataframe['hlRange'].shift(1, fill_value=False))
            )
        )
        dataframe['entryLongStr'] = (
            (~dataframe['hlRange']) &
            dataframe['diUp'] &
            dataframe['sigUp'] &
            (dataframe['plus_di'] >= self.hlTrend)
        )
        dataframe['entryShortSt'] = (
            (~dataframe['hlRange']) &
            dataframe['diDn'] &
            dataframe['sigUp'] &
            (dataframe['minus_di'] > self.hlTrend)
        )
        
        # 生成出场信号
        dataframe['exitLong'] = (
            (cross(dataframe['plus_di'], dataframe['minus_di']) & dataframe['diUp'].shift(1, fill_value=False)) |
            (dataframe['hlRange'] & (~dataframe['hlRange'].shift(1, fill_value=False)))
        )
        dataframe['exitShort'] = (
            (cross(dataframe['plus_di'], dataframe['minus_di']) & dataframe['diDn'].shift(1, fill_value=False)) |
            (dataframe['hlRange'] & (~dataframe['hlRange'].shift(1, fill_value=False)))
        )
        
        return dataframe



        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = pta.rsi(dataframe['close'], length=self.rsi_period)
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        informative = self.populate_indicators_1h(informative, metadata)
        dataframe = pd.merge(dataframe, informative[['date', 'rsi_1h', 'adx']], on='date', how='left').ffill()

        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 16) &  # RSI(6) < 30
                (dataframe['rsi_1h'] <= 25)   ADX > 20
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'buy_normal')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_entry(self, pair: str, current_rate: float, entry_tag: str, side: str, **kwargs) -> bool:

        trades = [trade for trade in Trade.get_open_trades() if trade.pair == pair and trade.is_open]
        if not trades:
            return entry_tag in ['buy_normal']
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
        """ 处理普通信号（可选，保留原有逻辑）
        if entry_tag == 'buy_normal':
           last_trade = trades[-1]
           if last_trade and last_trade.is_open:
               price_drop = (last_trade.open_rate - current_rate) / last_trade.open_rate
               return price_drop >= 0.05  # 从最新入仓价下跌 ≥ 3%
        """
        
    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', 
                    current_rate: float, current_profit: float, **kwargs) -> tuple[bool, str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.informative_timeframe)
        if len(dataframe) < 2:
            return False, "hold"
        last_row = dataframe.iloc[-1]
        prev_row = dataframe.iloc[-2]

        if (last_row['adx'] < prev_row['adx']) and (last_row['adx'] > 36):
            return True, "let_it_go"
        
        return False, "hold"
        
    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float, proposed_stake: float, **kwargs) -> float:
        
        total_balance = self.wallets.get_available_stake_amount()
        trades = [trade for trade in Trade.get_open_trades() if trade.pair == pair and trade.is_open]
        num_entries = len(trades)
        
        if num_entries == 0:
            return total_balance * self.initial_position_size.value
        
        last_trade = trades[-1]
        price_change = (last_trade.open_rate - current_rate) / last_trade.open_rate
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.informative_timeframe)
        last_row = dataframe.iloc[-1]
        prev_row = dataframe.iloc[-2]
        adx_increasing = last_row['adx'] < prev_row['adx']  # 当前1h ADX > 前一根1h ADX
        
        if entry_tag == 'buy_normal' and price_change >= 0.05 and adx_increasing and last_row['adx'] > 48:
            add_size = self.initial_position_size + (self.position_increment * num_entries)
            return min(total_balance * add_size, total_balance)
        return 0
        """if price_change <= -0.05:  # 下跌5%
            add_size = base_add_size + (self.position_increment.value * n_entries)
        elif price_change >= 0.05:  # 上涨5%
            add_size = max(base_add_size - (self.position_increment.value * n_entries), 0.01)
        else:
            return 0
        """