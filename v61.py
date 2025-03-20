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
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from functools import reduce
import talib.abstract as ta
import logging
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Optional
logger = logging.getLogger(__name__)

class v61(IStrategy):    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.08,
        "60": 0.25
    }
    
    # Stoploss
    stoploss = -0.15
    exit_profit_only = False
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True
    
    # Timeframe for the strategy
    timeframe = '15m'
    informative_timeframe = '1h'
    # Run "populate_indicators" only for new candle
    process_only_new_candles = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = 50
    
    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # Optional time in force for orders
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    # Define parameters
    rsi_period = 6
    adx_period = 14
    di_length = IntParameter(14, 28, default=14, space='buy')  # DI+ 和 DI- 的计算周期
    adx_period = IntParameter(10, 20, default=14, space='buy')  # ADX 的计算周期
    level_range = DecimalParameter(20, 30, default=25, decimals=1, space='buy')  # hlRange 的阈值
    level_trend = DecimalParameter(25, 60, default=48, decimals=1, space='buy')  # 趋势强度的阈值
    # Pyramid position sizing parameters
    max_entry_position_adjustment = 3  # Maximum number of additional entries
    pyramid_max_open_trades = 4  # Maximum number of open trades per pair
    min_stake = 10  
    max_stake = 5000000 
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several indicators to the given DataFrame
        """
        # RSI on 15m timeframe
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # Get 1h informative dataframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        informative['plus_di'] = ta.PLUS_DI(informative, timeperiod=self.di_length.value)
        informative['minus_di'] = ta.MINUS_DI(informative, timeperiod=self.di_length.value)
        informative['adx'] = ta.ADX(informative, timeperiod=self.adx_period.value)
        informative['rsi'] = ta.RSI(informative, timeperiod=self.rsi_period)
        # Previous ADX value
        last_candle = informative['adx'].iloc[-1]  # 最后一行 ADX
        previous_candle = informative['adx'].iloc[-2]  # 倒数第二行 ADX
        informative['adx_prev'] = last_candle < previous_candle
        # 将1小时数据合并到当前时间框架的dataframe（假设当前可能是15分钟或其他）
        # 将 self.merge_informative_pair 改为 merge_informative_pair        
        dataframe = merge_informative_pair(dataframe, informative[['date', 'rsi', 'adx', 'plus_di', 'minus_di', 'adx_prev']], 
                                               self.timeframe, '1h', ffill=True)        
        # 重命名列，避免名称冲突
        dataframe.rename(columns={
            'plus_di_1h': 'plus_di',
            'minus_di_1h': 'minus_di',
            'adx_1h': 'adx'
        }, inplace=True)
        
        # 定义辅助条件
        dataframe['hl_range'] = dataframe['adx'] <= self.level_range.value  # ADX低于阈值，表示震荡
        dataframe['di_up'] = dataframe['plus_di'] >= dataframe['minus_di']  # DI+ >= DI-，偏向上涨
        dataframe['di_dn'] = dataframe['minus_di'] > dataframe['plus_di']  # DI- > DI+，偏向下跌
        dataframe['sig_up'] = dataframe['adx'] > dataframe['adx'].shift(1)  # ADX上升，表示趋势增强
        
        # 计算DI+和DI-的交叉
        dataframe['cross_di'] = pd.Series(np.where(
            (dataframe['plus_di'] > dataframe['minus_di']) & 
            (dataframe['plus_di'].shift(1) <= dataframe['minus_di'].shift(1)), 1,  # DI+上穿DI-
            np.where((dataframe['plus_di'] < dataframe['minus_di']) & 
                     (dataframe['plus_di'].shift(1) >= dataframe['minus_di'].shift(1)), -1, 0)),  # DI+下穿DI-
            index=dataframe.index)
        
        # 生成趋势信号
        dataframe['trend_up'] = (dataframe['hl_range'] == False) & (dataframe['sig_up']) & (dataframe['di_up'])  # 上涨信号
        dataframe['exit_trend_up'] = ((dataframe['cross_di'] == -1) & (dataframe['di_up'].shift(1))) | ((dataframe['hl_range']) & (dataframe['hl_range'].shift(1) == False))  # 退出上涨信号
        dataframe['trend_down'] = (dataframe['hl_range'] == False) & (dataframe['sig_up']) & (dataframe['di_dn'])  # 下跌信号
        dataframe['exit_trend_down'] = ((dataframe['cross_di'] == 1) & (dataframe['di_dn'].shift(1))) | ((dataframe['hl_range']) & (dataframe['hl_range'].shift(1) == False)) # 退出下跌信号
    

               
        return dataframe
    

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, **kwargs) -> float:

        available_balance = self.wallets.get_available_stake_amount()  # 获取可用余额
        stake_percentage = 0.3  # 设置为 10%
        proposed_stake = available_balance * stake_percentage  # 计算百分比金额
        # Get current open trades for this pair
        open_trades = [trade for trade in Trade.get_open_trades() if trade.pair == pair and trade.is_open]        
        if not open_trades:
            # First entry - use normal stake
            return proposed_stake
        
        # Count open trades for this pair
        trade_count = len(open_trades)
        
        if trade_count > self.pyramid_max_open_trades:
            # Maximum trades reached for this pair
            return 0
        
        # Calculate decreasing position size for pyramid entries
        # Each subsequent entry will be smaller than the previous one
        position_adjustment = 1.0 / (trade_count + 1)
        adjusted_stake = proposed_stake * position_adjustment
        
        # Ensure stake amount is within allowed range
        return max(min(adjusted_stake, max_stake), min_stake)
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                             current_rate: float, current_profit: float, min_stake: float,
                             max_stake: float, **kwargs) -> Optional[float]:
        """
        Adjust position size if needed (add to position)
        """
        # Get 1h dataframe for ADX check
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, '1h')
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        
        # Check if this is a subsequent entry (not the first one)
        if trade.nr_of_successful_entries >= 1:
            # Check conditions for adding to position:
            # 1. Price dropped by 5% from entry
            # 2. ADX is decreasing but > 48
            price_dropped = current_rate < trade.open_rate * 0.95
            adx_condition = (last_candle['adx'] < previous_candle['adx']) and (dataframe['adx'] > 36)
            
            if price_dropped and adx_condition:
                # Calculate position size for additional entry
                # Using pyramid sizing - each entry gets smaller
                if trade.nr_of_successful_entries <= self.max_entry_position_adjustment:
                    # Calculate new stake amount
                    stake_amount = trade.stake_amount / (trade.nr_of_successful_entries + 1)
                    
                    # Ensure stake amount is within allowed range
                    stake_amount = max(min(stake_amount, max_stake), min_stake)
                    
                    return stake_amount
        
        # Return None to not adjust position
        return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < 16) &  # 15m RSI6 < 16
                (dataframe['rsi_1h'] < 35) &  # 1h RSI6 < 25
                (dataframe['volume'] > 0)  # Make sure volume is not 0
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe['adx_prev_1h']) &  # Current 1h ADX < previous 1h ADX
                (dataframe['adx'] > 36) &  # 1h ADX > 40
                (dataframe['volume'] > 0)  # Make sure volume is not 0
            ),
            'exit_long'] = 1
        
        return dataframe
    

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        Adjust leverage for each new position
        """
        # Using a conservative leverage of 2x
        return 2.0
