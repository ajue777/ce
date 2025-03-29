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
class v63H(IStrategy):
    # 策略接口版本
    INTERFACE_VERSION = 3
    
    # 最小投资回报率 (ROI)
    minimal_roi = {
        "0": 0.8  # 入场后立即追求80%的回报，若达不到则依赖出场信号
    }
    
    # 止损设置
    stoploss = -0.30  # 固定止损30%
    exit_profit_only = True  # 仅在盈利时退出
    
    # 时间框架
    timeframe = '15m'
    informative_timeframe = '1h'
    process_only_new_candles = True  # 仅处理新K线
    
    # 策略启动所需K线数量
    startup_candle_count = 50
    
    # 订单类型
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # 订单有效时间
    order_time_in_force = {
        'entry': 'gtc',  # 订单永久有效
        'exit': 'gtc'
    }
    
    # 参数定义
    rsi_period = 14  # RSI 计算周期
    adx_period = 14  # ADX 计算周期
    
    def informative_pairs(self):
        """定义额外的参考时间框架对"""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算技术指标"""
        # 计算15分钟周期的 RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        
        # 获取1小时周期数据
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        informative['adx'] = ta.ADX(informative, timeperiod=self.adx_period)
        informative['plus_di'] = ta.PLUS_DI(informative, timeperiod=self.adx_period)
        informative['minus_di'] = ta.MINUS_DI(informative, timeperiod=self.adx_period)
        informative['rsi'] = ta.RSI(informative, timeperiod=self.rsi_period)
        
        # 判断 ADX 是否上升
        informative['adx_rising'] = informative['adx'] > informative['adx'].shift(1)
        
        # 合并1小时数据到15分钟数据
        dataframe = merge_informative_pair(
            dataframe, 
            informative[['date', 'rsi', 'adx', 'plus_di', 'minus_di', 'adx_rising']], 
            self.timeframe, '1h', ffill=True
        )
        
        # 重命名列
        dataframe.rename(columns={
            'rsi_1h': 'rsi_1h',
            'adx_1h': 'adx',
            'plus_di_1h': 'plus_di',
            'minus_di_1h': 'minus_di',
            'adx_rising_1h': 'adx_rising'
        }, inplace=True)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义入场信号"""
        # 初始化 rsi_triggered 列
        if 'rsi_triggered' not in dataframe.columns:
            dataframe['rsi_triggered'] = False
        
        # 动态更新 rsi_triggered 状态
        dataframe['rsi_triggered'] = np.where(
            (dataframe['rsi'] < 30) & (dataframe['rsi_1h'] < 30), True,  # RSI 超卖触发
            np.where(dataframe['rsi'] > 70, False, dataframe['rsi_triggered'].shift(1).fillna(False))  # RSI 超买重置
        )
        
        # 入场条件：ADX 上升且 RSI 触发
        dataframe.loc[
            (dataframe['adx_rising']) & (dataframe['rsi_triggered']),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """定义出场信号"""
        # 出场条件：RSI 超买或 ADX 趋势减弱
        dataframe.loc[
            (dataframe['rsi'] > 70) | (dataframe['adx'] < 25),
            'exit_long'] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: str, **kwargs) -> float:
        """自定义仓位金额"""
        available_balance = self.wallets.get_available_stake_amount()
        stake_percentage = 0.3  # 使用30%的可用余额
        return max(min(available_balance * stake_percentage, max_stake), min_stake)
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """设置杠杆"""
        return 2.0  # 使用2倍杠杆