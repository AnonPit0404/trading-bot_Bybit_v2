import numpy as np
import logging
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, informative
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

class StrategySpotDCA_V2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False 
    
    # === РИСК-МЕНЕДЖМЕНТ ===
    stoploss = -0.25
    startup_candle_count = 50 
    position_adjustment_enable = True
    max_entry_position_adjustment = 3 
    use_custom_stoploss = True

    # === ПАРАМЕТРЫ ВХОДА ===
    adx_threshold = IntParameter(20, 35, default=23, space='buy')
    
    # === ТЕЙК-ПРОФИТЫ ===
    profit_exit_1 = 0.022 
    profit_exit_2 = 0.045 
    profit_exit_3 = 0.080 
    
    # === НАСТРОЙКИ СОСТОЯНИЯ ===
    STATE_SAVE_INTERVAL = 3600  # Секунд между сохранениями (1 час)
    CANDLE_RETENTION_HOURS = 48  # Сколько часов хранить обработанные свечи

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.state_file = "user_data/strategy_state.json"
        self.last_processed_timestamp = None
        self.processed_candles = set()
        self.load_state()

    @informative('5m')
    def populate_indicators_btc(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['safe_btc'] = (dataframe['close'] > dataframe['ema_200']) & (dataframe['rsi'] > 50)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Фишка: ATR для фильтра волатильности (не входим в "мертвый" рынок)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_mean'] = dataframe['atr'].rolling(30).mean()
        
        # Фишка: Relative Volume (всплеск объема должен быть > 1.4x от предыдущей свечи)
        dataframe['rel_volume'] = dataframe['volume'] > (dataframe['volume'].shift(1) * 1.1)
        
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        
        # Добавляем колонку с timestamp для отслеживания
        if 'date' in dataframe.columns:
            dataframe['timestamp'] = dataframe['date'].astype('int64') // 10**9
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Убираем все жесткие фильтры (BTC Shield, Volume, ADX)
        # Оставляем только пересечение средних и нормальный RSI
        
        # Получаем текущую пару
        pair = metadata.get('pair', 'unknown')
        
        # Создаем колонку enter_long по умолчанию = 0
        dataframe.loc[:, 'enter_long'] = 0
        
        # Находим свечи, где выполняются условия входа
        condition = (
            (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])) &
            (dataframe['rsi'] > 45) & (dataframe['rsi'] < 72)
        )
        
        # Получаем индексы свечей, где условие истинно
        condition_indices = dataframe.index[condition].tolist()
        
        current_time = time.time()
        
        for idx in condition_indices:
            # Получаем timestamp свечи
            candle_time = dataframe.loc[idx, 'date']
            candle_timestamp = int(candle_time.timestamp())
            
            # Создаем уникальный ключ для свечи (пара + timestamp)
            candle_key = f"{pair}_{candle_timestamp}"
            
            # Проверяем, не обрабатывали ли мы уже эту свечу
            if candle_key not in self.processed_candles:
                # Если свеча новая, разрешаем вход и добавляем в обработанные
                dataframe.loc[idx, 'enter_long'] = 1
                self.processed_candles.add(candle_key)
                
                # Обновляем последний обработанный timestamp
                if self.last_processed_timestamp is None or candle_timestamp > self.last_processed_timestamp:
                    self.last_processed_timestamp = candle_timestamp
                    
                logger.info(f"Новый сигнал на вход для {pair} на свече {candle_time}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit >= 0.010: # Безубыток при +1%
            return (1 + 0.001) / (1 + current_profit) - 1
        return self.stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                             current_profit: float, min_stake: float, max_stake: float, **kwargs) -> Optional[float]:
        filled_entries = trade.nr_of_successful_entries
        exit_count = trade.nr_of_successful_exits
        dca_count = filled_entries - 1
        safe_min = max(min_stake or 0, 5.0) * 1.1

        if current_profit > 0:
            current_pos_value = trade.stake_amount
            if exit_count == 0 and current_profit >= self.profit_exit_1:
                exit_amt = current_pos_value * 0.33
                return -exit_amt if exit_amt >= safe_min else None
            if exit_count == 1 and current_profit >= self.profit_exit_2:
                exit_amt = current_pos_value * 0.50
                return -exit_amt if exit_amt >= safe_min else None
            if exit_count == 2 and current_profit >= self.profit_exit_3:
                return -current_pos_value

        if dca_count < self.max_entry_position_adjustment and current_profit < 0:
            dca_thresholds = [-0.045, -0.09, -0.16]
            if current_profit <= dca_thresholds[dca_count]:
                # Берем первую ставку из истории ордеров сделки
                initial_stake = trade.entries[0].stake_amount
                multipliers = [0.5, 1.0, 1.5]
                stake_add = initial_stake * multipliers[dca_count]
                if stake_add >= safe_min:
                    return stake_add
        return None

    def cleanup_old_candles(self) -> None:
        """Удаляет старые записи свечей (старше CANDLE_RETENTION_HOURS часов)"""
        if not self.processed_candles:
            return
            
        current_time = time.time()
        cutoff_time = current_time - (self.CANDLE_RETENTION_HOURS * 3600)
        
        old_count = len(self.processed_candles)
        self.processed_candles = {
            candle_key for candle_key in self.processed_candles
            if self._get_timestamp_from_key(candle_key) > cutoff_time
        }
        
        removed_count = old_count - len(self.processed_candles)
        if removed_count > 0:
            logger.info(f"Очистка: удалено {removed_count} старых записей свечей")
    
    def _get_timestamp_from_key(self, candle_key: str) -> float:
        """Извлекает timestamp из ключа свечи (формат: "пара_timestamp")"""
        try:
            return float(candle_key.split('_')[-1])
        except:
            return 0  # Если не удалось извлечь, считаем очень старым
    
    def load_state(self) -> None:
        """Загружаем состояние из файла"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_processed_timestamp = state.get('last_processed_timestamp')
                    # Загружаем обработанные свечи как список и превращаем в множество
                    processed_list = state.get('processed_candles', [])
                    self.processed_candles = set(processed_list)
                    
                    # Сразу очищаем старые записи после загрузки
                    self.cleanup_old_candles()
                    
                    logger.info(f"Состояние загружено: обработано {len(self.processed_candles)} свечей")
                    logger.info(f"Последняя свеча: {self.last_processed_timestamp}")
            except Exception as e:
                logger.error(f"Ошибка загрузки состояния: {e}")
                self.processed_candles = set()
                self.last_processed_timestamp = None

    def save_state(self) -> None:
        """Сохраняем состояние в файл"""
        try:
            # Сначала очищаем старые записи
            self.cleanup_old_candles()
            
            # Создаем папку user_data, если её нет
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            # Превращаем множество в список для JSON
            processed_list = list(self.processed_candles)
            
            state = {
                'last_processed_timestamp': self.last_processed_timestamp,
                'processed_candles': processed_list,
                'strategy_version': '2.1'
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Состояние сохранено: {len(self.processed_candles)} свечей")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def bot_start(self, **kwargs) -> None:
        """Вызывается при старте бота"""
        self.load_state()
        logger.info("Бот запущен. Состояние загружено.")

    def bot_loop_start(self, **kwargs) -> None:
        """Вызывается в начале каждой итерации"""
        # Сохраняем состояние раз в STATE_SAVE_INTERVAL секунд
        current_time = datetime.now(timezone.utc)
        if not hasattr(self, 'last_save') or (current_time - self.last_save).seconds > self.STATE_SAVE_INTERVAL:
            self.save_state()
            self.last_save = current_time
            
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime, entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Подтверждение входа в сделку"""
        logger.info(f"Подтверждение входа: {pair} по цене {rate}")
        return True