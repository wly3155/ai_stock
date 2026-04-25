"""
股票过滤器基类
提供统一的股票筛选接口和数据获取能力
"""
import logging
import pandas as pd
import re

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional, Dict, Any

from stock_data_fetcher import StockDataFetcher

logger = logging.getLogger(__name__)


class StockFilter(ABC):
    """股票过滤器抽象基类"""

    def __init__(self, data_fetcher: Optional[StockDataFetcher] = None):
        """
        初始化过滤器

        Args:
            data_fetcher: 数据获取器实例，如果为None则创建新的
        """
        self.data_fetcher = data_fetcher or StockDataFetcher()
        self.config = self._create_config()

    @abstractmethod
    def _create_config(self) -> Any:
        """
        创建过滤器配置

        Returns:
            配置对象
        """
        pass

    @abstractmethod
    def filter_stocks(self, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        筛选股票的主要方法

        Args:
            market_data: 市场数据，如果为None则自动获取

        Returns:
            筛选后的股票DataFrame
        """
        pass

    def get_market_data(self, date_str: Optional[str] = None) -> pd.DataFrame:
        """
        获取市场数据

        Args:
            date_str: 日期字符串，格式为YYYYMMDD

        Returns:
            市场数据DataFrame
        """
        return self.data_fetcher.get_stock_market_data(date_str)

    def get_stock_history(
        self,
        stock_code: str,
        lookback_days: int = 100
    ) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            stock_code: 股票代码
            lookback_days: 回溯天数

        Returns:
            股票历史数据DataFrame
        """
        # 获取交易日历确定结束日期
        trade_dates = self.data_fetcher.get_trade_dates()
        if trade_dates.empty:
            logger.error("无法获取交易日历")
            return pd.DataFrame()

        end_date = trade_dates['trade_date'].iloc[-1]
        start_date = end_date - timedelta(days=lookback_days)
        return self.data_fetcher.get_stock_history_data(
            stock_code=stock_code,
            start_date=str(start_date),
            end_date=str(end_date)
        )

    def calculate_history_average_volume(
        self,
        history_data: pd.DataFrame,
        lookback_days: int = 5
    ) -> float:
        """
        计算历史平均成交量

        Args:
            history_data: 历史数据DataFrame
            lookback_days: 回溯天数

        Returns:
            历史平均成交量
        """
        if history_data.empty or 'volume' not in history_data.columns:
            return 0.0

        if len(history_data) < lookback_days:
            return history_data['volume'].mean()

        return history_data['volume'].rolling(lookback_days).mean().iloc[-1]

    def calculate_history_max_price(
        self,
        history_data: pd.DataFrame,
        lookback_days: int = 60
    ) -> float:
        """
        计算历史最低价

        Args:
            history_data: 历史数据DataFrame
            lookback_days: 回溯天数

        Returns:
            历史最低价
        """
        if history_data.empty or 'close' not in history_data.columns:
            return 0.0

        if len(history_data) < lookback_days:
            return history_data['close'].max()

        return history_data['close'].rolling(window=lookback_days, min_periods=1).max().iloc[-1]

    def calculate_history_min_price(
        self,
        history_data: pd.DataFrame,
        lookback_days: int = 60
    ) -> float:
        """
        计算历史最低价

        Args:
            history_data: 历史数据DataFrame
            lookback_days: 回溯天数

        Returns:
            历史最低价
        """
        if history_data.empty or 'close' not in history_data.columns:
            return 0.0

        if len(history_data) < lookback_days:
            return history_data['close'].min()

        return history_data['close'].rolling(window=lookback_days, min_periods=1).min().iloc[-1]

    def check_downtrend(self, history_data: pd.DataFrame) -> bool:
        """
        检查是否处于下跌趋势

        Args:
            history_data: 历史数据DataFrame

        Returns:
            是否处于下跌趋势
        """
        if history_data.empty or len(history_data) < 60:
            return False

        # 计算移动平均线
        ma5 = history_data['close'].rolling(5).mean().iloc[-1]
        ma10 = history_data['close'].rolling(10).mean().iloc[-1]
        ma20 = history_data['close'].rolling(20).mean().iloc[-1]
        ma60 = history_data['close'].rolling(60).mean().iloc[-1]

        # 空头排列：MA5 < MA10 < MA20 < MA60
        return ma5 < ma10 < ma20 < ma60

    def get_filter_info(self) -> Dict[str, Any]:
        """
        获取过滤器信息

        Returns:
            过滤器信息字典
        """
        return {
            'filter_name': self.__class__.__name__,
            'config_type': getattr(self.config, 'strategy_type', 'unknown'),
            'data_fetcher': self.data_fetcher.__class__.__name__
        }

    def validate_stock_code(self, stock_code: str) -> bool:
        """
        验证股票代码格式

        Args:
                stock_code: 股票代码

        Returns:
                是否有效
        """
        # 支持两种格式：
        # 1. 纯6位数字: 000001, 600000
        # 2. 前缀+6位数字: sh000001, sz000001
        pattern = r'^(sh|sz)?\d{6}$'
        return bool(re.match(pattern, str(stock_code), re.IGNORECASE))

    def save_filter_results(
        self,
        results: pd.DataFrame,
        filename: Optional[str] = None
    ) -> str:
        """
        保存筛选结果

        Args:
            results: 筛选结果DataFrame
            filename: 文件名，如果为None则自动生成

        Returns:
            保存的文件路径
        """
        from datetime import datetime

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filter_results_{timestamp}.csv"

        save_path = self.data_fetcher.cache_dir / filename
        results.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"筛选结果已保存到: {save_path}")

        return str(save_path)
