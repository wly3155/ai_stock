"""
过滤器配置类
定义各种筛选策略的配置参数
"""
from typing import Optional
import pandas as pd


class FilterConfig:
    """筛选配置基类 - 增强版"""

    def __init__(self):
        # 基础条件
        self.min_price = 10
        self.max_price = 50
        self.min_volume = 100000000
        self.min_change = 3.0
        self.max_change = 8.0
        self.min_turnover = None
        self.max_amplitude = None

        # 底部策略专用条件
        self.min_amount = None  # 最小成交额
        self.entity_ratio = None  # 实体涨幅比例
        self.exclude_st = False
        self.exclude_science_innovation = False
        self.exclude_gem = False

        # 策略类型标识
        self.strategy_type = "base"  # base, momentum, value, bottom_breakout

    def to_condition(self, data: pd.DataFrame) -> pd.Series:
        """统一的筛选条件接口"""
        if data.empty:
            return pd.Series(False, index=data.index)

        condition = pd.Series(True, index=data.index)

        # 通用基础条件
        condition = self._apply_base_conditions(data, condition)

        # 根据策略类型应用特定条件
        if self.strategy_type == "bottom_breakout":
            condition = self._apply_bottom_breakout_conditions(data, condition)
        elif self.strategy_type == "momentum":
            condition = self._apply_momentum_conditions(data, condition)
        elif self.strategy_type == "value":
            condition = self._apply_value_conditions(data, condition)

        return condition

    def _apply_base_conditions(self, data: pd.DataFrame, condition: pd.Series) -> pd.Series:
        """应用基础条件"""
        if '最新价' in data.columns:
            condition &= (data["最新价"] >= self.min_price)
            condition &= (data["最新价"] <= self.max_price)

        if '成交量' in data.columns and self.min_volume:
            condition &= (data["成交量"] > self.min_volume)

        if '涨跌幅' in data.columns:
            if self.min_change is not None:
                condition &= (data["涨跌幅"] > self.min_change)
            if self.max_change is not None:
                condition &= (data["涨跌幅"] < self.max_change)

        if self.min_turnover and '换手率' in data.columns:
            condition &= (data["换手率"] > self.min_turnover)

        if self.max_amplitude and '振幅' in data.columns:
            condition &= (data["振幅"] < self.max_amplitude)

        return condition

    def _apply_bottom_breakout_conditions(self, data: pd.DataFrame, condition: pd.Series) -> pd.Series:
        """应用底部突破策略特定条件"""
        # 1. 代码过滤
        if '代码' in data.columns:
            codes = data['代码'].astype(str)
            if self.exclude_science_innovation:
                condition &= ~codes.str.contains('^[a-zA-Z]{2}688')
                condition &= ~codes.str.startswith('688')
            if self.exclude_gem:
                condition &= ~codes.str.contains('^[a-zA-Z]{2}300')
                condition &= ~codes.str.startswith('300')
            if self.exclude_st and '名称' in data.columns:
                condition &= ~data['名称'].str.contains('ST', na=False)

        # 2. 成交额过滤
        if self.min_amount and '成交额' in data.columns:
            condition &= (data['成交额'] >= self.min_amount)

        # 3. K线形态过滤
        if self.entity_ratio and all(col in data.columns for col in ['今开', '最新价']):
            # 阳线条件
            condition &= (data['最新价'] > data['今开'])
            # 实体涨幅条件
            entity_change = (data['最新价'] - data['今开']) / data['今开'] * 100
            condition &= (entity_change > self.entity_ratio * 100)

        return condition

    def _apply_momentum_conditions(self, data: pd.DataFrame, condition: pd.Series) -> pd.Series:
        """应用动量策略条件"""
        # 动量策略特定逻辑
        if '成交量' in data.columns:
            condition &= (data['成交量'] > 150000000)  # 动量策略对量能要求更高

        if '换手率' in data.columns and self.min_turnover:
            condition &= (data['换手率'] > self.min_turnover)

        return condition

    def _apply_value_conditions(self, data: pd.DataFrame, condition: pd.Series) -> pd.Series:
        """应用价值策略条件"""
        # 价值策略特定逻辑
        if '市盈率' in data.columns:
            condition &= (data['市盈率'] < 30)  # 低估值

        if '市净率' in data.columns:
            condition &= (data['市净率'] < 2)  # 低市净率

        return condition

    def get_filtered_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取过滤后的股票"""
        condition = self.to_condition(data)
        return data[condition].copy()

    @classmethod
    def base_strategy(cls):
        """基础策略"""
        config = cls()
        config.strategy_type = "base"
        return config

    @classmethod
    def bottom_breakout_strategy(
        cls,
        min_price: float = 5.0,
        max_price: float = 50.0,
        min_change: float = 3.0,
        max_change: float = 8.0,
        min_amount: float = 100000000,
        entity_ratio: float = 0.02,
        min_turnover: Optional[float] = 2.0,
        exclude_st: bool = True,
        exclude_science_innovation: bool = True,
        exclude_gem: bool = True
    ):
        """底部放量上涨策略"""
        config = cls()
        config.strategy_type = "bottom_breakout"

        # 基础参数
        config.min_price = min_price
        config.max_price = max_price
        config.min_change = min_change
        config.max_change = max_change
        config.min_turnover = min_turnover

        # 底部策略专用参数
        config.min_amount = min_amount
        config.entity_ratio = entity_ratio
        config.exclude_st = exclude_st
        config.exclude_science_innovation = exclude_science_innovation
        config.exclude_gem = exclude_gem

        return config

    @classmethod
    def momentum_strategy(cls):
        """动量策略"""
        config = cls()
        config.strategy_type = "momentum"
        config.min_price = 10
        config.max_price = 50
        config.min_volume = 150000000
        config.min_change = 3.5
        config.max_change = 9.0
        config.min_turnover = 3.0
        return config

    @classmethod
    def value_strategy(cls):
        """价值策略"""
        config = cls()
        config.strategy_type = "value"
        config.min_price = 5
        config.max_price = 30
        config.min_volume = 50000000
        config.min_change = -2.0
        config.max_change = 5.0
        return config
