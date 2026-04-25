"""
底部成交量过滤器
实现底部放量上涨策略的股票筛选
"""

import logging

import pandas as pd

from filter_config import FilterConfig
from logger import getLogger
from stock_filter import StockFilter
from typing import List, Optional, Dict, Any


logger = getLogger(__name__, level=logging.DEBUG)


class BottomVolumeFilter(StockFilter):
    """底部成交量过滤器，继承自StockFilter基类"""

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        data_fetcher=None
    ):
        """
        初始化底部成交量过滤器

        Args:
            config: 过滤器配置，如果为None则使用默认配置
            data_fetcher: 数据获取器实例
        """
        self._config = config
        super().__init__(data_fetcher)

    def _create_config(self) -> FilterConfig:
        """创建底部突破策略配置"""
        if self._config is not None:
            return self._config

        return FilterConfig.bottom_breakout_strategy(
            min_price=5.0,
            max_price=50.0,
            min_change=3.0,
            max_change=8.0,
            min_amount=100000000,
            entity_ratio=0.02,
            min_turnover=2.0,
            exclude_st=True,
            exclude_science_innovation=True,
            exclude_gem=True
        )

    def filter_stocks(self, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        筛选底部放量上涨的股票

        Args:
            market_data: 市场数据，如果为None则自动获取

        Returns:
            筛选后的股票DataFrame
        """
        # 打印筛选策略配置
        self._print_filter_config()

        # 获取市场数据
        if market_data is None:
            market_data = self.get_market_data()

        if market_data.empty:
            logger.error("市场数据为空，无法进行筛选")
            return pd.DataFrame()

        # 第一步：基础条件筛选
        condition = self.config.to_condition(market_data)
        first_filter_data = market_data[condition].copy()

        if first_filter_data.empty:
            logger.info("第一步筛选后无符合条件的股票")
            return pd.DataFrame()

        logger.info(f"第一步筛选后剩余 {len(first_filter_data)} 只股票")
        logger.info(f"第一步筛选后剩余票 {first_filter_data}")
        # 第二步：技术指标筛选
        final_results = []
        # 可调整的阈值列表，从宽松到严格
        volume_thresholds = [1.5, 1.8, 2.0, 2.5]  # 成交量倍数
        price_thresholds = [1.05, 1.08, 1.12, 1.15, 1.20]  # 价格相对最低点涨幅
        price_lookback_days = 60
        volume_lookback_days = 10

        for index, row in first_filter_data.iterrows():
            stock_code = str(row['代码'])

            # 验证股票代码
            if not self.validate_stock_code(stock_code):
                logger.warning(f"无效的股票代码: {stock_code}")
                continue

            try:
                # 获取历史数据
                history_data = self.get_stock_history(stock_code, lookback_days=100)
                if history_data.empty:
                    logger.debug(f"股票 {stock_code} 无历史数据")
                    continue

                # 计算技术指标
                current_volume = row['成交量'] if '成交量' in row else 0
                current_price = row['最新价'] if '最新价' in row else 0

                # 计算历史平均成交量（最近10天）
                history_avg_volume = self.calculate_history_average_volume(
                    history_data, lookback_days=volume_lookback_days
                )

                # 计算历史最低价（最近60天）
                history_min_price = self.calculate_history_min_price(
                    history_data, lookback_days=price_lookback_days
                )

                # 计算历史最高价（最近60天）
                history_max_price = self.calculate_history_max_price(
                    history_data, lookback_days=price_lookback_days
                )
                price_position = (current_price - history_min_price) / (history_max_price - history_min_price) \
                    if (history_max_price / history_min_price) > 1.2 else 0

                # 应用底部放量上涨条件
                volume_ratio = current_volume / history_avg_volume if history_avg_volume > 0 else 0
                price_ratio = current_price / history_min_price if history_min_price > 0 else 0
                passed_conditions = []
                conditions_met = False
                if volume_ratio > 1.5 and price_ratio < 1.15 and price_position < 0.4:
                    conditions_met = True
                    condition_type = "量比>1.5, 价格比<1.15"
                else:
                    # 尝试不同的阈值组合
                    for vol_th in volume_thresholds:
                        for price_th in price_thresholds:
                            if volume_ratio > vol_th and price_ratio < price_th and price_position < 0.5:
                                conditions_met = True
                                condition_type = f"量比(vol>{vol_th}, 价格比<{price_th})"
                                break
                        if conditions_met:
                            break
                if not conditions_met:
                    logger.info(f"股票 {stock_code} 不符合底部放量上涨条件: "
                                f"成交量比={current_volume / history_avg_volume:.2f}, "
                                f"价格比={current_price / history_min_price:.2f}, "
                                f"价格位置={price_position:.2f}")
                    continue

                # 添加技术指标到结果
                conditions_met = True
                condition_type = condition_type
                result_row = row.to_dict()
                result_row.update({
                    f'{volume_lookback_days}_day_avg_volume': history_avg_volume,
                    f'{price_lookback_days}_day_min_price': history_min_price,
                    f'{price_lookback_days}_day_max_price': history_max_price,
                    'volume_ratio': volume_ratio,
                    'price_ratio': price_ratio,
                    'price_position': 'NA' if price_position == 0 else price_position,
                    'condition_type': condition_type
                })
                final_results.append(result_row)

                logger.info(f"股票 {stock_code} 符合条件[{condition_type}]: "
                            f"成交量比={volume_ratio:.2f}, "
                            f"价格比={price_ratio:.2f}, "
                            f"价格位置={price_position:.2f}")
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not final_results:
            logger.info("无股票符合底部放量上涨条件")
            return pd.DataFrame()

        # 创建结果DataFrame
        result_df = pd.DataFrame(final_results)

        # 按涨跌幅排序
        if '涨跌幅' in result_df.columns:
            result_df = result_df.sort_values('涨跌幅', ascending=False)

        logger.info(f"最终筛选出 {len(result_df)} 只符合底部放量上涨条件的股票")
        return result_df

    def _print_filter_config(self):
        """打印筛选策略配置"""
        print("\n" + "=" * 60)
        print("底部成交量筛选策略配置:")
        print("=" * 60)
        print(f"策略类型: {getattr(self.config, 'strategy_type', 'bottom_breakout')}")
        print(f"价格范围: {getattr(self.config, 'min_price', 0)} - {getattr(self.config, 'max_price', 0)} 元")
        print(f"涨幅范围: {getattr(self.config, 'min_change', 0)}% - {getattr(self.config, 'max_change', 0)}%")
        print(f"成交额: > {getattr(self.config, 'min_amount', 0) / 100000000:.2f} 亿元")
        print(f"阳线实体: > {getattr(self.config, 'entity_ratio', 0) * 100}%")
        print(f"排除ST: {getattr(self.config, 'exclude_st', True)}")
        print(f"排除科创板: {getattr(self.config, 'exclude_science_innovation', True)}")
        print(f"排除创业板: {getattr(self.config, 'exclude_gem', True)}")
        if getattr(self.config, 'min_turnover', None):
            print(f"换手率: > {getattr(self.config, 'min_turnover', 0)}%")
        print("=" * 60)

    def filter_with_custom_conditions(
        self,
        market_data: pd.DataFrame,
        volume_ratio_threshold: float = 1.8,
        price_ratio_threshold: float = 1.2,
        lookback_days: int = 100
    ) -> pd.DataFrame:
        """
        使用自定义条件进行筛选

        Args:
            market_data: 市场数据
            volume_ratio_threshold: 成交量比率阈值
            price_ratio_threshold: 价格比率阈值
            lookback_days: 回溯天数

        Returns:
            筛选结果
        """
        filtered_stocks = []

        for index, row in market_data.iterrows():
            stock_code = str(row['代码'])

            if not self.validate_stock_code(stock_code):
                continue

            try:
                history_data = self.get_stock_history(stock_code, lookback_days)

                if history_data.empty:
                    continue

                current_volume = row['成交量'] if '成交量' in row else 0
                current_price = row['最新价'] if '最新价' in row else 0

                history_avg_volume = self.calculate_history_average_volume(
                    history_data, lookback_days=10
                )
                history_min_price = self.calculate_history_min_price(
                    history_data, lookback_days=60
                )

                volume_condition = current_volume > history_avg_volume * volume_ratio_threshold
                price_condition = current_price > history_min_price * price_ratio_threshold

                if volume_condition and price_condition:
                    result_row = row.to_dict()
                    result_row.update({
                        'history_avg_volume': history_avg_volume,
                        'history_min_price': history_min_price,
                        'volume_ratio': current_volume / history_avg_volume if history_avg_volume > 0 else 0,
                        'price_ratio': current_price / history_min_price if history_min_price > 0 else 0,
                        'custom_volume_threshold': volume_ratio_threshold,
                        'custom_price_threshold': price_ratio_threshold
                    })
                    filtered_stocks.append(result_row)

            except Exception as e:
                logger.error(f"自定义筛选股票 {stock_code} 时出错: {e}")
                continue

        return pd.DataFrame(filtered_stocks)

    def analyze_stock(self, stock_code: str) -> Dict[str, Any]:
        """
        详细分析单只股票

        Args:
            stock_code: 股票代码

        Returns:
            分析结果字典
        """
        if not self.validate_stock_code(stock_code):
            return {"error": "无效的股票代码"}

        try:
            # 获取当前市场数据
            market_data = self.get_market_data()
            stock_data = market_data[market_data['代码'] == stock_code]

            if stock_data.empty:
                return {"error": "未找到该股票的市场数据"}

            row = stock_data.iloc[0]

            # 获取历史数据
            history_data = self.get_stock_history(stock_code, lookback_days=100)

            if history_data.empty:
                return {"error": "未找到该股票的历史数据"}

            # 计算各项指标
            current_volume = row['成交量'] if '成交量' in row else 0
            current_price = row['最新价'] if '最新价' in row else 0

            history_avg_volume_5 = self.calculate_history_average_volume(history_data, 5)
            history_avg_volume_10 = self.calculate_history_average_volume(history_data, 10)
            history_avg_volume_20 = self.calculate_history_average_volume(history_data, 20)

            history_min_price_30 = self.calculate_history_min_price(history_data, 30)
            history_min_price_60 = self.calculate_history_min_price(history_data, 60)
            history_min_price_100 = self.calculate_history_min_price(history_data, 100)

            is_downtrend = self.check_downtrend(history_data)

            analysis_result = {
                'stock_code': stock_code,
                'stock_name': row.get('名称', '未知'),
                'current_price': current_price,
                'current_volume': current_volume,
                'history_data_days': len(history_data),
                'volume_ratios': {
                    '5_day': current_volume / history_avg_volume_5 if history_avg_volume_5 > 0 else 0,
                    '10_day': current_volume / history_avg_volume_10 if history_avg_volume_10 > 0 else 0,
                    '20_day': current_volume / history_avg_volume_20 if history_avg_volume_20 > 0 else 0
                },
                'price_ratios': {
                    '30_day': current_price / history_min_price_30 if history_min_price_30 > 0 else 0,
                    '60_day': current_price / history_min_price_60 if history_min_price_60 > 0 else 0,
                    '100_day': current_price / history_min_price_100 if history_min_price_100 > 0 else 0
                },
                'is_downtrend': is_downtrend,
                'meets_volume_condition_1.8x': current_volume > history_avg_volume_10 * 1.8,
                'meets_price_condition_1.2x': current_price > history_min_price_60 * 1.2,
                'analysis_time': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return analysis_result

        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return {"error": f"分析失败: {str(e)}"}
