"""
股票数据获取基类
统一管理股票数据的获取、缓存和存储
"""

import os
import logging

import akshare as ak

import pandas as pd

from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from logger import getLogger

logger = getLogger(__name__, level=logging.INFO)


class StockDataFetcher:
    """股票数据获取基类，负责所有数据的获取和缓存"""

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        初始化数据获取器

        Args:
            cache_dir: 数据缓存目录
        """
        load_dotenv('.env')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stock_market_data(self, date_str: Optional[str] = None, skip_cache: bool = False) -> pd.DataFrame:
        """
        获取A股市场实时数据，支持缓存

        Args:
            date_str: 日期字符串，格式为YYYYMMDD，如果为None则使用当前日期
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            A股市场实时数据DataFrame
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        cache_file = self.cache_dir / f"stock_zh_a_spot_{date_str}.parquet"

        # 如果缓存文件存在且是今天的数据，则从缓存读取
        if not skip_cache and cache_file.exists():
            logger.info(f"从缓存读取股票市场数据: {cache_file}")
            return pd.read_parquet(str(cache_file))

        # 否则从API获取并缓存
        logger.info(f"从API获取新浪财经-沪深京 A 股数据,日期: {date_str}")
        stock_data = ak.stock_zh_a_spot()

        # 保存到缓存
        stock_data.to_parquet(cache_file, index=False, compression='snappy')
        logger.info(f"股票市场数据已缓存到: {cache_file}")

        return stock_data

    def get_stock_history_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        skip_cache: bool = False
    ) -> pd.DataFrame:
        """
        获取单只股票的历史数据，支持缓存

        Args:
            stock_code: 股票代码
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
            adjust: 复权类型，qfq(前复权), hfq(后复权), ""(不复权)
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            股票历史数据DataFrame
        """
        # 生成缓存文件名
        cache_key = f"{stock_code}_{start_date}_{end_date}_{adjust}"
        cache_file = self.cache_dir / f"stock_history_{cache_key}.parquet"

        # 如果缓存文件存在，则从缓存读取
        if not skip_cache and cache_file.exists():
            logger.debug(f"从缓存读取股票历史数据: {cache_file}")
            return pd.read_parquet(str(cache_file))

        # 否则从API获取
        logger.info(f"从API获取股票历史数据: {stock_code} {start_date} to {end_date}")
        try:
            history_data = ak.stock_zh_a_daily(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            # 如果获取到数据，保存到缓存
            if not history_data.empty:
                history_data.to_parquet(
                    cache_file, index=False, compression='snappy')
                logger.info(f"股票历史数据已缓存到: {cache_file}")

            return history_data

        except Exception as e:
            logger.error(f"获取股票历史数据失败: {stock_code}, 错误: {e}")
            return pd.DataFrame()

    def get_trade_dates(self, cache_days: int = 30, skip_cache: bool = False) -> pd.DataFrame:
        """
        获取交易日历，支持缓存

        Args:
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            交易日历DataFrame
        """
        cache_file = self.cache_dir / f"trade_dates.parquet"

        # 检查缓存是否过期
        if not skip_cache and cache_file.exists():
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - file_mtime).days < cache_days:
                logger.debug(f"从缓存读取交易日历: {cache_file}")
                return pd.read_parquet(str(cache_file))

        # 从API获取最新的交易日历
        logger.info("从API获取交易日历")
        trade_dates = ak.tool_trade_date_hist_sina()

        # 保存到缓存
        trade_dates.to_parquet(cache_file, index=False, compression='snappy')
        logger.info(f"交易日历已缓存到: {cache_file}")

        return trade_dates

    def get_stock_distrubution(self, stock_code: str, cache_days: int = 1, skip_cache: bool = False):
        """
        获取获取筹码分布，支持缓存

        Args:
            stock_code: 股票代码
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            筹码分布DataFrame
        """
        plain_code = self._normalize_stock_code(stock_code)
        cache_file = self.cache_dir / f"{plain_code}_distrubution.parquet"

        # 检查缓存是否过期
        if not skip_cache and cache_file.exists():
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - file_mtime).days < cache_days:
                logger.debug(f"从缓存读取获取筹码分布: {cache_file}")
                return pd.read_parquet(str(cache_file))

        # 从API获取最新的交易日历
        try:
            logger.info("从API获取筹码分布")
            stock_distrubution_dates = ak.stock_cyq_em(symbol=plain_code)

            # 保存到缓存
            stock_distrubution_dates.to_parquet(
                cache_file, index=False, compression='snappy')
            logger.info(f"取筹码分布已缓存到: {cache_file}")
            return stock_distrubution_dates
        except Exception as e:
            logger.error(f"获取 {stock_code} 筹码分布失败 {e}")
            return None

    def get_stock_news(self, stock_code: str, cache_days: int = 1, skip_cache: bool = False):
        """
        获取股票新闻，支持缓存

        Args:
            stock_code: 股票代码
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            新闻DataFrame
        """
        cache_file = self.cache_dir / f"{stock_code}_news.parquet"

        # 检查缓存是否过期
        if not skip_cache and cache_file.exists():
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - file_mtime).days < cache_days:
                logger.debug(f"从缓存读取股票新闻: {cache_file}")
                return pd.read_parquet(str(cache_file))
        # 从API获取股票新闻
        try:
            logger.info("从API获取股票新闻")
            stock_news = ak.stock_news_em(symbol=stock_code)

            # 保存到缓存
            stock_news.to_parquet(
                cache_file, index=False, compression='snappy')
            logger.info(f"新闻已缓存到: {cache_file}")
            return stock_news
        except Exception as e:
            logger.error(f"获取 {stock_code} 新闻失败 {e}")
            return None

    @staticmethod
    def _normalize_stock_code(stock_code: str) -> str:
        """
        统一股票代码格式：去除 sh/sz 前缀，统一小写。
        例如: 'SZ002549' -> '002549', 'sh600519' -> '600519'
        """
        code = stock_code.lower()
        for prefix in ('sh', 'sz'):
            if code.startswith(prefix):
                return code[len(prefix):]
        return code

    @staticmethod
    def _code_formatters() -> Dict[str, List]:
        """
        返回不同格式的 stock_code 生成器列表。
        每种格式是一个可调用对象，接收原始 stock_code，返回格式化后的代码。

        支持的格式：
        - 'raw': 原样传入（如 SZ002549 / sz002549 / 002549）
        - 'lower': 统一小写（如 sz002549）
        - 'upper': 统一大写（如 SZ002549）
        - 'plain': 纯数字（如 002549）
        """
        return {
            'raw':      lambda code: code,
            'lower':    lambda code: code.lower(),
            'upper':    lambda code: code.upper(),
            'plain':    lambda code: StockDataFetcher._normalize_stock_code(code),
        }

    def _get_cache_path(self, stock_code: str, api_name: str) -> Path:
        """生成单个 API 数据缓存文件路径"""
        # 缓存文件名统一用纯数字，避免大小写/前缀导致重复缓存
        plain_code = self._normalize_stock_code(stock_code)
        return self.cache_dir / f"{plain_code}_{api_name}.parquet"

    def _fetch_and_cache(
        self,
        stock_code: str,
        api_name: str,
        fetch_func,
        cache_days: int = 30,
        skip_cache: bool = False,
        code_formats: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        通用：从缓存读取或从 API 获取单个数据源，并保存到独立缓存文件。

        自动尝试多种 stock_code 格式（如 raw / lower / upper / plain），
        直到某个格式调用成功为止。

        Args:
            stock_code: 股票代码
            api_name: API 名称（用于缓存文件名和日志）
            fetch_func: 可调用对象，用于获取数据
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存
            code_formats: 要尝试的 stock_code 格式列表，默认 ['plain', 'raw']
                          可选: 'raw', 'lower', 'upper', 'plain'
            kwargs: 传递给 fetch_func 的额外参数

        Returns:
            DataFrame 或 None（所有格式均获取失败时）
        """
        if code_formats is None:
            code_formats = ['raw']

        cache_file = self._get_cache_path(stock_code, api_name)
        formatters = self._code_formatters()

        # 检查缓存
        if not skip_cache and cache_file.exists():
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - file_mtime).days < cache_days:
                logger.debug(f"从缓存读取 {api_name}: {cache_file}")
                return pd.read_parquet(str(cache_file))

        # 从 API 获取，尝试多种 code 格式
        last_error = None
        for fmt in code_formats:
            if fmt not in formatters:
                continue
            formatted_code = formatters[fmt](stock_code)
            # 将 kwargs 中的 stock_code 占位符替换为格式化后的代码
            updated_kwargs = {
                k: (formatted_code if v == '__stock_code__' else v)
                for k, v in kwargs.items()
            }
            try:
                logger.info(f"从API获取 {formatted_code} {api_name} (format={fmt})")
                df = fetch_func(**updated_kwargs)

                if df is not None and not df.empty:
                    df.to_parquet(cache_file, index=False, compression='snappy')
                    logger.info(f"{api_name} 已缓存到: {cache_file}")
                    return df

                logger.warning(f"{api_name}({fmt}) 返回空数据")
                last_error = f"empty data for format {fmt}"

            except Exception as e:
                logger.debug(f"{api_name}({fmt}) 失败: {e}")
                last_error = e
                continue

        logger.error(f"获取 {stock_code} {api_name} 失败，已尝试格式 {code_formats}: {last_error}")
        return None

    def get_stock_basic_info(
        self,
        stock_code: str,
        cache_days: int = 30,
        skip_cache: bool = False
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        获取股票基本面综合信息，返回多个数据源的原始 DataFrame 列表。

        每个元素为 (api_name, DataFrame) 元组，保留 API 返回的原始结构。

        数据来源：
        - 基本信息: ak.stock_individual_info_em(symbol=stock_code)
        - 公司规模: ak.stock_zh_scale_comparison_em(symbol=stock_code)
        - 主营构成: ak.stock_zygc_em(symbol=stock_code)
        - 财务分析指标: ak.stock_financial_analysis_indicator(symbol=stock_code)
        - 同行业成长性比较: ak.stock_zh_growth_comparison_em(symbol=stock_code)
        - 同行业估值比较: ak.stock_zh_valuation_comparison_em(symbol=stock_code)
        - 同行业杜邦分析比较: ak.stock_zh_dupont_comparison_em(symbol=stock_code)

        Args:
            stock_code: 股票代码
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            List[Tuple[str, pd.DataFrame]]: [(api_name, DataFrame), ...]
        """
        results: List[Tuple[str, pd.DataFrame]] = []

        # 定义要获取的数据源列表：(api_name, fetch_func, kwargs, code_formats)
        # kwargs 中使用 '__stock_code__' 占位符，_fetch_and_cache 会自动按 code_formats 顺序尝试
        # code_formats: 尝试顺序，可选 'raw'(原样), 'lower'(小写), 'upper'(大写), 'plain'(纯数字)
        #               None 表示使用默认值 ['raw']
        api_sources = [
            ('基本信息', ak.stock_individual_info_em, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('公司规模', ak.stock_zh_scale_comparison_em, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('主营构成', ak.stock_zygc_em_after_date, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('财务分析指标', ak.stock_financial_analysis_indicator, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('同行业成长性比较', ak.stock_zh_growth_comparison_em, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('同行业估值比较', ak.stock_zh_valuation_comparison_em, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
            ('同行业杜邦分析比较', ak.stock_zh_dupont_comparison_em, {'symbol': '__stock_code__'}, ['raw', 'upper', 'lower', 'plain']),
        ]

        for api_name, fetch_func, kwargs, code_formats in api_sources:
            df = self._fetch_and_cache(
                stock_code=stock_code,
                api_name=f"basic_info_{api_name}",
                fetch_func=fetch_func,
                cache_days=cache_days,
                skip_cache=skip_cache,
                code_formats=code_formats,
                **kwargs
            )
            if df is not None and not df.empty:
                results.append((api_name, df))

        if not results:
            logger.warning(f"stock_code {stock_code}, 所有数据源均获取失败")

        return results

    def get_stock_financial_info(
        self,
        stock_code: str,
        cache_days: int = 30,
        skip_cache: bool = False
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        获取股票深度财务/基本面信息，返回多个数据源的原始 DataFrame 列表。

        每个元素为 (api_name, DataFrame) 元组，保留 API 返回的原始结构。

        数据来源：
        - 研究报告: ak.stock_research_report_em(symbol=stock_code)
        - 新浪财务报表: ak.stock_financial_report_sina(stock=stock_code)
        - 财务摘要: ak.stock_financial_abstract(symbol=stock_code)
        - 财务分析指标: ak.stock_financial_analysis_indicator(symbol=stock_code)

        Args:
            stock_code: 股票代码
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            List[Tuple[str, pd.DataFrame]]: [(api_name, DataFrame), ...]
        """
        results: List[Tuple[str, pd.DataFrame]] = []

        api_sources = [
            ('研究报告', ak.stock_research_report_em, {'symbol': '__stock_code__'}, ['upper', 'lower', 'plain']),
            ('新浪财务报表', ak.stock_financial_report_sina, {'stock': '__stock_code__'}, ['upper', 'lower', 'plain']),
            ('财务摘要', ak.stock_financial_abstract, {'symbol': '__stock_code__'}, ['upper', 'lower', 'plain']),
            ('财务分析指标', ak.stock_financial_analysis_indicator, {'symbol': '__stock_code__'}, ['upper', 'lower', 'plain']),
        ]

        for api_name, fetch_func, kwargs, code_formats in api_sources:
            df = self._fetch_and_cache(
                stock_code=stock_code,
                api_name=f"financial_info_{api_name}",
                fetch_func=fetch_func,
                cache_days=cache_days,
                skip_cache=skip_cache,
                code_formats=code_formats,
                **kwargs
            )
            if df is not None and not df.empty:
                results.append((api_name, df))

        if not results:
            logger.warning(f"stock_code {stock_code}, 所有财务数据源均获取失败")

        return results

    def get_stock_market_misc_info(
        self,
        stock_code: str,
        cache_days: int = 30,
        skip_cache: bool = False
    ) -> List[Tuple[str, pd.DataFrame]]:
        """
        获取股票杂项市场数据，返回多个数据源的原始 DataFrame 列表。

        每个元素为 (api_name, DataFrame) 元组，保留 API 返回的原始结构。

        数据来源：
        - 股票质押: ak.stock_gpzy_profile_em()
        - 质押比例: ak.stock_gpzy_pledge_ratio_em()
        - 资金流向: ak.stock_individual_fund_flow()

        Args:
            stock_code: 股票代码
            cache_days: 缓存天数
            skip_cache: 是否跳过缓存，强制从API获取

        Returns:
            List[Tuple[str, pd.DataFrame]]: [(api_name, DataFrame), ...]
        """
        results: List[Tuple[str, pd.DataFrame]] = []

        # 这些 API 不需要 stock_code 参数，所以 code_formats 用默认 ['raw'] 即可
        api_sources = [
            ('股票质押', ak.stock_gpzy_profile_em, {}, None),
            ('股票质押比例', ak.stock_gpzy_pledge_ratio_em, {}, None),
            ('资金流向', ak.stock_individual_fund_flow, {}, None),
        ]

        for api_name, fetch_func, kwargs, code_formats in api_sources:
            df = self._fetch_and_cache(
                stock_code=stock_code,
                api_name=f"misc_{api_name}",
                fetch_func=fetch_func,
                cache_days=cache_days,
                skip_cache=skip_cache,
                code_formats=code_formats,
                **kwargs
            )
            if df is not None and not df.empty:
                results.append((api_name, df))

        if not results:
            logger.warning(f"stock_code {stock_code}, 所有杂项数据源均获取失败")

        return results

    def clear_cache(self, days_old: int = 30) -> int:
        """
        清理指定天数前的缓存文件

        Args:
            days_old: 清理多少天前的文件

        Returns:
            清理的文件数量
        """
        deleted_count = 0
        cutoff_time = datetime.now() - timedelta(days=days_old)

        for cache_file in self.cache_dir.glob("*.parquet"):
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_mtime < cutoff_time:
                cache_file.unlink()
                deleted_count += 1
                logger.info(f"清理缓存文件: {cache_file}")

        logger.info(f"共清理 {deleted_count} 个缓存文件")
        return deleted_count
