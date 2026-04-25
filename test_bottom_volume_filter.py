"""
股票筛选系统主程序
使用重构后的模块进行股票筛选
"""
import logging
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from filter_config import FilterConfig
from bottom_volume_filter import BottomVolumeFilter
from stock_data_fetcher import StockDataFetcher


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    print("=" * 60)
    print("股票底部放量上涨筛选系统")
    print("=" * 60)

    # 1. 创建数据获取器
    print("1. 初始化数据获取器...")
    data_fetcher = StockDataFetcher(cache_dir="./data/cache")

    # 2. 创建过滤器配置
    print("2. 创建筛选配置...")
    filter_config = FilterConfig.bottom_breakout_strategy(
        min_price=5.0,
        max_price=50.0,
        min_change=3.0,
        max_change=8.0,
        min_amount=100000000,  # 1亿元
        entity_ratio=0.02,  # 2%
        min_turnover=2.0,  # 2%
        exclude_st=True,
        exclude_science_innovation=True,
        exclude_gem=True
    )

    # 3. 创建底部成交量过滤器
    print("3. 创建底部成交量过滤器...")
    bottom_filter = BottomVolumeFilter(
        config=filter_config,
        data_fetcher=data_fetcher
    )

    # 4. 执行筛选
    print("4. 开始筛选股票...")
    print("-" * 60)

    start_time = datetime.now()
    filtered_stocks = bottom_filter.filter_stocks()
    end_time = datetime.now()

    print("-" * 60)
    print(f"筛选完成，耗时: {(end_time - start_time).total_seconds():.2f}秒")

    # 5. 显示结果
    if not filtered_stocks.empty:
        print(f"\n找到 {len(filtered_stocks)} 只符合底部放量上涨条件的股票:")
        print("=" * 100)

        # 显示关键列
        display_columns = ['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额',
                           'volume_ratio', 'price_ratio', 'is_downtrend']

        # 只显示存在的列
        available_columns = [col for col in display_columns if col in filtered_stocks.columns]

        if available_columns:
            display_df = filtered_stocks[available_columns].head(20)  # 只显示前20只
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(display_df.to_string(index=False))
        else:
            print(filtered_stocks.head(20).to_string(index=False))

        print("=" * 100)

        import akshare as ak
        stock_codes = filtered_stocks["代码"].tolist()
        stock_codes_numbers = [code[2:] for code in stock_codes]  # 去掉sh/sz前缀
        for stock_code in stock_codes:
            print(f"ak stock cyq: {ak.stock_cyq_em(stock_codes)}")

        # 6. 保存结果
        print("\n5. 保存筛选结果...")
        save_path = bottom_filter.save_filter_results(
            filtered_stocks,
            f"bottom_volume_stocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        print(f"结果已保存到: {save_path}")

        # 7. 显示详细信息
        print("\n6. 前3只股票的详细分析:")
        print("-" * 60)

        for i, (_, row) in enumerate(filtered_stocks.head(3).iterrows()):
            stock_code = str(row['代码'])
            print(f"\n股票 {i + 1}: {row.get('名称', '未知')} ({stock_code})")
            print(f"当前价格: {row.get('最新价', 'N/A')} 元")
            print(f"涨跌幅: {row.get('涨跌幅', 'N/A')}%")
            print(f"成交量: {row.get('成交量', 'N/A'):,.0f} 手")

            if 'volume_ratio' in row:
                print(f"成交量比率: {row['volume_ratio']:.2f}x (当前/历史平均)")
            if 'price_ratio' in row:
                print(f"价格比率: {row['price_ratio']:.2f}x (当前/历史最低)")
            if 'is_downtrend' in row:
                print(f"下跌趋势: {'是' if row['is_downtrend'] else '否'}")

            # 详细分析
            analysis = bottom_filter.analyze_stock(stock_code)
            if 'error' not in analysis:
                print(f"成交量比率(5/10/20天): {analysis['volume_ratios']['5_day']:.2f}x / "
                      f"{analysis['volume_ratios']['10_day']:.2f}x / "
                      f"{analysis['volume_ratios']['20_day']:.2f}x")
                print(f"价格比率(30/60/100天): {analysis['price_ratios']['30_day']:.2f}x / "
                      f"{analysis['price_ratios']['60_day']:.2f}x / "
                      f"{analysis['price_ratios']['100_day']:.2f}x")
            print("-" * 40)

    else:
        print("\n未找到符合底部放量上涨条件的股票")

    # 8. 清理缓存
    print("\n7. 清理旧缓存文件...")
    deleted_count = data_fetcher.clear_cache(days_old=7)
    print(f"清理了 {deleted_count} 个7天前的缓存文件")

    print("\n" + "=" * 60)
    print("程序执行完成")
    print("=" * 60)


def test_single_stock():
    """测试单只股票分析"""
    print("\n测试单只股票分析功能")
    print("-" * 40)

    # 创建数据获取器和过滤器
    data_fetcher = StockDataFetcher()
    bottom_filter = BottomVolumeFilter(data_fetcher=data_fetcher)

    # 测试股票代码（示例：贵州茅台）
    test_stock = "600519"

    print(f"分析股票: {test_stock}")
    result = bottom_filter.analyze_stock(test_stock)

    if 'error' in result:
        print(f"分析失败: {result['error']}")
    else:
        print(f"股票名称: {result.get('stock_name', '未知')}")
        print(f"当前价格: {result.get('current_price', 'N/A')} 元")
        print(f"成交量比率(10天): {result.get('volume_ratios', {}).get('10_day', 0):.2f}x")
        print(f"价格比率(60天): {result.get('price_ratios', {}).get('60_day', 0):.2f}x")
        print(f"是否下跌趋势: {'是' if result.get('is_downtrend') else '否'}")
        print(f"是否符合成交量条件(1.8x): {result.get('meets_volume_condition_1.8x', False)}")
        print(f"是否符合价格条件(1.2x): {result.get('meets_price_condition_1.2x', False)}")


if __name__ == "__main__":
    import pandas as pd

    try:
        # 运行主筛选程序
        main()

        # 可选：运行单只股票测试
        # test_single_stock()

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        print(f"\n程序执行出错: {e}")
