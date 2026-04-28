"""
股票筛选系统主程序
使用重构后的模块进行股票筛选
"""
import json
import logging
import math
import os
import sys
import urllib.request
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from filter_config import FilterConfig
from bottom_volume_filter import BottomVolumeFilter
from stock_data_fetcher import StockDataFetcher

# 加载 .env 文件（必须在读取环境变量之前）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 配置日志：同时输出到控制台和文件
LOG_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {LOG_FILE}")

# 企业微信 Webhook URL
WECHAT_WEBHOOK_URL = os.environ.get(
    "WECHAT_WEBHOOK_URL",
    "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=df7d16a8-ba60-4f3b-a827-1d6669a21cdd"
)


def _df_to_json_safe(df: pd.DataFrame) -> str:
    """
    将 DataFrame 转为 JSON 字符串，NaN/NaT/Inf 转为 null。
    """
    if df is None or df.empty:
        return json.dumps(None, ensure_ascii=False)
    # 将 NaN/NaT/inf 替换为 None（JSON 中输出为 null）
    cleaned = df.where(pd.notna(df), None).where(~df.isin([float('inf'), float('-inf')]), None)
    return json.dumps(cleaned.to_dict(orient='records'), ensure_ascii=False, default=str)


def _load_prompt() -> str:
    """加载 AI 评分 prompt"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _build_ai_input(
    row_dict: Dict[str, Any],
    basic_info: Optional[Dict[str, Any]] = None,
    financial_info: Optional[Dict[str, Any]] = None,
    distrubution: Optional[Dict[str, Any]] = None,
    stock_news: Optional[Dict[str, Any]] = None,
) -> str:
    """将多维度数据组装成 AI 分析的输入文本"""
    parts = []

    parts.append("=== row_dict（技术面筛选结果）===")
    parts.append(json.dumps(row_dict, ensure_ascii=False, default=str))

    if basic_info:
        parts.append("\n=== basic_info（基本面综合信息）===")
        parts.append(json.dumps(basic_info, ensure_ascii=False, default=str))

    if financial_info:
        parts.append("\n=== financial_info（深度财务信息）===")
        parts.append(json.dumps(financial_info, ensure_ascii=False, default=str))

    if distrubution:
        parts.append("\n=== distrubution（筹码分布）===")
        parts.append(json.dumps(distrubution, ensure_ascii=False, default=str))

    if stock_news:
        parts.append("\n=== stock_news（股票新闻）===")
        parts.append(json.dumps(stock_news, ensure_ascii=False, default=str))

    return "\n".join(parts)


def _call_llm(system_prompt: str, user_message: str) -> str:
    """调用 litellm + deepseek 模型进行 AI 分析"""
    from litellm import completion

    response = completion(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_ai_result(ai_result: str, stock_code: str, stock_name: str) -> Dict[str, Any]:
    """解析 AI 返回的 JSON 结果，提取关键字段"""
    # 尝试从 markdown 代码块中提取 JSON
    import re
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', ai_result)
    json_str = json_match.group(1) if json_match else ai_result

    try:
        result = json.loads(json_str.strip())
    except json.JSONDecodeError:
        # 如果解析失败，返回原始文本
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "score": None,
            "level": "未知",
            "summary": ai_result.strip(),
        }

    return {
        "stock_code": result.get("stock_code", stock_code),
        "stock_name": result.get("stock_name", stock_name),
        "score": result.get("score"),
        "level": result.get("level", "未知"),
        "summary": result.get("summary", ""),
        "reason_technical": result.get("reason_technical", ""),
        "reason_fundamental": result.get("reason_fundamental", ""),
        "reason_valuation": result.get("reason_valuation", ""),
        "reason_distribution": result.get("reason_distribution", ""),
        "reason_news": result.get("reason_news", ""),
    }


def _send_wechat_webhook(results: List[Dict[str, Any]]) -> bool:
    """将 AI 分析结果汇总推送到企业微信（text 类型，兼容个人微信）"""
    if not results:
        return False

    # 按评分排序（高分在前）
    sorted_results = sorted(
        [r for r in results if r.get("score") is not None],
        key=lambda r: r["score"],
        reverse=True,
    )
    # 评分失败的放最后
    failed = [r for r in results if r.get("score") is None]
    sorted_results.extend(failed)

    # 构建消息内容（纯文本，兼容个人微信）
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"📊 股票AI分析日报 ({today})"]
    lines.append("=" * 30)

    for r in sorted_results:
        score = r.get("score")
        level = r.get("level", "未知")
        summary = r.get("summary", "")

        if score is not None:
            if score >= 80:
                emoji = "🟢"
            elif score >= 60:
                emoji = "🔵"
            elif score >= 40:
                emoji = "🟡"
            else:
                emoji = "🔴"
            score_str = f"{emoji} {r['stock_name']}({r['stock_code']})  评分:{score} ({level})"
        else:
            score_str = f"⚪ {r['stock_name']}({r['stock_code']})  评分失败"

        lines.append("")
        lines.append(score_str)
        if summary:
            lines.append(f"  {summary}")

    lines.append("")
    lines.append("=" * 30)
    content = "\n".join(lines)

    # 企业微信消息体（text类型，兼容个人微信转发）
    payload = json.dumps({
        "msgtype": "text",
        "text": {
            "content": content,
        }
    }, ensure_ascii=False).encode("utf-8")

    try:
        req = urllib.request.Request(
            WECHAT_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = resp.read().decode("utf-8")
            result = json.loads(resp_body)
            if result.get("errcode") == 0:
                logger.info("企业微信推送成功")
                return True
            else:
                logger.error(f"企业微信推送失败: {resp_body}")
                return False
    except Exception as e:
        logger.error(f"企业微信推送异常: {e}")
        return False


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

    if filtered_stocks.empty:
        print("无符合条件的股票，程序退出")
        return

    # 5. 多维度数据获取 & 6. AI 分析
    print(f"\n筛选出 {len(filtered_stocks)} 只股票，开始获取多维度数据并 AI 分析...\n")

    # 加载 AI prompt
    system_prompt = _load_prompt()

    # 收集所有 AI 分析结果
    all_results: List[Dict[str, Any]] = []

    for index, row in filtered_stocks.iterrows():
        stock_code = str(row["代码"])
        stock_name = str(row.get("名称", ""))
        row_dict = row.to_dict()

        print(f"\n{'='*60}")
        print(f"AI 分析: {stock_name}({stock_code})")
        print(f"{'='*60}")

        # 获取多维度数据
        print("  获取基本面信息...")
        basic_info_list = data_fetcher.get_stock_basic_info(stock_code)
        basic_info = {
            api_name: json.loads(_df_to_json_safe(df))
            for api_name, df in basic_info_list
        } if basic_info_list else None

        print("  获取财务信息...")
        financial_info_list = data_fetcher.get_stock_financial_info(stock_code)
        financial_info = {
            api_name: json.loads(_df_to_json_safe(df))
            for api_name, df in financial_info_list
        } if financial_info_list else None

        print("  获取筹码分布...")
        distrubution_df = data_fetcher.get_stock_distrubution(stock_code)
        distrubution = json.loads(_df_to_json_safe(distrubution_df)) if distrubution_df is not None else None

        print("  获取新闻...")
        news_df = data_fetcher.get_stock_news(stock_code)
        stock_news = json.loads(_df_to_json_safe(news_df)) if news_df is not None else None

        # 组装 AI 输入
        user_message = _build_ai_input(
            row_dict=row_dict,
            basic_info=basic_info,
            financial_info=financial_info,
            distrubution=distrubution,
            stock_news=stock_news,
        )

        # 调用 LLM
        print("  调用 AI 分析中...")
        try:
            ai_result = _call_llm(system_prompt, user_message)
            print(f"\n  AI 评分结果:\n{ai_result}\n")

            # 解析结果
            parsed = _parse_ai_result(ai_result, stock_code, stock_name)
            all_results.append(parsed)

        except Exception as e:
            logger.error(f"AI 分析 {stock_code} 失败: {e}")
            print(f"  AI 分析失败: {e}\n")
            all_results.append({
                "stock_code": stock_code,
                "stock_name": stock_name,
                "score": None,
                "level": "失败",
                "summary": f"AI 分析异常: {e}",
            })

    # 7. 推送到企业微信
    print("\n" + "=" * 60)
    print("推送汇总结果到企业微信...")
    if all_results:
        success = _send_wechat_webhook(all_results)
        if success:
            print("✅ 企业微信推送成功")
        else:
            print("❌ 企业微信推送失败，请查看日志")
    else:
        print("无分析结果，跳过推送")


if __name__ == "__main__":
    # 运行主筛选程序
    main()
