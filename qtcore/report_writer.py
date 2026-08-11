"""
DeepSeek 报告生成
=================
日报/周报/月报/突发报告均由 DeepSeek(deepseek-chat) 根据真实数据撰写。
API 失败时回退为结构化纯文本, 保证邮件一定能发出。
密钥: config/llm_config.json 或环境变量 DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import json
from typing import Any

import requests

from qtcore.deepseek_tuner import load_api_key


def _ask(system: str, user: str, max_tokens: int = 1200) -> str:
    """调用 DeepSeek 聊天接口, 返回文本。"""
    key = load_api_key()
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _fallback(title: str, data: dict[str, Any]) -> str:
    """API 失败时的纯文本回退(保证报告/邮件可用)。"""
    lines = [f"【{title}】(DeepSeek 生成失败, 以下为结构化摘要)"]
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def write_daily_report(data: dict[str, Any]) -> str:
    system = (
        "你是量化交易助手, 用通俗中文给非专业用户写每日交易总结。"
        "语气平和、客观, 先说结论再说细节; 避免术语堆砌; 不构成投资建议。"
    )
    user = (
        "请根据以下今日运行数据写一份约300字的日报告: 今天是否操作、"
        "操作了什么(买卖哪些、为什么)、今日盈亏、当前持仓、与大盘对比、风险提示。\n"
        + json.dumps(data, ensure_ascii=False)
    )
    try:
        return _ask(system, user)
    except Exception:
        return _fallback("每日交易总结", data)


def write_weekly_report(data: dict[str, Any]) -> str:
    system = (
        "你是量化交易助手, 用通俗中文写周度总结。"
        "先给结论(本周赚亏、是否跑赢大盘), 再分点说明本周操作、做得好的/不好的地方、下周关注。"
        "不构成投资建议。"
    )
    user = "请写一份约500字的周报告:\n" + json.dumps(data, ensure_ascii=False)
    try:
        return _ask(system, user)
    except Exception:
        return _fallback("本周交易总结", data)


def write_monthly_report(data: dict[str, Any]) -> str:
    system = (
        "你是量化交易助手, 用通俗中文写月度复盘。"
        "结构: 本月总览(收益/回撤/是否跑赢大盘) -> 本月最大盈利与最大亏损交易 -> "
        "策略表现复盘(哪些规则贡献了收益, 哪些造成了亏损) -> 下月建议。"
        "不构成投资建议。"
    )
    user = "请写一份约800字的月度报告与复盘:\n" + json.dumps(data, ensure_ascii=False)
    try:
        return _ask(system, user)
    except Exception:
        return _fallback("本月交易总结与复盘", data)


def write_incident_report(context: dict[str, Any]) -> str:
    system = (
        "你是量化交易系统的告警助手。用简洁中文说明发生了什么、可能的影响、建议的处理动作。"
        "控制在200字内, 重点突出。"
    )
    user = "请写突发情况说明:\n" + json.dumps(context, ensure_ascii=False)
    try:
        return _ask(system, user, max_tokens=500)
    except Exception:
        return _fallback("突发情况告警", context)
