"""
DeepSeek 调参引擎
=================

用 DeepSeek(OpenAI 兼容接口)做参数提案:
    每轮: 把搜索空间、数据集切分、历史提案与训练集指标发给 DeepSeek,
          返回一个 JSON 提案 -> 系统评估 -> 追加历史 -> 下一轮。

API: POST {base_url}/chat/completions, model 默认 deepseek-chat。
密钥优先从环境变量 DEEPSEEK_API_KEY 读取(可在项目根目录 .env 中配置),
其次回退到 config/llm_config.json, 不硬编码在代码里。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from qtcore.dotenv import load_dotenv


SYSTEM_PROMPT = """你是资深量化研究员, 正在为 A 股"双均线交叉 + RSI 过滤"趋势策略做参数调优。

【数据划分】训练集与测试集年份不重叠, 且各自都包含牛市/熊市/横盘年份。你只能依据训练集指标做决策, 严禁推测或使用测试集信息。

【可调参数与范围】(JSON 输出)
{
  "fast": 3~60(整数, 快线窗口),
  "slow": 10~120(整数, 必须大于 fast),
  "timeframe": "daily"/"60min"/"2h"/"4h"/"6h"(K线周期: 日线/60分钟/2小时/4小时/6小时; A股一天4小时, 4h=一整个交易时段),
  "use_rsi": true/false,
  "rsi_window": 5~30(整数),
  "rsi_buy": 20~45(浮点, RSI 低于此值才做多),
  "rsi_sell": 60~85(浮点, RSI 高于此值离场),
  "top_k": 3~10(整数, 选股持仓数量),
  "select_metric": "sharpe"或"total_return"或"profit_factor"(选股排序指标),
  "position_ratio": 0.1~1.0(单次建仓资金比例),
  "max_position_ratio": 0.2~1.0(单标的持仓上限, 控制集中度),
  "rebalance": "daily"/"weekly"/"monthly"(调仓周期),
  "order_type": "market"/"limit"(订单类型),
  "slippage_tolerance_pct": 0.0~0.01(滑点容忍, 0=不限制),
  "leverage": 1.0~2.0(杠杆),
  "stop_loss_pct": 0.0~0.2(单笔止损, 0=关闭),
  "take_profit_pct": 0.0~0.5(单笔止盈, 0=关闭),
  "max_drawdown_halt": 0.0~0.3(账户回撤熔断, 0=关闭),
  "halt_cooldown_days": 0~20(整数, 熔断后强制空仓的交易日数),
  "halt_resume_drawdown": 0.0~0.2(熔断恢复阈值: 回撤回到该比例以内才允许重新建仓)
}

【评估口径】每轮给你: 提案 JSON -> 训练集组合绩效(等权 Top-K: 总收益/年化/夏普/回撤/胜率/盈亏比/覆盖率)。
注意: 胜率低但盈亏比高是趋势策略的正常形态; 优先提升夏普与降低回撤, 不要只看收益;
参数要稳健(相邻参数也合理), 不要为拟合历史而选取极端值。

【优化目标 - 硬性要求】训练集年化收益 >= 6% 是硬性门槛, 达不到即判无效(即使夏普很高)。
在满足年化收益的前提下, 再最大化夏普并控制回撤。特别警告:
- 过度保守(几乎不持仓、资金利用率过低, 例如 position_ratio 过小导致年化远低于门槛)会被直接淘汰;
- 过度激进(高杠杆且无风控)同样危险; 止损/止盈/回撤熔断是加分项, 但不应过度到错过行情。
请给出能真正赚钱且稳健的参数, 而不是"高夏普的躺平组合"。

【关键调优建议 - 必须遵守】
- 默认从活跃基线出发: use_rsi=false, sampling=daily, rebalance=daily,
- 默认从活跃基线出发: use_rsi=false, timeframe=daily, rebalance=daily,
  position_ratio>=0.8, fast 3~10, slow 20~40(日内周期 fast/slow 可相应缩小);
- 训练期交易次数 <20 笔通常意味着过度保守, 会被直接判无效;
- 止损/止盈/回撤熔断是加分项, 但设置过紧会错过行情:
  建议止损 0.08~0.15、止盈 0.15~0.4 范围内探索, 不要同时把三者都开得很激进;
- RSI 过滤会大幅降低交易频率: 若启用, 建议 rsi_buy 靠近 40~45、rsi_sell 靠近 55~60,
  或干脆 use_rsi=false。

【输出要求】只输出一个合法 JSON 对象, 不要包含任何解释文字或 markdown 代码块。
"""


def parse_json_content(content: str) -> dict[str, Any]:
    """从模型回复中提取 JSON(兼容代码块/前后缀文本)。"""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"回复中无 JSON: {content[:200]}")
    return json.loads(text[start : end + 1])


class DeepSeekTuner:
    """DeepSeek 参数提案器。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def propose(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """给定历史(提案+评估结果), 请求 DeepSeek 返回下一个提案。"""
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    "请基于以上历史给出下一个提案 JSON。硬性要求: "
                    "不要与任何历史提案完全相同, 主动在不同参数区域探索; "
                    "若上一轮被判无效, 必须按提示切换到有效方向; "
                    "若上一轮有效但分数偏低, 尝试小幅扰动并观察方向。"
                ),
            }
        )

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.9,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_json_content(content)


def load_api_key(path: Path | None = None) -> str:
    """读取 DeepSeek 密钥: 环境变量优先, 其次配置文件。"""
    load_dotenv()
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    key_path = path or Path(__file__).resolve().parent.parent / "config" / "llm_config.json"
    if key_path.exists():
        data = json.loads(key_path.read_text(encoding="utf-8"))
        key = data.get("deepseek_api_key", "")
        if key:
            return key
    raise RuntimeError(
        f"未找到 DeepSeek API Key, 请写入 {key_path} 或设置环境变量 DEEPSEEK_API_KEY"
    )
