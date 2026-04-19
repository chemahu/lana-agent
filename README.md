# Lana-Style AI Agent Trading System

> 复刻「拉哪」核心思路的开源实现：用 AI Agent 在山寨币合约市场中实现「少数大赚、多数小亏」的趋势跟随策略。

⚠️ **免责声明**：本项目仅用于学习和研究目的。加密货币合约交易具有极高风险，使用本代码产生的任何盈亏由使用者自行承担。强烈建议先在测试网/小资金环境下验证。

---

## 一、设计哲学

```
不预测、只跟随     ←  捕捉已经启动的趋势
刚性止损、动态止盈 ←  亏的时候亏得起，盈的时候拿得住
信息隔离评估       ←  AI 决策时不知道自己的持仓和成本，避免认知偏差
少数大赚多数小亏   ←  接受低胜率（个位数），靠长尾收益赚钱
```

## 二、系统架构

```
┌─────────────────────────────────────────────────────┐
│                   主调度器 main.py                   │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
   ┌───────▼────────┐         ┌───────▼────────┐
   │  选标漏斗       │         │ 持仓评估循环    │
   │  Scanner       │         │ Cron 15min     │
   └───────┬────────┘         └───────┬────────┘
           │                          │
   ┌───────▼────────────────────────────▼────────┐
   │  AI 评估层 (Claude + GPT 并行交叉验证)          │
   │  - 入场判断（evaluate_entry）                │
   │  - 持仓"继续持有/减仓/全平"（evaluate_hold） │
   │  - 两模型均可用时并行调用、保守合并结果       │
   │  - 无 LLM 时降级为规则引擎兜底               │
   └───────┬────────────────────────┬────────────┘
           │                        │
   ┌───────▼────────┐     ┌─────────▼────────┐
   │  下单 + 止损    │     │ 分批止盈/平仓     │
   │  Position Mgr  │     │ TrailingEvaluator │
   └────────────────┘     └──────────────────┘
```

## 三、详细交易策略（逻辑闭合）

### 3.1 选标漏斗（三层，全部为硬过滤）

| 层级 | 数据源 | 通过条件 |
|---|---|---|
| **价格层** | 涨幅榜 + K线 | 24h 涨幅前 20（量 > 100 万 U），且 1h 涨幅 > 3% |
| **资金层** | 合约 OI | 4h OI 增长 ≥ 15%，且价格涨幅 < OI 涨幅 × 1.5（资金未充分反应） |
| **舆论层** | 币安广场 | 1h 帖子数 ≥ 50 且独立作者 ≥ 30 且 1h 增长 ≥ 100% |

> **注**：三层均为硬过滤，任意一层不通过则该标的被跳过。

### 3.2 入场规则

- 触发条件：候选池标的 + AI 评估 `should_enter=true`，且 `p_up ≥ 0.6`，且 `confidence ≥ 0.65`
- 仓位规模：每笔最大风险敞口 = min(账户净值 × 1%, 200 USDT)；新币最大亏损翻倍（×2）
- 杠杆：默认 5×，新币（上市 ≤ 14 天）3×
- 方向：**只做多，不做空**
- 最多同时持仓 5 个标的

### 3.3 止损规则（刚性）

```python
单笔最大亏损 = min(账户净值 × 1%, 200 USDT)
止损幅度 = 2%（新币 4%）          # 合约价格相对入场价的下跌幅度，与杠杆无关
止损价 = 入场价 × (1 - 止损幅度)
```

> **止损幅度说明（价格幅度 vs 账户亏损）**
>
> 止损幅度 **2%（新币 4%）** 指合约市场价格相对入场价的跌幅，与杠杆倍数无关。
> 例：入场价 100 USDT，止损触发价 = 100 × (1 - 2%) = **98 USDT**（价格跌 2% 触发）。
>
> 账户实际最大亏损金额受 `max_loss`（账户净值 × 1%，上限 200 USDT）控制，
> 系统通过反推仓位大小（`notional = max_loss / stop_pct`）来保证：
> 止损触发时账户亏损 ≈ `max_loss`，而非"账户净值 × 杠杆 × 2%"。
>
> 以 5× 杠杆为例：止损被打时保证金损失 ≈ `margin × 10%`（= `margin × stop_pct × leverage`），
> 但绝对金额上限由仓位 sizing 保证 ≤ `max_loss`。

开仓后立即挂 `stop_market reduceOnly` 止损单。平仓（全仓或减仓）后自动撤销或按剩余仓位重新挂止损单，避免止损单残留误触发。

### 3.4 持仓追踪规则（动态，Cron 每 15 分钟）

每轮遍历所有持仓，按以下优先级处理：

1. **黑天鹅检测**（最高优先）：若 1 分钟内最低价 / 最高价 ≤ -15%，立即全平并推送告警
2. **高盈利止盈**：若无杠杆 ROI ≥ 100%，立即减仓 50%
3. **AI 决策**：调用 LLM（Claude 优先，否则 OpenAI，否则规则引擎）返回：
   - `action = "close"` → 全部平仓
   - `action = "scale_out"` + `scale_out_pct`（限制在 10%–90%）→ 按比例减仓
   - `action = "hold"` → 继续持有

### 3.5 AI 评估模型（双模型交叉验证）

- **双模型并行**：同时配置 `ANTHROPIC_API_KEY` 和 `OPENAI_API_KEY` 时，Claude 与 GPT 通过 `ThreadPoolExecutor` **并行**调用，对同一"市场快照"各自独立判断，结果保守合并：
  - **入场**：两个模型均返回 `should_enter=true` 才最终入场（AND 逻辑），避免单模型幻觉误触发；`p_up` 与 `confidence` 取均值。
  - **持仓**：取两个模型操作中更保守（close > scale_out > hold）的那一侧；`scale_out_pct` 取均值。
- **单模型降级**：只配置其中一个 API Key 则使用该单模型，行为与之前一致。
- **规则引擎兜底**：两者均未配置时使用内置动量规则（基于 momentum、OI、KOL 提及、成交量计算 p_up）。

### 3.6 市场快照（5 个维度）

AI 决策的输入为标准化的 5 维市场快照，每轮评估前实时抓取：

| 维度 | 字段 | 说明 |
|---|---|---|
| **price** | current_price, change_1h/4h/24h, distance_from_24h_high, consecutive_green_candles, upper_wick_ratio_avg | 价格动量与 K 线形态 |
| **derivatives** | oi_change_1h/4h, funding_rate, current_oi_usd | 合约持仓量与资金费率 |
| **volume** | volume_24h_usdt, avg_hourly_volume_usdt, volume_change_1h, buy_volume_ratio | 成交量强度与多空比 |
| **social** | posts_1h/24h, posts_growth_rate, unique_authors, bullish_tag_ratio, kol_mentioned, trade_widget_count | 币安广场舆论热度 |
| **relative** | rank_in_gainers, gainers_count | 标的在涨幅榜中的相对排名 |

### 3.6 黑天鹅兜底

| 触发条件 | 操作 |
|---|---|
| 5 分钟内价格闪跌 ≥ 15% | 立即全平，推送 Telegram 告警 |
| （计划）交易所 API 异常 > 3 分钟 | 暂未实现 |

## 四、目录结构

```
lana-agent/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── core/
│   ├── data_fetcher.py
│   ├── scanner.py
│   ├── ai_evaluator.py
│   ├── position_manager.py
│   ├── risk_manager.py
│   └── trailing.py
├── scrapers/
│   └── binance_square/   ← 币安广场爬虫（已实现）
└── utils/
    └── notifier.py
```

## 五、快速开始

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 API Key
python main.py --testnet --dry-run  # 测试
python main.py --capital 100        # 实盘
```

## 六、环境变量 (.env)

```
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
ANTHROPIC_API_KEY=xxx
OPENAI_API_KEY=xxx
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

## 七、已知限制 / 待实现

| 功能 | 状态 | 说明 |
|---|---|---|
| 双模型交叉验证 | ✅ 已实现 | Claude + GPT 并行调用，保守合并结果 |
| 5 维市场快照 | ✅ 已实现 | price / derivatives / volume / social / relative |
| 模糊区加快评估 (5min) | ❌ 未实现 | `AMBIGUOUS_INTERVAL_MINUTES` 已定义但未启用 |
| 高市值币 10× 杠杆 | ❌ 未实现 | 当前仅区分新币(3×)/默认(5×) |
| API 异常超时兜底 | ❌ 未实现 | `API_FAILURE_TIMEOUT_SEC` 已定义但无对应监控逻辑 |
| 策略自适应进化 | ❌ 未实现 | AI 决策为无状态调用，不存在历史学习机制 |
| Hyperliquid 聪明钱 | ❌ 未实现 | 占位符，需自行接入 |

## License

MIT

---