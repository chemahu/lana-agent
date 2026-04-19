# scrapers/binance_square

币安广场（Binance Square）社交数据爬虫子模块。

## 模块概览

| 文件 | 说明 |
|------|------|
| `endpoints.py` | 所有已知的公开 API 端点（逆向自 Web 端） |
| `client.py` | HTTP 客户端（带重试、限速、代理轮换） |
| `parser.py` | 原始 JSON → 标准化帖子记录 |
| `bot_filter.py` | 机器人 / 刷帖账号识别与过滤 |
| `aggregator.py` | 高层聚合器，供 `DataFetcher` 调用 |
| `storage.py` | 帖子持久化（SQLite） |
| `scheduler.py` | 后台定时爬取调度器 |

## 快速开始

```python
from scrapers.binance_square import SquareAggregator

agg = SquareAggregator()
features = agg.features_for("BTC/USDT:USDT")
print(features)
# {
#   "posts_1h": 12,
#   "posts_24h": 87,
#   "posts_growth_rate": 0.45,
#   "unique_authors": 10,
#   "bullish_tag_ratio": 0.72,
#   "kol_mentioned": False,
#   "trade_widget_count": 3,
# }
```

## 独立调度爬取

```python
from scrapers.binance_square.scheduler import SquareScheduler

scheduler = SquareScheduler(symbols=["BTC", "ETH", "SOL"], interval_sec=300)
scheduler.start()   # 后台线程，每 5 分钟抓一次
# ...
scheduler.stop()
```

## 注意事项

- 所有接口均为非官方逆向接口，Binance 可能随时修改路径或增加风控。
- 建议配合代理池（`SquareClient(proxies=[...])`) 避免 IP 封禁。
- 爬取频率建议不低于 5 分钟间隔，以免触发 429 限流。
- 帖子数据默认存储在 `data/binance_square.db`（SQLite）。
