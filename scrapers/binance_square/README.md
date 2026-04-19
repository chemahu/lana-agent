# scrapers/binance_square

币安广场（Binance Square）社交数据抓取与分析子模块。

## 目录结构

```
scrapers/binance_square/
├── __init__.py        公开 API：SquareAggregator、SquareClient
├── client.py          带限速/重试/代理轮换的 HTTP 客户端（同步+异步双模）
├── endpoints.py       逆向自 web 端的 API 端点常量
├── parser.py          原始帖子 JSON → 标准化记录解析器
├── bot_filter.py      启发式 bot 过滤器
├── storage.py         SQLite 去重存储（WAL 模式，线程安全）
├── aggregator.py      高层聚合器：帖子列表 → 社交热度特征向量
└── scheduler.py       常驻后台抓取进程（2 分钟轮询）
```

## 快速使用

### 获取某 symbol 的社交特征（供 DataFetcher 调用）

```python
from scrapers.binance_square import SquareAggregator

agg = SquareAggregator()
features = agg.features_for("BTC/USDT")
# {
#   "posts_1h": 120,
#   "posts_24h": 1800,
#   "posts_growth_rate": 2.5,
#   "unique_authors": 87,
#   "bullish_tag_ratio": 0.71,
#   "kol_mentioned": True,
#   "trade_widget_count": 4,
# }
```

### 直接使用 HTTP 客户端

```python
from scrapers.binance_square.client import SquareClient

with SquareClient() as client:
    data = client.search(keyword="ETH", page=1, page_size=20)
    hot  = client.fetch_hot_topics()
```

### 启动常驻抓取进程

```bash
# 作为独立进程（每 2 分钟拉一次，写入 data/square_posts.db）
python -m scrapers.binance_square.scheduler
```

或在程序内嵌入：

```python
from scrapers.binance_square.scheduler import SquareScheduler

sched = SquareScheduler(interval_sec=120)
sched.start()          # 后台 daemon 线程
# ... 主程序运行 ...
sched.stop()
```

## 模块说明

### client.py — `SquareClient`

- 令牌桶限速（默认 1 req/s，突发 5）
- 可配置代理列表，循环轮换
- 同步方法：`fetch_feed_list`, `fetch_topic_feed`, `search`, `fetch_hot_topics`, `fetch_user_profile`
- 对应异步版本：`async_fetch_feed_list` 等

### parser.py — `parse_post(raw)`

将 API 返回的单帖原始 JSON 解析为标准化字段：

| 字段 | 说明 |
|------|------|
| `id` | 帖子唯一 ID |
| `author_id` | 作者 UID |
| `author_is_kol` | 是否 KOL（需在 `parser.KOL_UIDS` 中维护） |
| `sentiment` | `bullish` / `bearish` / `neutral` |
| `symbols` | 提取出的代币符号列表 |
| `created_at_ms` | UTC 毫秒时间戳 |
| `has_trade_widget` | 是否附带交易挂件 |

### bot_filter.py — `BotFilter`

启发式过滤规则（满足一条即丢弃）：
1. 默认用户名 + 零互动
2. 内容完全重复超过 3 次
3. 内容极短（< 5 字符）且零互动

### storage.py — `PostStorage`

- SQLite WAL 模式，线程安全
- `save_post(post, symbol)` / `save_posts(posts, symbol)` — 去重写入
- `get_posts(symbol, since_ms, limit)` — 查询
- `prune_old(older_than_ms)` — 清理旧数据

### aggregator.py — `SquareAggregator`

对外暴露 `features_for(symbol: str) -> Dict`，返回 `DataFetcher.get_social_features` 所需的 7 个特征字段。

### scheduler.py — `SquareScheduler`

- 后台线程，默认每 120 秒抓取一轮广场 feed
- 自动调用 `BotFilter` + `PostStorage`
- 每 30 个周期清理 48 小时前的旧数据

## 注意事项

- API 端点来源于对 web 端的逆向分析，**可能随时失效**，需定期校对 `endpoints.py`。
- 建议配合代理池（`SquareClient(proxies=[...])`)）以规避 Cloudflare / Akamai 封锁。
- `parser.KOL_UIDS` 默认为空集合，需根据实际需求填入已知 KOL 的 UID。
