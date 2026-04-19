# scrapers/binance_square

Binance Square 舆情爬虫子模块，负责从币安广场抓取帖子并提供下游所需的社会热度特征向量。

## 模块结构

```
scrapers/binance_square/
├── __init__.py        # 导出 SquareAggregator / SquareClient
├── endpoints.py       # API 端点常量
├── client.py          # 同步 + 异步 HTTP 客户端（含限速 / 重试 / 代理）
├── parser.py          # 帖子解析 + symbol 提取 + 情绪识别
├── bot_filter.py      # 启发式 bot / 水军过滤
├── storage.py         # SQLite 去重增量存储
├── aggregator.py      # 高层聚合器（唯一对外入口 SquareAggregator）
└── scheduler.py       # 常驻 2min 拉取进程
```

## 快速开始

### 获取某代币的社交特征

```python
from scrapers.binance_square import SquareAggregator

agg = SquareAggregator()
features = agg.features_for("BTC")
# {'posts_1h': 42, 'posts_24h': 380, 'posts_growth_rate': 0.8,
#  'unique_authors': 31, 'bullish_tag_ratio': 0.65,
#  'kol_mentioned': True, 'trade_widget_count': 5}
```

### 启动常驻后台调度器

```bash
# 命令行
python -m scrapers.binance_square.scheduler --symbols BTC ETH SOL --interval 2

# 代码中后台线程
from scrapers.binance_square.scheduler import SquareScheduler
scheduler = SquareScheduler(symbols=["BTC", "ETH", "SOL"])
scheduler.start()   # 非阻塞
```

### 直接操作 SQLite 存储

```python
from scrapers.binance_square.storage import PostStorage

with PostStorage() as db:
    db.save_posts(posts, symbol="BTC")
    recent = db.get_posts("BTC", since_ms=..., limit=200)
    db.prune_old(keep_days=7)
```

## 各模块说明

### `client.py` — SquareClient

- 同步（`search / fetch_feed_list / fetch_topic_feed / fetch_user_profile / fetch_hot_topics`）+ 异步（`async_*`）双模 API。
- 令牌桶限速（默认 1 req/s，burst 5）。
- 指数退避重试，最多 5 次，对 429 / 418 / 403 自动重试。
- 支持代理列表轮换。

### `parser.py` — parse_post

将原始 JSON 解析为标准化记录，包含：
- `id`, `author_id`, `author_nickname`, `author_is_default`, `author_is_kol`
- `content`, `created_at` (ISO-8601), `created_at_ms`
- `likes`, `comments`, `shares`, `views`
- `symbols`（从内容提取的代币列表）
- `sentiment`（`bullish` / `bearish` / `neutral`）
- `has_trade_widget`

### `bot_filter.py` — filter_posts / is_bot_post

启发式规则：
1. 默认用户名 + 零互动
2. 内容长度 < 5 字符
3. 重复 token 占比 ≥ 60%
4. 去掉 URL/hashtag 后无实质文字
5. 同一作者相邻帖子间隔 < 3 秒

### `storage.py` — PostStorage

SQLite 单表 `posts`：
- `INSERT OR IGNORE` 天然去重。
- `get_posts(symbol, since_ms)` 增量查询。
- `prune_old(keep_days)` 定期清理。

### `aggregator.py` — SquareAggregator

对外暴露 `features_for(symbol) -> dict`，汇总 `DataFetcher` 所需的七项社会热度指标。

### `scheduler.py` — SquareScheduler

- 默认每 2 分钟抓取一轮，支持 `start()` 后台线程 / `run()` 阻塞运行。
- 每次写入前经 `bot_filter` 过滤，再落库。
- 每轮循环后自动调用 `prune_old(7)` 清理过期数据。

## 环境依赖

见仓库根目录 `requirements.txt`，主要：

- `httpx>=0.27` — HTTP 客户端
- `loguru` — 日志

## 注意事项

- 所有端点均为**公开无鉴权** POST 接口，逆向自 web 端，**可能随 Binance 更新失效**，需定期校对 `endpoints.py`。
- Cloudflare/Akamai 防护下直连可能被限速；生产环境建议配合代理池使用（`SquareClient(proxies=[...])`）。
- SQLite 存储不支持多进程并发写入，如需横向扩展请替换为 PostgreSQL 或 Redis。
