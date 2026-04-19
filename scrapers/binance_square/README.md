# scrapers/binance_square

币安广场（Binance Square）社交数据爬虫子包。

## 模块结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包入口，导出 `SquareAggregator` 和 `SquareClient` |
| `endpoints.py` | 所有已知公开 API 路径常量 |
| `client.py` | 带限速/重试/代理轮换的同步+异步 HTTP 客户端 |
| `parser.py` | 原始帖子 JSON → 标准化记录（symbol 提取、情绪识别） |
| `bot_filter.py` | Lana 启发式 Bot 过滤（默认昵称、高频刷屏、模板匹配） |
| `storage.py` | SQLite 去重增量存储（`PostStorage`） |
| `aggregator.py` | 高层聚合器 `SquareAggregator`：帖子 → 社会热度特征向量 |
| `scheduler.py` | 常驻 2 min 拉取进程 `SquareScheduler` |

## 快速开始

```python
from scrapers.binance_square import SquareAggregator

agg = SquareAggregator()
features = agg.features_for("BTC")
# {'posts_1h': 42, 'posts_24h': 380, 'posts_growth_rate': 1.3,
#  'unique_authors': 38, 'bullish_tag_ratio': 0.72,
#  'kol_mentioned': True, 'trade_widget_count': 5}
```

## 常驻进程

```bash
python -m scrapers.binance_square.scheduler --symbols BTC ETH SOL --interval 120
```

## 数据流

```
SquareClient (HTTP)
       │
       ▼
   parser.parse_post()   ──►  BotFilter.filter()
                                      │
                                      ▼
                              PostStorage.bulk_insert()
                                      │
                         ─────────────────────────────
                         │
                         ▼
               SquareAggregator.features_for()
                         │
                         ▼
               DataFetcher.get_social_features()
```

## 接口说明

### SquareAggregator

```python
class SquareAggregator:
    def features_for(self, symbol: str) -> dict:
        """返回社会热度特征向量，供 DataFetcher 调用。"""
```

返回字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `posts_1h` | int | 最近 1 小时帖子数 |
| `posts_24h` | int | 最近 24 小时帖子数 |
| `posts_growth_rate` | float | 近 1h 相对 24h 基线的增速 |
| `unique_authors` | int | 近 1h 不重复作者数 |
| `bullish_tag_ratio` | float | 看涨情绪占比 `[0, 1]` |
| `kol_mentioned` | bool | 是否有 KOL 提及 |
| `trade_widget_count` | int | 含交易挂件的帖子数 |

### BotFilter

```python
bf = BotFilter()
clean, bots = bf.filter(posts)
```

过滤规则：
1. 默认昵称（`User-xxxxxxxx` 格式）
2. 内容过短（< 10 字符）
3. 命中已知 bot 模板（广告/信号群/推广链接等）
4. 单作者在 1 小时内连续发帖超过 10 条

### PostStorage

```python
storage = PostStorage()           # 默认存储路径 data/square_posts.db
new_count = storage.bulk_insert(posts)
recent = storage.query_recent(hours=1)
by_sym = storage.query_by_symbol("ETH", hours=4)
```

## 注意事项

- 所有接口均为 POST + JSON，无需身份验证，但需模拟浏览器 Headers。
- Binance 会对高频请求返回 `429`/`403`，`SquareClient` 内置令牌桶限速与指数退避重试。
- 代理池通过 `SquareClient(proxies=["http://..."])` 注入。
- `endpoints.py` 中的路径可能随 Binance 前端版本更新而失效，需定期核对。
