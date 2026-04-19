"""币安广场已知公开接口（逆向自 web 端，可能随时失效，需定期校对）。

所有路径都是 POST + JSON body，不需 KYC，但需要带送浏览器类似的 headers，
且会被 Cloudflare/Akamai 识别。建议配合代理池使用。
"""

BASE = "https://www.binance.com"

ENDPOINTS = {
    # 广场首页推荐信息流
    "feed_list": f"{BASE}/bapi/composite/v1/public/content/community/square/feed/list",
    # 话题流（用于按 symbol/hashtag 拉取）
    "topic_feed": f"{BASE}/bapi/composite/v1/public/content/community/square/topic/feed/list",
    # 全文搜索（用 symbol 作为关键词）
    "search": f"{BASE}/bapi/composite/v1/public/content/community/search",
    # 单帖详情（含点赞/评论数）
    "feed_detail": f"{BASE}/bapi/composite/v1/public/content/community/square/feed/detail",
    # 热门话题榜
    "hot_topics": f"{BASE}/bapi/composite/v1/public/content/community/square/hot/topic/list",
    # 用户信息（判断是否改过名）
    "user_profile": f"{BASE}/bapi/composite/v1/public/content/community/profile",
}