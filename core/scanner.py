"""三层选标漏斗"""
from typing import List, Dict
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher


class Scanner:
    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def scan(self) -> List[Dict]:
        candidates = []
        gainers = self.fetcher.get_top_gainers(CFG.GAINERS_TOP_N)
        logger.info(f"[Scanner] {len(gainers)} gainers fetched")

        for g in gainers:
            symbol = g["symbol"]
            try:
                price_feat = self.fetcher.get_price_features(symbol)
                if price_feat["change_1h"] < CFG.MIN_PRICE_CHANGE_1H:
                    continue
            except Exception as e:
                logger.warning(f"price feat fail {symbol}: {e}")
                continue

            deriv = self.fetcher.get_derivatives_features(symbol)
            if deriv["oi_change_4h"] < CFG.MIN_OI_CHANGE_4H:
                continue
            if price_feat["change_4h"] >= deriv["oi_change_4h"] * 1.5:
                continue

            social = self.fetcher.get_social_features(symbol)
            if (social["posts_1h"] < CFG.MIN_SOCIAL_POSTS_1H
                    or social["unique_authors"] < CFG.MIN_UNIQUE_AUTHORS
                    or social["posts_growth_rate"] < CFG.MIN_POSTS_GROWTH):
                logger.debug(f"{symbol} weak social, kept for ranking")

            candidates.append({
                "symbol": symbol,
                "price_features": price_feat,
                "derivatives": deriv,
                "social": social,
                "relative": {"rank": gainers.index(g) + 1},
            })

        logger.info(f"[Scanner] {len(candidates)} candidates passed filters")
        return candidates
