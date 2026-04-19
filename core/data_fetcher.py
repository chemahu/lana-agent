"""行情、合约、舆论数据抓取"""
import ccxt
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from config import CFG
from scrapers.binance_square import SquareAggregator


class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            "apiKey": CFG.BINANCE_API_KEY,
            "secret": CFG.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if CFG.TESTNET:
            self.exchange.set_sandbox_mode(True)
        self.square = SquareAggregator()

    def get_price_features(self, symbol: str) -> Dict:
        ohlcv_1h = self.exchange.fetch_ohlcv(symbol, "1h", limit=24)
        ticker = self.exchange.fetch_ticker(symbol)
        closes = [c[4] for c in ohlcv_1h]
        highs = [c[2] for c in ohlcv_1h]
        lows = [c[3] for c in ohlcv_1h]
        opens_1h = [c[1] for c in ohlcv_1h]
        price = ticker["last"]
        consecutive_green = 0
        for i in range(len(closes) - 1, -1, -1):
            if closes[i] > opens_1h[i]:
                consecutive_green += 1
            else:
                break
        wick_ratios = []
        for i in range(-3, 0):
            body = abs(closes[i] - opens_1h[i]) or 1e-9
            upper_wick = highs[i] - max(closes[i], opens_1h[i])
            wick_ratios.append(upper_wick / body)
        return {
            "current_price": price,
            "change_1h": (price / closes[-2] - 1) if len(closes) > 1 else 0,
            "change_4h": (price / closes[-5] - 1) if len(closes) > 4 else 0,
            "change_24h": (price / closes[0] - 1),
            "distance_from_24h_high": price / max(highs) - 1,
            "distance_from_24h_low": price / min(lows) - 1,
            "consecutive_green_candles": consecutive_green,
            "upper_wick_ratio_avg": sum(wick_ratios) / len(wick_ratios),
        }

    def get_derivatives_features(self, symbol: str) -> Dict:
        try:
            mark = symbol.replace("/USDT", "USDT").replace(":USDT", "")
            funding = self.exchange.fetch_funding_rate(symbol)
            oi_history = self.exchange.fapiPublicGetOpenInterestHist({
                "symbol": mark, "period": "1h", "limit": 5
            })
            oi_now = float(oi_history[-1]["sumOpenInterest"])
            oi_1h_ago = float(oi_history[-2]["sumOpenInterest"])
            oi_4h_ago = float(oi_history[0]["sumOpenInterest"])
            return {
                "oi_change_1h": oi_now / oi_1h_ago - 1,
                "oi_change_4h": oi_now / oi_4h_ago - 1,
                "funding_rate": funding["fundingRate"],
                "current_oi_usd": oi_now,
            }
        except Exception as e:
            logger.warning(f"derivatives fetch failed for {symbol}: {e}")
            return {"oi_change_1h": 0, "oi_change_4h": 0,
                    "funding_rate": 0, "current_oi_usd": 0}

    def get_top_gainers(self, n: int = 20) -> List[Dict]:
        tickers = self.exchange.fetch_tickers()
        usdt_perps = [
            {"symbol": s, "change_24h": t.get("percentage", 0) or 0,
             "volume": t.get("quoteVolume", 0) or 0}
            for s, t in tickers.items() if s.endswith(":USDT")
        ]
        usdt_perps.sort(key=lambda x: x["change_24h"], reverse=True)
        return [x for x in usdt_perps if x["volume"] > 1_000_000][:n]

    def get_social_features(self, symbol: str) -> Dict:
        try:
            return self.square.features_for(symbol)
        except Exception as e:
            logger.warning(f"square features fail {symbol}: {e}")
            return {"posts_1h": 0, "posts_24h": 0, "posts_growth_rate": 0,
                    "unique_authors": 0, "bullish_tag_ratio": 0.5,
                    "kol_mentioned": False, "trade_widget_count": 0}

    def get_volume_features(self, symbol: str) -> Dict:
        """获取成交量特征（市场快照第5维）。"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1h", limit=25)
            volumes = [c[5] for c in ohlcv]
            if len(volumes) < 2:
                raise ValueError("insufficient OHLCV data for volume features")
            last_vol = volumes[-1]
            prev_vol = volumes[-2]
            # 用除最新 K 线外的所有历史 K 线计算均量，排除当前未完成的 K 线
            historical = volumes[:-1]
            avg_vol = sum(historical) / len(historical) if historical else last_vol or 1
            return {
                "volume_1h": last_vol,
                "volume_change_1h": (last_vol / prev_vol - 1) if prev_vol > 0 else 0.0,
                "volume_vs_avg_24h": (last_vol / avg_vol) if avg_vol > 0 else 1.0,
                "volume_24h_total": sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes),
            }
        except Exception as e:
            logger.warning(f"volume features fetch failed for {symbol}: {e}")
            return {
                "volume_1h": 0,
                "volume_change_1h": 0.0,
                "volume_vs_avg_24h": 1.0,
                "volume_24h_total": 0,
            }

    def get_relative_features(self, symbol: str, gainers: List[Dict]) -> Dict:
        rank = next((i for i, g in enumerate(gainers, 1)
                     if g["symbol"] == symbol), 999)
        return {"rank_in_gainers": rank, "gainers_count": len(gainers)}

    def snapshot(self, symbol: str, gainers: Optional[List] = None) -> Dict:
        gainers = gainers or self.get_top_gainers()
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "price": self.get_price_features(symbol),
            "derivatives": self.get_derivatives_features(symbol),
            "social": self.get_social_features(symbol),
            "volume": self.get_volume_features(symbol),
            "relative": self.get_relative_features(symbol, gainers),
        }

    def is_new_listing(self, symbol: str, days: int = 14) -> bool:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1d", limit=days + 1)
            return len(ohlcv) <= days
        except Exception:
            return False
