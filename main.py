"""主调度器"""
import time
import argparse
import schedule
from loguru import logger
from config import CFG
from core.data_fetcher import DataFetcher
from core.scanner import Scanner
from core.ai_evaluator import AIEvaluator
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from core.trailing import TrailingEvaluator
from utils.notifier import notify


class LanaAgent:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.scanner = Scanner(self.fetcher)
        self.evaluator = AIEvaluator()
        self.risk = RiskManager(self.fetcher)
        self.position = PositionManager(self.fetcher, self.risk)
        self.trailing = TrailingEvaluator(
            self.fetcher, self.position, self.evaluator, self.risk
        )

    def scan_and_enter(self):
        logger.info("=" * 60)
        logger.info("SCAN CYCLE START")
        candidates = self.scanner.scan()
        held = {p["symbol"] for p in self.position.get_positions()}
        for c in candidates:
            symbol = c["symbol"]
            if symbol in held:
                continue
            snapshot = self.fetcher.snapshot(symbol)
            decision = self.evaluator.evaluate_entry(symbol, snapshot)
            if decision["should_enter"]:
                result = self.position.open_long(symbol)
                if result:
                    notify(f"OPEN LONG {symbol} entry={{result.get('entry_price','?')}} stop={{result['stop_price']:.6f}} reason={{decision['key_reason']}}")

    def trailing_check(self):
        logger.info("-" * 60)
        logger.info("TRAILING CHECK")
        try:
            self.trailing.evaluate_all()
        except Exception as e:
            logger.error(f"trailing crashed: {{e}}")

    def run(self):
        notify(f"Lana Agent started (testnet={{CFG.TESTNET}})")
        self.scan_and_enter()
        self.trailing_check()
        schedule.every(CFG.SCAN_INTERVAL_MINUTES).minutes.do(self.scan_and_enter)
        schedule.every(CFG.EVALUATION_INTERVAL_MINUTES).minutes.do(self.trailing_check)
        logger.info(f"scheduler running: scan/{{CFG.SCAN_INTERVAL_MINUTES}}min trailing/{{CFG.EVALUATION_INTERVAL_MINUTES}}min")
        while True:
            try:
                schedule.run_pending()
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("shutting down")
                notify("Lana Agent stopped")
                break
            except Exception as e:
                logger.error(f"main loop error: {{e}}")
                time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testnet', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--capital', type=float, default=100)
    args = parser.parse_args()
    if args.testnet:
        CFG.TESTNET = True
    if args.dry_run:
        CFG.DRY_RUN = True
    logger.add("logs/lana_{time}.log", rotation="1 day", retention="30 days")
    logger.info(f"starting with capital reference: {{args.capital}} USDT")
    agent = LanaAgent()
    agent.run()
