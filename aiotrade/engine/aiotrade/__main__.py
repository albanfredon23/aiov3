"""CLI : ``python -m aiotrade --shock flash_crash --seed 7``."""
from __future__ import annotations

import argparse
import json

from .copilot import simulate
from .market import MarketConfig, Shock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulation AIOTrade (TAP + SCG + filtre χ²)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--shock", choices=[s.value for s in Shock], default=Shock.FLASH_CRASH.value)
    parser.add_argument("--events", action="store_true", help="affiche le journal d'audit")
    args = parser.parse_args(argv)

    result = simulate(MarketConfig(steps=args.steps, seed=args.seed, shock=Shock(args.shock)))
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    if args.events:
        for event in result.events:
            if event.kind != "rebalance":
                print(f"[{event.step:>4}] {event.kind:<14} {event.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
