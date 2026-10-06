"""Ligne de commande AIOTrade.

    python -m aiotrade benchmark --out reports/benchmark.json
    python -m aiotrade simulate --shock flash_crash
    python -m aiotrade broker-check --symbol BTCUSDT
    python -m aiotrade trade --symbol BTCUSDT            # papier : données publiques, aucun ordre
    python -m aiotrade trade --broker binance --loop     # testnet, ordres validés sans exécution
    python -m aiotrade ledger-verify ledger/decisions.jsonl

Les clés ne passent jamais en argument : elles sont lues dans l'environnement
(AIOTRADE_BINANCE_API_KEY, AIOTRADE_BINANCE_API_SECRET). Voir README.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .market import Shock


def _benchmark(args: argparse.Namespace) -> int:
    from .backtest import BacktestConfig, walk_forward

    seeds = args.seeds or [11]
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    report = walk_forward(BacktestConfig(seed=seeds[0], folds=args.folds), progress=log)
    if len(seeds) > 1:
        runs = [{"seed": seeds[0], "aggregate": report["aggregate"], "targets": report["targets"],
                 "scg_pruning_rate": report["scg"]["scg_pruning_rate"]}]
        for seed in seeds[1:]:
            log(f"graine {seed}")
            other = walk_forward(BacktestConfig(seed=seed, folds=args.folds))
            runs.append({"seed": seed, "aggregate": other["aggregate"], "targets": other["targets"],
                         "scg_pruning_rate": other["scg"]["scg_pruning_rate"]})
        report["robustness"] = runs
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"rapport écrit : {args.out}", file=sys.stderr)
    summary = {
        "aggregate": report["aggregate"],
        "targets": report["targets"],
        "scg_pruning_rate": report["scg"]["scg_pruning_rate"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from .api import MandateIn, _simulation

    result = _simulation(args.seed, args.shock, MandateIn().model_dump_json())
    print(json.dumps({"summary": result["summary"], "scg": result["scg"], "ledger": {
        k: v for k, v in result["ledger"].items() if k in ("records", "head_hash", "verified")
    }}, indent=2, ensure_ascii=False))
    if args.ledger:
        print(json.dumps(result["ledger"]["latest"], indent=2, ensure_ascii=False))
    return 0


def _broker(name: str, ledger: str):  # noqa: ANN202
    from .broker import BinanceSpot, PaperBroker

    if name == "paper":
        return PaperBroker(BinanceSpot(environment="public"), state_path=Path(ledger).parent / "paper_state.json")
    return BinanceSpot.from_env()


def _broker_check(args: argparse.Namespace) -> int:
    from .broker import BinanceSpot, OrderRequest, Side

    broker = BinanceSpot.from_env()
    print(f"plateforme : {broker.base_url} (mode {broker.mode})")
    broker.ping()
    print(f"ping : ok ; décalage d'horloge : {broker.sync_time()} ms")
    rules = broker.rules(args.symbol)
    quote = broker.get_quote(args.symbol)
    candles = broker.get_candles(args.symbol, "15m", 5)
    print(f"{args.symbol} : bid {quote.bid} / ask {quote.ask} ; pas {rules.step_size}, minimum {rules.min_notional} {rules.quote_asset}")
    print(f"dernier chandelier 15m : {candles[-1].open_time.isoformat()} clôture {candles[-1].close}")
    if not broker.has_credentials:
        print("clés absentes : vérification du compte et des ordres ignorée")
        return 0
    balances = broker.balances()
    print("soldes : " + ", ".join(f"{k} {v:g}" for k, v in sorted(balances.items()) if k in (rules.base_asset, rules.quote_asset)))
    qty = max(rules.min_qty, rules.floor_qty(rules.min_notional * 1.5 / quote.ask + rules.step_size))
    if not broker.dry_run:
        print("AIOTRADE_SEND_ORDERS=1 : test d'ordre ignoré (la vérification n'envoie jamais d'ordre réel)")
        return 0
    result = broker.place_order(OrderRequest(args.symbol, Side.BUY, qty, stop_loss=quote.bid * 0.99))
    print(f"ordre de test ({qty} {rules.base_asset}, stop -1 %) : {result.status} — {result.message}")
    return 0


def _trade(args: argparse.Namespace) -> int:
    from .live import LiveRunner

    broker = _broker(args.broker, args.ledger)
    send = args.broker != "paper" and getattr(broker, "dry_run", True) is False
    runner = LiveRunner(
        broker,
        symbol=args.symbol,
        interval=args.interval,
        ledger_path=args.ledger,
        calendar_path=args.calendar,
        send_orders=send or args.broker == "paper",
    )
    print(f"mode : {getattr(broker, 'mode', '?')} ; registre : {args.ledger}", file=sys.stderr)
    if args.loop:
        runner.run_forever()
    else:
        step = runner.step()
        print(json.dumps(step.record, indent=2, ensure_ascii=False))
    return 0


def _ledger_verify(args: argparse.Namespace) -> int:
    from .ledger import XAILedger, verify_chain

    result = verify_chain(XAILedger.read_jsonl(args.path))
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("aiotrade.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aiotrade", description="AIOTrade : co-pilote de gestion des risques")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("benchmark", help="backtest walk-forward à 4 bras (12 mois / 3 mois)")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13], help="graines (la première est détaillée)")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--out", default="")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("simulate", help="démonstration courte avec choc injecté")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shock", choices=[s.value for s in Shock], default=Shock.FLASH_CRASH.value)
    p.add_argument("--ledger", action="store_true", help="affiche le dernier enregistrement XAI")
    p.set_defaults(func=_simulate)

    p = sub.add_parser("broker-check", help="vérifie la connexion Binance (testnet par défaut)")
    p.add_argument("--symbol", default="BTCUSDT")
    p.set_defaults(func=_broker_check)

    p = sub.add_parser("trade", help="décision sur la dernière barre clôturée (papier par défaut)")
    p.add_argument("--broker", choices=["paper", "binance"], default="paper")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--ledger", default="ledger/decisions.jsonl")
    p.add_argument("--calendar", default=str(Path(__file__).resolve().parent.parent / "data" / "macro_calendar.example.json"))
    p.add_argument("--loop", action="store_true", help="tourne à chaque clôture de barre")
    p.set_defaults(func=_trade)

    p = sub.add_parser("ledger-verify", help="vérifie la chaîne SHA-256 d'un registre XAI")
    p.add_argument("path")
    p.set_defaults(func=_ledger_verify)

    p = sub.add_parser("serve", help="lance l'API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    from .broker.base import BrokerError

    try:
        return int(args.func(args))
    except BrokerError as exc:
        print(f"erreur courtier : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
