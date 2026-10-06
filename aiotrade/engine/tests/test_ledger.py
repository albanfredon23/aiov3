import json

import pytest

from aiotrade.ledger import GENESIS, XAILedger, record_hash, verify_chain

SAMPLE = {
    "timestamp_utc": "2026-10-06T17:15:02.104Z",
    "market_integrity_d2": 4.12,
    "chi2_threshold": 13.28,
    "status": "NOMINAL",
    "tap_trajectories_tested": 16,
    "scg_rejections": {"drawdown_violation": 2, "cvar_violation": 1},
    "admissibility_ratio": 0.8125,
    "selected_allocation": 0.35,
    "active_constraints": ["MAX_LEVERAGE_1.5X", "STOP_LOSS_1.08410"],
}


def filled(n=5):
    ledger = XAILedger()
    for i in range(n):
        ledger.append({**SAMPLE, "selected_allocation": 0.1 * i})
    return ledger


def test_chainage_sha256():
    ledger = filled()
    recs = ledger.records()
    assert recs[0]["prev_hash"] == GENESIS
    assert all(recs[i]["prev_hash"] == recs[i - 1]["hash"] for i in range(1, len(recs)))
    assert all(r["hash"] == record_hash(r) for r in recs)
    assert ledger.verify().ok and ledger.head_hash == recs[-1]["hash"]


@pytest.mark.parametrize("tamper", ["modify", "delete", "swap"])
def test_falsification_detectee(tamper):
    recs = filled().records()
    if tamper == "modify":
        recs[2]["selected_allocation"] = 0.9
    elif tamper == "delete":
        del recs[2]
    else:
        recs[1], recs[2] = recs[2], recs[1]
    result = verify_chain(recs)
    assert not result.ok
    assert result.first_invalid_seq is not None


def test_cles_reservees():
    with pytest.raises(ValueError):
        XAILedger().append({**SAMPLE, "hash": "x"})


def test_fichier_jsonl_ajout_seul(tmp_path):
    path = tmp_path / "audit" / "decisions.jsonl"
    a = XAILedger(path)
    a.append(SAMPLE)
    a.append(SAMPLE)
    b = XAILedger(path)  # reprise : la chaîne continue
    rec = b.append(SAMPLE)
    assert rec["seq"] == 2 and len(b) == 3
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["status"] == "NOMINAL"
    assert b.verify().ok
    assert XAILedger.parse_jsonl(b.export_jsonl())[-1]["hash"] == rec["hash"]


def test_fichier_corrompu_refuse(tmp_path):
    path = tmp_path / "decisions.jsonl"
    ledger = XAILedger(path)
    ledger.append(SAMPLE)
    ledger.append(SAMPLE)
    lines = path.read_text().splitlines()
    path.write_text(lines[1] + "\n" + lines[0] + "\n")
    with pytest.raises(ValueError):
        XAILedger(path)


def test_format_de_la_specification_conserve():
    rec = filled(1).records()[0]
    for key in SAMPLE:
        assert rec[key] == SAMPLE[key] or key == "selected_allocation"
