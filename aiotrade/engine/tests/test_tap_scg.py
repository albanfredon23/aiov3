import numpy as np
import pytest

from aiotrade.scg import REJECTION_KEYS, RiskMandate, SphericalConstraintGraph
from aiotrade.tap import TAPConfig, TAPPlanner, TrajectoryBundle


class FixedForecaster:
    name = "fixe"

    def __init__(self, paths):
        self.paths = np.asarray(paths, dtype=float)

    def sample(self, ctx, n_paths, horizon, rng):
        return self.paths

    def observe(self, ctx, realized_return):
        return None


def alban_bundle() -> TrajectoryBundle:
    """16 trajectoires : 13 admissibles, 2 violent le drawdown, 1 viole le budget CVaR."""
    paths = [[0.001, 0.001]] * 13 + [[0.03, -0.025]] * 2 + [[-0.004, -0.0125]]
    return TrajectoryBundle(np.array(paths), "fixe")


def test_exemple_du_registre_xai_reproduit():
    mandate = RiskMandate(max_path_drawdown=0.02, max_loss_per_trade=0.03, var_confidence=0.9)
    report = SphericalConstraintGraph(mandate).evaluate(
        alban_bundle(), exposure=1.0, equity=1.0, peak_equity=1.0, stop_distance=0.5, cost_per_unit=0.0
    )
    assert report.rejections["drawdown_violation"] == 2
    assert report.rejections["cvar_violation"] == 1
    assert sum(report.rejections.values()) == 3
    assert report.admissibility_ratio == pytest.approx(0.8125)
    assert report.admitted


def test_ratio_sous_75_pourcent_rejete():
    paths = [[0.001]] * 11 + [[-0.03]] * 5  # 11/16 = 68,75 %
    report = SphericalConstraintGraph().evaluate(
        TrajectoryBundle(np.array(paths), "x"), 1.0, 1.0, 1.0, stop_distance=0.5, cost_per_unit=0.0
    )
    assert report.admissibility_ratio == pytest.approx(11 / 16)
    assert not report.admitted
    assert any("Ratio admissible" in r for r in report.reasons)


def test_levier_excessif_rejette_tout():
    report = SphericalConstraintGraph().evaluate(alban_bundle(), 2.0, 1.0, 1.0, 0.5, 0.0)
    assert report.rejections["leverage_violation"] == 16
    assert report.admissibility_ratio == 0.0 and not report.admitted


def test_exposition_nulle_toujours_admise():
    report = SphericalConstraintGraph().evaluate(alban_bundle(), 0.0, 1.0, 1.0, 0.01, 0.0)
    assert report.admitted and report.admissibility_ratio == 1.0


def test_drawdown_du_compte_pris_en_compte():
    # Le compte est déjà à -9,9 % de son pic : une perte de 0,2 % franchit le mandat de 10 %.
    paths = np.array([[-0.002]] * 16)
    report = SphericalConstraintGraph().evaluate(TrajectoryBundle(paths, "x"), 1.0, 0.901, 1.0, 0.5, 0.0)
    assert report.rejections["drawdown_violation"] == 16


def test_stop_loss_tronque_la_trajectoire_et_frictions():
    scg = SphericalConstraintGraph()
    bundle = TrajectoryBundle(np.array([[-0.01, 0.05, 0.05]]), "x")
    curves = scg.equity_paths(bundle, 1.0, stop_distance=0.005, cost_per_unit=0.0002)
    # Sortie au stop sur la première barre (gap jusqu'à -1 %), plus de rebond ensuite.
    assert curves[0, -1] == pytest.approx(1.0 - 0.01 - 0.0002)
    assert curves[0, 0] == pytest.approx(1.0 - 0.0001)


def test_vente_a_decouvert_symetrique():
    scg = SphericalConstraintGraph()
    bundle = TrajectoryBundle(np.array([[0.002, 0.002]]), "x")
    long_end = scg.equity_paths(bundle, 1.0, 0.5, 0.0)[0, -1]
    short_end = scg.equity_paths(bundle, -1.0, 0.5, 0.0)[0, -1]
    assert long_end > 1.0 > short_end


def test_budgets_ramenes_a_l_horizon():
    assert RiskMandate.horizon_scale(12, 15) == pytest.approx(np.sqrt(180 / 1440))
    with pytest.raises(ValueError):
        SphericalConstraintGraph().evaluate(alban_bundle(), 1.0, 1.0, 1.0, 0.5, 0.0, budget_scale=2.0)


def test_mandat_valide():
    with pytest.raises(ValueError):
        RiskMandate(min_admissibility=1.5)
    labels = RiskMandate().constraint_labels()
    assert "MAX_LEVERAGE_1.5X" in labels


def test_cles_de_rejet():
    assert set(REJECTION_KEYS) == {
        "leverage_violation", "drawdown_violation", "loss_violation", "cvar_violation", "var_violation"
    }


def test_planner_controle_la_sortie_du_modele():
    planner = TAPPlanner(TAPConfig(n_paths=8, horizon=3))
    rng = np.random.default_rng(0)
    ok = planner.plan(FixedForecaster(np.full((8, 3), 0.001)), None, rng)
    assert ok.n_paths == 8 and ok.horizon == 3
    with pytest.raises(ValueError):
        planner.plan(FixedForecaster(np.zeros((4, 3))), None, rng)
    bad = np.zeros((8, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        planner.plan(FixedForecaster(bad), None, rng)


def test_distance_de_stop_plancher_deux_sigma():
    bundle = TrajectoryBundle(np.full((16, 4), 0.0001), "x")
    assert TAPPlanner.stop_distance(bundle, 1, sigma=0.001) == pytest.approx(0.002)
    assert np.all(bundle.adverse_excursion(1) == 0.0)
    assert np.all(bundle.adverse_excursion(-1) > 0.0)
