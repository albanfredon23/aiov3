import math

import numpy as np
import pytest

from aiotrade.execution import ExecutionConfig, ExecutionModel
from aiotrade.sizing import FractionalKellySizer, KellyEstimate, SizingConfig


def sizer(**kw):
    return FractionalKellySizer(SizingConfig(**kw))


def test_formule_de_kelly_fractionnaire():
    # p = 0,6 ; b = 1,5 → f* = (0,6·1,5 − 0,4) / 1,5 = 1/3 ; λ = 0,2 → 0,0667
    outcomes = np.array([0.015] * 60 + [-0.01] * 40)
    k = sizer(confidence_z=0.0).estimate(outcomes)
    assert k.p == pytest.approx(0.6)
    assert k.b == pytest.approx(1.5)
    assert k.full_kelly == pytest.approx(1 / 3)
    assert k.fraction == pytest.approx(0.2 / 3)


def test_kelly_prudent_borne_basse():
    outcomes = np.array([0.015] * 60 + [-0.01] * 40)
    k = sizer(confidence_z=1.0).estimate(outcomes)
    assert k.p_lower == pytest.approx(0.6 - math.sqrt(0.24 / 100))
    assert k.fraction < sizer(confidence_z=0.0).estimate(outcomes).fraction


def test_cas_limites_sans_division_par_zero():
    s = sizer()
    assert s.estimate(np.array([])).fraction == 0.0
    assert s.estimate(np.array([-0.01, -0.02])).fraction == 0.0
    assert s.estimate(np.array([0.01, 0.02])).fraction == pytest.approx(0.20)


def test_lambda_borne_a_un_quart():
    with pytest.raises(ValueError):
        SizingConfig(kelly_fraction=0.3)


def test_taille_tient_compte_du_prix_et_du_contrat():
    k = KellyEstimate(0.6, 1.5, 1 / 3, 0.005)
    # risque 0,5 % / perte au stop 0,5 % → exposition 1× ; 100 000 € à 1,0850 → 0,92 lot
    size = sizer().size(k, equity=100_000, price=1.0850, loss_at_stop=0.005)
    assert size.lots == pytest.approx(0.92)
    assert size.notional <= 100_000 * 1.0 + 1e-6
    assert size.capped_by == "KELLY_FRACTIONNAIRE"


def test_arrondi_par_defaut_ne_depasse_jamais_le_plafond():
    k = KellyEstimate(0.9, 3.0, 0.9, 0.2)
    size = sizer(max_leverage=1.5).size(k, equity=100_000, price=1.0, loss_at_stop=0.001)
    assert size.exposure <= 1.5 + 1e-12
    assert size.lots == pytest.approx(1.5)
    size = sizer(max_leverage=1.5).size(k, equity=100_333, price=1.0, loss_at_stop=0.001)
    assert size.exposure <= 1.5 + 1e-12  # 1,50499 lot arrondi à 1,50 et non 1,51


def test_plafond_de_perte_par_trade():
    k = KellyEstimate(0.9, 3.0, 0.9, 0.2)
    size = sizer(max_leverage=10.0, max_lot=100).size(k, equity=100_000, price=1.0, loss_at_stop=0.01)
    assert size.risk_fraction <= 0.01 + 1e-12
    assert size.capped_by == "PERTE_MAX_PAR_TRADE"


def test_multiplicateur_macro_et_lot_minimal():
    k = KellyEstimate(0.6, 1.5, 1 / 3, 0.005)
    full = sizer().size(k, 100_000, 1.0, 0.005)
    half = sizer().size(k, 100_000, 1.0, 0.005, exposure_multiplier=0.5)
    assert half.lots == pytest.approx(full.lots / 2, abs=0.01)
    assert sizer().size(k, 100, 1.0, 0.005).lots == 0.0  # sous le lot minimal
    assert sizer().shrink(full, 100_000, 1.0, 0.005).lots == pytest.approx(full.lots / 2, abs=0.01)


def test_impact_almgren_chriss():
    m = ExecutionModel(ExecutionConfig(impact_gamma=0.8, impact_alpha=0.6))
    assert m.impact(1e6, 5e7, 0.001) == pytest.approx(0.8 * 0.001 * (1e6 / 5e7) ** 0.6)
    assert m.impact(0, 5e7, 0.001) == 0.0


def test_maker_moins_cher_que_taker():
    m = ExecutionModel()
    taker = m.cost(1e6, 1.0, 0.0001, 0.001, 5e7)
    maker = m.cost(1e6, 1.0, 0.0001, 0.001, 5e7, maker=True)
    assert taker.spread == pytest.approx(0.00005)
    assert maker.total < taker.total
    assert m.round_trip(1e6, 1.0, 0.0001, 0.001, 5e7) == pytest.approx(2 * taker.total)


def test_remplissage_maker_simule():
    m = ExecutionModel(ExecutionConfig(queue_fill_probability=0.0))
    rng = np.random.default_rng(0)
    assert m.maker_fill(+1, 1.0, next_low=0.999, next_high=1.01, rng=rng)  # traversée
    assert not m.maker_fill(+1, 1.0, next_low=1.0, next_high=1.01, rng=rng)  # simple contact, file pleine
    assert not m.maker_fill(+1, 1.0, next_low=1.001, next_high=1.01, rng=rng)
    assert m.maker_fill(-1, 1.0, next_low=0.99, next_high=1.001, rng=rng)
    assert 0.0 < m.fill_probability(+1, 0.999, 1.0, 0.001) < 1.0
