import numpy as np
import pytest

from aiotrade.integrity import (
    FEATURES,
    IntegrityConfig,
    IntegrityFilter,
    IntegrityStatus,
    ewma_volatility,
    microstructure_features,
)
from aiotrade.market import MarketConfig, Shock, generate_market, macro_event_mask


def test_seuil_theorique_chi2_4_ddl():
    f = IntegrityFilter()
    assert f.threshold_theory == pytest.approx(13.28, abs=0.01)
    assert f.threshold_freeze_theory == pytest.approx(23.51, abs=0.01)


def test_volatilite_ewma_causale():
    r = np.zeros(100)
    r[50] = 0.05
    vol = ewma_volatility(r)
    assert vol[50] == vol[49]  # le choc de t n'entre qu'à t + 1
    assert vol[51] > vol[50]


def test_prechauffage_puis_lectures():
    rng = np.random.default_rng(0)
    f = IntegrityFilter(IntegrityConfig(warmup=50))
    readings = f.run(rng.normal(size=(120, len(FEATURES))))
    assert all(r.status == IntegrityStatus.WARMUP for r in readings[:50])
    assert all(r.status != IntegrityStatus.WARMUP for r in readings[50:])
    assert set(readings[-1].contributions) == set(FEATURES)
    assert sum(readings[-1].contributions.values()) == pytest.approx(readings[-1].d2, rel=1e-6)


def test_gaussien_peu_de_gels():
    rng = np.random.default_rng(1)
    f = IntegrityFilter()
    readings = f.run(rng.normal(size=(6_000, 4)))
    freeze = np.mean([r.status == IntegrityStatus.FREEZE for r in readings[200:]])
    assert freeze < 0.01


def test_calibrage_exige_500_barres():
    with pytest.raises(ValueError):
        IntegrityFilter().calibrate(np.random.default_rng(2).normal(size=(400, 4)))


@pytest.fixture(scope="module")
def calibrated():
    m = generate_market(MarketConfig(bars=8_000, seed=4, shock=Shock.FLASH_CRASH, shock_at=7_000))
    x = microstructure_features(m)
    f = IntegrityFilter()
    info = f.calibrate(x[:6_000], exclude=macro_event_mask(m)[:6_000])
    readings = [f.update(row) for row in x[6_000:]]
    return m, f, info, readings


def test_calibrage_empirique_au_moins_aussi_strict(calibrated):
    _, f, info, _ = calibrated
    assert info["chi2_threshold_theoretical"] == pytest.approx(13.28, abs=0.01)
    assert f.threshold >= f.threshold_theory
    assert f.threshold_freeze >= f.threshold


def test_flash_crash_gele_des_la_premiere_barre(calibrated):
    m, _, _, readings = calibrated
    a, b = m.shock_window
    window = readings[a - 6_000 : b - 6_000]
    assert window[0].status == IntegrityStatus.FREEZE
    assert window[0].d2 > 10 * window[0].threshold
    assert np.mean([r.status == IntegrityStatus.FREEZE for r in window]) > 0.8


def test_peu_de_faux_gels_hors_choc(calibrated):
    m, _, _, readings = calibrated
    a = m.shock_window[0] - 6_000
    calm = [r for r in readings[:a]]
    assert np.mean([r.status == IntegrityStatus.FREEZE for r in calm]) < 0.05


def test_etat_de_taille_fixe():
    f = IntegrityFilter(IntegrityConfig(warmup=10))
    before = (f._mu.shape, f._cov.shape, f._inv.shape)
    f.run(np.random.default_rng(3).normal(size=(500, 4)))
    assert (f._mu.shape, f._cov.shape, f._inv.shape) == before
