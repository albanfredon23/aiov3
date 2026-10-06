import numpy as np
import pytest

from aiotrade.stats import chi2_sf, historical_cvar, historical_var, max_drawdown


@pytest.mark.parametrize(
    ("statistic", "df", "expected"),
    [
        (3.841458820694124, 1, 0.05),
        (6.634896601021214, 1, 0.01),
        (11.070497693516351, 5, 0.05),
        (23.209251158954356, 10, 0.01),
        (79.08194397720072, 60, 0.05),
        (0.5, 3, 0.9188914),
    ],
)
def test_chi2_sf_matches_reference_values(statistic, df, expected):
    assert chi2_sf(statistic, df) == pytest.approx(expected, rel=1e-6)


def test_chi2_sf_edges():
    assert chi2_sf(0.0, 4) == 1.0
    assert chi2_sf(1e4, 4) == pytest.approx(0.0, abs=1e-300)
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)


def test_var_and_cvar_on_known_distribution():
    pnl = np.linspace(-0.10, 0.09, 20)  # pertes de 10 % à -9 %
    var = historical_var(pnl, 0.95)
    cvar = historical_cvar(pnl, 0.95)
    assert 0.09 <= var <= 0.10
    assert cvar >= var


def test_var_is_zero_for_flat_portfolio():
    assert historical_var(np.zeros(50)) == 0.0


def test_max_drawdown():
    assert max_drawdown(np.array([100, 120, 90, 130, 117])) == pytest.approx(0.25)
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0
