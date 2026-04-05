"""
Tests de calidad para el Generador de Números Aleatorios (RNG).
Incluye test KS, autocorrelación y análisis de distribución.
"""

import numpy as np
from scipy import stats
from rng_module import ReproducibleRNG


def test_uniformity_ks(rng: ReproducibleRNG, n_samples: int = 100_000) -> dict:
    """
    Test Kolmogorov-Smirnov para verificar uniformidad.
    H0: Los datos siguen una distribución uniforme [0,1).
    Se espera p-value > 0.05 para no rechazar H0.
    """
    samples = rng.uniform(0, 1, size=n_samples)
    statistic, p_value = stats.kstest(samples, 'uniform')
    return {
        'test': 'Kolmogorov-Smirnov',
        'statistic': statistic,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'n_samples': n_samples,
        'interpretation': (
            f"{'APROBADO' if p_value > 0.05 else 'RECHAZADO'}: "
            f"p-value = {p_value:.6f} "
            f"({'>' if p_value > 0.05 else '<='} 0.05)"
        )
    }


def test_autocorrelation(rng: ReproducibleRNG, n_samples: int = 10_000, max_lag: int = 50) -> dict:
    """
    Calcula autocorrelación para diferentes lags.
    Valores cercanos a 0 indican independencia entre muestras.
    """
    samples = rng.uniform(0, 1, size=n_samples)
    mean = np.mean(samples)
    var = np.var(samples)

    autocorr = []
    for lag in range(1, max_lag + 1):
        c = np.mean((samples[:-lag] - mean) * (samples[lag:] - mean)) / var
        autocorr.append(c)

    # Intervalo de confianza al 95%: +-1.96/sqrt(n)
    ci = 1.96 / np.sqrt(n_samples)
    n_outside = sum(1 for a in autocorr if abs(a) > ci)

    return {
        'test': 'Autocorrelación',
        'autocorrelations': autocorr,
        'confidence_interval': ci,
        'lags': list(range(1, max_lag + 1)),
        'n_outside_ci': n_outside,
        'passed': n_outside <= max_lag * 0.1,  # Menos del 10% fuera del IC
        'interpretation': (
            f"{'APROBADO' if n_outside <= max_lag * 0.1 else 'RECHAZADO'}: "
            f"{n_outside}/{max_lag} lags fuera del IC 95% "
            f"(umbral: {int(max_lag * 0.1)})"
        )
    }


def test_histogram_uniformity(rng: ReproducibleRNG, n_samples: int = 100_000, n_bins: int = 50) -> dict:
    """
    Test Chi-cuadrado sobre histograma de muestras uniformes.
    """
    samples = rng.uniform(0, 1, size=n_samples)
    observed, bin_edges = np.histogram(samples, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, n_samples / n_bins)

    chi2, p_value = stats.chisquare(observed, expected)

    return {
        'test': 'Chi-cuadrado (histograma)',
        'chi2_statistic': chi2,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'observed': observed,
        'expected': expected[0],
        'bin_edges': bin_edges,
        'n_bins': n_bins,
        'n_samples': n_samples,
        'interpretation': (
            f"{'APROBADO' if p_value > 0.05 else 'RECHAZADO'}: "
            f"Chi2 = {chi2:.2f}, p-value = {p_value:.6f}"
        )
    }


def run_all_tests(seed: int = 20260211) -> list:
    """Ejecuta todos los tests de calidad RNG y retorna resultados."""
    rng = ReproducibleRNG(seed=seed)
    results = []

    rng.reset()
    results.append(test_uniformity_ks(rng))

    rng.reset()
    results.append(test_autocorrelation(rng))

    rng.reset()
    results.append(test_histogram_uniformity(rng))

    return results
