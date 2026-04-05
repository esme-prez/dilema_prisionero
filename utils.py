"""
Utilidades para exportación de datos, estadísticas y formateo.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, List
import json
import io


# --- Exportación ---

def export_to_csv(df: pd.DataFrame) -> str:
    """Convierte DataFrame a string CSV para descarga."""
    return df.to_csv(index=False)


def export_to_json(data: dict) -> str:
    """Convierte diccionario a JSON string para descarga."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return str(obj)
    return json.dumps(data, default=convert, indent=2, ensure_ascii=False)


# --- Estadísticas ---

def compute_anova(groups: Dict[str, List[float]]) -> Dict:
    """
    Realiza ANOVA de una vía sobre grupos de datos.
    Retorna estadístico F, p-value e interpretación.
    """
    group_values = list(groups.values())
    if len(group_values) < 2:
        return {'f_statistic': None, 'p_value': None, 'significant': False}

    f_stat, p_value = stats.f_oneway(*group_values)
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_groups': len(groups),
        'group_sizes': {k: len(v) for k, v in groups.items()},
        'interpretation': (
            f"F({len(groups)-1}, {sum(len(v) for v in group_values)-len(groups)}) = {f_stat:.4f}, "
            f"p = {p_value:.6f}. "
            f"{'Diferencias significativas' if p_value < 0.05 else 'Sin diferencias significativas'} "
            f"entre grupos (alpha=0.05)."
        )
    }


def logistic_regression_fit(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Ajusta modelo logístico: y = L / (1 + exp(-k*(x - x0)))
    Para predecir cooperación en función de parámetros.
    """
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    try:
        popt, pcov = curve_fit(logistic, x, y, p0=[1.0, 5.0, np.median(x)],
                               maxfev=10000, bounds=([0, -50, -10], [1.5, 50, 100]))
        y_pred = logistic(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
        return {
            'params': {'L': popt[0], 'k': popt[1], 'x0': popt[2]},
            'r_squared': r_squared,
            'y_pred': y_pred,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def summary_statistics(df: pd.DataFrame, group_col: str, value_col: str = 'coop_rate') -> pd.DataFrame:
    """Genera tabla resumen con estadísticas por grupo."""
    summary = df.groupby(group_col)[value_col].agg([
        'count', 'mean', 'std', 'min',
        ('q25', lambda x: x.quantile(0.25)),
        'median',
        ('q75', lambda x: x.quantile(0.75)),
        'max'
    ]).reset_index()
    summary.columns = [group_col, 'Replicas', 'Media', 'Desv.Est.', 'Min', 'Q25', 'Mediana', 'Q75', 'Max']
    return summary


# --- Intervalos de confianza ---

def confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """Calcula intervalo de confianza para la media."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h
