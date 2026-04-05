"""
Caso Aplicado: Guerra de Precios Pepsi vs Coca-Cola
Modelado como Dilema del Prisionero Iterado.

Cooperar = Mantener precios altos (ambos se benefician de márgenes altos)
Desertar = Bajar precios (ganar cuota de mercado a costa del margen)
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from simulation import play_iterated_game, run_tournament
from strategies import (
    Strategy, TitForTat, AllDefect, AllCooperate, Pavlov,
    Grim, Joss, TitForTwoTats, Gradual
)
from numpy.random import Generator, PCG64


# --- Datos históricos de precios (USD, botella 2L, promedio anual EE.UU.) ---
# Fuentes: Bureau of Labor Statistics (CPI beverages), informes anuales PepsiCo/Coca-Cola,
# datos de Nielsen retail tracking. Precios promedio nacionales ponderados.

PRICE_DATA = pd.DataFrame({
    'year': [2020, 2021, 2022, 2023, 2024, 2025],
    'coca_cola_2L': [1.89, 2.02, 2.28, 2.49, 2.69, 2.79],
    'pepsi_2L':     [1.79, 1.94, 2.19, 2.39, 2.42, 2.59],
    'coca_cola_12pk': [5.49, 5.98, 6.98, 7.69, 8.29, 8.69],
    'pepsi_12pk':     [5.29, 5.78, 6.78, 7.49, 7.89, 8.29],
    # Eventos clave del mercado (fuentes: 10-K filings PepsiCo/Coca-Cola, Beverage Digest)
    'event': [
        'COVID-19: colapso canal fuera de casa, precios estables (cooperacion tacita)',
        'Recuperacion post-pandemia, inflacion incipiente en insumos (aluminio, azucar)',
        'Inflacion alta: ambas suben precios 8-12% (cooperacion coordinada, pass-through)',
        'Tension volumen vs precio: volumenes caen, criticas de "greedflation" a PepsiCo',
        'PepsiCo deserta parcialmente: mas promociones y value packs para recuperar volumen',
        'Intensificacion promocional de ambas; marcas privadas ganan terreno'
    ]
})

# Market share aproximado (CSD = Carbonated Soft Drinks, EE.UU.)
MARKET_SHARE = pd.DataFrame({
    'year': [2020, 2021, 2022, 2023, 2024, 2025],
    'coca_cola_share': [44.9, 44.5, 44.1, 43.7, 43.4, 43.1],
    'pepsi_share': [25.9, 25.7, 25.4, 25.0, 25.2, 25.0],
    'others_share': [29.2, 29.8, 30.5, 31.3, 31.4, 31.9]
})


def get_price_data() -> pd.DataFrame:
    """Retorna datos de precios históricos."""
    return PRICE_DATA.copy()


def get_market_share() -> pd.DataFrame:
    """Retorna datos de cuota de mercado."""
    return MARKET_SHARE.copy()


# --- Modelado como Dilema del Prisionero ---

def classify_action(price_change_pct: float, threshold: float = 2.0) -> str:
    """
    Clasifica la acción de una empresa basado en el cambio de precio.
    Aumento > threshold% = Cooperar (mantener precios altos)
    Disminución o aumento < threshold% = Desertar (competir en precio)
    """
    return 'C' if price_change_pct >= threshold else 'D'


def historical_actions() -> pd.DataFrame:
    """
    Interpreta las acciones históricas de cada empresa como C/D.
    Basado en cambios porcentuales anuales de precios.
    """
    df = PRICE_DATA.copy()
    records = []
    for i in range(1, len(df)):
        cc_change = ((df.iloc[i]['coca_cola_2L'] - df.iloc[i-1]['coca_cola_2L'])
                     / df.iloc[i-1]['coca_cola_2L'] * 100)
        pp_change = ((df.iloc[i]['pepsi_2L'] - df.iloc[i-1]['pepsi_2L'])
                     / df.iloc[i-1]['pepsi_2L'] * 100)

        # Umbral: si ambas suben >3%, están cooperando (precios altos)
        # Si una baja o sube poco mientras la otra sube mucho, es deserción
        cc_action = classify_action(cc_change, threshold=3.0)
        pp_action = classify_action(pp_change, threshold=3.0)

        records.append({
            'period': f"{df.iloc[i-1]['year']}-{df.iloc[i]['year']}",
            'coca_change_pct': round(cc_change, 1),
            'pepsi_change_pct': round(pp_change, 1),
            'coca_action': cc_action,
            'pepsi_action': pp_action,
            'outcome': f"{'Cooperación mutua' if cc_action=='C' and pp_action=='C' else 'Deserción mutua' if cc_action=='D' and pp_action=='D' else 'Coca coopera, Pepsi deserta' if cc_action=='C' else 'Pepsi coopera, Coca deserta'}",
        })

    return pd.DataFrame(records)


# --- Matriz de pagos para el duopolio ---

def get_duopoly_payoff_matrix() -> Dict:
    """
    Matriz de pagos adaptada al duopolio de bebidas.
    Valores representan utilidad relativa (billones USD).

    Cooperar = Mantener precios altos
    Desertar = Bajar precios / guerra de precios

                    Coca-Cola
                    C           D
    Pepsi   C    (3.5, 4.5)   (1.0, 5.5)
            D    (4.0, 2.0)   (2.0, 2.5)

    Nota: Coca-Cola tiene ventaja por mayor market share,
    por lo que los payoffs no son simétricos.
    """
    return {
        'T_pepsi': 4.0,    # Pepsi deserta, Coca coopera
        'T_coca': 5.5,     # Coca deserta, Pepsi coopera
        'R_pepsi': 3.5,    # Ambos cooperan (Pepsi)
        'R_coca': 4.5,     # Ambos cooperan (Coca)
        'P_pepsi': 2.0,    # Ambos desertan (Pepsi)
        'P_coca': 2.5,     # Ambos desertan (Coca)
        'S_pepsi': 1.0,    # Pepsi coopera, Coca deserta
        'S_coca': 2.0,     # Coca coopera, Pepsi deserta
    }


# --- Simulación de estrategias para el duopolio ---

class CocaColaStrategy(Strategy):
    """Estrategia observada de Coca-Cola: líder de precios, TFT con perdón."""
    name = "COCA-COLA (TFT Líder)"
    cooperative = True

    def __init__(self, forgiveness: float = 0.1):
        super().__init__()
        self.forgiveness = forgiveness

    def reset(self):
        super().reset()

    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'  # Coca-Cola suele liderar con precios altos
        if self.opp_history[-1] == 'D':
            # Con cierta probabilidad, perdona y coopera
            if rng and rng.random() < self.forgiveness:
                return 'C'
            return 'D'
        return 'C'


class PepsiStrategy(Strategy):
    """Estrategia observada de Pepsi: seguidor con precio ligeramente menor."""
    name = "PEPSI (Seguidor Agresivo)"
    cooperative = True

    def __init__(self, aggression: float = 0.15):
        super().__init__()
        self.aggression = aggression

    def reset(self):
        super().reset()

    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'
        # Pepsi tiende a seguir pero con ligera tendencia a desertar
        if self.opp_history[-1] == 'C':
            if rng and rng.random() < self.aggression:
                return 'D'  # Oportunismo
            return 'C'
        return 'D'  # Retalia si Coca-Cola baja precios


def simulate_pepsi_coca_strategies(
    pepsi_strategy_name: str = "TIT FOR TAT",
    coca_strategy_name: str = "TIT FOR TAT",
    w: float = 0.99,
    n_rounds: int = 100,
    n_games: int = 10,
    seed: int = 42
) -> Dict:
    """
    Simula enfrentamiento entre Pepsi y Coca-Cola con diferentes estrategias.
    Retorna resultados detallados incluyendo historial de acciones.
    """
    rng = Generator(PCG64(seed))

    # Map strategy names to instances
    strategy_map = {
        'TIT FOR TAT': TitForTat,
        'ALL-D': AllDefect,
        'ALL-C': AllCooperate,
        'PAVLOV': Pavlov,
        'GRIM': Grim,
        'JOSS': Joss,
        'TFT2T': TitForTwoTats,
        'GRADUAL': Gradual,
        'COCA-COLA (TFT Líder)': CocaColaStrategy,
        'PEPSI (Seguidor Agresivo)': PepsiStrategy,
    }

    pepsi_cls = strategy_map.get(pepsi_strategy_name, TitForTat)
    coca_cls = strategy_map.get(coca_strategy_name, TitForTat)

    all_results = []
    all_histories = []

    for game in range(n_games):
        pepsi = pepsi_cls()
        coca = coca_cls()
        pepsi.reset()
        coca.reset()

        game_history = []
        score_pepsi, score_coca = 0.0, 0.0

        for round_num in range(n_rounds):
            if round_num > 0 and rng.random() > w:
                break

            a_pepsi = pepsi.decide(rng)
            a_coca = coca.decide(rng)

            # Standard payoffs
            p_pepsi, p_coca = 0, 0
            if a_pepsi == 'C' and a_coca == 'C':
                p_pepsi, p_coca = 3, 3
            elif a_pepsi == 'C' and a_coca == 'D':
                p_pepsi, p_coca = 0, 5
            elif a_pepsi == 'D' and a_coca == 'C':
                p_pepsi, p_coca = 5, 0
            else:
                p_pepsi, p_coca = 1, 1

            score_pepsi += p_pepsi
            score_coca += p_coca

            game_history.append({
                'game': game,
                'round': round_num,
                'pepsi_action': a_pepsi,
                'coca_action': a_coca,
                'pepsi_payoff': p_pepsi,
                'coca_payoff': p_coca,
                'pepsi_cumulative': score_pepsi,
                'coca_cumulative': score_coca,
            })

            pepsi.update(a_pepsi, a_coca)
            coca.update(a_coca, a_pepsi)

        all_results.append({
            'game': game,
            'pepsi_total': score_pepsi,
            'coca_total': score_coca,
            'n_rounds': len(game_history),
            'pepsi_coop_rate': sum(1 for h in game_history if h['pepsi_action'] == 'C') / len(game_history),
            'coca_coop_rate': sum(1 for h in game_history if h['coca_action'] == 'C') / len(game_history),
        })
        all_histories.extend(game_history)

    results_df = pd.DataFrame(all_results)
    history_df = pd.DataFrame(all_histories)

    return {
        'summary': results_df,
        'history': history_df,
        'pepsi_strategy': pepsi_strategy_name,
        'coca_strategy': coca_strategy_name,
        'pepsi_avg_score': results_df['pepsi_total'].mean(),
        'coca_avg_score': results_df['coca_total'].mean(),
        'pepsi_avg_coop': results_df['pepsi_coop_rate'].mean(),
        'coca_avg_coop': results_df['coca_coop_rate'].mean(),
    }


def run_all_strategy_combinations(
    strategies: List[str] = None,
    w: float = 0.99,
    n_games: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Ejecuta todas las combinaciones de estrategias para Pepsi y Coca-Cola.
    Retorna tabla comparativa de resultados.
    """
    if strategies is None:
        strategies = ['TIT FOR TAT', 'ALL-D', 'ALL-C', 'PAVLOV', 'GRIM',
                       'COCA-COLA (TFT Líder)', 'PEPSI (Seguidor Agresivo)']

    results = []
    for pepsi_strat in strategies:
        for coca_strat in strategies:
            sim = simulate_pepsi_coca_strategies(
                pepsi_strategy_name=pepsi_strat,
                coca_strategy_name=coca_strat,
                w=w, n_games=n_games, seed=seed
            )
            results.append({
                'Pepsi Strategy': pepsi_strat,
                'Coca-Cola Strategy': coca_strat,
                'Pepsi Avg Score': round(sim['pepsi_avg_score'], 1),
                'Coca-Cola Avg Score': round(sim['coca_avg_score'], 1),
                'Pepsi Coop Rate': round(sim['pepsi_avg_coop'] * 100, 1),
                'Coca-Cola Coop Rate': round(sim['coca_avg_coop'] * 100, 1),
                'Total Value': round(sim['pepsi_avg_score'] + sim['coca_avg_score'], 1),
            })

    return pd.DataFrame(results)
