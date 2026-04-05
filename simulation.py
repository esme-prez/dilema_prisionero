"""
Motor de simulación para el Dilema del Prisionero Iterado.
Implementa torneo round-robin, dinámica evolutiva y análisis de sensibilidad.
Opción 5: Análisis de Sensibilidad y Condiciones Límite.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from numpy.random import Generator, PCG64
from joblib import Parallel, delayed
from strategies import Strategy, get_all_strategies, get_strategy_by_name


# --- Matriz de pagos ---

def get_payoff(action1: str, action2: str, T=5, R=3, P=1, S=0) -> Tuple[float, float]:
    """Retorna (payoff_jugador1, payoff_jugador2) según la matriz de pagos."""
    if action1 == 'C' and action2 == 'C':
        return (R, R)
    elif action1 == 'C' and action2 == 'D':
        return (S, T)
    elif action1 == 'D' and action2 == 'C':
        return (T, S)
    else:  # D, D
        return (P, P)


# --- Juego iterado ---

def play_iterated_game(
    s1: Strategy, s2: Strategy,
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    rng: Optional[Generator] = None,
    max_rounds: int = 500
) -> Dict:
    """
    Juega un juego iterado del Dilema del Prisionero entre dos estrategias.
    El juego termina probabilísticamente: en cada ronda, continúa con prob w.
    """
    if rng is None:
        rng = Generator(PCG64(42))

    s1.reset()
    s2.reset()

    score1, score2 = 0.0, 0.0
    coop1, coop2 = 0, 0
    n_rounds = 0

    for round_num in range(max_rounds):
        # Decidir si continuar (excepto primera ronda)
        if round_num > 0 and rng.random() > w:
            break

        # Obtener decisiones
        a1 = s1.decide(rng)
        a2 = s2.decide(rng)

        # Aplicar ruido (trembling hand)
        if error_rate > 0:
            if rng.random() < error_rate:
                a1 = 'D' if a1 == 'C' else 'C'
            if rng.random() < error_rate:
                a2 = 'D' if a2 == 'C' else 'C'

        # Calcular pagos
        p1, p2 = get_payoff(a1, a2, T, R, P, S)
        score1 += p1
        score2 += p2

        # Contar cooperación
        if a1 == 'C':
            coop1 += 1
        if a2 == 'C':
            coop2 += 1

        # Actualizar historiales
        s1.update(a1, a2)
        s2.update(a2, a1)

        n_rounds += 1

    return {
        'score1': score1,
        'score2': score2,
        'coop_rate1': coop1 / max(n_rounds, 1),
        'coop_rate2': coop2 / max(n_rounds, 1),
        'n_rounds': n_rounds
    }


# --- Torneo Round-Robin ---

def run_tournament(
    strategies: List[Strategy],
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    n_games: int = 5,
    seed: int = 42
) -> Dict:
    """
    Ejecuta un torneo round-robin completo.
    Cada par juega n_games juegos.
    """
    rng = Generator(PCG64(seed))
    n = len(strategies)
    scores = np.zeros((n, n))
    coop_rates = np.zeros((n, n))
    total_scores = np.zeros(n)
    total_coop = np.zeros(n)
    total_games = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            s1_total, s2_total = 0.0, 0.0
            c1_total, c2_total = 0.0, 0.0

            for _ in range(n_games):
                result = play_iterated_game(
                    strategies[i], strategies[j],
                    w=w, T=T, R=R, P=P, S=S,
                    error_rate=error_rate, rng=rng
                )
                s1_total += result['score1']
                s2_total += result['score2']
                c1_total += result['coop_rate1']
                c2_total += result['coop_rate2']

            scores[i, j] = s1_total / n_games
            scores[j, i] = s2_total / n_games
            coop_rates[i, j] = c1_total / n_games
            coop_rates[j, i] = c2_total / n_games

            total_scores[i] += s1_total
            total_scores[j] += s2_total
            total_coop[i] += c1_total
            total_coop[j] += c2_total
            total_games[i] += n_games
            total_games[j] += n_games

    avg_coop = total_coop / np.maximum(total_games, 1)
    overall_coop = np.mean(avg_coop)

    return {
        'scores_matrix': scores,
        'coop_matrix': coop_rates,
        'total_scores': total_scores,
        'avg_coop_rates': avg_coop,
        'overall_coop_rate': overall_coop,
        'strategy_names': [s.name for s in strategies]
    }


# --- Dinámica Evolutiva ---

def run_evolutionary(
    strategy_names: List[str],
    N: int = 100,
    generations: int = 50,
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    seed: int = 42,
    matches_per_gen: int = 5
) -> Dict:
    """
    Simulación evolutiva con dinámica de imitación.
    Cada generación, agentes juegan contra vecinos aleatorios,
    luego imitan la estrategia del vecino más exitoso.
    """
    rng = Generator(PCG64(seed))

    # Inicializar población: distribución equitativa de estrategias
    n_strategies = len(strategy_names)
    population = []
    for i in range(N):
        idx = i % n_strategies
        population.append(strategy_names[idx])

    history = []  # Track composition over generations

    for gen in range(generations):
        # Count current composition
        counts = {name: 0 for name in strategy_names}
        for s in population:
            counts[s] += 1
        coop_count = sum(
            counts[name] for name in strategy_names
            if get_strategy_by_name(name).cooperative
        )
        history.append({
            'generation': gen,
            'coop_fraction': coop_count / N,
            **{f'frac_{name}': counts[name] / N for name in strategy_names}
        })

        # Each agent plays against random opponents
        fitness = np.zeros(N)
        for i in range(N):
            s1 = get_strategy_by_name(population[i])
            opponents = rng.choice(N, size=matches_per_gen, replace=False)
            for j_idx in opponents:
                if j_idx == i:
                    continue
                s2 = get_strategy_by_name(population[j_idx])
                result = play_iterated_game(s1, s2, w=w, T=T, R=R, P=P, S=S,
                                            error_rate=error_rate, rng=rng, max_rounds=50)
                fitness[i] += result['score1']

        # Imitation dynamics: each agent looks at a random neighbor
        new_population = population.copy()
        for i in range(N):
            j = rng.integers(0, N)
            if j == i:
                continue
            # Fermi function: probability of switching
            beta = 0.1  # Selection intensity
            prob_switch = 1.0 / (1.0 + np.exp(-beta * (fitness[j] - fitness[i])))
            if rng.random() < prob_switch:
                new_population[i] = population[j]

        population = new_population

    # Final composition
    counts = {name: 0 for name in strategy_names}
    for s in population:
        counts[s] += 1
    coop_count = sum(
        counts[name] for name in strategy_names
        if get_strategy_by_name(name).cooperative
    )
    history.append({
        'generation': generations,
        'coop_fraction': coop_count / N,
        **{f'frac_{name}': counts[name] / N for name in strategy_names}
    })

    return {
        'history': pd.DataFrame(history),
        'final_coop_fraction': coop_count / N,
        'final_counts': counts
    }


# --- Análisis de Sensibilidad ---

def _single_replica_tournament(params):
    """Ejecuta una réplica del torneo para paralelización."""
    w, T, R, P, S, error_rate, seed = params
    strategies = get_all_strategies()
    result = run_tournament(strategies, w=w, T=T, R=R, P=P, S=S,
                            error_rate=error_rate, n_games=3, seed=seed)
    return result['overall_coop_rate'], result['total_scores'], result['strategy_names']


def _single_replica_evolutionary(params):
    """Ejecuta una réplica evolutiva para paralelización."""
    strategy_names, N, generations, w, T, R, P, S, error_rate, seed = params
    result = run_evolutionary(
        strategy_names, N=N, generations=generations,
        w=w, T=T, R=R, P=P, S=S, error_rate=error_rate,
        seed=seed, matches_per_gen=min(5, N - 1)
    )
    return result['final_coop_fraction']


def sensitivity_vary_w(
    w_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999],
    n_replicas: int = 50,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Experimento 1: Variar probabilidad de interacción futura (w)."""
    results = []
    for w in w_values:
        params_list = [(w, T, R, P, S, error_rate, 42 + i) for i in range(n_replicas)]
        coop_rates = Parallel(n_jobs=n_jobs)(
            delayed(_single_replica_tournament)(p) for p in params_list
        )
        rates = [r[0] for r in coop_rates]
        for i, rate in enumerate(rates):
            results.append({'w': w, 'replica': i, 'coop_rate': rate})

    return pd.DataFrame(results)


def sensitivity_vary_payoff(
    tr_ratios: List[float] = [1.2, 1.5, 2.0, 2.5, 3.0],
    n_replicas: int = 50,
    w: float = 0.995,
    error_rate: float = 0.0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Experimento 2: Variar ratio T/R en la matriz de pagos."""
    R = 3
    results = []
    for ratio in tr_ratios:
        T = R * ratio
        params_list = [(w, T, R, 1, 0, error_rate, 42 + i) for i in range(n_replicas)]
        coop_rates = Parallel(n_jobs=n_jobs)(
            delayed(_single_replica_tournament)(p) for p in params_list
        )
        rates = [r[0] for r in coop_rates]
        for i, rate in enumerate(rates):
            results.append({'T_R_ratio': ratio, 'T': T, 'replica': i, 'coop_rate': rate})

    return pd.DataFrame(results)


def sensitivity_vary_noise(
    error_rates: List[float] = [0.0, 0.01, 0.05, 0.10],
    n_replicas: int = 50,
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Experimento 3: Variar ruido (trembling hand)."""
    results = []
    for err in error_rates:
        params_list = [(w, T, R, P, S, err, 42 + i) for i in range(n_replicas)]
        coop_rates = Parallel(n_jobs=n_jobs)(
            delayed(_single_replica_tournament)(p) for p in params_list
        )
        rates = [r[0] for r in coop_rates]
        for i, rate in enumerate(rates):
            results.append({'error_rate': err, 'replica': i, 'coop_rate': rate})

    return pd.DataFrame(results)


def sensitivity_vary_population(
    n_values: List[int] = [10, 50, 100, 500, 1000],
    n_replicas: int = 30,
    generations: int = 30,
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Experimento 4: Variar tamaño de población (drift estocástico)."""
    strategy_names = ['TIT FOR TAT', 'ALL-D', 'PAVLOV', 'GRIM', 'ALL-C', 'RANDOM']
    results = []
    for N in n_values:
        params_list = [
            (strategy_names, N, generations, w, T, R, P, S, error_rate, 42 + i)
            for i in range(n_replicas)
        ]
        coop_fracs = Parallel(n_jobs=n_jobs)(
            delayed(_single_replica_evolutionary)(p) for p in params_list
        )
        for i, frac in enumerate(coop_fracs):
            results.append({'N': N, 'replica': i, 'coop_rate': frac})

    return pd.DataFrame(results)


def sensitivity_surface_w_noise(
    w_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99],
    error_rates: List[float] = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10],
    n_replicas: int = 20,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Genera datos para superficie 3D: w vs ruido vs cooperación."""
    results = []
    all_params = []
    for w in w_values:
        for err in error_rates:
            for i in range(n_replicas):
                all_params.append((w, 5, 3, 1, 0, err, 42 + i, w, err))

    # Flatten for parallel execution
    params_flat = [(w, T, R, P, S, err, seed)
                   for (w, T, R, P, S, err, seed, _, _) in all_params]
    coop_rates = Parallel(n_jobs=n_jobs)(
        delayed(_single_replica_tournament)(p) for p in params_flat
    )

    idx = 0
    for w in w_values:
        for err in error_rates:
            rates = []
            for _ in range(n_replicas):
                rates.append(coop_rates[idx][0])
                idx += 1
            results.append({
                'w': w,
                'error_rate': err,
                'coop_rate_mean': np.mean(rates),
                'coop_rate_std': np.std(rates),
            })

    return pd.DataFrame(results)


def sensitivity_surface_w_payoff(
    w_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99],
    tr_ratios: List[float] = [1.2, 1.5, 2.0, 2.5, 3.0],
    n_replicas: int = 20,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Genera datos para superficie 3D: w vs T/R vs cooperación."""
    R = 3
    results = []
    all_params = []
    for w in w_values:
        for ratio in tr_ratios:
            T = R * ratio
            for i in range(n_replicas):
                all_params.append((w, T, R, 1, 0, 0.0, 42 + i))

    coop_rates = Parallel(n_jobs=n_jobs)(
        delayed(_single_replica_tournament)(p) for p in all_params
    )

    idx = 0
    for w in w_values:
        for ratio in tr_ratios:
            rates = []
            for _ in range(n_replicas):
                rates.append(coop_rates[idx][0])
                idx += 1
            results.append({
                'w': w,
                'T_R_ratio': ratio,
                'coop_rate_mean': np.mean(rates),
                'coop_rate_std': np.std(rates),
            })

    return pd.DataFrame(results)


# --- Matriz de Desviación respecto a TIT FOR TAT ---

def compute_tft_deviation_matrix(
    w: float = 0.995,
    T: float = 5, R: float = 3, P: float = 1, S: float = 0,
    error_rate: float = 0.0,
    n_games: int = 10,
    seed: int = 42
) -> Dict:
    """
    Calcula la desviación de score de cada estrategia respecto a TFT.

    Para cada par (estrategia S, oponente O):
        delta(S, O) = Score(S vs O) - Score(TFT vs O)

    Positivo = supera a TFT contra ese oponente
    Negativo = peor que TFT contra ese oponente

    Retorna:
        - deviation_matrix: matriz NxN de desviaciones
        - tft_scores: vector de scores de TFT contra cada oponente (referencia)
        - scores_matrix: matriz NxN de scores absolutos
        - aggregate_deviation: desviación total por estrategia (suma de columnas)
        - strategy_names: lista de nombres
    """
    rng = Generator(PCG64(seed))
    strategies = get_all_strategies()
    names = [s.name for s in strategies]
    n = len(strategies)

    # Scores absolutos: scores[i,j] = score de estrategia i contra oponente j
    scores = np.zeros((n, n))
    coop_rates = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-play
                s1 = get_strategy_by_name(names[i])
                s2 = get_strategy_by_name(names[i])
            else:
                s1 = get_strategy_by_name(names[i])
                s2 = get_strategy_by_name(names[j])

            total_score = 0.0
            total_coop = 0.0
            for _ in range(n_games):
                result = play_iterated_game(s1, s2, w=w, T=T, R=R, P=P, S=S,
                                            error_rate=error_rate, rng=rng)
                total_score += result['score1']
                total_coop += result['coop_rate1']

            scores[i, j] = total_score / n_games
            coop_rates[i, j] = total_coop / n_games

    # Encontrar índice de TFT
    tft_idx = names.index('TIT FOR TAT')
    tft_scores = scores[tft_idx, :]  # Score de TFT contra cada oponente

    # Matriz de desviación: delta[i,j] = score[i,j] - score[TFT,j]
    deviation = scores - tft_scores[np.newaxis, :]

    # Normalizar: desviación porcentual respecto a TFT
    tft_safe = np.where(tft_scores > 0, tft_scores, 1.0)
    deviation_pct = (deviation / tft_safe) * 100

    # Agregados por estrategia
    aggregate_deviation = np.mean(deviation, axis=1)
    aggregate_deviation_pct = np.mean(deviation_pct, axis=1)

    # Ranking por score total
    total_scores = np.sum(scores, axis=1)
    ranking_idx = np.argsort(-total_scores)

    # Identificar estrategia dominante
    dominant_idx = ranking_idx[0]
    dominant_name = names[dominant_idx]

    return {
        'scores_matrix': scores,
        'coop_matrix': coop_rates,
        'deviation_matrix': deviation,
        'deviation_pct_matrix': deviation_pct,
        'tft_scores': tft_scores,
        'aggregate_deviation': aggregate_deviation,
        'aggregate_deviation_pct': aggregate_deviation_pct,
        'total_scores': total_scores,
        'ranking_idx': ranking_idx,
        'dominant_strategy': dominant_name,
        'dominant_deviation_from_tft': aggregate_deviation[dominant_idx],
        'strategy_names': names,
        'tft_idx': tft_idx,
    }


def compute_equilibrium_analysis(
    w_values: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999],
    n_games: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Para cada valor de w, identifica la estrategia ganadora del torneo
    y calcula su desviación respecto a TFT.
    Muestra cómo el equilibrio cambia con w.
    """
    results = []
    for w in w_values:
        dev = compute_tft_deviation_matrix(w=w, n_games=n_games, seed=seed)

        # Top 3
        ranking = dev['ranking_idx']
        names = dev['strategy_names']

        # w crítico teórico para TFT
        T, R, P, S = 5, 3, 1, 0
        w_crit = max((T - R) / (T - P), (T - R) / (R - S))

        results.append({
            'w': w,
            'w_critico': w_crit,
            'tft_es_estable': w >= w_crit,
            'ganador': names[ranking[0]],
            'ganador_score': dev['total_scores'][ranking[0]],
            'tft_score': dev['total_scores'][dev['tft_idx']],
            'tft_rank': list(ranking).index(dev['tft_idx']) + 1,
            'desviacion_ganador_vs_tft': dev['aggregate_deviation'][ranking[0]],
            'desviacion_ganador_vs_tft_pct': dev['aggregate_deviation_pct'][ranking[0]],
            'top2': names[ranking[1]],
            'top3': names[ranking[2]],
        })

    return pd.DataFrame(results)


# --- Detección de Phase Transitions ---

def detect_phase_transition(df: pd.DataFrame, param_col: str, value_col: str = 'coop_rate') -> Dict:
    """
    Detecta transiciones de fase encontrando el punto donde la derivada es máxima.
    """
    grouped = df.groupby(param_col)[value_col].agg(['mean', 'std']).reset_index()
    grouped.columns = [param_col, 'mean', 'std']

    # Calcular derivada numérica
    x = grouped[param_col].values
    y = grouped['mean'].values

    if len(x) < 3:
        return {'transition_point': None, 'data': grouped}

    dy = np.gradient(y, x)
    max_deriv_idx = np.argmax(np.abs(dy))

    return {
        'transition_point': x[max_deriv_idx],
        'transition_value': y[max_deriv_idx],
        'max_derivative': dy[max_deriv_idx],
        'data': grouped,
        'derivatives': dy
    }
