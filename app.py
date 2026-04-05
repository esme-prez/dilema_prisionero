"""
Dashboard Interactivo - Dilema del Prisionero Iterado
Opcion 5: Analisis de Sensibilidad y Condiciones Limite
Replicacion Axelrod & Hamilton (1981)

Ejecutar con: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Configuracion de pagina
st.set_page_config(
    page_title="Dilema del Prisionero - Analisis de Sensibilidad",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Imports del proyecto ---
from rng_module import ReproducibleRNG
from strategies import (
    get_all_strategies, get_strategy_by_name,
    AXELROD_PROPERTIES, get_properties_dataframe
)
from simulation import (
    run_tournament, play_iterated_game, run_evolutionary,
    sensitivity_vary_w, sensitivity_vary_payoff,
    sensitivity_vary_noise, sensitivity_vary_population,
    sensitivity_surface_w_noise, sensitivity_surface_w_payoff,
    detect_phase_transition,
    compute_tft_deviation_matrix, compute_equilibrium_analysis,
    get_payoff
)
from rng_tests import run_all_tests
from utils import (
    export_to_csv, export_to_json, compute_anova,
    logistic_regression_fit, summary_statistics, confidence_interval
)
from pepsi_coca import (
    get_price_data, get_market_share, historical_actions,
    simulate_pepsi_coca_strategies, run_all_strategy_combinations,
    get_duopoly_payoff_matrix
)

# --- CSS ---
st.markdown("""
<style>
    .main-header {font-size:2.2rem;font-weight:bold;color:#1f4e79;text-align:center;margin-bottom:0.5rem;}
    .sub-header {font-size:1.1rem;color:#555;text-align:center;margin-bottom:2rem;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {padding:8px 16px;border-radius:4px 4px 0 0;}
    .converge-yes {color:#009988;font-weight:bold;}
    .converge-no {color:#CC3311;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# --- Colorblind-friendly palette ---
COLORS = {
    'blue': '#0077BB', 'orange': '#EE7733', 'green': '#009988',
    'red': '#CC3311', 'purple': '#AA3377', 'cyan': '#33BBEE',
    'grey': '#BBBBBB', 'yellow': '#EE3377',
}
COLOR_LIST = list(COLORS.values())


# ============================================================
# TABS PRINCIPALES
# ============================================================
tabs = st.tabs([
    "Inicio",
    "Marco Teorico y Equilibrio",
    "Simulacion",
    "Resultados",
    "Analisis",
    "Pepsi vs Coca-Cola",
    "Referencias",
    "Anexos Tecnicos"
])

# ============================================================
# TAB 0: INICIO
# ============================================================
with tabs[0]:
    st.markdown('<div class="main-header">Evolucion de la Cooperacion: Dilema del Prisionero Iterado</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Replicacion Axelrod & Hamilton (1981) - Opcion 5: Analisis de Sensibilidad y Condiciones Limite</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### El Experimento de Axelrod (1981)

        En 1980, Robert Axelrod invito a expertos en teoria de juegos a enviar estrategias
        computacionales para un **torneo de Dilema del Prisionero Iterado (IPD)**. Cada estrategia
        jugaba contra todas las demas en multiples rondas, acumulando puntos segun la matriz de pagos.

        **El resultado fue sorprendente:** la estrategia mas simple, **TIT FOR TAT** (enviada por
        Anatol Rapoport), gano ambos torneos. TFT tiene solo 4 lineas de codigo:

        > *Coopera en la primera ronda. Despues, haz lo que el oponente hizo en la ronda anterior.*

        Axelrod identifico **4 propiedades** que explican por que TFT es la estrategia de equilibrio:

        | Propiedad | Significado | Por que importa |
        |-----------|-------------|-----------------|
        | **Amable** | Nunca deserta primero | Evita provocar espirales de retaliacion |
        | **Provocable** | Castiga la desercion inmediatamente | Disuade la explotacion |
        | **Perdonadora** | Vuelve a cooperar si el oponente coopera | Permite restaurar cooperacion |
        | **Clara** | Comportamiento predecible | El oponente puede anticipar consecuencias |

        ### Opcion 5: Analisis de Sensibilidad

        Este proyecto explora **bajo que condiciones** la cooperacion (y la dominancia de TFT)
        se mantiene, colapsa o es superada:

        1. **w (sombra del futuro)**: Cuando la probabilidad de interaccion futura es baja, la desercion paga
        2. **T/R (tentacion relativa)**: Mayor tentacion erosiona la cooperacion
        3. **Ruido (trembling hand)**: Errores aleatorios destruyen la reciprocidad de TFT
        4. **N (tamano de poblacion)**: Poblaciones pequenas sufren drift estocastico
        """)

    with col2:
        st.markdown("### Matriz de Pagos Estandar")
        st.markdown("*(Jugador fila, Jugador columna)*")
        payoff_df = pd.DataFrame(
            [["(R=3, R=3)", "(S=0, T=5)"],
             ["(T=5, S=0)", "(P=1, P=1)"]],
            columns=["Cooperar (C)", "Desertar (D)"],
            index=["Cooperar (C)", "Desertar (D)"]
        )
        st.table(payoff_df)

        st.markdown("""
        **Restricciones del juego:**
        - T > R > P > S  *(desertar unilateral > cooperar > mutual desercion > ser explotado)*
        - 2R > T + S  *(cooperacion mutua > alternar explotacion)*

        **Condicion de estabilidad de TFT:**

        w >= max((T-R)/(T-P), (T-R)/(R-S))

        Con pagos estandar: **w >= 0.5**

        *Esto significa que TFT es estrategia de equilibrio
        cuando hay al menos 50% de probabilidad de
        volver a interactuar.*
        """)

    st.divider()
    st.markdown("### Estrategias Implementadas y Propiedades de Axelrod")
    props_df = get_properties_dataframe()
    st.dataframe(props_df, use_container_width=True, hide_index=True)
    st.caption("Las estrategias con 4/4 propiedades de Axelrod son candidatas a equilibrio. "
               "TFT es la referencia (benchmark) contra la cual se miden todas las demas.")


# ============================================================
# TAB 1: MARCO TEORICO Y EQUILIBRIO
# ============================================================
with tabs[1]:
    st.header("Marco Teorico: Estrategia de Equilibrio y Desviacion respecto a TFT")

    marco_tabs = st.tabs([
        "Analisis Cualitativo vs TFT",
        "Matriz de Desviacion (Escalar)",
        "Equilibrio segun w",
        "Convergencia y Divergencia"
    ])

    # --- Tab: Análisis Cualitativo ---
    with marco_tabs[0]:
        st.subheader("Diferencia Cualitativa de Cada Estrategia respecto a TIT FOR TAT")

        st.markdown("""
        Axelrod demostro que TFT es la **estrategia de equilibrio** del IPD porque
        maximiza el score total en un torneo round-robin. A continuacion se analiza
        **cualitativamente** por que cada estrategia se desvía del equilibrio de TFT.
        """)

        for name, props in AXELROD_PROPERTIES.items():
            with st.expander(f"{'🟢' if props['convergence_bool'] else '🔴'} {name} — "
                             f"{'Converge' if props['convergence_bool'] else 'No converge'} al equilibrio",
                             expanded=(name == 'TIT FOR TAT')):

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Propiedades de Axelrod:**")
                    for prop_name, prop_key in [('Amable', 'nice'), ('Provocable', 'retaliatory'),
                                                 ('Perdonadora', 'forgiving'), ('Clara', 'clear')]:
                        icon = "✅" if props[prop_key] else "❌"
                        match = " *(= TFT)*" if props[prop_key] == AXELROD_PROPERTIES['TIT FOR TAT'][prop_key] else " ***(≠ TFT)***"
                        st.markdown(f"- {icon} {prop_name}{match}")

                    n_match = sum(1 for k in ['nice','retaliatory','forgiving','clear']
                                  if props[k] == AXELROD_PROPERTIES['TIT FOR TAT'][k])
                    st.metric("Similitud con TFT", f"{n_match}/4 propiedades")

                with col2:
                    st.markdown(f"**Diferencia clave vs TFT:**")
                    st.info(props['description_vs_tft'])
                    st.markdown(f"**Convergencia al equilibrio:**")
                    if props['convergence_bool']:
                        st.success(props['convergence'])
                    else:
                        st.error(props['convergence'])

    # --- Tab: Matriz de Desviación ---
    with marco_tabs[1]:
        st.subheader("Matriz de Desviacion Escalar: Score(S, O) - Score(TFT, O)")

        st.markdown("""
        Esta matriz muestra, para cada par *(estrategia S, oponente O)*, cuanto
        **mejor o peor** es S comparada con TFT cuando ambas enfrentan al mismo oponente O.

        - **Verde (positivo)**: S supera a TFT contra ese oponente
        - **Rojo (negativo)**: S pierde frente a TFT contra ese oponente
        - **Blanco (cero)**: Rendimiento identico a TFT

        La fila de TFT es siempre cero (referencia). La columna de la derecha
        muestra la **desviacion agregada** (promedio contra todos los oponentes).
        """)

        col1, col2 = st.columns([1, 3])
        with col1:
            w_dev = st.slider("w", 0.1, 1.0, 0.995, 0.005, key="w_dev")
            err_dev = st.slider("Tasa de error", 0.0, 0.15, 0.0, 0.01, key="err_dev")
            ng_dev = st.slider("Juegos por par", 3, 20, 8, key="ng_dev")

        if st.button("Calcular Matriz de Desviacion", key="calc_dev"):
            with st.spinner("Calculando desviaciones..."):
                dev = compute_tft_deviation_matrix(w=w_dev, error_rate=err_dev,
                                                    n_games=ng_dev, seed=42)
            st.session_state['dev_matrix'] = dev

        if 'dev_matrix' in st.session_state:
            dev = st.session_state['dev_matrix']
            names = dev['strategy_names']

            with col2:
                # Heatmap de desviación
                fig = go.Figure(data=go.Heatmap(
                    z=dev['deviation_matrix'],
                    x=names, y=names,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(dev['deviation_matrix'], 1),
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title='δ vs TFT'),
                    hovertemplate='%{y} vs %{x}<br>Desviacion: %{z:.1f}<extra></extra>'
                ))
                fig.update_layout(
                    title=f"Matriz de Desviacion respecto a TFT (w={w_dev}, error={err_dev:.0%})",
                    xaxis_title="Oponente",
                    yaxis_title="Estrategia",
                    height=550,
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tabla resumen de desviación agregada
            st.markdown("#### Desviacion Agregada (promedio contra todos los oponentes)")
            agg_df = pd.DataFrame({
                'Estrategia': names,
                'Score Total': dev['total_scores'].round(1),
                'Desv. vs TFT': dev['aggregate_deviation'].round(1),
                'Desv. % vs TFT': [f"{v:+.1f}%" for v in dev['aggregate_deviation_pct']],
                'Rank': [0]*len(names)
            })
            # Fill rank
            for rank, idx in enumerate(dev['ranking_idx']):
                agg_df.loc[idx, 'Rank'] = rank + 1
            agg_df = agg_df.sort_values('Rank')
            st.dataframe(agg_df, use_container_width=True, hide_index=True)

            # Dominante vs TFT
            dominant = dev['dominant_strategy']
            tft_rank = list(dev['ranking_idx']).index(dev['tft_idx']) + 1
            dom_dev = dev['dominant_deviation_from_tft']

            st.markdown("---")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Estrategia Dominante", dominant)
            mc2.metric("Desv. Dominante vs TFT", f"{dom_dev:+.1f} pts/oponente")
            mc3.metric("Rank de TFT", f"#{tft_rank} de {len(names)}")

            # Explicación
            if dominant == 'TIT FOR TAT':
                st.success("TFT es la estrategia dominante en estas condiciones, "
                           "confirmando la prediccion de Axelrod.")
            elif tft_rank <= 3:
                st.info(f"TFT esta en el Top 3 (#{tft_rank}). {dominant} la supera por "
                        f"{dom_dev:+.1f} puntos promedio por oponente. "
                        f"Esto puede deberse a condiciones de ruido/pagos que favorecen "
                        f"mecanismos diferentes al reciproco exacto de TFT.")
            else:
                st.warning(f"TFT cae al puesto #{tft_rank}. En estas condiciones "
                           f"(w={w_dev}, error={err_dev:.0%}), las propiedades de TFT "
                           f"no son suficientes para dominar. {dominant} se beneficia de "
                           f"un mecanismo mejor adaptado a estos parametros.")

            # Bar chart de desviación
            fig2 = go.Figure()
            colors = [COLORS['green'] if v > 0 else COLORS['red'] if v < 0 else COLORS['grey']
                      for v in dev['aggregate_deviation']]
            sorted_idx = np.argsort(dev['aggregate_deviation'])[::-1]
            fig2.add_trace(go.Bar(
                x=[names[i] for i in sorted_idx],
                y=[dev['aggregate_deviation'][i] for i in sorted_idx],
                marker_color=[colors[i] for i in sorted_idx],
                text=[f"{dev['aggregate_deviation'][i]:+.1f}" for i in sorted_idx],
                textposition='outside'
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig2.update_layout(
                title="Desviacion Agregada vs TFT (positivo = supera a TFT)",
                xaxis_title="Estrategia",
                yaxis_title="Desviacion promedio vs TFT",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

            csv_dev = export_to_csv(agg_df)
            st.download_button("Descargar CSV", csv_dev, "desviacion_tft.csv", "text/csv")

    # --- Tab: Equilibrio según w ---
    with marco_tabs[2]:
        st.subheader("Como Cambia la Estrategia de Equilibrio con w")

        st.markdown("""
        La prediccion teorica de Axelrod dice que **TFT es estable cuando w >= 0.5**.
        Pero en la practica, la estrategia *ganadora* del torneo depende de la composicion
        del campo de competidores. Aqui se muestra que estrategia gana para cada valor de w
        y como se compara con TFT.
        """)

        if st.button("Calcular Equilibrio para Todos los w", key="calc_eq"):
            with st.spinner("Analizando equilibrio para 8 valores de w..."):
                eq_df = compute_equilibrium_analysis(n_games=5, seed=42)
            st.session_state['eq_analysis'] = eq_df

        if 'eq_analysis' in st.session_state:
            eq_df = st.session_state['eq_analysis']

            # Tabla principal
            display_df = eq_df[['w', 'tft_es_estable', 'ganador', 'tft_rank',
                                'desviacion_ganador_vs_tft', 'top2', 'top3']].copy()
            display_df.columns = ['w', 'TFT Estable (teoria)', 'Ganador Torneo',
                                  'Rank TFT', 'Desv. Ganador vs TFT', 'Top 2', 'Top 3']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Gráfica: rank de TFT vs w
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df['w'], y=eq_df['tft_rank'],
                mode='lines+markers+text',
                text=eq_df['ganador'],
                textposition='top center',
                textfont=dict(size=9),
                line=dict(color=COLORS['blue'], width=3),
                marker=dict(size=12)
            ))
            fig.add_hline(y=1, line_dash="dash", line_color=COLORS['green'],
                          annotation_text="TFT gana (Rank #1)")
            fig.add_vline(x=0.5, line_dash="dot", line_color=COLORS['red'],
                          annotation_text="w critico = 0.5")
            fig.update_layout(
                title="Ranking de TFT en el Torneo segun w",
                xaxis_title="w (probabilidad de interaccion futura)",
                yaxis_title="Ranking de TFT (1 = ganador)",
                yaxis=dict(autorange='reversed'),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # Gráfica: desviación del ganador vs TFT
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[str(w) for w in eq_df['w']],
                y=eq_df['desviacion_ganador_vs_tft'],
                text=eq_df['ganador'],
                textposition='outside',
                textfont=dict(size=9),
                marker_color=[COLORS['green'] if g == 'TIT FOR TAT' else COLORS['orange']
                              for g in eq_df['ganador']]
            ))
            fig2.add_hline(y=0, line_dash="dash", line_color="black")
            fig2.update_layout(
                title="Desviacion del Ganador respecto a TFT (por valor de w)",
                xaxis_title="w",
                yaxis_title="Desv. score ganador vs TFT (promedio/oponente)",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
            #### Interpretacion

            - Cuando **w es bajo** (< 0.5), los juegos son cortos. La primera ronda domina
              el resultado total. Estrategias que desertan primero (TESTER, ALL-D) pueden
              obtener T=5 sin sufrir retaliacion suficiente.

            - Cuando **w es alto** (> 0.5), los juegos son largos. La cooperacion sostenida
              vale mas que una desercion rapida. TFT y estrategias similares (PAVLOV, GRADUAL)
              dominan porque la retaliacion repetida castiga a los desertores.

            - **TFT no siempre gana** el torneo porque su score depende del *campo de competidores*.
              Contra un campo con muchos cooperadores ingenuos (ALL-C), estrategias parcialmente
              explotadoras pueden superar a TFT. Pero TFT es **evolutivamente estable**: en
              dinamica evolutiva, una poblacion de TFT no puede ser invadida.
            """)

    # --- Tab: Convergencia ---
    with marco_tabs[3]:
        st.subheader("Por que Converge o No Converge al Equilibrio")

        st.markdown("""
        ### Condicion Formal de Convergencia

        Una estrategia S converge al equilibrio cooperativo del IPD si cumple:

        1. **Contra cooperadores**: S obtiene score >= R por ronda (no autolesiona)
        2. **Contra desertores**: S castiga lo suficiente para que la desercion no pague
        3. **Estabilidad evolutiva**: Una poblacion de S no puede ser invadida por mutantes

        Formalmente, TFT es un **Equilibrio de Nash** del juego iterado cuando:

        **w >= max((T-R)/(T-P), (T-R)/(R-S))**

        Con T=5, R=3, P=1, S=0: **w >= max(2/4, 2/3) = 0.667**

        Esto significa que si ambos jugadores usan TFT y w >= 0.667, ninguno tiene
        incentivo unilateral para cambiar de estrategia.
        """)

        st.markdown("### Analisis de Convergencia por Estrategia")

        # Tabla resumen
        conv_data = []
        for name, props in AXELROD_PROPERTIES.items():
            missing = []
            for prop_name, prop_key in [('Amable', 'nice'), ('Provocable', 'retaliatory'),
                                         ('Perdonadora', 'forgiving'), ('Clara', 'clear')]:
                if not props[prop_key]:
                    missing.append(prop_name)

            conv_data.append({
                'Estrategia': name,
                'Converge': 'Si' if props['convergence_bool'] else 'No',
                'Propiedades faltantes': ', '.join(missing) if missing else 'Ninguna (4/4)',
                'Razon principal': {
                    'TIT FOR TAT': 'Referencia: cumple las 4 propiedades',
                    'ALL-C': 'Sin retaliacion → explotable por desertores',
                    'ALL-D': 'Sin amabilidad ni perdon → atrapada en desercion mutua',
                    'GRIM': 'Sin perdon → un error causa desercion permanente',
                    'PAVLOV': 'Autocorreccion Win-Stay/Lose-Shift',
                    'TFT2T': 'Retraso en retaliacion → vulnerable a explotacion intermitente',
                    'RANDOM': 'Sin estructura → no puede establecer patrones',
                    'JOSS': 'Desercion aleatoria → rompe reciprocidad con reactivas',
                    'GRADUAL': 'Castigo proporcional + reconciliacion explicita',
                    'ADAPTIVE': 'Depende de condiciones iniciales (path-dependent)',
                    'FRIEDMAN': 'Sin perdon → fragil ante ruido',
                    'TESTER': 'Desercion inicial → parasito del ecosistema cooperativo',
                }[name]
            })
        st.dataframe(pd.DataFrame(conv_data), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Por que TFT es el Equilibrio y Otras No

        El argumento central de Axelrod es que la cooperacion emerge cuando se cumplen
        **simultaneamente** las 4 propiedades. Perder cualquiera tiene consecuencias:

        | Propiedad perdida | Consecuencia | Ejemplo |
        |-------------------|--------------|---------|
        | **Amabilidad** | Provoca retaliacion innecesaria, reduce score mutuo | ALL-D, JOSS, TESTER |
        | **Provocabilidad** | Es explotada sin costo, incentiva desercion | ALL-C |
        | **Perdon** | Errores causan castigo permanente, cooperacion irreversible | GRIM, FRIEDMAN |
        | **Claridad** | Oponente no puede predecir, dificulta coordinacion | RANDOM, ADAPTIVE |

        ### El Caso Especial de PAVLOV

        PAVLOV (Nowak & Sigmund, 1993) es la unica estrategia que puede **superar a TFT**
        en ciertas condiciones. Su regla Win-Stay/Lose-Shift le permite:

        - **Autocorregir errores** sin necesidad de que el oponente coopere primero
        - **Explotar cooperadores ingenuos** (a diferencia de TFT que coopera indefinidamente con ALL-C)

        Sin embargo, PAVLOV es vulnerable ante ALL-D en juegos cortos porque su ciclo
        D-C-D-C contra ALL-D le da score medio (T+S)/2 = 2.5, peor que P=1 pero no tan
        bueno como R=3.
        """)


# ============================================================
# TAB 2: SIMULACION
# ============================================================
with tabs[2]:
    st.header("Simulacion Interactiva")
    sim_tab1, sim_tab2 = st.tabs(["Juego Individual", "Torneo Completo"])

    with sim_tab1:
        st.subheader("Juego Iterado entre Dos Estrategias")
        col1, col2, col3 = st.columns(3)
        strat_names = [s.name for s in get_all_strategies()]

        with col1:
            s1_name = st.selectbox("Estrategia Jugador 1", strat_names, index=0)
            s2_name = st.selectbox("Estrategia Jugador 2", strat_names, index=2)
        with col2:
            w_sim = st.slider("Probabilidad w", 0.0, 1.0, 0.995, 0.005, key="w_single")
            error_sim = st.slider("Tasa de error", 0.0, 0.20, 0.0, 0.01, key="err_single")
        with col3:
            T_sim = st.number_input("T (Tentacion)", value=5.0, min_value=0.1, key="T_single")
            R_sim = st.number_input("R (Recompensa)", value=3.0, min_value=0.1, key="R_single")
            P_sim = st.number_input("P (Castigo)", value=1.0, min_value=0.0, key="P_single")
            S_sim = st.number_input("S (Sucker)", value=0.0, min_value=0.0, key="S_single")

        if st.button("Jugar", key="play_single"):
            if not (T_sim > R_sim > P_sim > S_sim):
                st.error("Error: Se requiere T > R > P > S")
            else:
                s1 = get_strategy_by_name(s1_name)
                s2 = get_strategy_by_name(s2_name)
                from numpy.random import Generator as Gen, PCG64 as P64
                rng_game = Gen(P64(42))
                s1.reset(); s2.reset()
                history = []
                sc1, sc2 = 0, 0
                for r in range(500):
                    if r > 0 and rng_game.random() > w_sim:
                        break
                    a1, a2 = s1.decide(rng_game), s2.decide(rng_game)
                    if error_sim > 0:
                        if rng_game.random() < error_sim: a1 = 'D' if a1 == 'C' else 'C'
                        if rng_game.random() < error_sim: a2 = 'D' if a2 == 'C' else 'C'
                    p1, p2 = get_payoff(a1, a2, T_sim, R_sim, P_sim, S_sim)
                    sc1 += p1; sc2 += p2
                    s1.update(a1, a2); s2.update(a2, a1)
                    history.append({'Ronda': r+1, s1_name: a1, s2_name: a2,
                                    f'Score {s1_name}': sc1, f'Score {s2_name}': sc2})
                hdf = pd.DataFrame(history)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(f"Score {s1_name}", f"{sc1:.0f}")
                mc2.metric(f"Score {s2_name}", f"{sc2:.0f}")
                mc3.metric("Rondas jugadas", len(history))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hdf['Ronda'], y=hdf[f'Score {s1_name}'],
                                         name=s1_name, line=dict(color=COLORS['blue'], width=2)))
                fig.add_trace(go.Scatter(x=hdf['Ronda'], y=hdf[f'Score {s2_name}'],
                                         name=s2_name, line=dict(color=COLORS['orange'], width=2)))
                fig.update_layout(title="Scores Acumulados", xaxis_title="Ronda",
                                  yaxis_title="Score", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Acciones
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=hdf['Ronda'], y=[1.1]*len(hdf), mode='markers', name=s1_name,
                    marker=dict(color=[COLORS['green'] if a=='C' else COLORS['red'] for a in hdf[s1_name]],
                                size=6, symbol='square')))
                fig2.add_trace(go.Scatter(
                    x=hdf['Ronda'], y=[0.9]*len(hdf), mode='markers', name=s2_name,
                    marker=dict(color=[COLORS['green'] if a=='C' else COLORS['red'] for a in hdf[s2_name]],
                                size=6, symbol='square')))
                fig2.update_layout(title="Acciones (Verde=C, Rojo=D)", height=200,
                                   yaxis=dict(tickvals=[0.9,1.1], ticktext=[s2_name, s1_name]),
                                   showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

                # Contexto teórico del resultado
                p1_props = AXELROD_PROPERTIES.get(s1_name, {})
                p2_props = AXELROD_PROPERTIES.get(s2_name, {})
                if p1_props and p2_props:
                    st.markdown("---")
                    st.markdown("#### Interpretacion Teorica")
                    if p1_props.get('nice') and p2_props.get('nice'):
                        st.success("Ambas estrategias son **amables** (nunca desertan primero). "
                                   "Segun Axelrod, dos estrategias amables siempre convergen a "
                                   "cooperacion mutua sostenida (score por ronda ≈ R=3).")
                    elif not p1_props.get('nice') and not p2_props.get('nice'):
                        st.error("Ninguna estrategia es amable. Ambas pueden desertar primero, "
                                 "lo que tiende a producir espirales de desercion mutua "
                                 "(score por ronda ≈ P=1).")
                    else:
                        nice_one = s1_name if p1_props.get('nice') else s2_name
                        mean_one = s2_name if p1_props.get('nice') else s1_name
                        st.warning(f"**{nice_one}** es amable pero **{mean_one}** no lo es. "
                                   f"El resultado depende de si {nice_one} tiene suficiente "
                                   f"provocabilidad para castigar las deserciones de {mean_one}.")

    with sim_tab2:
        st.subheader("Torneo Round-Robin")
        col1, col2 = st.columns(2)
        with col1:
            selected_strats = st.multiselect("Selecciona estrategias", strat_names, default=strat_names[:8])
            w_tourn = st.slider("Probabilidad w", 0.0, 1.0, 0.995, 0.005, key="w_tourn")
        with col2:
            n_games_tourn = st.slider("Juegos por par", 1, 20, 5, key="ngames")
            error_tourn = st.slider("Tasa de error", 0.0, 0.20, 0.0, 0.01, key="err_tourn")

        if st.button("Ejecutar Torneo", key="run_tourn"):
            if len(selected_strats) < 2:
                st.error("Selecciona al menos 2 estrategias.")
            else:
                with st.spinner("Ejecutando torneo..."):
                    strategies = [get_strategy_by_name(n) for n in selected_strats]
                    result = run_tournament(strategies, w=w_tourn, error_rate=error_tourn,
                                           n_games=n_games_tourn, seed=42)
                fig = px.imshow(result['scores_matrix'], x=result['strategy_names'],
                                y=result['strategy_names'],
                                labels=dict(x="Oponente", y="Estrategia", color="Score"),
                                title="Matriz de Scores", color_continuous_scale='Viridis', aspect='auto')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                ranking = pd.DataFrame({
                    'Estrategia': result['strategy_names'],
                    'Score Total': result['total_scores'],
                    'Tasa Cooperacion': [f"{r:.1%}" for r in result['avg_coop_rates']]
                }).sort_values('Score Total', ascending=False).reset_index(drop=True)
                ranking.index += 1; ranking.index.name = 'Rank'
                st.dataframe(ranking, use_container_width=True)

                # Interpretación del ganador
                winner = ranking.iloc[0]['Estrategia']
                w_props = AXELROD_PROPERTIES.get(winner, {})
                if w_props:
                    n_axelrod = sum([w_props.get('nice',0), w_props.get('retaliatory',0),
                                    w_props.get('forgiving',0), w_props.get('clear',0)])
                    st.info(f"**Ganador: {winner}** — Cumple {n_axelrod}/4 propiedades de Axelrod. "
                            f"{'Consistente' if n_axelrod >= 3 else 'Inconsistente'} con la teoria de Axelrod.")


# ============================================================
# TAB 3: RESULTADOS
# ============================================================
with tabs[3]:
    st.header("Resultados de Analisis de Sensibilidad")
    st.markdown("Mueve el slider del parametro principal para explorar como cambian los resultados. "
                "Cada ejecucion corre un torneo round-robin completo con las 12 estrategias.")

    exp_tabs = st.tabs(["Exp 1: Probabilidad w", "Exp 2: Matriz de Pagos",
                         "Exp 3: Ruido", "Exp 4: Poblacion"])

    # --- Helper: ejecutar torneo y mostrar resultados completos ---
    def show_tournament_results(result, param_name, param_value, theory_text, key_prefix):
        """Muestra resultados completos de un torneo para un valor de parametro."""
        names = result['strategy_names']
        n = len(names)
        tft_idx = names.index('TIT FOR TAT')

        col_res, col_theory = st.columns([3, 1])

        with col_theory:
            st.markdown(theory_text)

        with col_res:
            # Métricas principales
            ranking_idx = np.argsort(-result['total_scores'])
            winner = names[ranking_idx[0]]
            tft_rank = list(ranking_idx).index(tft_idx) + 1
            overall_coop = result['overall_coop_rate']

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Estrategia Ganadora", winner)
            m2.metric("Rank de TFT", f"#{tft_rank} / {n}")
            m3.metric("Coop. Global", f"{overall_coop:.1%}")
            m4.metric(param_name, f"{param_value}")

            # Interpretación automática
            winner_props = AXELROD_PROPERTIES.get(winner, {})
            if winner == 'TIT FOR TAT':
                st.success(f"TFT gana el torneo con {param_name}={param_value}. "
                           f"Consistente con la prediccion de Axelrod.")
            elif winner_props.get('convergence_bool', False):
                st.info(f"**{winner}** gana (converge al equilibrio). TFT queda #{tft_rank}. "
                        f"{winner} comparte propiedades cooperativas con TFT pero tiene "
                        f"ventaja en estas condiciones especificas.")
            else:
                st.warning(f"**{winner}** gana (NO converge al equilibrio). TFT queda #{tft_rank}. "
                           f"Condiciones limite: la cooperacion se degrada en estos parametros.")

        # Heatmap de scores
        fig_heat = px.imshow(
            result['scores_matrix'], x=names, y=names,
            labels=dict(x="Oponente", y="Estrategia", color="Score"),
            title=f"Matriz de Scores ({param_name} = {param_value})",
            color_continuous_scale='Viridis', aspect='auto'
        )
        fig_heat.update_layout(height=480)
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{key_prefix}")

        # Ranking + desviación respecto a TFT
        col_rank, col_dev = st.columns(2)
        with col_rank:
            tft_total = result['total_scores'][tft_idx]
            rank_df = pd.DataFrame({
                'Rank': range(1, n+1),
                'Estrategia': [names[i] for i in ranking_idx],
                'Score Total': [result['total_scores'][i] for i in ranking_idx],
                'Tasa Coop.': [f"{result['avg_coop_rates'][i]:.1%}" for i in ranking_idx],
                'Desv. vs TFT': [f"{result['total_scores'][i] - tft_total:+.1f}" for i in ranking_idx],
            })
            st.markdown("#### Ranking y Desviacion vs TFT")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

        with col_dev:
            # Bar chart de desviación
            devs = result['total_scores'] - tft_total
            sorted_idx = np.argsort(devs)[::-1]
            colors = [COLORS['green'] if devs[i] > 0 else COLORS['red'] if devs[i] < 0
                      else COLORS['grey'] for i in sorted_idx]
            fig_dev = go.Figure()
            fig_dev.add_trace(go.Bar(
                x=[names[i] for i in sorted_idx],
                y=[devs[i] for i in sorted_idx],
                marker_color=colors,
                text=[f"{devs[i]:+.0f}" for i in sorted_idx],
                textposition='outside', textfont=dict(size=9)
            ))
            fig_dev.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig_dev.update_layout(
                title="Desviacion de Score Total vs TFT",
                yaxis_title="Score - Score_TFT",
                height=400, xaxis_tickangle=-45
            )
            st.plotly_chart(fig_dev, use_container_width=True, key=f"dev_{key_prefix}")

        # Heatmap de cooperación
        fig_coop = px.imshow(
            result['coop_matrix'], x=names, y=names,
            labels=dict(x="Oponente", y="Estrategia", color="Coop Rate"),
            title="Tasa de Cooperacion por Enfrentamiento",
            color_continuous_scale='RdYlGn', aspect='auto',
            zmin=0, zmax=1
        )
        fig_coop.update_layout(height=480)
        st.plotly_chart(fig_coop, use_container_width=True, key=f"coop_{key_prefix}")

        # Exportar
        export_df = rank_df.copy()
        export_df[param_name] = param_value
        csv = export_to_csv(export_df)
        st.download_button("Descargar CSV", csv, f"exp_{key_prefix}_{param_value}.csv",
                           "text/csv", key=f"dl_{key_prefix}_{param_value}")

    # ============================================================
    # EXPERIMENTO 1: VARIAR w
    # ============================================================
    with exp_tabs[0]:
        st.subheader("Experimento 1: Sensibilidad a la Probabilidad de Interaccion Futura (w)")

        w_val = st.slider(
            "**w** — Probabilidad de interaccion futura",
            min_value=0.05, max_value=1.0, value=0.995, step=0.005,
            key="e1_w_slider",
            help="w=0.05: juegos de ~1 ronda. w=0.995: juegos de ~200 rondas."
        )
        # w crítico teórico
        w_crit = max((5-3)/(5-1), (5-3)/(3-0))
        st.markdown(f"w critico teorico (pagos estandar): **{w_crit:.3f}** — "
                    f"{'✅ w > w* → TFT deberia ser estable' if w_val >= w_crit else '⚠️ w < w* → TFT puede NO ser estable'}")

        if st.button("Ejecutar Torneo", key="run_exp1"):
            with st.spinner(f"Ejecutando torneo con w = {w_val}..."):
                strategies = get_all_strategies()
                result = run_tournament(strategies, w=w_val, n_games=5, seed=42)
            st.session_state['exp1_result'] = result
            st.session_state['exp1_w'] = w_val

        if 'exp1_result' in st.session_state:
            theory = f"""
            #### Prediccion Teorica

            **w = {st.session_state['exp1_w']}**

            Longitud esperada del juego:
            **~{1/(1-min(st.session_state['exp1_w'], 0.9999)):.0f} rondas**

            Condicion de Axelrod:
            w* = {w_crit:.3f}

            {'**w > w***: La sombra del futuro es suficiente. La retaliacion futura de TFT compensa la ganancia inmediata de desertar. La cooperacion es sostenible.' if st.session_state['exp1_w'] >= w_crit else '**w < w***: Juegos demasiado cortos. La ganancia de desertar en la primera ronda (T=5) no es compensada por la retaliacion (que dura pocas rondas). Estrategias agresivas dominan.'}
            """
            show_tournament_results(
                st.session_state['exp1_result'], "w", st.session_state['exp1_w'],
                theory, "exp1"
            )

    # ============================================================
    # EXPERIMENTO 2: VARIAR T/R
    # ============================================================
    with exp_tabs[1]:
        st.subheader("Experimento 2: Sensibilidad al Ratio T/R de la Matriz de Pagos")

        tr_val = st.slider(
            "**T/R** — Tentacion relativa a la recompensa",
            min_value=1.1, max_value=4.0, value=1.67, step=0.1,
            key="e2_tr_slider",
            help="T/R=1.67: pagos estandar (T=5,R=3). Mayor T/R = mayor incentivo a desertar."
        )
        T_val = 3.0 * tr_val
        w_crit_tr = max((T_val-3)/(T_val-1), (T_val-3)/3)
        st.markdown(f"Con T/R={tr_val:.1f}: T={T_val:.1f}, R=3, P=1, S=0 — "
                    f"w critico = **{w_crit_tr:.3f}** — "
                    f"{'✅ w=0.995 > w*' if 0.995 >= w_crit_tr else '⚠️ w=0.995 < w* → TFT inestable'}")

        if st.button("Ejecutar Torneo", key="run_exp2"):
            with st.spinner(f"Ejecutando torneo con T/R = {tr_val}..."):
                strategies = get_all_strategies()
                result = run_tournament(strategies, w=0.995, T=T_val, R=3, P=1, S=0, n_games=5, seed=42)
            st.session_state['exp2_result'] = result
            st.session_state['exp2_tr'] = tr_val

        if 'exp2_result' in st.session_state:
            tr_show = st.session_state['exp2_tr']
            T_show = 3.0 * tr_show
            w_crit_show = max((T_show-3)/(T_show-1), (T_show-3)/3)
            theory = f"""
            #### Prediccion Teorica

            **T/R = {tr_show:.1f}** (T={T_show:.1f})

            w critico = **{w_crit_show:.3f}**

            {'**Cooperacion sostenible**: Con w=0.995 > w*, la retaliacion de TFT es suficiente para disuadir la desercion.' if 0.995 >= w_crit_show else '**Cooperacion insostenible**: Con w=0.995 < w*, la tentacion de desertar (T=' + str(T_show) + ') es tan alta que ni la retaliacion a largo plazo compensa. Estrategias agresivas dominan.'}

            A mayor T/R, el incentivo
            a desertar crece. Con T/R > 3.0,
            la ganancia unilateral de desertar
            es 3x la recompensa de cooperar.
            """
            show_tournament_results(
                st.session_state['exp2_result'], "T/R", st.session_state['exp2_tr'],
                theory, "exp2"
            )

    # ============================================================
    # EXPERIMENTO 3: VARIAR RUIDO
    # ============================================================
    with exp_tabs[2]:
        st.subheader("Experimento 3: Sensibilidad al Ruido (Trembling Hand)")

        err_val = st.slider(
            "**Error rate** — Probabilidad de ejecutar la accion opuesta",
            min_value=0.0, max_value=0.25, value=0.0, step=0.01,
            key="e3_err_slider",
            help="0%: sin errores. 10%: 1 de cada 10 acciones se invierte accidentalmente."
        )
        st.markdown(f"Error rate = **{err_val:.0%}** — "
                    f"{'Sin ruido: TFT funciona perfectamente' if err_val == 0 else f'~1 de cada {int(1/err_val) if err_val > 0 else 0} acciones se invierte. TFT vulnerable a espirales de retaliacion.'}")

        if st.button("Ejecutar Torneo", key="run_exp3"):
            with st.spinner(f"Ejecutando torneo con error = {err_val:.0%}..."):
                strategies = get_all_strategies()
                result = run_tournament(strategies, w=0.995, error_rate=err_val, n_games=5, seed=42)
            st.session_state['exp3_result'] = result
            st.session_state['exp3_err'] = err_val

        if 'exp3_result' in st.session_state:
            err_show = st.session_state['exp3_err']
            theory = f"""
            #### Prediccion Teorica

            **Error = {err_show:.0%}**

            {'**Sin ruido**: TFT reciproca perfectamente. Cooperacion mutua estable con otras estrategias amables.' if err_show == 0 else f'**Con ruido**: En un juego de ~200 rondas, se esperan ~{200*err_show:.0f} errores por jugador. Cada error de C→D provoca retaliacion de TFT, generando ciclos D-D que reducen el score de ambos.'}

            {'PAVLOV (Win-Stay/Lose-Shift) supera a TFT con ruido porque: si ambos cooperan y uno comete error (D accidental), PAVLOV puede autocorregir en la siguiente ronda sin esperar que el oponente coopere primero.' if err_show > 0.02 else ''}

            {'**Condicion limite**: Con error > 15%, la senal se pierde en el ruido. Todas las estrategias tienden a comportamiento aleatorio.' if err_show > 0.15 else ''}
            """
            show_tournament_results(
                st.session_state['exp3_result'], "Error", f"{err_show:.0%}",
                theory, "exp3"
            )

    # ============================================================
    # EXPERIMENTO 4: VARIAR N (POBLACION)
    # ============================================================
    with exp_tabs[3]:
        st.subheader("Experimento 4: Sensibilidad al Tamano de Poblacion (Drift Estocastico)")

        n_val = st.slider(
            "**N** — Tamano de la poblacion",
            min_value=10, max_value=5000, value=100, step=10,
            key="e4_n_slider",
            help="N pequeno: susceptible a drift estocastico. N grande: seleccion natural domina."
        )
        n_gens = st.slider("Generaciones de evolucion", 10, 100, 30, 10, key="e4_gens")

        st.markdown(f"N = **{n_val}** agentes, **{n_gens}** generaciones — "
                    f"{'⚠️ Poblacion muy pequena: alta variabilidad por drift' if n_val < 50 else '✅ Poblacion suficiente para seleccion estable' if n_val >= 100 else 'Poblacion moderada'}")

        if st.button("Ejecutar Simulacion Evolutiva", key="run_exp4"):
            with st.spinner(f"Ejecutando 10 replicas con N={n_val}, {n_gens} generaciones..."):
                strategy_names = ['TIT FOR TAT', 'ALL-D', 'PAVLOV', 'GRIM', 'ALL-C', 'RANDOM']
                results_evo = []
                for rep in range(10):
                    evo = run_evolutionary(strategy_names, N=n_val, generations=n_gens,
                                           seed=42 + rep, matches_per_gen=min(5, n_val-1))
                    results_evo.append(evo)
                st.session_state['exp4_results'] = results_evo
                st.session_state['exp4_n'] = n_val
                st.session_state['exp4_gens'] = n_gens

        if 'exp4_results' in st.session_state:
            results_evo = st.session_state['exp4_results']
            n_show = st.session_state['exp4_n']
            gens_show = st.session_state['exp4_gens']

            # Métricas
            final_coops = [r['final_coop_fraction'] for r in results_evo]
            m1, m2, m3 = st.columns(3)
            m1.metric("Coop. Final (media)", f"{np.mean(final_coops):.1%}")
            m2.metric("Desv. Est.", f"{np.std(final_coops):.1%}")
            m3.metric("Rango", f"{min(final_coops):.1%} — {max(final_coops):.1%}")

            col_evo, col_theory = st.columns([3, 1])

            with col_theory:
                st.markdown(f"""
                #### Prediccion Teorica

                **N = {n_show}** agentes

                {'**Drift estocastico domina**: Con N < 50, fluctuaciones aleatorias pueden eliminar TFT por azar, independientemente de su fitness.' if n_show < 50 else '**Seleccion natural domina**: Con N >= 100, las estrategias cooperativas tienen suficiente masa critica para resistir fluctuaciones.' if n_show >= 100 else '**Zona de transicion**: Los resultados son moderadamente variables.'}

                La variabilidad entre replicas
                es evidencia directa del drift:
                - **Desv.Est. alta** → drift fuerte
                - **Desv.Est. baja** → seleccion domina
                """)

            with col_evo:
                # Gráfica de evolución temporal (primer réplica)
                hist_df = results_evo[0]['history']
                strategy_names = ['TIT FOR TAT', 'ALL-D', 'PAVLOV', 'GRIM', 'ALL-C', 'RANDOM']
                fig = go.Figure()
                for i, sname in enumerate(strategy_names):
                    col_name = f'frac_{sname}'
                    if col_name in hist_df.columns:
                        fig.add_trace(go.Scatter(
                            x=hist_df['generation'], y=hist_df[col_name],
                            name=sname, line=dict(color=COLOR_LIST[i % len(COLOR_LIST)], width=2)
                        ))
                fig.update_layout(
                    title=f"Evolucion de Estrategias (Replica 1, N={n_show})",
                    xaxis_title="Generacion",
                    yaxis_title="Fraccion de Poblacion",
                    yaxis=dict(range=[0, 1]),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

            # Composición final de todas las réplicas
            final_data = []
            for i, r in enumerate(results_evo):
                for sname, count in r['final_counts'].items():
                    final_data.append({'Replica': i+1, 'Estrategia': sname,
                                       'Fraccion': count / n_show})
            final_df = pd.DataFrame(final_data)

            fig2 = px.bar(final_df, x='Replica', y='Fraccion', color='Estrategia',
                          title="Composicion Final por Replica",
                          color_discrete_sequence=COLOR_LIST)
            fig2.update_layout(height=400, barmode='stack')
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 4: ANALISIS
# ============================================================
with tabs[4]:
    st.header("Analisis Estadistico y Comparaciones")
    st.markdown("Este tab ejecuta analisis batch con multiples valores de cada parametro y "
                "multiples replicas para obtener estadisticas robustas.")

    analysis_tabs = st.tabs(["Batch & Phase Transitions", "Superficie 3D",
                              "Regresion & ANOVA", "Comparador"])

    # --- Tab: Batch + Phase Transitions ---
    with analysis_tabs[0]:
        st.subheader("Analisis Batch: Barrido de Parametros con Replicas")
        st.markdown("Ejecuta torneos para **multiples valores** de cada parametro con "
                    "**multiples replicas** por valor. Esto genera los datos para detectar "
                    "phase transitions, ajustar regresiones y hacer ANOVA.")

        col_cfg, col_run = st.columns([2, 1])
        with col_cfg:
            an_replicas = st.slider("Replicas por valor", 5, 50, 15, 5, key="an_rep")
        with col_run:
            st.markdown("")  # spacer

        batch_tabs = st.tabs(["Barrido w", "Barrido T/R", "Barrido Ruido", "Barrido N"])

        # Barrido w
        with batch_tabs[0]:
            if st.button("Ejecutar barrido de w", key="batch_w"):
                with st.spinner("Barriendo w en [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999]..."):
                    st.session_state['df_w'] = sensitivity_vary_w(
                        w_values=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999],
                        n_replicas=an_replicas, n_jobs=-1)
            if 'df_w' in st.session_state:
                df = st.session_state['df_w']
                grouped = df.groupby('w')['coop_rate'].agg(['mean','std']).reset_index()
                pt = detect_phase_transition(df, 'w')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grouped['w'], y=grouped['mean'], mode='lines+markers',
                    line=dict(color=COLORS['blue'], width=3), marker=dict(size=10), name='Media',
                    error_y=dict(type='data', array=grouped['std'].values, visible=True)))
                fig.add_vline(x=0.667, line_dash="dash", line_color=COLORS['red'],
                              annotation_text="w* teorico = 0.667")
                if pt['transition_point'] is not None:
                    fig.add_vline(x=pt['transition_point'], line_dash="dot", line_color=COLORS['orange'],
                                  annotation_text=f"Trans. observada: {pt['transition_point']:.3f}")
                fig.update_layout(title="Cooperacion vs w", xaxis_title="w",
                                  yaxis_title="Cooperacion", yaxis=dict(range=[0,1]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                if pt['transition_point'] is not None:
                    st.info(f"**Phase transition** en w = {pt['transition_point']:.3f} "
                            f"(derivada max = {pt['max_derivative']:.4f})")
                st.dataframe(summary_statistics(df, 'w').round(4), use_container_width=True, hide_index=True)
                st.download_button("CSV", export_to_csv(df), "batch_w.csv", "text/csv", key="dl_bw")

        # Barrido T/R
        with batch_tabs[1]:
            if st.button("Ejecutar barrido de T/R", key="batch_tr"):
                with st.spinner("Barriendo T/R en [1.2, 1.5, 2.0, 2.5, 3.0]..."):
                    st.session_state['df_tr'] = sensitivity_vary_payoff(
                        tr_ratios=[1.2, 1.5, 2.0, 2.5, 3.0],
                        n_replicas=an_replicas, n_jobs=-1)
            if 'df_tr' in st.session_state:
                df = st.session_state['df_tr']
                grouped = df.groupby('T_R_ratio')['coop_rate'].agg(['mean','std']).reset_index()
                pt = detect_phase_transition(df, 'T_R_ratio')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grouped['T_R_ratio'], y=grouped['mean'], mode='lines+markers',
                    line=dict(color=COLORS['orange'], width=3), marker=dict(size=10), name='Media',
                    error_y=dict(type='data', array=grouped['std'].values, visible=True)))
                if pt['transition_point'] is not None:
                    fig.add_vline(x=pt['transition_point'], line_dash="dot", line_color=COLORS['red'],
                                  annotation_text=f"Trans: {pt['transition_point']:.2f}")
                fig.update_layout(title="Cooperacion vs T/R", xaxis_title="T/R",
                                  yaxis_title="Cooperacion", yaxis=dict(range=[0,1]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                if pt['transition_point'] is not None:
                    st.info(f"**Phase transition** en T/R = {pt['transition_point']:.2f}")
                st.dataframe(summary_statistics(df, 'T_R_ratio').round(4), use_container_width=True, hide_index=True)
                st.download_button("CSV", export_to_csv(df), "batch_tr.csv", "text/csv", key="dl_btr")

        # Barrido Ruido
        with batch_tabs[2]:
            if st.button("Ejecutar barrido de ruido", key="batch_noise"):
                with st.spinner("Barriendo error en [0%, 1%, 5%, 10%]..."):
                    st.session_state['df_noise'] = sensitivity_vary_noise(
                        error_rates=[0.0, 0.01, 0.05, 0.10],
                        n_replicas=an_replicas, n_jobs=-1)
            if 'df_noise' in st.session_state:
                df = st.session_state['df_noise']
                grouped = df.groupby('error_rate')['coop_rate'].agg(['mean','std']).reset_index()
                pt = detect_phase_transition(df, 'error_rate')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grouped['error_rate']*100, y=grouped['mean'], mode='lines+markers',
                    line=dict(color=COLORS['green'], width=3), marker=dict(size=10), name='Media',
                    error_y=dict(type='data', array=grouped['std'].values, visible=True)))
                if pt['transition_point'] is not None:
                    fig.add_vline(x=pt['transition_point']*100, line_dash="dot", line_color=COLORS['red'],
                                  annotation_text=f"Trans: {pt['transition_point']*100:.1f}%")
                fig.update_layout(title="Cooperacion vs Ruido", xaxis_title="Error Rate (%)",
                                  yaxis_title="Cooperacion", yaxis=dict(range=[0,1]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                if pt['transition_point'] is not None:
                    st.info(f"**Phase transition** en error = {pt['transition_point']*100:.1f}%")
                st.dataframe(summary_statistics(df, 'error_rate').round(4), use_container_width=True, hide_index=True)
                st.download_button("CSV", export_to_csv(df), "batch_noise.csv", "text/csv", key="dl_bn")

        # Barrido N
        with batch_tabs[3]:
            if st.button("Ejecutar barrido de N", key="batch_pop"):
                with st.spinner("Barriendo N en [10, 50, 100, 500, 1000, 5000]..."):
                    st.session_state['df_pop'] = sensitivity_vary_population(
                        n_values=[10, 50, 100, 500, 1000, 5000],
                        n_replicas=min(an_replicas, 15), generations=30, n_jobs=-1)
            if 'df_pop' in st.session_state:
                df = st.session_state['df_pop']
                grouped = df.groupby('N')['coop_rate'].agg(['mean','std']).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grouped['N'], y=grouped['mean'], mode='lines+markers',
                    line=dict(color=COLORS['purple'], width=3), marker=dict(size=10), name='Media',
                    error_y=dict(type='data', array=grouped['std'].values, visible=True)))
                fig.update_layout(title="Cooperacion vs N", xaxis_title="N", xaxis_type="log",
                                  yaxis_title="Cooperacion", yaxis=dict(range=[0,1]), height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Variabilidad como evidencia de drift
                fig2 = px.bar(grouped, x='N', y='std', title="Variabilidad (Desv.Est.) por N",
                              color_discrete_sequence=[COLORS['purple']])
                fig2.update_layout(height=300, xaxis_type="log")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(summary_statistics(df, 'N').round(4), use_container_width=True, hide_index=True)
                st.download_button("CSV", export_to_csv(df), "batch_pop.csv", "text/csv", key="dl_bp")

    # --- Tab: Superficie 3D ---
    with analysis_tabs[1]:
        st.subheader("Superficie 3D: Interaccion entre Parametros")
        surface_type = st.radio("Variables", ["w vs Ruido vs Cooperacion", "w vs T/R vs Cooperacion"], horizontal=True)
        n_rep_3d = st.slider("Replicas por punto", 5, 30, 10, 5, key="nrep_3d")

        if st.button("Generar Superficie 3D", key="run_3d"):
            if surface_type == "w vs Ruido vs Cooperacion":
                with st.spinner("Calculando superficie w vs ruido..."):
                    st.session_state['df_surf_wn'] = sensitivity_surface_w_noise(n_replicas=n_rep_3d, n_jobs=-1)
            else:
                with st.spinner("Calculando superficie w vs T/R..."):
                    st.session_state['df_surf_wp'] = sensitivity_surface_w_payoff(n_replicas=n_rep_3d, n_jobs=-1)

        for key, x_col, y_col, x_label, y_label, cond in [
            ('df_surf_wn', 'w', 'error_rate', 'w', 'Error Rate (%)', "w vs Ruido vs Cooperacion"),
            ('df_surf_wp', 'w', 'T_R_ratio', 'w', 'T/R ratio', "w vs T/R vs Cooperacion")
        ]:
            if key in st.session_state and surface_type == cond:
                df_s = st.session_state[key]
                x_vals = sorted(df_s[x_col].unique())
                y_vals = sorted(df_s[y_col].unique())
                Z = np.zeros((len(y_vals), len(x_vals)))
                for i, yv in enumerate(y_vals):
                    for j, xv in enumerate(x_vals):
                        row = df_s[(df_s[x_col]==xv) & (df_s[y_col]==yv)]
                        if len(row) > 0: Z[i,j] = row['coop_rate_mean'].values[0]
                y_display = [e*100 for e in y_vals] if y_col == 'error_rate' else y_vals
                fig = go.Figure(data=[go.Surface(z=Z, x=x_vals, y=y_display,
                                                  colorscale='Viridis', colorbar=dict(title='Coop'))])
                fig.update_layout(title=f"Superficie: {cond}",
                                  scene=dict(xaxis_title=x_label, yaxis_title=y_label,
                                             zaxis_title='Cooperacion'), height=600)
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab: Regresión & ANOVA ---
    with analysis_tabs[2]:
        st.subheader("Regresion Logistica y ANOVA")
        st.markdown("Requiere haber ejecutado los barridos en el tab 'Batch & Phase Transitions'.")

        has_data = False
        for key, name, param in [('df_w','w','w'), ('df_tr','T/R','T_R_ratio'), ('df_noise','Error','error_rate')]:
            if key in st.session_state:
                has_data = True
                df = st.session_state[key]
                grouped = df.groupby(param)['coop_rate'].mean().reset_index()
                x, y = grouped[param].values.astype(float), grouped['coop_rate'].values

                st.markdown(f"---")
                st.markdown(f"### {name}")
                col_reg, col_anova = st.columns(2)

                # Regresión
                with col_reg:
                    result = logistic_regression_fit(x, y)
                    if result['success']:
                        x_s = np.linspace(x.min(), x.max(), 100)
                        def logistic(x, L, k, x0): return L / (1 + np.exp(-k*(x-x0)))
                        y_s = logistic(x_s, **result['params'])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Datos', marker=dict(size=10)))
                        fig.add_trace(go.Scatter(x=x_s, y=y_s, mode='lines', name='Regresion',
                                                 line=dict(color=COLORS['red'], width=2)))
                        fig.update_layout(title=f"Regresion Logistica: {name}", height=350,
                                          xaxis_title=param, yaxis_title="Cooperacion")
                        st.plotly_chart(fig, use_container_width=True)
                        st.metric("R²", f"{result['r_squared']:.4f}")
                        for pk, pv in result['params'].items():
                            st.caption(f"{pk} = {pv:.4f}")
                    else:
                        st.warning(f"No se pudo ajustar regresion para {name}")

                # ANOVA
                with col_anova:
                    groups = {str(v): df[df[param]==v]['coop_rate'].tolist()
                              for v in sorted(df[param].unique())}
                    anova = compute_anova(groups)
                    st.markdown("#### ANOVA de una via")
                    st.metric("Estadistico F", f"{anova['f_statistic']:.4f}")
                    st.metric("p-value", f"{anova['p_value']:.6f}")
                    if anova['significant']:
                        st.success("Diferencias significativas (p < 0.05)")
                    else:
                        st.warning("Sin diferencias significativas (p >= 0.05)")

                    # Box plot
                    fig2 = px.box(df, x=param, y='coop_rate', title=f"Distribucion: {name}")
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)

        # ANOVA para N (si existe)
        if 'df_pop' in st.session_state:
            has_data = True
            df = st.session_state['df_pop']
            st.markdown("---")
            st.markdown("### N (Poblacion)")
            groups = {str(v): df[df['N']==v]['coop_rate'].tolist() for v in sorted(df['N'].unique())}
            anova = compute_anova(groups)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, x='N', y='coop_rate', title="Distribucion por N")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("#### ANOVA")
                st.metric("F", f"{anova['f_statistic']:.4f}")
                st.metric("p-value", f"{anova['p_value']:.6f}")
                if anova['significant']:
                    st.success("Diferencias significativas")
                else:
                    st.warning("Sin diferencias significativas")

        if not has_data:
            st.warning("Ejecuta los barridos en 'Batch & Phase Transitions' primero.")

    # --- Tab: Comparador ---
    with analysis_tabs[3]:
        st.subheader("Comparador de Condiciones")
        available = {"Barrido w": ('df_w','w'), "Barrido T/R": ('df_tr','T_R_ratio'),
                     "Barrido Ruido": ('df_noise','error_rate'), "Barrido N": ('df_pop','N')}
        avail_keys = [k for k in available if available[k][0] in st.session_state]
        if avail_keys:
            exp_sel = st.selectbox("Selecciona barrido", avail_keys)
            k, param = available[exp_sel]
            df = st.session_state[k]
            vals = sorted(df[param].unique())
            sel = st.multiselect(f"Valores de {param} a comparar", vals,
                                  default=vals[:2] if len(vals)>=2 else vals)
            if len(sel) >= 2:
                fig = px.violin(df[df[param].isin(sel)], x=param, y='coop_rate', box=True, points='all',
                                title=f"Comparacion: {param}", color_discrete_sequence=COLOR_LIST)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(summary_statistics(df[df[param].isin(sel)], param).round(4),
                             use_container_width=True, hide_index=True)
                if len(sel) == 2:
                    from scipy.stats import mannwhitneyu
                    g1 = df[df[param]==sel[0]]['coop_rate']
                    g2 = df[df[param]==sel[1]]['coop_rate']
                    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                    st.markdown(f"**Mann-Whitney U**: U={stat:.2f}, p={p:.6f}")
                    if p < 0.05:
                        st.success("Diferencia significativa entre estas condiciones")
                    else:
                        st.info("Sin diferencia significativa")
        else:
            st.warning("Ejecuta los barridos en 'Batch & Phase Transitions' primero.")


# ============================================================
# TAB 5: PEPSI VS COCA-COLA
# ============================================================
with tabs[5]:
    st.header("Caso Aplicado: Guerra de Precios Pepsi vs Coca-Cola")
    st.markdown("""
    La competencia Pepsi/Coca-Cola es un **ejemplo clasico de IPD en el mundo real**.
    Ambas empresas enfrentan constantemente la decision de mantener precios altos (cooperar)
    o bajar precios para ganar cuota (desertar).
    """)

    pepsi_tabs = st.tabs(["Precios Historicos", "Modelo IPD", "Simulacion", "Analisis"])

    with pepsi_tabs[0]:
        st.subheader("Evolucion de Precios 2020-2025")
        prices = get_price_data()
        market = get_market_share()

        product = st.radio("Producto", ["Botella 2L", "12-Pack"], horizontal=True)
        coca_col = 'coca_cola_2L' if product == "Botella 2L" else 'coca_cola_12pk'
        pepsi_col = 'pepsi_2L' if product == "Botella 2L" else 'pepsi_12pk'
        unit = "USD / 2L" if product == "Botella 2L" else "USD / 12-pack"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices['year'], y=prices[coca_col], name='Coca-Cola',
                                  line=dict(color='#E61A27', width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=prices['year'], y=prices[pepsi_col], name='Pepsi',
                                  line=dict(color='#004B93', width=3), marker=dict(size=10)))
        fig.update_layout(title=f"Precios {product} (EE.UU.)", xaxis_title="Ano",
                          yaxis_title=unit, height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Diferencia
        prices['diff'] = prices[coca_col] - prices[pepsi_col]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=prices['year'], y=prices['diff'],
                               marker_color=[COLORS['green'] if d>0 else COLORS['red'] for d in prices['diff']],
                               text=[f"${d:.2f}" for d in prices['diff']], textposition='outside'))
        fig2.update_layout(title="Diferencia (Coca - Pepsi)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

        # Market share
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=market['year'], y=market['coca_cola_share'], name='Coca-Cola',
                                   fill='tozeroy', line=dict(color='#E61A27'), fillcolor='rgba(230,26,39,0.2)'))
        fig3.add_trace(go.Scatter(x=market['year'], y=market['pepsi_share'], name='Pepsi',
                                   fill='tozeroy', line=dict(color='#004B93'), fillcolor='rgba(0,75,147,0.2)'))
        fig3.update_layout(title="Cuota de Mercado CSD (%)", height=350)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Eventos Clave")
        st.dataframe(prices[['year','event']].rename(columns={'year':'Ano','event':'Evento'}),
                     use_container_width=True, hide_index=True)

    with pepsi_tabs[1]:
        st.subheader("Modelado como Dilema del Prisionero")
        st.markdown("""
        | Accion | Significado | Ejemplo real |
        |--------|------------|--------------|
        | **Cooperar (C)** | Mantener/subir precios | Ambas suben precios 2022-2023 (inflacion) |
        | **Desertar (D)** | Bajar precios/promociones agresivas | PepsiCo 2024 (value packs) |

        **Porque es un Dilema del Prisionero:**
        - **T > R**: Bajar precios mientras el rival mantiene → ganas cuota rapidamente
        - **R > P**: Ambos con precios altos → mejores margenes que una guerra de precios
        - **P > S**: Guerra de precios mutua → mejor que ser el unico con precios altos

        **Porque se mantiene la cooperacion:**
        - **w ≈ 1.0**: Coca-Cola y Pepsi interactuan *cada dia, indefinidamente*
        - **Retaliacion creible**: Ambas pueden bajar precios en semanas
        - **Observabilidad**: Los precios son publicos, desviaciones se detectan inmediatamente
        """)

        st.subheader("Acciones Historicas Interpretadas")
        actions = historical_actions()
        st.dataframe(actions, use_container_width=True, hide_index=True)
        st.caption("Umbral: cambio de precio anual >= 3% se interpreta como Cooperar (mantener precios altos)")

    with pepsi_tabs[2]:
        st.subheader("Simulacion de Estrategias")
        col1, col2 = st.columns(2)
        strat_opts = ['TIT FOR TAT', 'ALL-D', 'ALL-C', 'PAVLOV', 'GRIM', 'JOSS', 'TFT2T', 'GRADUAL',
                      'COCA-COLA (TFT Líder)', 'PEPSI (Seguidor Agresivo)']
        with col1:
            pepsi_strat = st.selectbox("Estrategia Pepsi", strat_opts,
                                        index=strat_opts.index('PEPSI (Seguidor Agresivo)'))
        with col2:
            coca_strat = st.selectbox("Estrategia Coca-Cola", strat_opts,
                                       index=strat_opts.index('COCA-COLA (TFT Líder)'))
        col3, col4 = st.columns(2)
        with col3: n_games_pc = st.slider("Juegos", 5, 50, 10, key="ngpc")
        with col4: w_pc = st.slider("w", 0.9, 1.0, 0.99, 0.005, key="wpc")

        if st.button("Simular", key="run_pc"):
            with st.spinner("Simulando..."):
                sim = simulate_pepsi_coca_strategies(pepsi_strat, coca_strat,
                                                     w=w_pc, n_games=n_games_pc, seed=42)
            st.session_state['pc_sim'] = sim

        if 'pc_sim' in st.session_state:
            sim = st.session_state['pc_sim']
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Pepsi Score", f"{sim['pepsi_avg_score']:.1f}")
            mc2.metric("Coca Score", f"{sim['coca_avg_score']:.1f}")
            mc3.metric("Pepsi Coop", f"{sim['pepsi_avg_coop']:.1%}")
            mc4.metric("Coca Coop", f"{sim['coca_avg_coop']:.1%}")

            summary = sim['summary']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"J{i+1}" for i in summary['game']], y=summary['pepsi_total'],
                                  name='Pepsi', marker_color='#004B93'))
            fig.add_trace(go.Bar(x=[f"J{i+1}" for i in summary['game']], y=summary['coca_total'],
                                  name='Coca-Cola', marker_color='#E61A27'))
            fig.update_layout(title="Scores por Juego", barmode='group', height=350)
            st.plotly_chart(fig, use_container_width=True)

            # --- Gráfica de precios simulados ---
            st.markdown("#### Proyeccion de Precios segun Estrategia")
            st.markdown("Cada ronda representa un trimestre. "
                        "**C** (cooperar) = subir precio ~3%. "
                        "**D** (desertar) = bajar precio ~5%.")

            h0 = sim['history'][sim['history']['game'] == 0].copy()

            # Precio base: último precio real (2025)
            pepsi_base = 2.59
            coca_base = 2.79

            # Simular evolución de precios según acciones
            pepsi_prices = [pepsi_base]
            coca_prices = [coca_base]
            for _, row in h0.iterrows():
                # C = subir 3%, D = bajar 5%
                p_mult = 1.03 if row['pepsi_action'] == 'C' else 0.95
                c_mult = 1.03 if row['coca_action'] == 'C' else 0.95
                pepsi_prices.append(pepsi_prices[-1] * p_mult)
                coca_prices.append(coca_prices[-1] * c_mult)

            trimestres = list(range(len(pepsi_prices)))
            # Etiquetas: Q1 2026, Q2 2026, etc.
            trim_labels = []
            for i in range(len(trimestres)):
                year = 2025 + (i // 4)
                q = (i % 4) + 1
                trim_labels.append(f"Q{q} {year}")

            fig_prices = go.Figure()
            fig_prices.add_trace(go.Scatter(
                x=trim_labels, y=pepsi_prices,
                name='Pepsi', line=dict(color='#004B93', width=3),
                marker=dict(size=8,
                            color=['#009988' if a == 'C' else '#CC3311'
                                   for a in ['C'] + h0['pepsi_action'].tolist()],
                            line=dict(width=1, color='white')),
                mode='lines+markers'
            ))
            fig_prices.add_trace(go.Scatter(
                x=trim_labels, y=coca_prices,
                name='Coca-Cola', line=dict(color='#E61A27', width=3),
                marker=dict(size=8,
                            color=['#009988' if a == 'C' else '#CC3311'
                                   for a in ['C'] + h0['coca_action'].tolist()],
                            line=dict(width=1, color='white')),
                mode='lines+markers'
            ))

            # Banda de referencia: precio si ambos cooperan siempre
            coop_prices = [coca_base]
            for _ in range(len(h0)):
                coop_prices.append(coop_prices[-1] * 1.03)
            fig_prices.add_trace(go.Scatter(
                x=trim_labels, y=coop_prices,
                name='Referencia (ambos cooperan)',
                line=dict(color='grey', width=1, dash='dot'),
                mode='lines'
            ))

            fig_prices.update_layout(
                title=f"Proyeccion de Precios (Botella 2L) — Pepsi: {sim['pepsi_strategy']}, Coca-Cola: {sim['coca_strategy']}",
                xaxis_title="Trimestre",
                yaxis_title="Precio (USD)",
                height=450,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                xaxis=dict(tickangle=-45)
            )
            # Limitar a 20 trimestres para legibilidad
            if len(trim_labels) > 20:
                fig_prices.update_xaxes(range=[0, 20])

            st.plotly_chart(fig_prices, use_container_width=True)

            st.caption("Marcadores: verde = coopero (subio precio), rojo = deserto (bajo precio). "
                       "Linea gris punteada = escenario ideal donde ambos cooperan siempre.")

        st.subheader("Todas las Combinaciones")
        if st.button("Generar Tabla Completa", key="all_pc"):
            with st.spinner("..."):
                st.session_state['all_combos'] = run_all_strategy_combinations(
                    strategies=['TIT FOR TAT','ALL-D','ALL-C','PAVLOV','GRIM',
                                'COCA-COLA (TFT Líder)','PEPSI (Seguidor Agresivo)'], w=0.99, n_games=5)
        if 'all_combos' in st.session_state:
            st.dataframe(st.session_state['all_combos'], use_container_width=True, hide_index=True)
            pivot = st.session_state['all_combos'].pivot_table(values='Total Value',
                    index='Pepsi Strategy', columns='Coca-Cola Strategy', aggfunc='mean')
            fig = px.imshow(pivot, title="Valor Total por Combinacion", color_continuous_scale='RdYlGn', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)

    with pepsi_tabs[3]:
        st.subheader("Analisis: Conexion con Teoria de Axelrod")
        st.markdown("""
        ### Hallazgo Principal

        El duopolio Pepsi/Coca-Cola **confirma la prediccion de Axelrod**: en un juego repetido
        con w alto (interaccion diaria, indefinida), la cooperacion tacita emerge y se mantiene.

        **Estrategia observada de cada empresa:**

        | Empresa | Estrategia mas similar | Propiedades |
        |---------|----------------------|-------------|
        | **Coca-Cola** | TFT con perdon (lider) | Amable, provocable, perdonadora, clara |
        | **Pepsi** | Seguidor agresivo | Amable (mayormente), provocable, perdonadora, menos clara |

        ### Evento 2024: Desercion Parcial de PepsiCo

        En 2024, PepsiCo "deserto parcialmente" al invertir en promociones y value packs
        mientras Coca-Cola mantenia precios. Esto es analogo a **JOSS** (TFT con desercion
        ocasional). Segun Axelrod, este tipo de comportamiento **no paga a largo plazo**
        porque provoca retaliacion del rival.

        **Resultado**: Coca-Cola respondio con sus propias promociones en 2025 (retaliacion TFT),
        y ambas marcas perdieron margen. El mercado se ajusto y ambas volvieron a cooperar,
        consistente con la dinamica TFT de desercion → retaliacion → reconciliacion.

        ### Por que No Hay Guerra de Precios Permanente

        Aplicando el **Folk Theorem**: en un juego infinitamente repetido con w suficientemente
        alto, la cooperacion mutua es sostenible como equilibrio de Nash. Para Pepsi/Coca-Cola:

        - **w ≈ 1.0** (interactuan diariamente, sin fin previsible)
        - **Retaliacion inmediata** (pueden ajustar precios en semanas)
        - **Monitoreo perfecto** (precios son publicos)

        Estas condiciones satisfacen exactamente los requisitos de Axelrod para que TFT sea estable.
        """)


# ============================================================
# TAB 6: REFERENCIAS
# ============================================================
with tabs[6]:
    st.header("Referencias y Metodologia")
    st.markdown("""
    ### Referencias Principales

    1. **Axelrod, R. & Hamilton, W.D. (1981)**. "The Evolution of Cooperation".
       *Science*, 211(4489), 1390-1396.

    2. **Axelrod, R. (2006)**. *The Evolution of Cooperation* (Revised Ed.). Basic Books.

    3. **Nowak, M.A. (2006)**. *Evolutionary Dynamics*. Harvard Univ. Press.

    4. **Nowak, M.A. & Sigmund, K. (1993)**. "A strategy of win-stay, lose-shift
       that outperforms tit-for-tat". *Nature*, 364, 56-58.

    5. **Santos, F.C. & Pacheco, J.M. (2005)**. "Scale-free networks provide a
       unifying framework". *PRL*, 95, 098104.

    ### Caso Aplicado

    6. **Harrington, J.E. (2008)**. *Games, Strategies, and Decision Making*. Worth Publishers.

    7. **Tirole, J. (1988)**. *The Theory of Industrial Organization*. MIT Press.

    8. **PepsiCo 10-K Annual Reports (2020-2025)**. SEC EDGAR.

    9. **The Coca-Cola Company 10-K Annual Reports (2020-2025)**. SEC EDGAR.

    10. **Bureau of Labor Statistics**. CPI Average Price Data, Carbonated Drinks.

    ### Metodologia

    **Diseno experimental:** Para cada experimento, se ejecutan N replicas (configurable, hasta 50)
    con semillas reproducibles (PCG64, seed = 42 + replica_id).

    **Motor de simulacion:**
    - Torneo round-robin con 12 estrategias
    - Juego iterado con terminacion probabilistica (w)
    - Ruido trembling hand (inversion de accion con probabilidad error_rate)
    - Dinamica evolutiva con imitacion (funcion de Fermi, beta=0.1)

    **Analisis estadistico:** ANOVA, regresion logistica, deteccion de phase transitions
    por derivada maxima, test Mann-Whitney U para comparacion de pares.

    **Metrica principal:** Desviacion respecto a TFT: delta(S,O) = Score(S vs O) - Score(TFT vs O)

    **Arquitectura:**
    ```
    dilema_prisionero/
    ├── app.py              # Dashboard (Streamlit)
    ├── strategies.py       # 12 estrategias + propiedades de Axelrod
    ├── simulation.py       # Torneo, sensibilidad, matriz de desviacion
    ├── rng_module.py       # PCG64 reproducible
    ├── rng_tests.py        # KS, autocorrelacion, chi²
    ├── pepsi_coca.py       # Caso aplicado duopolio
    ├── utils.py            # Exportacion, ANOVA, regresion
    └── requirements.txt
    ```
    """)


# ============================================================
# TAB 7: ANEXOS TECNICOS
# ============================================================
with tabs[7]:
    st.header("Anexos Tecnicos")
    anexo_tabs = st.tabs(["Tests RNG", "Codigo Fuente"])

    with anexo_tabs[0]:
        st.subheader("Tests de Calidad del RNG (PCG64)")
        seed_test = st.number_input("Semilla", value=20260211, key="seed_rng")

        if st.button("Ejecutar Tests", key="run_rng"):
            with st.spinner("..."):
                st.session_state['rng_tests'] = run_all_tests(seed=int(seed_test))

        if 'rng_tests' in st.session_state:
            tests = st.session_state['rng_tests']

            # KS
            ks = tests[0]
            st.markdown("#### 1. Kolmogorov-Smirnov")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("KS Statistic", f"{ks['statistic']:.6f}")
                st.metric("p-value", f"{ks['p_value']:.6f}")
                (st.success if ks['passed'] else st.error)(ks['interpretation'])
            with col2:
                rng_viz = ReproducibleRNG(seed=int(seed_test))
                samples = rng_viz.uniform(0, 1, size=100_000)
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=samples, nbinsx=50, marker_color=COLORS['blue'], opacity=0.7))
                fig.add_hline(y=100_000/50, line_dash="dash", line_color=COLORS['red'])
                fig.update_layout(title="Histograma 100K muestras", height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Autocorrelación
            ac = tests[1]
            st.markdown("#### 2. Autocorrelacion")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Lags fuera IC 95%", f"{ac['n_outside_ci']}/{len(ac['lags'])}")
                (st.success if ac['passed'] else st.error)(ac['interpretation'])
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=ac['lags'], y=ac['autocorrelations'], marker_color=COLORS['blue']))
                fig.add_hline(y=ac['confidence_interval'], line_dash="dash", line_color=COLORS['red'])
                fig.add_hline(y=-ac['confidence_interval'], line_dash="dash", line_color=COLORS['red'])
                fig.update_layout(title="Autocorrelacion por Lag", height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Chi²
            chi = tests[2]
            st.markdown("#### 3. Chi-cuadrado")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chi²", f"{chi['chi2_statistic']:.2f}")
                st.metric("p-value", f"{chi['p_value']:.6f}")
                (st.success if chi['passed'] else st.error)(chi['interpretation'])
            with col2:
                bc = (chi['bin_edges'][:-1]+chi['bin_edges'][1:])/2
                fig = go.Figure()
                fig.add_trace(go.Bar(x=bc, y=chi['observed'], marker_color=COLORS['blue'], opacity=0.7))
                fig.add_hline(y=chi['expected'], line_dash="dash", line_color=COLORS['red'])
                fig.update_layout(title="Observado vs Esperado", height=300)
                st.plotly_chart(fig, use_container_width=True)

    with anexo_tabs[1]:
        st.subheader("Codigo Fuente Destacado")
        st.markdown("#### ReproducibleRNG (obligatorio)")
        st.code("""
class ReproducibleRNG:
    def __init__(self, seed: int = 20260211):
        self.seed = seed
        self.rng = Generator(PCG64(seed))
    def uniform(self, low=0.0, high=1.0, size=None):
        return self.rng.uniform(low, high, size)
    def choice(self, a, size=None, replace=True, p=None):
        return self.rng.choice(a, size, replace, p)
        """, language='python')

        st.markdown("#### TIT FOR TAT (estrategia de equilibrio)")
        st.code("""
class TitForTat(Strategy):
    name = "TIT FOR TAT"
    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'  # Coopera primero
        return self.opp_history[-1]  # Reciproca exactamente
        """, language='python')

        st.markdown("#### Matriz de Desviacion vs TFT")
        st.code("""
# delta(S, O) = Score(S vs O) - Score(TFT vs O)
# Positivo: S supera a TFT contra O
# Negativo: S pierde vs TFT contra O
deviation = scores_matrix - tft_scores[np.newaxis, :]
aggregate = np.mean(deviation, axis=1)  # Desviacion promedio
        """, language='python')


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Dilema del Prisionero Iterado")
    st.markdown("**Opcion 5**: Sensibilidad y Condiciones Limite")
    st.divider()
    st.markdown("""
    **Pagos estandar:** T=5, R=3, P=1, S=0

    **Condicion de equilibrio TFT:**
    w >= max((T-R)/(T-P), (T-R)/(R-S))
    Con pagos estandar: **w >= 0.667**

    **RNG:** PCG64, seed=20260211
    """)
    st.divider()
    st.markdown("**Exportar Resultados**")
    if st.button("Exportar JSON"):
        data = {k: st.session_state[k].to_dict('records')
                for k in ['df_w','df_tr','df_noise','df_pop'] if k in st.session_state}
        if data:
            st.download_button("Descargar", export_to_json(data), "resultados.json", "application/json")
        else:
            st.warning("Ejecuta experimentos primero.")
    st.divider()
    st.markdown("**Curso:** Tecnicas Computacionales\n\n**Ref:** Axelrod & Hamilton (1981)")
