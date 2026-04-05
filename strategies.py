"""
Estrategias para el Dilema del Prisionero Iterado.
Incluye estrategias clásicas del torneo de Axelrod (1981) y variantes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from numpy.random import Generator


# --- Clase base ---

class Strategy(ABC):
    """Clase base abstracta para estrategias del Dilema del Prisionero."""

    name: str = "Base"
    cooperative: bool = True  # Si la estrategia es considerada "cooperativa"

    def __init__(self):
        self.my_history: List[str] = []
        self.opp_history: List[str] = []

    def reset(self):
        """Reinicia historiales para un nuevo juego."""
        self.my_history = []
        self.opp_history = []

    @abstractmethod
    def decide(self, rng: Optional[Generator] = None) -> str:
        """Retorna 'C' (cooperar) o 'D' (desertar)."""
        pass

    def update(self, my_action: str, opp_action: str):
        """Actualiza historiales tras una ronda."""
        self.my_history.append(my_action)
        self.opp_history.append(opp_action)

    def __repr__(self):
        return self.name


# --- Estrategias clásicas ---

class TitForTat(Strategy):
    """Coopera en la primera ronda, luego copia la acción previa del oponente."""
    name = "TIT FOR TAT"
    cooperative = True

    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'
        return self.opp_history[-1]


class AllCooperate(Strategy):
    """Siempre coopera."""
    name = "ALL-C"
    cooperative = True

    def decide(self, rng=None) -> str:
        return 'C'


class AllDefect(Strategy):
    """Siempre deserta."""
    name = "ALL-D"
    cooperative = False

    def decide(self, rng=None) -> str:
        return 'D'


class Grim(Strategy):
    """Coopera hasta que el oponente deserta una vez, luego deserta para siempre."""
    name = "GRIM"
    cooperative = True

    def __init__(self):
        super().__init__()
        self.triggered = False

    def reset(self):
        super().reset()
        self.triggered = False

    def decide(self, rng=None) -> str:
        if self.triggered:
            return 'D'
        if len(self.opp_history) > 0 and self.opp_history[-1] == 'D':
            self.triggered = True
            return 'D'
        return 'C'


class Pavlov(Strategy):
    """Win-Stay, Lose-Shift. Coopera si ambos hicieron lo mismo, deserta si no."""
    name = "PAVLOV"
    cooperative = True

    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'
        if self.my_history[-1] == self.opp_history[-1]:
            return 'C'
        return 'D'


class TitForTwoTats(Strategy):
    """Deserta solo si el oponente desertó en las últimas 2 rondas consecutivas."""
    name = "TFT2T"
    cooperative = True

    def decide(self, rng=None) -> str:
        if len(self.opp_history) < 2:
            return 'C'
        if self.opp_history[-1] == 'D' and self.opp_history[-2] == 'D':
            return 'D'
        return 'C'


class Random(Strategy):
    """Coopera o deserta aleatoriamente con probabilidad 0.5."""
    name = "RANDOM"
    cooperative = False

    def decide(self, rng=None) -> str:
        if rng is None:
            return np.random.choice(['C', 'D'])
        return rng.choice(['C', 'D'])


class Joss(Strategy):
    """TIT FOR TAT pero con 10% de probabilidad de desertar en vez de cooperar."""
    name = "JOSS"
    cooperative = False

    def decide(self, rng=None) -> str:
        if len(self.opp_history) == 0:
            return 'C'
        if self.opp_history[-1] == 'C':
            if rng is not None:
                if rng.random() < 0.1:
                    return 'D'
            return 'C'
        return 'D'


class Gradual(Strategy):
    """Deserta N veces después de la N-ésima deserción del oponente, luego coopera 2 veces."""
    name = "GRADUAL"
    cooperative = True

    def __init__(self):
        super().__init__()
        self.opp_defections = 0
        self.punish_count = 0
        self.calm_count = 0

    def reset(self):
        super().reset()
        self.opp_defections = 0
        self.punish_count = 0
        self.calm_count = 0

    def decide(self, rng=None) -> str:
        if self.punish_count > 0:
            self.punish_count -= 1
            return 'D'
        if self.calm_count > 0:
            self.calm_count -= 1
            return 'C'
        if len(self.opp_history) > 0 and self.opp_history[-1] == 'D':
            self.opp_defections += 1
            self.punish_count = self.opp_defections - 1  # -1 porque esta ronda ya cuenta
            self.calm_count = 2
            return 'D'
        return 'C'


class Adaptive(Strategy):
    """Comienza con una secuencia fija, luego elige la acción con mayor payoff histórico."""
    name = "ADAPTIVE"
    cooperative = True

    INITIAL_SEQUENCE = ['C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D']

    def __init__(self):
        super().__init__()
        self.c_score = 0
        self.d_score = 0
        self.c_count = 0
        self.d_count = 0

    def reset(self):
        super().reset()
        self.c_score = 0
        self.d_score = 0
        self.c_count = 0
        self.d_count = 0

    def update(self, my_action: str, opp_action: str):
        super().update(my_action, opp_action)
        # Track payoffs for each action type
        # Using standard payoffs for tracking (T=5, R=3, P=1, S=0)
        if my_action == 'C':
            self.c_count += 1
            self.c_score += (3 if opp_action == 'C' else 0)
        else:
            self.d_count += 1
            self.d_score += (5 if opp_action == 'C' else 1)

    def decide(self, rng=None) -> str:
        round_num = len(self.my_history)
        if round_num < len(self.INITIAL_SEQUENCE):
            return self.INITIAL_SEQUENCE[round_num]
        # Choose action with higher average payoff
        avg_c = self.c_score / max(self.c_count, 1)
        avg_d = self.d_score / max(self.d_count, 1)
        return 'C' if avg_c >= avg_d else 'D'


class Friedman(Strategy):
    """Idéntico a GRIM - coopera hasta la primera deserción, luego siempre deserta."""
    name = "FRIEDMAN"
    cooperative = True

    def __init__(self):
        super().__init__()
        self.triggered = False

    def reset(self):
        super().reset()
        self.triggered = False

    def decide(self, rng=None) -> str:
        if self.triggered:
            return 'D'
        if len(self.opp_history) > 0 and self.opp_history[-1] == 'D':
            self.triggered = True
            return 'D'
        return 'C'


class Tester(Strategy):
    """Deserta en la primera ronda. Si el oponente castiga, juega TFT. Si no, explota."""
    name = "TESTER"
    cooperative = False

    def __init__(self):
        super().__init__()
        self.is_tft_mode = False

    def reset(self):
        super().reset()
        self.is_tft_mode = False

    def decide(self, rng=None) -> str:
        if len(self.my_history) == 0:
            return 'D'  # Test on first move
        if len(self.my_history) == 1:
            if self.opp_history[-1] == 'D':
                self.is_tft_mode = True
                return 'C'
            else:
                return 'C'  # Opponent didn't retaliate
        if self.is_tft_mode:
            return self.opp_history[-1]
        # Exploit: alternate C and D
        return 'D' if len(self.my_history) % 2 == 0 else 'C'


# --- Función de utilidad ---

def get_all_strategies() -> List[Strategy]:
    """Retorna una lista con instancias de todas las estrategias disponibles."""
    return [
        TitForTat(),
        AllCooperate(),
        AllDefect(),
        Grim(),
        Pavlov(),
        TitForTwoTats(),
        Random(),
        Joss(),
        Gradual(),
        Adaptive(),
        Friedman(),
        Tester(),
    ]


def get_strategy_by_name(name: str) -> Strategy:
    """Retorna una instancia de estrategia por nombre."""
    mapping = {s.name: type(s) for s in get_all_strategies()}
    if name in mapping:
        return mapping[name]()
    raise ValueError(f"Estrategia '{name}' no encontrada. Disponibles: {list(mapping.keys())}")


def classify_cooperative(strategy: Strategy) -> bool:
    """Clasifica si una estrategia es cooperativa."""
    return strategy.cooperative


# ============================================================
# Propiedades cualitativas de Axelrod para cada estrategia
# Axelrod (1984) identificó 4 propiedades clave de estrategias exitosas:
#   1. Amable (Nice): Nunca deserta primero
#   2. Provocable (Retaliatory): Castiga la deserción del oponente
#   3. Perdonadora (Forgiving): Vuelve a cooperar si el oponente coopera
#   4. Clara (Clear): Comportamiento predecible/transparente
# ============================================================

AXELROD_PROPERTIES = {
    'TIT FOR TAT': {
        'nice': True, 'retaliatory': True, 'forgiving': True, 'clear': True,
        'description_vs_tft': (
            'Estrategia de referencia (equilibrio). Coopera primero, luego reciproca exactamente. '
            'Es la unica estrategia que cumple las 4 propiedades de Axelrod simultaneamente.'
        ),
        'convergence': (
            'CONVERGE al equilibrio cooperativo. TFT establece cooperacion mutua con cualquier '
            'estrategia amable y castiga la desercion sin escalar. Su simplicidad la hace robusta '
            'contra la mayoria de oponentes en torneos de Axelrod.'
        ),
        'convergence_bool': True,
    },
    'ALL-C': {
        'nice': True, 'retaliatory': False, 'forgiving': True, 'clear': True,
        'description_vs_tft': (
            'Comparte con TFT la amabilidad y el perdon, pero carece de provocabilidad. '
            'Nunca castiga la desercion, lo que la hace explotable por estrategias agresivas. '
            'Obtiene el mismo score que TFT contra cooperadores, pero pierde contra desertores.'
        ),
        'convergence': (
            'NO CONVERGE al equilibrio. Sin capacidad de retaliacion, es invadida por desertores. '
            'En dinamica evolutiva, ALL-C es eliminada progresivamente porque ALL-D la explota '
            'obteniendo T=5 en cada ronda mientras ALL-C obtiene S=0.'
        ),
        'convergence_bool': False,
    },
    'ALL-D': {
        'nice': False, 'retaliatory': True, 'forgiving': False, 'clear': True,
        'description_vs_tft': (
            'Opuesta a TFT en amabilidad y perdon. Siempre deserta independientemente del contexto. '
            'Contra TFT, obtiene T=5 en la primera ronda pero luego cae a P=1 por ronda '
            '(vs R=3 que obtendria cooperando). A largo plazo pierde contra TFT.'
        ),
        'convergence': (
            'CONVERGE a un equilibrio diferente (desercion mutua). Es Nash equilibrium del juego '
            'one-shot pero NO del juego iterado con w alto. Solo domina cuando w < (T-R)/(T-P) = 0.5. '
            'Con w alto, la retaliacion de TFT la castiga suficientemente.'
        ),
        'convergence_bool': False,
    },
    'GRIM': {
        'nice': True, 'retaliatory': True, 'forgiving': False, 'clear': True,
        'description_vs_tft': (
            'Comparte amabilidad y provocabilidad con TFT, pero es implacable: una sola desercion '
            'del oponente causa desercion permanente. Carece del perdon de TFT. '
            'Contra TFT coopera perfectamente, pero un solo error (ruido) destruye la cooperacion.'
        ),
        'convergence': (
            'CONVERGE CONDICIONALMENTE. Sin ruido, GRIM sostiene cooperacion igual que TFT. '
            'Con ruido (trembling hand), NO converge porque un error accidental causa espiral '
            'de desercion permanente. Esta fragilidad es la razon principal por la que '
            'TFT supera a GRIM en el torneo de Axelrod.'
        ),
        'convergence_bool': True,
    },
    'PAVLOV': {
        'nice': True, 'retaliatory': True, 'forgiving': True, 'clear': True,
        'description_vs_tft': (
            'Cumple las 4 propiedades como TFT pero con mecanismo diferente: Win-Stay, Lose-Shift. '
            'Repite la accion si obtuvo R o T, cambia si obtuvo P o S. '
            'Supera a TFT en ambientes ruidosos (Nowak & Sigmund, 1993) porque puede '
            'autocorregir errores sin necesidad de que el oponente tambien coopere primero.'
        ),
        'convergence': (
            'CONVERGE al equilibrio. PAVLOV converge a cooperacion mutua y puede recuperarse '
            'de errores por su mecanismo de autocorreccion. Nowak & Sigmund (1993) demostraron '
            'que PAVLOV domina en ambientes con ruido, superando incluso a TFT. '
            'Sin embargo, es explotable por ALL-D en juegos cortos.'
        ),
        'convergence_bool': True,
    },
    'TFT2T': {
        'nice': True, 'retaliatory': True, 'forgiving': True, 'clear': True,
        'description_vs_tft': (
            'Version mas tolerante de TFT: requiere 2 deserciones consecutivas para retaliar. '
            'Mas perdonadora que TFT pero menos provocable. Mejor que TFT contra ruido bajo '
            'pero vulnerable a desertores intermitentes que explotan su paciencia alternando C/D.'
        ),
        'convergence': (
            'CONVERGE PARCIALMENTE. Coopera bien con estrategias amables y tolera errores '
            'esporadicos mejor que TFT. Pero falla ante explotacion sistematica: JOSS y TESTER '
            'pueden desertar cada 2 rondas sin ser castigados. '
            'Por eso TFT2T pierde ante poblaciones mixtas con explotadores.'
        ),
        'convergence_bool': True,
    },
    'RANDOM': {
        'nice': False, 'retaliatory': False, 'forgiving': False, 'clear': False,
        'description_vs_tft': (
            'No cumple ninguna propiedad de Axelrod. Acciones completamente impredecibles. '
            'Contra TFT, produce cooperacion ~50% del tiempo (cuando TFT copia su C anterior). '
            'Score esperado es inferior a TFT porque deserta gratuitamente ~50% del tiempo.'
        ),
        'convergence': (
            'NO CONVERGE. Sin estructura ni memoria, RANDOM no puede establecer patrones '
            'cooperativos. Genera payoff promedio de (R+S+T+P)/4 ≈ 2.25 por ronda, '
            'inferior al R=3 de cooperacion mutua. Es eliminada evolutivamente por '
            'cualquier estrategia con estructura.'
        ),
        'convergence_bool': False,
    },
    'JOSS': {
        'nice': False, 'retaliatory': True, 'forgiving': True, 'clear': False,
        'description_vs_tft': (
            'TFT "tramposa" que deserta aleatoriamente 10% del tiempo. Pierde la amabilidad '
            'y claridad de TFT. Contra TFT, las deserciones aleatorias de JOSS provocan '
            'retaliacion de TFT, generando espirales de castigo mutuo que reducen '
            'el score de ambos.'
        ),
        'convergence': (
            'NO CONVERGE. Las deserciones aleatorias de JOSS rompen la cooperacion con '
            'cualquier estrategia reactiva. Contra TFT, genera ciclos D-D que degradan '
            'el score promedio. Axelrod demostro que "ser tramposo no paga" porque '
            'la ganancia marginal de desertar se pierde en la retaliacion resultante.'
        ),
        'convergence_bool': False,
    },
    'GRADUAL': {
        'nice': True, 'retaliatory': True, 'forgiving': True, 'clear': True,
        'description_vs_tft': (
            'Castigo proporcional: tras la N-esima desercion, castiga con N rondas de D, '
            'luego ofrece 2 rondas de C. Mas sofisticada que TFT en la graduacion del castigo. '
            'Contra TFT coopera perfectamente. Contra desertores, el castigo creciente '
            'eventualmente disuade mas que el castigo unitario de TFT.'
        ),
        'convergence': (
            'CONVERGE al equilibrio. El castigo proporcional es mas efectivo que el de TFT '
            'contra oponentes que prueban limites. Las 2 rondas de reconciliacion garantizan '
            'que la cooperacion se restaure. En algunos torneos modernos, GRADUAL supera a TFT '
            'por su mejor manejo de oponentes parcialmente cooperativos.'
        ),
        'convergence_bool': True,
    },
    'ADAPTIVE': {
        'nice': True, 'retaliatory': True, 'forgiving': True, 'clear': False,
        'description_vs_tft': (
            'Aprende empiricamente que accion da mejor payoff. Empieza cooperando (6C, 5D), '
            'luego elige la accion con mayor payoff promedio historico. '
            'Menos transparente que TFT porque su comportamiento depende del historial acumulado. '
            'Contra TFT, tiende a converger a cooperacion porque C da R=3 > promedio de D.'
        ),
        'convergence': (
            'CONVERGE CONDICIONALMENTE. Si los primeros oponentes son cooperativos, ADAPTIVE '
            'aprende que C es rentable y converge a cooperacion. Si enfrenta desertores primero, '
            'puede converger a desercion. Es sensible a las condiciones iniciales, '
            'a diferencia de TFT que es robusta independientemente del orden de oponentes.'
        ),
        'convergence_bool': True,
    },
    'FRIEDMAN': {
        'nice': True, 'retaliatory': True, 'forgiving': False, 'clear': True,
        'description_vs_tft': (
            'Identica a GRIM. Coopera hasta la primera desercion, luego deserta permanentemente. '
            'Difiere de TFT solo en la falta de perdon. Contra TFT coopera perfectamente, '
            'pero con ruido tiene el mismo problema que GRIM: un error destruye la relacion.'
        ),
        'convergence': (
            'CONVERGE CONDICIONALMENTE (mismo que GRIM). Sin ruido mantiene cooperacion. '
            'Con ruido falla porque la falta de perdon convierte errores temporales en '
            'desercion permanente. Axelrod la incluyo en su primer torneo como ejemplo '
            'de estrategia que castiga pero no perdona.'
        ),
        'convergence_bool': True,
    },
    'TESTER': {
        'nice': False, 'retaliatory': True, 'forgiving': True, 'clear': False,
        'description_vs_tft': (
            'Explora al oponente desertando en ronda 1. Si es castigada, cambia a TFT. '
            'Si no, explota alternando C/D. Pierde la amabilidad y claridad de TFT. '
            'Contra TFT: deserta en R1, TFT retalia en R2, TESTER cambia a TFT → cooperacion '
            'desde R3, pero con un deficit inicial de 1 ronda.'
        ),
        'convergence': (
            'NO CONVERGE establemente. Aunque se adapta a TFT, su desercion inicial la penaliza. '
            'Contra estrategias amables sin retaliacion (ALL-C, TFT2T), las explota '
            'obteniendo ganancia a costa de la cooperacion grupal. '
            'Es un "parasito" del ecosistema cooperativo.'
        ),
        'convergence_bool': False,
    },
}


def get_strategy_properties() -> dict:
    """Retorna las propiedades de Axelrod para todas las estrategias."""
    return AXELROD_PROPERTIES.copy()


def get_properties_dataframe():
    """Retorna DataFrame con propiedades cualitativas de cada estrategia."""
    import pandas as pd
    rows = []
    for name, props in AXELROD_PROPERTIES.items():
        rows.append({
            'Estrategia': name,
            'Amable': 'Si' if props['nice'] else 'No',
            'Provocable': 'Si' if props['retaliatory'] else 'No',
            'Perdonadora': 'Si' if props['forgiving'] else 'No',
            'Clara': 'Si' if props['clear'] else 'No',
            'Props. Axelrod (de 4)': sum([props['nice'], props['retaliatory'],
                                          props['forgiving'], props['clear']]),
            'Converge': 'Si' if props['convergence_bool'] else 'No',
        })
    return pd.DataFrame(rows)
