"""
Módulo de Generador de Números Aleatorios Reproducible.
Implementación obligatoria según especificaciones del curso.
Usa numpy.random.Generator con PCG64 para reproducibilidad.
"""

import numpy as np
from numpy.random import PCG64, Generator


class ReproducibleRNG:
    """Generador de números aleatorios reproducible con PCG64."""

    def __init__(self, seed: int = 20260211):
        self.seed = seed
        self.rng = Generator(PCG64(seed))

    def uniform(self, low=0.0, high=1.0, size=None):
        """Genera valores uniformes en [low, high)."""
        return self.rng.uniform(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        """Selecciona elementos aleatorios de un arreglo."""
        return self.rng.choice(a, size, replace, p)

    def integers(self, low, high=None, size=None):
        """Genera enteros aleatorios en [low, high)."""
        return self.rng.integers(low, high, size)

    def random(self, size=None):
        """Genera valores aleatorios en [0, 1)."""
        return self.rng.random(size)

    def reset(self):
        """Reinicia el generador con la semilla original."""
        self.rng = Generator(PCG64(self.seed))

    def spawn(self, n_children: int = 1):
        """Crea generadores hijos independientes para paralelización."""
        bit_gen = self.rng.bit_generator
        child_states = bit_gen.spawn(n_children)
        return [Generator(s) for s in child_states]
