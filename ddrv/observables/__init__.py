"""
This module contains the implementations for the observable function for the ResDMD algorithm.

for basic observables, it is defined a scalar function of state variables (vector).
"""

from .poly_observ import PolyObservable

__all__ = ["PolyObservable"]
