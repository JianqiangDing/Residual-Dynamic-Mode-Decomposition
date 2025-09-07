"""
This module contains the implementations for the observable function for the ResDMD algorithm.

for basic observables, it is defined a scalar function of state variables (vector).
"""

from .concat_observ import ConcatObservable
from .poly_observ import PolyObservable
from .rbf_observ import RBFObservable

__all__ = ["PolyObservable", "RBFObservable", "ConcatObservable"]
