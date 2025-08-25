"""
ResDMD - Residual Dynamic Mode Decomposition Python Module

This module implements core ResDMD algorithms and Koopman operator analysis,
based on MATLAB implementations in main_routines.

Main functions:
- resdmd_algorithm: Core ResDMD Algorithm 1
- resdmd_with_rbf_features: Complete ResDMD with RBF feature construction
- koop_pseudospec_qr: Koopman pseudospectrum computation using QR decomposition
- analyze_eigenvalues: Eigenvalue analysis utilities

Usage examples:
    import resdmd

    # Core ResDMD algorithm
    LAM, V, residuals = resdmd.resdmd(X, Y)

    # Compute pseudospectrum
    RES, RES2, V2 = resdmd.koop_pseudospec_qr(PX, PY, z_pts, W)
"""

from .koop_pseudospec_qr import koop_pseudospec_qr
from .resdmd_algorithm import analyze_eigenvalues, resdmd, resdmd_algorithm

# Define module version
__version__ = "1.0.0"

# Define public API
__all__ = [
    "koop_pseudospec_qr",
    "resdmd_algorithm",
    "resdmd",
    "analyze_eigenvalues",
]


def get_version():
    """Return module version"""
    return __version__


def list_functions():
    """List all available functions"""
    return __all__


def get_module_info():
    """Get module information"""
    info = {
        "name": "ResDMD",
        "version": __version__,
        "description": "Residual Dynamic Mode Decomposition - Core Algorithm and Koopman Pseudospectra",
        "functions": __all__,
        "matlab_correspondence": {
            "koop_pseudospec_qr": "KoopPseudoSpecQR.m",
            "resdmd_algorithm": "Core ResDMD Algorithm 1",
            "resdmd": "Core ResDMD Algorithm",
            "analyze_eigenvalues": "Eigenvalue analysis utilities",
        },
    }
    return info


print(f"ResDMD module loaded (version {__version__})")
print(f"Available functions: {', '.join(__all__)}")
