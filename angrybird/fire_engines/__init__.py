"""
angrybird.fire_engines
======================

Fire spread engines used by the AB Protocol.

  GPUFireEngine   — PyTorch GPU engine with crown fire, Nelson FMC, terrain slope
"""

from .gpu_fire_engine import GPUFireEngine

__all__ = ["GPUFireEngine"]
