# model/__init__.py
"""
FitTrack Pro ML Model Package

This package contains:
- train_model.py: Train Random Forest classifier
- predictor.py: Make predictions for new users
- data/: CSV training data
"""

from .predictor import DietPredictor, get_predictor

__all__ = ['DietPredictor', 'get_predictor']