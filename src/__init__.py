"""
src package initializer for Credit Card Fraud Detection project.

This package contains:
- Data preparation utilities
- Preprocessing & sampling
- Model training (ML + DL)
- Evaluation & threshold calibration
- GAN augmentation (optional)
- HuggingFace push helpers
"""

import logging

__version__ = "0.1.0"

# Configure default logger for all modules under src
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("cc_fraud")

__all__ = ["logger", "__version__"]