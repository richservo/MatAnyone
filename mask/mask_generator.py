"""
# mask_generator.py - v1.1716998903
# Updated: Tuesday, May 21, 2025 at 17:15:03 PST
# Changes in this version:
# - Updated to import from new modularized mask generator code
# - Maintains compatibility as a central entry point for the mask generation system
# - Re-exports the necessary classes for backwards compatibility

Compatibility layer for mask generation functionality.
"""

# Import from the modularized mask generator code
from mask.sam_generator import SAMMaskGenerator
from ui.mask_generator_ui_main import MaskGeneratorUI

# Re-export for backward compatibility
__all__ = ['MaskGeneratorUI', 'SAMMaskGenerator']
