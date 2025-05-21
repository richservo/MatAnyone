"""
# chunk_optimizer.py - v1.1735054800
# Updated: December 24, 2024
# Changes in this version:
# - Refactored into multiple specialized modules for better code organization
# - This file now serves as a compatibility layer that imports and re-exports all functions
# - No functional changes - all existing functionality is maintained
# - Improved maintainability by splitting large file into logical components
"""

# Import all functions from the specialized modules
from chunking.chunk_mask_analysis import optimize_chunk_masks
from mask.apply_mask_optimization import apply_mask_optimization
from chunking.chunk_weight_optimization import optimize_reassembly_weights
from chunking.chunk_mask_propagation import propagate_mask_data
from chunking.chunk_visualization import create_composite_masks, create_maximized_alpha_mask

# Re-export all functions for backward compatibility
__all__ = [
    'optimize_chunk_masks',
    'apply_mask_optimization', 
    'optimize_reassembly_weights',
    'propagate_mask_data',
    'create_composite_masks',
    'create_maximized_alpha_mask'
]
