"""
# memory_utils.py - v1.1734482910
# Updated: Tuesday, May 21, 2025
# Changes in this version:
# - Added detailed memory tracking for both CUDA and MPS
# - Implemented tensor pooling to reduce memory fragmentation
# - Added force_full_cleanup option for aggressive memory clearing
# - Improved garbage collection with better tensor handling
# - Added memory usage reporting functions

Memory management utilities for MatAnyone video processing.
Contains functions for clearing memory and optimizing resource usage.
"""

import gc
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

# Global tensor pool
_tensor_pool = {}
_memory_stats = {"peak": 0, "current": 0, "cleared_total": 0, "last_cleared": 0}
_processors_pool = {}

def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import torch
        
        result = {
            "available": True,
            "device": "unknown",
            "allocated": 0,
            "reserved": 0,
            "peak_allocated": 0,
            "peak_reserved": 0,
        }
        
        # Check for CUDA
        if torch.cuda.is_available():
            result["device"] = f"CUDA ({torch.cuda.get_device_name(0)})"
            result["allocated"] = torch.cuda.memory_allocated() / (1024**2)  # MB
            result["reserved"] = torch.cuda.memory_reserved() / (1024**2)  # MB
            result["peak_allocated"] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            result["peak_reserved"] = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result["device"] = "MPS (Apple Silicon)"
            # MPS doesn't have built-in memory tracking like CUDA
            # We can only track system memory usage as a proxy
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            result["allocated"] = memory_info.rss / (1024**2)  # MB
            result["reserved"] = memory_info.vms / (1024**2)  # MB
            # No reliable peak tracking for MPS
            result["peak_allocated"] = _memory_stats["peak"]
            result["peak_reserved"] = _memory_stats["peak"]
        
        # Update global stats
        _memory_stats["current"] = result["allocated"]
        if result["allocated"] > _memory_stats["peak"]:
            _memory_stats["peak"] = result["allocated"]
        
        return result
    
    except Exception as e:
        return {
            "available": False,
            "device": "unknown",
            "error": str(e)
        }

def clear_gpu_memory(processor=None, force_full_cleanup=False) -> Dict[str, Any]:
    """
    Clear memory aggressively for both CUDA and MPS
    
    Args:
        processor: Optional processor instance to clear
        force_full_cleanup: Whether to perform aggressive tensor cleanup
        
    Returns:
        Dictionary with memory statistics before and after cleanup
    """
    print("Clearing memory...")
    
    # Get memory stats before clearing
    before_stats = get_memory_usage()
    before_mem = before_stats.get("allocated", 0)
    
    # Clear the processor's internal memory cache if provided
    if processor is not None:
        if hasattr(processor, 'clear_internal_memory'):
            processor.clear_internal_memory()
        if hasattr(processor, 'model') and hasattr(processor.model, 'reset_kv'):
            processor.model.reset_kv()
    
    # Force CUDA cache clearing
    try:
        import torch
        if torch.cuda.is_available():
            # Empty cache
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared")
            
            # Force collection of tensor fragments in aggressive mode
            if force_full_cleanup:
                # Explicitly clear any dangling tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj):
                            # Detach tensors that might have gradients
                            if obj.requires_grad:
                                obj = obj.detach()
                            # Move tensors to CPU to free GPU memory
                            if obj.is_cuda:
                                obj = obj.cpu()
                    except Exception:
                        pass
                
                # Call garbage collector
                gc.collect()
                
                # Empty cache again after GC
                torch.cuda.empty_cache()
                print("CUDA aggressive cleanup completed")
    except Exception as e:
        print(f"CUDA memory clearing error: {e}")
    
    # Try to force MPS cache clearing for Apple Silicon
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Explicitly clear any references to MPS tensors
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.device.type == 'mps':
                        obj = obj.cpu()
                except Exception:
                    pass
            
            # Force garbage collection
            gc.collect()
            print("MPS memory cache cleared")
    except Exception as e:
        print(f"MPS memory clearing error: {e}")
    
    # Force garbage collection
    try:
        gc.collect()
    except Exception:
        pass
    
    # Clear tensor pool in aggressive mode
    if force_full_cleanup:
        global _tensor_pool
        _tensor_pool = {}
        print("Tensor pool cleared")
    
    # Get memory stats after clearing
    after_stats = get_memory_usage()
    after_mem = after_stats.get("allocated", 0)
    
    # Update global stats
    memory_cleared = max(0, before_mem - after_mem)
    _memory_stats["last_cleared"] = memory_cleared
    _memory_stats["cleared_total"] += memory_cleared
    
    print(f"Memory clearing completed. Freed approximately {memory_cleared:.2f} MB")
    
    return {
        "before": before_stats,
        "after": after_stats,
        "cleared": memory_cleared
    }

def get_cached_processor(model_path: str, model_type: str = "matanyone", processor_cls=None):
    """
    Get or create a cached processor instance
    
    Args:
        model_path: Path to the model
        model_type: Type of model ('matanyone' or any installed plugin)
        processor_cls: Processor class to instantiate if not cached
        
    Returns:
        Processor instance
    """
    global _processors_pool
    
    # Use both model path and type as key for the cache
    cache_key = f"{model_type}:{model_path}"
    
    if cache_key not in _processors_pool:
        print(f"Creating new processor instance for {model_type} model at {model_path}")
        
        if processor_cls is None:
            # Import here to avoid circular imports
            from core.inference_core import InterruptibleInferenceCore
            processor_cls = InterruptibleInferenceCore
        
        # Create new processor with model type
        _processors_pool[cache_key] = processor_cls(model_path, model_type=model_type)
    else:
        print(f"Reusing cached processor for {model_type} model at {model_path}")
    
    return _processors_pool[cache_key]

def clear_processor_pool():
    """
    Clear the processor pool to release model memory
    """
    global _processors_pool
    
    # Clear each processor's internal memory
    for processor in _processors_pool.values():
        if hasattr(processor, 'clear_internal_memory'):
            processor.clear_internal_memory()
    
    # Clear the pool
    _processors_pool = {}
    
    # Force memory cleanup
    clear_gpu_memory(force_full_cleanup=True)
    
    print("Processor pool cleared")

def get_tensor_from_pool(shape, dtype, device="cuda", zero_filled=True):
    """
    Get a tensor from the pool or create a new one
    
    Args:
        shape: Tensor shape
        dtype: Tensor data type
        device: Tensor device
        zero_filled: Whether to fill with zeros
        
    Returns:
        Tensor of the specified shape and type
    """
    global _tensor_pool
    
    # Create key for the tensor pool
    key = (tuple(shape), dtype, str(device))
    
    if key in _tensor_pool:
        # Reuse existing tensor
        tensor = _tensor_pool[key]
        
        # Ensure tensor is on the correct device
        if tensor.device != device:
            tensor = tensor.to(device)
        
        # Zero-fill if requested
        if zero_filled:
            tensor.zero_()
    else:
        # Create new tensor
        try:
            import torch
            tensor = torch.zeros(shape, dtype=dtype, device=device) if zero_filled else torch.empty(shape, dtype=dtype, device=device)
            _tensor_pool[key] = tensor
        except Exception as e:
            print(f"Error creating tensor: {e}")
            import torch
            # Fallback to CPU
            tensor = torch.zeros(shape, dtype=dtype, device="cpu") if zero_filled else torch.empty(shape, dtype=dtype, device="cpu")
    
    return tensor

def return_tensor_to_pool(tensor):
    """
    Return a tensor to the pool for reuse
    
    Args:
        tensor: Tensor to return
    """
    global _tensor_pool
    
    # Create key for the tensor pool
    import torch
    key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
    
    # Only store tensors that are likely to be reused
    if tensor.numel() > 1000:  # Only cache larger tensors
        # Detach the tensor to remove any computational graph
        tensor = tensor.detach()
        _tensor_pool[key] = tensor

def get_memory_operations_history(count=10) -> List[Dict[str, Any]]:
    """Get recent memory operations for diagnosis"""
    with _memory_stats_lock:
        # Return the most recent operations
        operations = _memory_stats.get("operations", [])[-count:]
        return operations

def print_memory_stats(include_operations=False):
    """
    Print current memory usage statistics
    """
    stats = get_memory_usage()
    
    if stats.get("available", False):
        print(f"\nMEMORY USAGE ({stats['device']}):")
        print(f"  Current:  {stats['allocated']:.2f} MB allocated, {stats['reserved']:.2f} MB reserved")
        print(f"  Peak:     {stats['peak_allocated']:.2f} MB allocated, {stats['peak_reserved']:.2f} MB reserved")
        print(f"  Tracking: {_memory_stats['peak']:.2f} MB peak, {_memory_stats['cleared_total']:.2f} MB cleared total\n")
        
        # Print recent operations if requested
        if include_operations:
            recent_ops = get_memory_operations_history(5)  # Last 5 operations
            print("\nRECENT MEMORY OPERATIONS:")
            for op in recent_ops:
                timestamp = time.strftime("%H:%M:%S", time.localtime(op["timestamp"]))
                print(f"  [{timestamp}] {op['operation']} (Thread {op['thread_id']})")
                if "error" in op.get("details", {}):
                    print(f"    Error: {op['details']['error']}")
            print()
    else:
        print("\nMemory tracking unavailable:", stats.get("error", "Unknown error"))