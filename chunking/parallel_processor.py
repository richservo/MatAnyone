"""
# parallel_processor.py - v1.1734506012
# Updated: Tuesday, May 21, 2025
# Changes in this version:
# - Added timeout for GPU lock acquisition to prevent deadlocks
# - Improved error handling and recovery in chunk processing
# - Added status tracking for better diagnostic information
# - Enhanced lock management with safety checks to prevent hanging
# - Added force_continue option to proceed despite chunk failures

Module for parallel processing of video chunks.
"""

import os
import time
import traceback
import multiprocessing
import threading
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait, FIRST_EXCEPTION
from threading import Lock, Event

# Default number of parallel workers (adjust based on testing)
DEFAULT_MAX_WORKERS = min(multiprocessing.cpu_count(), 4)

# Global lock for thread-safe printing
print_lock = Lock()

# Global lock for GPU access with timeout tracking
gpu_lock = Lock()

# Lock acquisition status tracking
_lock_acquire_time = {}
_lock_holder_thread = None  # Global variable to track which thread holds the lock
_lock_status_lock = Lock()


def thread_safe_print(*args, **kwargs):
    """
    Thread-safe version of print
    """
    with print_lock:
        print(*args, **kwargs)


class ParallelChunkProcessor:
    """
    Helper class for processing video chunks in parallel
    """
    
    def __init__(self, max_workers=None, use_gpu_lock=True, lock_timeout=30, force_continue=False):
        """
        Initialize the parallel chunk processor
        
        Args:
            max_workers: Maximum number of parallel workers (defaults to CPU count)
            use_gpu_lock: Whether to use a lock for GPU operations
        """
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.use_gpu_lock = use_gpu_lock
        self.lock_timeout = lock_timeout  # Timeout in seconds
        self.force_continue = force_continue  # Continue processing even if some chunks fail
        self.results = []
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.lock_timeouts = 0
        self.stop_event = Event()  # For emergency shutdown
        
        # Try to determine if CUDA is available for better defaults
        try:
            import torch
            if torch.cuda.is_available():
                # For CUDA, limit concurrent GPU usage more strictly
                self.max_workers = min(self.max_workers, 2)
                thread_safe_print(f"CUDA detected, limiting parallel workers to {self.max_workers}")
        except:
            pass
    
    def process_chunks(self, chunk_infos: List[Dict[str, Any]], 
                       process_func: Callable[[Dict[str, Any], int, Dict[str, Any]], Any],
                       shared_args: Dict[str, Any] = None,
                       timeout: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process chunks in parallel
        
        Args:
            chunk_infos: List of chunk information dictionaries
            process_func: Function to process each chunk (args: chunk_info, chunk_idx, shared_args)
            shared_args: Shared arguments passed to all chunk processing functions
            
        Returns:
            List of results from chunk processing
        """
        if shared_args is None:
            shared_args = {}
            
        total_chunks = len(chunk_infos)
        self.started = 0
        self.completed = 0
        self.failed = 0
        self.results = []
        
        thread_safe_print(f"Starting parallel processing of {total_chunks} chunks with {self.max_workers} workers")
        start_time = time.time()
        
        # Reset stop event
        self.stop_event.clear()
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = []
            for idx, chunk_info in enumerate(chunk_infos):
                futures.append(executor.submit(
                    self._process_chunk_wrapper, 
                    chunk_info, 
                    idx, 
                    process_func, 
                    shared_args,
                    total_chunks
                ))
            
            # Wait for futures with timeout monitoring
            if timeout is not None:
                # Add overall timeout
                done, not_done = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
                if not_done:
                    # Cancel remaining tasks
                    thread_safe_print(f"WARNING: Timeout occurred, canceling {len(not_done)} remaining tasks")
                    self.stop_event.set()  # Signal all threads to stop
                    for future in not_done:
                        future.cancel()
                    # Try to force-release GPU lock if it's held too long
                    self._emergency_lock_release()
            
            # Collect results as they complete
            for future in futures:
                if future.done():
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        thread_safe_print(f"Error in parallel processing: {str(e)}")
                        traceback.print_exc()
                        self.failed += 1
                        if not self.force_continue:
                            thread_safe_print("Stopping processing due to error (force_continue=False)")
                            self.stop_event.set()  # Signal all threads to stop
                            break
        
        # Sort results by chunk index
        self.results.sort(key=lambda x: x.get('chunk_idx', 0))
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        thread_safe_print(f"Parallel processing completed in {elapsed:.2f} seconds")
        thread_safe_print(f"Chunks: {total_chunks}, Completed: {self.completed}, Failed: {self.failed}")
        
        return self.results
    
    def _emergency_lock_release(self):
        """Emergency method to release GPU lock if held too long"""
        global _lock_holder_thread  # Declare global at the start of the function
        with _lock_status_lock:
            current_time = time.time()
            if _lock_holder_thread is not None:
                lock_time = _lock_acquire_time.get(_lock_holder_thread, 0)
                # Check if the lock has been held for more than double the timeout
                if current_time - lock_time > 2 * self.lock_timeout:
                    thread_safe_print(f"WARNING: EMERGENCY - Attempting to force-release GPU lock held by thread {_lock_holder_thread}")
                    if gpu_lock.locked():
                        try:
                            # Try to force cleanup and release lock
                            from utils.memory_utils import clear_gpu_memory
                            clear_gpu_memory(force_full_cleanup=True)
                            gpu_lock.release()
                            thread_safe_print("WARNING: GPU lock forcibly released")
                            # Clear the holder info (global was declared at function start)
                            _lock_holder_thread = None
                        except Exception as e:
                            thread_safe_print(f"Failed to force-release GPU lock: {e}")
    
    def _process_chunk_wrapper(self, chunk_info: Dict[str, Any], chunk_idx: int, 
                              process_func: Callable, shared_args: Dict[str, Any],
                              total_chunks: int) -> Dict[str, Any]:
        # Declare globals at the start of the function
        global _lock_holder_thread
        """
        Wrapper for processing a single chunk
        
        Args:
            chunk_info: Information about the chunk
            chunk_idx: Index of the chunk
            process_func: Function to process the chunk
            shared_args: Shared arguments for processing
            total_chunks: Total number of chunks (for progress reporting)
            
        Returns:
            Processing result
        """
        # Report start
        with print_lock:
            self.started += 1
            thread_safe_print(f"Starting chunk {chunk_idx+1}/{total_chunks} ({self.started} started, {self.completed} completed)")
        
        # Import utils here to avoid circular imports
        from utils.memory_utils import clear_gpu_memory
        
        start_time = time.time()
        result = None
        
        # Check if we've been asked to stop
        if self.stop_event.is_set():
            thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: Skipped due to stop event")
            return None
            
        # Track thread ID for lock monitoring
        thread_id = threading.get_ident()
        lock_acquired = False
            
        try:
            # Acquire GPU lock if needed
            if self.use_gpu_lock:
                thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: Waiting for GPU access")
                if self.lock_timeout > 0:
                    # Try to acquire with timeout
                    lock_acquired = gpu_lock.acquire(timeout=self.lock_timeout)
                    if not lock_acquired:
                        with _lock_status_lock:
                            thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: TIMEOUT waiting for GPU access after {self.lock_timeout}s")
                            self.lock_timeouts += 1
                            # Try to detect who is holding the lock
                            if _lock_holder_thread is not None:
                                thread_safe_print(f"GPU lock appears to be held by thread {_lock_holder_thread} for {time.time() - _lock_acquire_time.get(_lock_holder_thread, 0):.1f}s")
                            # Raise exception instead of hanging forever
                            raise TimeoutError(f"Failed to acquire GPU lock within {self.lock_timeout} seconds")
                else:
                    # No timeout, use standard acquire
                    gpu_lock.acquire()
                    lock_acquired = True
                
                # Record this thread as the lock holder
                with _lock_status_lock:
                    _lock_holder_thread = thread_id
                    _lock_acquire_time[thread_id] = time.time()
                
                thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: GPU access granted")
            
            # Process the chunk
            result = process_func(chunk_info, chunk_idx, shared_args)
            
            # Include chunk index in result
            if result is not None and isinstance(result, dict):
                result['chunk_idx'] = chunk_idx
            
            # Report completion
            end_time = time.time()
            elapsed = end_time - start_time
            
            with print_lock:
                self.completed += 1
                completion_percent = (self.completed / total_chunks) * 100
                thread_safe_print(f"Completed chunk {chunk_idx+1}/{total_chunks} in {elapsed:.2f}s ({completion_percent:.1f}% done)")
        
        except Exception as e:
            thread_safe_print(f"Error processing chunk {chunk_idx+1}/{total_chunks}: {str(e)}")
            traceback.print_exc()
            self.failed += 1
        
        finally:
            # Release GPU lock if we acquired it
            if self.use_gpu_lock and lock_acquired:
                # Clear memory before releasing lock
                clear_gpu_memory(force_full_cleanup=True)
                
                # Clear the lock holder info before releasing
                with _lock_status_lock:
                    if _lock_holder_thread == thread_id:
                        _lock_holder_thread = None
                        _lock_acquire_time.pop(thread_id, None)
                
                # Actually release the lock
                try:
                    gpu_lock.release()
                    thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: GPU access released")
                except RuntimeError as e:
                    # Lock wasn't actually held by this thread
                    thread_safe_print(f"Chunk {chunk_idx+1}/{total_chunks}: Error releasing GPU lock: {e}")
        
        return result


def process_chunks_parallel(chunk_infos: List[Dict[str, Any]], 
                           process_func: Callable[[Dict[str, Any], int, Dict[str, Any]], Any],
                           shared_args: Dict[str, Any] = None,
                           max_workers: int = None,
                           use_gpu_lock: bool = True,
                           lock_timeout: int = 30,
                           force_continue: bool = False,
                           timeout: int = None) -> List[Dict[str, Any]]:
    """
    Convenience function to process chunks in parallel
    
    Args:
        chunk_infos: List of chunk information dictionaries
        process_func: Function to process each chunk (args: chunk_info, chunk_idx, shared_args)
        shared_args: Shared arguments passed to all chunk processing functions
        max_workers: Maximum number of parallel workers
        use_gpu_lock: Whether to use a lock for GPU operations
        
    Returns:
        List of results from chunk processing
    """
    processor = ParallelChunkProcessor(max_workers=max_workers, use_gpu_lock=use_gpu_lock, 
                                lock_timeout=lock_timeout, force_continue=force_continue)
    return processor.process_chunks(chunk_infos, process_func, shared_args, timeout=timeout)


# Example usage:
"""
def process_chunk(chunk_info, chunk_idx, shared_args):
    # Get shared processor from args
    processor = shared_args.get('processor')
    input_path = shared_args.get('input_path')
    mask_path = shared_args.get('mask_path')
    
    # Process the chunk
    start_frame = chunk_info.get('start_frame')
    end_frame = chunk_info.get('end_frame')
    
    # ... processing logic ...
    
    return {
        'output_path': output_path,
        'start_frame': start_frame,
        'end_frame': end_frame
    }

# Define chunks
chunks = [
    {'start_frame': 0, 'end_frame': 30},
    {'start_frame': 30, 'end_frame': 60},
    # ...
]

# Define shared args
shared_args = {
    'processor': processor,
    'input_path': input_path,
    'mask_path': mask_path
}

# Process chunks in parallel
results = process_chunks_parallel(chunks, process_chunk, shared_args)
"""