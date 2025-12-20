#!/usr/bin/env python3
"""
Wrapper script to start vLLM server with torch.compile disabled.

This script must be run with environment variables set BEFORE Python starts.
Use the shell script start_vllm_server.sh instead, or set env vars manually.
"""

import os
import sys

# These should be set before Python starts, but set them here as backup
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

# Try to patch torch before it's imported by vLLM
# This is a last resort - environment variables should be set earlier
try:
    # Import torch early and patch it
    import torch
    
    # Disable dynamo
    if hasattr(torch, "_dynamo"):
        try:
            torch._dynamo.config.disable = True
        except Exception:
            pass
    
    # Patch torch.compile to be a no-op
    if hasattr(torch, "compile"):
        _original_compile = torch.compile
        
        def _noop_compile(fn=None, *args, **kwargs):
            if fn is not None and callable(fn):
                return fn
            # If called as decorator without args
            if fn is None:
                def decorator(func):
                    return func
                return decorator
            return _original_compile(fn, *args, **kwargs)
        
        torch.compile = _noop_compile
        
        # Also try to patch at module level
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.disable = True
            
except Exception as e:
    print(f"Warning: Could not patch torch.compile: {e}", file=sys.stderr)
    print("Make sure to set TORCHDYNAMO_DISABLE=1 before starting Python", file=sys.stderr)

# Now import and run vLLM
if __name__ == "__main__":
    from vllm.entrypoints.openai.api_server import main
    main()

