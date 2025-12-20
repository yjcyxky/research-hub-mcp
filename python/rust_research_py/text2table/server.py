"""
vLLM server for text2table model serving.

This module provides a vLLM-based server that keeps models loaded in memory,
avoiding repeated initialization overhead.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _try_wrapper_script_command(
    model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    cache_dir: Optional[Path],
) -> None:
    """Try starting vLLM server using shell script that sets env vars before Python starts."""
    # Find the shell script
    script_dir = Path(__file__).parent
    shell_script = script_dir / "start_vllm_server.sh"
    
    if not shell_script.exists():
        raise FileNotFoundError("Shell wrapper script not found")
    
    # Make sure script is executable
    os.chmod(shell_script, 0o755)
    
    cmd = [
        str(shell_script),
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if cache_dir:
        cmd.extend(["--download-dir", str(cache_dir)])
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    logger.info("Trying shell wrapper script (env vars set before Python): %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def start_vllm_server(
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    cache_dir: Optional[Path] = None,
) -> None:
    """Start a vLLM server for the specified model.
    
    Args:
        model_name: HuggingFace model name or path
        host: Server host address
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        trust_remote_code: Whether to trust remote code
        cache_dir: Optional cache directory for model weights
    """
    logger.info("Starting vLLM server for model: %s", model_name)
    logger.info("Server will listen on %s:%d", host, port)

    # Try multiple methods to start vLLM server
    methods = [
        # Method 1: Try wrapper script (disables torch.compile before import)
        _try_wrapper_script_command,
        # Method 2: Try vllm serve command (newer versions)
        _try_vllm_serve_command,
        # Method 3: Try python -m vllm.entrypoints.openai.api_server
        _try_python_module_command,
    ]

    for method in methods:
        try:
            method(
                model_name=model_name,
                host=host,
                port=port,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
            return  # Success, exit
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.debug("Method failed, trying next: %s", e)
            continue
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            return

    # All methods failed
    logger.error(
        "Failed to start vLLM server with all available methods.\n"
        "This might be due to:\n"
        "1. PyTorch/Triton version incompatibility (try: pip install --upgrade torch triton)\n"
        "2. vLLM not installed (install with: pip install vllm)\n"
        "3. Environment issues\n\n"
        "You can try starting vLLM server manually with torch.compile disabled:\n"
        "  TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 vllm serve %s --host %s --port %d --tensor-parallel-size %d --gpu-memory-utilization %s%s%s\n\n"
        "Or use Python module directly:\n"
        "  TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 python -m vllm.entrypoints.openai.api_server --model %s --host %s --port %d --tensor-parallel-size %d --gpu-memory-utilization %s%s%s",
        model_name,
        host,
        port,
        tensor_parallel_size,
        gpu_memory_utilization,
        " --trust-remote-code" if trust_remote_code else "",
        f" --max-model-len {max_model_len}" if max_model_len else "",
        model_name,
        host,
        port,
        tensor_parallel_size,
        gpu_memory_utilization,
        " --trust-remote-code" if trust_remote_code else "",
        f" --max-model-len {max_model_len}" if max_model_len else "",
    )
    sys.exit(1)


def _try_vllm_serve_command(
    model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    cache_dir: Optional[Path],
) -> None:
    """Try starting vLLM server using 'vllm serve' command."""
    # Set environment variables to disable torch.compile
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    
    cmd = [
        "vllm", "serve",
        model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if cache_dir:
        cmd.extend(["--download-dir", str(cache_dir)])
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    logger.info("Trying vLLM serve command with torch.compile disabled: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _try_python_module_command(
    model_name: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    cache_dir: Optional[Path],
) -> None:
    """Try starting vLLM server using Python module."""
    # Set environment variables to potentially avoid Triton/torch.compile issues
    env = os.environ.copy()
    # Disable torch.compile to avoid Triton template duplication issues
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env.setdefault("TORCH_COMPILE_DEBUG", "0")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    # Disable torch inductor compilation
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if cache_dir:
        cmd.extend(["--download-dir", str(cache_dir)])
    
    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    logger.info("Trying Python module command with torch.compile disabled: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    """CLI entry point for vLLM server."""
    parser = argparse.ArgumentParser(
        description="Start vLLM server for text2table models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name or path (default: Qwen/Qwen3-30B-A3B-Instruct-2507; set to your served model id)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length (default: model's max)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code (default: True)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for model weights",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_vllm_server(
        model_name=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
