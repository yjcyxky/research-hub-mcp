"""
vLLM server for text2table model serving.

Simplified server that starts vLLM with essential parameters.
Environment variables for torch.compile are set before subprocess execution.
"""

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


def start_vllm_server(
    model: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    download_dir: Optional[Path] = None,
    # Backward compatibility aliases
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    """Start a vLLM server for the specified model.

    Args:
        model: HuggingFace model name or path
        host: Server host address
        port: Server port
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        trust_remote_code: Whether to trust remote code
        download_dir: Optional directory for model weights
        model_name: Deprecated alias for model
        cache_dir: Deprecated alias for download_dir
    """
    # Handle backward compatibility
    if model is None:
        model = model_name or "Qwen/Qwen3-30B-A3B-Instruct-2507"
    if download_dir is None:
        download_dir = cache_dir
    logger.info("Starting vLLM server for model: %s", model)
    logger.info("Server will listen on %s:%d", host, port)

    # Set environment variables to disable torch.compile (avoids Triton issues)
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"

    # Build command
    cmd = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if download_dir:
        cmd.extend(["--download-dir", str(download_dir)])

    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    logger.info("Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError:
        logger.error(
            "vLLM not found. Install with: pip install vllm\n"
            "Or start manually:\n"
            "  TORCH_COMPILE_DISABLE=1 vllm serve %s --host %s --port %d",
            model, host, port
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("vLLM server failed with exit code %d", e.returncode)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


@click.command()
@click.option(
    "--model", "-m",
    default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    show_default=True,
    help="Model name or path",
)
@click.option(
    "--port", "-p",
    default=8000,
    type=int,
    show_default=True,
    help="Server port",
)
@click.option(
    "--host", "-H",
    default="0.0.0.0",
    show_default=True,
    help="Server host",
)
@click.option(
    "--tensor-parallel-size", "-tp",
    default=1,
    type=int,
    show_default=True,
    help="Number of GPUs for tensor parallelism",
)
@click.option(
    "--gpu-memory-utilization",
    default=0.9,
    type=float,
    show_default=True,
    help="Fraction of GPU memory to use (0.0-1.0)",
)
@click.option(
    "--max-model-len",
    default=None,
    type=int,
    help="Maximum sequence length (default: model's max)",
)
@click.option(
    "--trust-remote-code/--no-trust-remote-code",
    default=True,
    show_default=True,
    help="Whether to trust remote code",
)
@click.option(
    "--download-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for model weights",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level",
)
def main(
    model: str,
    port: int,
    host: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    download_dir: Optional[Path],
    log_level: str,
) -> None:
    """Start vLLM server for text2table."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_vllm_server(
        model=model,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        download_dir=download_dir,
    )


if __name__ == "__main__":
    main()
