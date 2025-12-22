"""
Unit tests for text2table server module.
"""

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from rust_research_py.text2table.server import start_vllm_server


class TestStartVllmServer:
    """Tests for start_vllm_server function."""

    def test_backward_compatible_params(self):
        """Test backward compatibility with model_name and cache_dir params."""
        # Should accept both old and new parameter names
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()  # Stop immediately

            # Using new params
            try:
                start_vllm_server(
                    model="test-model",
                    port=8000,
                    download_dir=Path("/tmp/models"),
                )
            except KeyboardInterrupt:
                pass

            # Using old params (backward compat)
            try:
                start_vllm_server(
                    model_name="test-model-old",
                    port=8001,
                    cache_dir=Path("/tmp/cache"),
                )
            except KeyboardInterrupt:
                pass

    def test_default_model(self):
        """Test default model is used when not specified."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(port=8000)
            except KeyboardInterrupt:
                pass

            # Check that vllm serve was called
            call_args = mock_run.call_args
            assert call_args is not None
            cmd = call_args[0][0]
            # Default model should be used
            assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in cmd

    def test_environment_variables_set(self):
        """Test that torch.compile disable env vars are set."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(model="test", port=8000)
            except KeyboardInterrupt:
                pass

            call_args = mock_run.call_args
            env = call_args[1].get("env", {})
            assert env.get("TORCH_COMPILE_DISABLE") == "1"
            assert env.get("TORCHDYNAMO_DISABLE") == "1"

    def test_command_construction(self):
        """Test vLLM command is constructed correctly."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(
                    model="my-model",
                    host="0.0.0.0",
                    port=8080,
                    tensor_parallel_size=2,
                    gpu_memory_utilization=0.85,
                    trust_remote_code=True,
                )
            except KeyboardInterrupt:
                pass

            call_args = mock_run.call_args
            cmd = call_args[0][0]

            assert "vllm" in cmd
            assert "serve" in cmd
            assert "my-model" in cmd
            assert "--host" in cmd
            assert "0.0.0.0" in cmd
            assert "--port" in cmd
            assert "8080" in cmd
            assert "--tensor-parallel-size" in cmd
            assert "2" in cmd
            assert "--gpu-memory-utilization" in cmd
            assert "0.85" in cmd
            assert "--trust-remote-code" in cmd

    def test_optional_max_model_len(self):
        """Test max_model_len is included when specified."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(
                    model="test",
                    port=8000,
                    max_model_len=4096,
                )
            except KeyboardInterrupt:
                pass

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--max-model-len" in cmd
            assert "4096" in cmd

    def test_optional_download_dir(self):
        """Test download_dir is included when specified."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(
                    model="test",
                    port=8000,
                    download_dir=Path("/models"),
                )
            except KeyboardInterrupt:
                pass

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--download-dir" in cmd
            assert "/models" in cmd

    def test_no_trust_remote_code(self):
        """Test trust_remote_code=False excludes the flag."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                start_vllm_server(
                    model="test",
                    port=8000,
                    trust_remote_code=False,
                )
            except KeyboardInterrupt:
                pass

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "--trust-remote-code" not in cmd


class TestServerErrorHandling:
    """Tests for error handling in server module."""

    def test_vllm_not_found(self, capsys):
        """Test handling when vLLM is not installed."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(SystemExit) as exc_info:
                start_vllm_server(model="test", port=8000)

            assert exc_info.value.code == 1

    def test_subprocess_error(self, capsys):
        """Test handling subprocess errors."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "vllm")

            with pytest.raises(SystemExit) as exc_info:
                start_vllm_server(model="test", port=8000)

            assert exc_info.value.code == 1

    def test_keyboard_interrupt(self):
        """Test graceful handling of keyboard interrupt."""
        with mock.patch("subprocess.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            # Should not raise, just return
            start_vllm_server(model="test", port=8000)
