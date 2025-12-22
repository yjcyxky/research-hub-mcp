"""
Unit tests for text2table config module.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from rust_research_py.text2table.config import (
    Text2TableConfig,
    detect_input_format,
    detect_output_format,
)


class TestText2TableConfig:
    """Tests for Text2TableConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Text2TableConfig()

        assert config.server_url == ""
        assert config.labels == []
        assert config.gliner_model == "Ihor/gliner-biomed-large-v1.0"
        assert config.threshold == 0.5
        assert config.gliner_soft_threshold == 0.3
        assert config.max_new_tokens == 4096
        assert config.temperature == 0.2
        assert config.top_p == 0.9
        assert config.request_timeout == 600
        assert config.concurrency == 4
        assert config.use_gliner is True
        assert config.enable_thinking is False
        assert config.enable_row_validation is False

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = Text2TableConfig(
            server_url="http://localhost:8000/v1",
            labels=["Gene", "Disease", "Drug"],
            threshold=0.7,
            max_new_tokens=2048,
            concurrency=8,
        )

        assert config.server_url == "http://localhost:8000/v1"
        assert config.labels == ["Gene", "Disease", "Drug"]
        assert config.threshold == 0.7
        assert config.max_new_tokens == 2048
        assert config.concurrency == 8

    def test_from_env(self):
        """Test loading config from environment variables."""
        with mock.patch.dict(os.environ, {
            "TEXT2TABLE_VLLM_URL": "http://test:8000/v1",
            "TEXT2TABLE_GLINER_URL": "http://gliner:9000",
        }):
            config = Text2TableConfig.from_env()
            assert config.server_url == "http://test:8000/v1"
            assert config.gliner_url == "http://gliner:9000"

    def test_from_env_missing_vars(self):
        """Test loading config with missing environment variables."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove the env vars if they exist
            env = os.environ.copy()
            env.pop("TEXT2TABLE_VLLM_URL", None)
            env.pop("TEXT2TABLE_GLINER_URL", None)

            with mock.patch.dict(os.environ, env, clear=True):
                config = Text2TableConfig.from_env()
                assert config.server_url == ""
                assert config.gliner_url is None


class TestDetectInputFormat:
    """Tests for detect_input_format function."""

    def test_txt_file(self):
        """Test .txt file detection."""
        assert detect_input_format(Path("input.txt")) == "single"
        assert detect_input_format(Path("/path/to/file.TXT")) == "single"

    def test_tsv_file(self):
        """Test .tsv file detection."""
        assert detect_input_format(Path("data.tsv")) == "batch"
        assert detect_input_format(Path("DATA.TSV")) == "batch"

    def test_csv_file(self):
        """Test .csv file detection."""
        assert detect_input_format(Path("data.csv")) == "batch"
        assert detect_input_format(Path("export.CSV")) == "batch"

    def test_jsonl_file(self):
        """Test .jsonl file detection."""
        assert detect_input_format(Path("records.jsonl")) == "batch"
        assert detect_input_format(Path("DATA.JSONL")) == "batch"

    def test_unsupported_format(self):
        """Test unsupported file format raises error."""
        with pytest.raises(ValueError, match="Unsupported input format"):
            detect_input_format(Path("file.pdf"))

        with pytest.raises(ValueError, match="Unsupported input format"):
            detect_input_format(Path("data.xlsx"))

        with pytest.raises(ValueError, match="Unsupported input format"):
            detect_input_format(Path("file.json"))


class TestDetectOutputFormat:
    """Tests for detect_output_format function."""

    def test_none_defaults_to_tsv(self):
        """Test None path defaults to TSV."""
        assert detect_output_format(None) == "tsv"

    def test_tsv_output(self):
        """Test .tsv output detection."""
        assert detect_output_format(Path("output.tsv")) == "tsv"
        assert detect_output_format(Path("OUTPUT.TSV")) == "tsv"

    def test_csv_output(self):
        """Test .csv output detection."""
        assert detect_output_format(Path("output.csv")) == "csv"
        assert detect_output_format(Path("export.CSV")) == "csv"

    def test_jsonl_output(self):
        """Test .jsonl output detection."""
        assert detect_output_format(Path("results.jsonl")) == "jsonl"
        assert detect_output_format(Path("OUTPUT.JSONL")) == "jsonl"

    def test_unknown_defaults_to_tsv(self):
        """Test unknown extension defaults to TSV."""
        assert detect_output_format(Path("output.dat")) == "tsv"
        assert detect_output_format(Path("file.txt")) == "tsv"


class TestIntegration:
    """Integration tests for config module."""

    def test_config_serialization(self):
        """Test config can be serialized to dict."""
        from dataclasses import asdict

        config = Text2TableConfig(
            server_url="http://localhost:8000",
            labels=["Entity"],
            threshold=0.8,
        )

        config_dict = asdict(config)
        assert config_dict["server_url"] == "http://localhost:8000"
        assert config_dict["labels"] == ["Entity"]
        assert config_dict["threshold"] == 0.8

    def test_path_handling(self):
        """Test various path formats."""
        # Unix paths
        assert detect_input_format(Path("/home/user/data.txt")) == "single"
        assert detect_input_format(Path("/data/batch.jsonl")) == "batch"

        # Relative paths
        assert detect_input_format(Path("./input.tsv")) == "batch"
        assert detect_input_format(Path("../data.csv")) == "batch"
