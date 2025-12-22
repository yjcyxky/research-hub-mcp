use include_dir::{include_dir, Dir};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, error, info};
use uuid::Uuid;

// Embed the entire rust_research_py package
static RUST_RESEARCH_PY_PACKAGE: Dir<'_> = include_dir!("python");

/// Install the embedded Python package into the current environment.
pub fn install_python_package() -> Result<(), String> {
    let temp_dir =
        std::env::temp_dir().join(format!("rust-research-py-install-{}", Uuid::new_v4()));
    info!(
        "Extracting embedded Python package to: {}",
        temp_dir.display()
    );

    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| format!("Failed to create temp dir for embedded package: {e}"))?;

    // Extract the package to the temporary directory
    RUST_RESEARCH_PY_PACKAGE
        .extract(&temp_dir)
        .map_err(|e| format!("Failed to extract embedded package: {e}"))?;

    info!("Installing Python package using pip...");

    // Run pip install
    let status = Command::new("pip")
        .arg("install")
        .arg(".")
        .current_dir(&temp_dir)
        .status()
        .map_err(|e| format!("Failed to execute pip: {e}"))?;

    // Cleanup temp dir
    let _ = std::fs::remove_dir_all(&temp_dir);

    if status.success() {
        info!("Successfully installed rust_research_py package.");
        Ok(())
    } else {
        Err(format!("pip install failed with status: {}", status))
    }
}

pub struct PluginDownloadResult {
    pub success: bool,
    pub file_path: Option<PathBuf>,
    pub file_size: Option<u64>,
    pub publisher: Option<String>,
    pub error: Option<String>,
}

pub fn run_plugin_download(
    doi: &str,
    output_dir: &Path,
    filename: Option<&str>,
    wait_time: f64,
    headless: bool,
) -> Result<PluginDownloadResult, String> {
    Python::with_gil(|py| {
        // Import modules
        let utils = py
            .import("rust_research_py.plugins.utils")
            .map_err(|e| format!("Failed to import rust_research_py.plugins.utils: {e}"))?;
        let asyncio = py
            .import("asyncio")
            .map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Create coroutine
        let coroutine = utils
            .call_method1(
                "download_with_detected_plugin",
                (
                    doi,
                    output_dir.to_string_lossy().to_string(), // Convert Path to string
                    filename,
                    wait_time,
                    headless,
                ),
            )
            .map_err(|e| {
                format!("Failed to create download_with_detected_plugin coroutine: {e}")
            })?;

        // Run coroutine
        let result = asyncio
            .call_method1("run", (coroutine,))
            .map_err(|e| format!("Failed to run async loop: {e}"))?;

        // Parse result
        let success: bool = result
            .getattr("success")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let error: Option<String> = result
            .getattr("error")
            .and_then(|v| v.extract())
            .unwrap_or(None);
        let file_path_str: Option<String> = result
            .getattr("file_path")
            .and_then(|v| v.extract())
            .unwrap_or(None);
        let file_size: Option<u64> = result
            .getattr("file_size")
            .and_then(|v| v.extract())
            .unwrap_or(None);
        let publisher: Option<String> = result
            .getattr("publisher")
            .and_then(|v| v.extract())
            .unwrap_or(None);

        Ok(PluginDownloadResult {
            success,
            file_path: file_path_str.map(PathBuf::from),
            file_size,
            publisher,
            error,
        })
    })
}

pub fn run_cdp_download(
    url: &str,
    output_path: &Path,
    headless: bool,
    timeout: u64,
    wait_time: f64,
) -> Result<PluginDownloadResult, String> {
    Python::with_gil(|py| {
        // Import modules
        let utils = py
            .import("rust_research_py.plugins.utils")
            .map_err(|e| format!("Failed to import rust_research_py.plugins.utils: {e}"))?;
        let asyncio = py
            .import("asyncio")
            .map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Create coroutine
        let coroutine = utils
            .call_method1(
                "execute_download_by_cdp",
                (
                    url,
                    output_path.to_string_lossy().to_string(),
                    headless,
                    timeout,
                    wait_time,
                ),
            )
            .map_err(|e| format!("Failed to create execute_download_by_cdp coroutine: {e}"))?;

        // Run coroutine
        let result = asyncio
            .call_method1("run", (coroutine,))
            .map_err(|e| format!("Failed to run async loop: {e}"))?;

        // Parse result
        let success: bool = result
            .getattr("success")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        let error: Option<String> = result
            .getattr("error")
            .and_then(|v| v.extract())
            .unwrap_or(None);
        let file_path_str: Option<String> = result
            .getattr("file_path")
            .and_then(|v| v.extract())
            .unwrap_or(None);
        let file_size: Option<u64> = result
            .getattr("file_size")
            .and_then(|v| v.extract())
            .unwrap_or(None);

        // Publisher not returned here by python, but consistent struct used

        Ok(PluginDownloadResult {
            success,
            file_path: file_path_str.map(PathBuf::from),
            file_size,
            publisher: None,
            error,
        })
    })
}

pub struct Pdf2TextResult {
    pub success: bool,
    pub json_path: Option<String>,
    pub markdown_path: Option<String>,
    pub error: Option<String>,
}

pub fn run_pdf2text(
    pdf_file: &Path,
    output_dir: &Path,
    grobid_url: Option<&str>,
    no_auto_start: bool,
    no_figures: bool,
    no_tables: bool,
    copy_pdf: bool,
    overwrite: bool,
    no_markdown: bool,
) -> Result<Pdf2TextResult, String> {
    Python::with_gil(|py| {
        // Import using standard Python import mechanism
        let pdf2text = py
            .import("rust_research_py.pdf2text.pdf2text")
            .map_err(|e| format!("Failed to import rust_research_py.pdf2text.pdf2text: {e}"))?;

        let kwargs = PyDict::new(py);
        kwargs
            .set_item("pdf_file", pdf_file.to_string_lossy().to_string())
            .map_err(|e| format!("Failed to set pdf_file: {e}"))?;
        kwargs
            .set_item("output_dir", output_dir.to_string_lossy().to_string())
            .map_err(|e| format!("Failed to set output_dir: {e}"))?;

        if let Some(url) = grobid_url {
            kwargs
                .set_item("grobid_url", url)
                .map_err(|e| format!("Failed to set grobid_url: {e}"))?;
        }

        kwargs
            .set_item("auto_start_grobid", !no_auto_start)
            .unwrap();
        kwargs.set_item("extract_figures", !no_figures).unwrap();
        kwargs.set_item("extract_tables", !no_tables).unwrap();
        kwargs.set_item("copy_pdf", copy_pdf).unwrap();
        kwargs.set_item("overwrite", overwrite).unwrap();
        kwargs.set_item("generate_markdown", !no_markdown).unwrap();

        let result = pdf2text
            .call_method("extract_fulltext", (), Some(&kwargs))
            .map_err(|e| format!("Failed to call extract_fulltext: {e}"));

        match result {
            Ok(res) => {
                let path_str: Option<String> = res.extract().unwrap_or(None);

                let json_path = path_str.clone();
                let markdown_path = if !no_markdown {
                    path_str.as_ref().map(|p| p.replace(".json", ".md"))
                } else {
                    None
                };

                Ok(Pdf2TextResult {
                    success: path_str.is_some(),
                    json_path,
                    markdown_path,
                    error: None,
                })
            }
            Err(e) => {
                // If the python call raised an exception
                Ok(Pdf2TextResult {
                    success: false,
                    json_path: None,
                    markdown_path: None,
                    error: Some(e.to_string()),
                })
            }
        }
    })
}

pub struct Text2TableResult {
    pub success: bool,
    pub table: Option<String>,
    pub entities: Option<String>,
    pub thinking: Option<String>,
    pub error: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_text2table(
    text: &str,
    labels: &[String],
    server_url: &str,
    model_name: Option<&str>,
    enable_thinking: bool,
    prompt: Option<&str>,
    threshold: f64,
    use_gliner: bool,
    gliner_url: Option<&str>,
    gliner_model: &str,
    gliner_soft_threshold: Option<f64>,
    enable_row_validation: bool,
    row_validation_mode: &str,
    api_key: Option<&str>,
    gliner_api_key: Option<&str>,
) -> Result<Text2TableResult, String> {
    Python::with_gil(|py| {
        // Import
        let t2t_module = py
            .import("rust_research_py.text2table.text2table")
            .map_err(|e| format!("Failed to import rust_research_py.text2table.text2table: {e}"))?;

        let t2t_class = t2t_module
            .getattr("Text2Table")
            .map_err(|e| format!("Failed to get Text2Table class: {e}"))?;

        // Constructor args
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("labels", labels)
            .map_err(|e| format!("Failed to set labels: {e}"))?;
        kwargs
            .set_item("server_url", server_url)
            .map_err(|e| format!("Failed to set server_url: {e}"))?;

        if let Some(m) = model_name {
            kwargs
                .set_item("model_name", m)
                .map_err(|e| format!("Failed to set model_name: {e}"))?;
        }

        kwargs
            .set_item("enable_thinking", enable_thinking)
            .map_err(|e| format!("Failed to set enable_thinking: {e}"))?;
        kwargs
            .set_item("threshold", threshold)
            .map_err(|e| format!("Failed to set threshold: {e}"))?;
        kwargs
            .set_item("use_gliner", use_gliner)
            .map_err(|e| format!("Failed to set use_gliner: {e}"))?;
        kwargs
            .set_item("gliner_model_name", gliner_model)
            .map_err(|e| format!("Failed to set gliner_model_name: {e}"))?;
        kwargs
            .set_item("enable_row_validation", enable_row_validation)
            .map_err(|e| format!("Failed to set enable_row_validation: {e}"))?;
        kwargs
            .set_item("row_validation_mode", row_validation_mode)
            .map_err(|e| format!("Failed to set row_validation_mode: {e}"))?;

        if let Some(url) = gliner_url {
            kwargs
                .set_item("gliner_url", url)
                .map_err(|e| format!("Failed to set gliner_url: {e}"))?;
        }

        if let Some(t) = gliner_soft_threshold {
            kwargs
                .set_item("gliner_soft_threshold", t)
                .map_err(|e| format!("Failed to set gliner_soft_threshold: {e}"))?;
        }

        if let Some(k) = api_key {
            kwargs
                .set_item("api_key", k)
                .map_err(|e| format!("Failed to set api_key: {e}"))?;
        }

        if let Some(k) = gliner_api_key {
            kwargs
                .set_item("gliner_api_key", k)
                .map_err(|e| format!("Failed to set gliner_api_key: {e}"))?;
        }

        // Initialize instance
        let extractor = t2t_class
            .call((), Some(&kwargs))
            .map_err(|e| format!("Failed to instantiate Text2Table: {e}"))?;

        // Run
        let run_kwargs = PyDict::new(py);
        if let Some(p) = prompt {
            run_kwargs
                .set_item("user_prompt", p)
                .map_err(|e| format!("Failed to set user_prompt: {e}"))?;
        }

        let result = if enable_thinking {
            let res_tuple = extractor
                .call_method("run_with_thinking", (text,), Some(&run_kwargs))
                .map_err(|e| format!("Failed to call run_with_thinking: {e}"))?;

            // Tuple: (thinking, table, entities)
            let thinking: String = res_tuple
                .get_item(0)
                .and_then(|i| i.extract())
                .unwrap_or_default();
            let table: String = res_tuple
                .get_item(1)
                .and_then(|i| i.extract())
                .unwrap_or_default();
            let entities_obj = res_tuple
                .get_item(2)
                .map_err(|e| format!("Failed to get entities: {e}"))?;

            // Convert entities to JSON string for passing back
            let json_mod = py
                .import("json")
                .map_err(|e| format!("Failed to import json: {e}"))?;
            let entities_json: String = json_mod
                .call_method1("dumps", (entities_obj,))
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "[]".to_string());

            Text2TableResult {
                success: true,
                table: Some(table),
                entities: Some(entities_json),
                thinking: Some(thinking),
                error: None,
            }
        } else {
            let res_tuple = extractor
                .call_method("run", (text,), Some(&run_kwargs))
                .map_err(|e| format!("Failed to call run: {e}"))?;

            // Tuple: (table, entities)
            let table: String = res_tuple
                .get_item(0)
                .and_then(|i| i.extract())
                .unwrap_or_default();
            let entities_obj = res_tuple
                .get_item(1)
                .map_err(|e| format!("Failed to get entities: {e}"))?;

            let json_mod = py
                .import("json")
                .map_err(|e| format!("Failed to import json: {e}"))?;
            let entities_json: String = json_mod
                .call_method1("dumps", (entities_obj,))
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| "[]".to_string());

            Text2TableResult {
                success: true,
                table: Some(table),
                entities: Some(entities_json),
                thinking: None,
                error: None,
            }
        };

        // Try close
        let _ = extractor.call_method0("close");

        Ok(result)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_text2table_batch(
    input_file: &Path,
    output_file: &Path,
    output_format: &str,
    labels: &[String],
    labels_file: Option<&Path>,
    prompt: Option<&str>,
    threshold: f64,
    gliner_model: &str,
    gliner_soft_threshold: Option<f64>,
    model_name: Option<&str>,
    enable_thinking: bool,
    server_url: &str,
    gliner_url: Option<&str>,
    disable_gliner: bool,
    enable_row_validation: bool,
    row_validation_mode: &str,
    api_key: Option<&str>,
    gliner_api_key: Option<&str>,
    concurrency: usize,
) -> Result<usize, String> {
    Python::with_gil(|py| {
        let t2t_module = py
            .import("rust_research_py.text2table.text2table")
            .map_err(|e| format!("Failed to import rust_research_py.text2table.text2table: {e}"))?;

        let kwargs = PyDict::new(py);
        kwargs
            .set_item("input_file", input_file.to_string_lossy().to_string())
            .map_err(|e| format!("Failed to set input_file: {e}"))?;
        kwargs
            .set_item("output_file", output_file.to_string_lossy().to_string())
            .map_err(|e| format!("Failed to set output_file: {e}"))?;
        kwargs
            .set_item("output_format", output_format)
            .map_err(|e| format!("Failed to set output_format: {e}"))?;
        kwargs
            .set_item("labels", labels)
            .map_err(|e| format!("Failed to set labels: {e}"))?;
        kwargs
            .set_item("threshold", threshold)
            .map_err(|e| format!("Failed to set threshold: {e}"))?;
        kwargs
            .set_item("gliner_model", gliner_model)
            .map_err(|e| format!("Failed to set gliner_model: {e}"))?;
        kwargs
            .set_item("enable_thinking", enable_thinking)
            .map_err(|e| format!("Failed to set enable_thinking: {e}"))?;
        kwargs
            .set_item("server_url", server_url)
            .map_err(|e| format!("Failed to set server_url: {e}"))?;
        kwargs
            .set_item("disable_gliner", disable_gliner)
            .map_err(|e| format!("Failed to set disable_gliner: {e}"))?;
        kwargs
            .set_item("enable_row_validation", enable_row_validation)
            .map_err(|e| format!("Failed to set enable_row_validation: {e}"))?;
        kwargs
            .set_item("row_validation_mode", row_validation_mode)
            .map_err(|e| format!("Failed to set row_validation_mode: {e}"))?;
        kwargs
            .set_item("concurrency", concurrency)
            .map_err(|e| format!("Failed to set concurrency: {e}"))?;

        if let Some(path) = labels_file {
            kwargs
                .set_item("labels_file", path.to_string_lossy().to_string())
                .map_err(|e| format!("Failed to set labels_file: {e}"))?;
        }

        if let Some(p) = prompt {
            kwargs
                .set_item("prompt", p)
                .map_err(|e| format!("Failed to set prompt: {e}"))?;
        }

        if let Some(t) = gliner_soft_threshold {
            kwargs
                .set_item("gliner_soft_threshold", t)
                .map_err(|e| format!("Failed to set gliner_soft_threshold: {e}"))?;
        }

        if let Some(m) = model_name {
            kwargs
                .set_item("model_name", m)
                .map_err(|e| format!("Failed to set model_name: {e}"))?;
        }

        if let Some(url) = gliner_url {
            kwargs
                .set_item("gliner_url", url)
                .map_err(|e| format!("Failed to set gliner_url: {e}"))?;
        }

        if let Some(k) = api_key {
            kwargs
                .set_item("api_key", k)
                .map_err(|e| format!("Failed to set api_key: {e}"))?;
        }

        if let Some(k) = gliner_api_key {
            kwargs
                .set_item("gliner_api_key", k)
                .map_err(|e| format!("Failed to set gliner_api_key: {e}"))?;
        }

        let func = t2t_module
            .getattr("run_text2table_batch")
            .map_err(|e| format!("Failed to load run_text2table_batch: {e}"))?;
        let result = func
            .call((), Some(&kwargs))
            .map_err(|e| format!("Failed to call run_text2table_batch: {e}"))?;

        result
            .extract::<usize>()
            .map_err(|e| format!("Failed to parse batch result: {e}"))
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_text2table_server(
    model: &str,
    host: &str,
    port: u16,
    tensor_parallel_size: usize,
    gpu_memory_utilization: f32,
    max_model_len: Option<usize>,
    trust_remote_code: bool,
    cache_dir: Option<&Path>,
) -> Result<(), String> {
    Python::with_gil(|py| {
        let server_module = py
            .import("rust_research_py.text2table.server")
            .map_err(|e| format!("Failed to import rust_research_py.text2table.server: {e}"))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("model_name", model).unwrap();
        kwargs.set_item("host", host).unwrap();
        kwargs.set_item("port", port).unwrap();
        kwargs
            .set_item("tensor_parallel_size", tensor_parallel_size)
            .unwrap();
        kwargs
            .set_item("gpu_memory_utilization", gpu_memory_utilization)
            .unwrap();
        kwargs
            .set_item("trust_remote_code", trust_remote_code)
            .unwrap();

        if let Some(mml) = max_model_len {
            kwargs.set_item("max_model_len", mml).unwrap();
        }

        if let Some(cd) = cache_dir {
            kwargs
                .set_item("cache_dir", cd.to_string_lossy().to_string())
                .unwrap();
        }

        server_module
            .call_method("start_vllm_server", (), Some(&kwargs))
            .map_err(|e| format!("Failed to call start_vllm_server: {e}"))?;

        Ok(())
    })
}

pub fn check_python_available() -> Result<String, String> {
    Python::with_gil(|py| {
        let sys = py
            .import("sys")
            .map_err(|e| format!("Failed to import sys: {e}"))?;
        let version = sys
            .getattr("version")
            .map_err(|e| format!("Failed to get version: {e}"))?;
        let version_str: String = version
            .extract()
            .map_err(|e| format!("Failed to extract version: {e}"))?;
        Ok(version_str)
    })
}
