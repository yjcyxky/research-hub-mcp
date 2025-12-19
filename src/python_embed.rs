//! Python embedding module for executing embedded Python code via PyO3.
//!
//! This module embeds the `plugins/` and `pdf2text/` Python sources at compile time
//! using `include_dir!` and extracts them to a temporary directory at runtime.
//! Python's standard import mechanism handles all dependency resolution automatically.

use include_dir::{include_dir, Dir};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Embedded plugins directory - all .py files are included at compile time
static PLUGINS_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/plugins");

/// Embedded pdf2text directory - all .py files are included at compile time
static PDF2TEXT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/pdf2text");

/// Embedded text2table directory - all .py files are included at compile time
static TEXT2TABLE_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/text2table");

/// Track if Python path has been initialized
static PYTHON_PATH_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Base directory for extracted Python modules
static PYTHON_LIB_PATH: Lazy<PathBuf> = Lazy::new(|| {
    let base = std::env::temp_dir().join("rust-research-mcp-python");

    // Extract both directories
    extract_embedded_dir(&PLUGINS_DIR, &base.join("plugins"));
    extract_embedded_dir(&PDF2TEXT_DIR, &base.join("pdf2text"));
    extract_embedded_dir(&TEXT2TABLE_DIR, &base.join("text2table"));

    tracing::debug!("Extracted Python modules to: {}", base.display());
    base
});

/// Result of a plugin download operation
#[derive(Debug, Clone)]
pub struct PluginDownloadResult {
    pub success: bool,
    pub file_path: Option<String>,
    pub file_size: u64,
    pub error: Option<String>,
    pub publisher: Option<String>,
    pub doi: String,
}

/// Result of a pdf2text extraction operation
#[derive(Debug, Clone)]
pub struct Pdf2TextResult {
    pub success: bool,
    pub json_path: Option<String>,
    pub markdown_path: Option<String>,
    pub error: Option<String>,
}

/// Result of a text2table operation
#[derive(Debug, Clone)]
pub struct Text2TableResult {
    pub success: bool,
    pub table: Option<String>,
    pub entities: Option<String>, // JSON string
    pub thinking: Option<String>,
    pub error: Option<String>,
}

/// Recursively extract an embedded directory to the filesystem.
fn extract_embedded_dir(dir: &Dir<'_>, target: &Path) {
    // Create target directory
    if let Err(e) = fs::create_dir_all(target) {
        tracing::warn!("Failed to create directory {}: {}", target.display(), e);
        return;
    }

    // Extract all files
    for file in dir.files() {
        if let Some(filename) = file.path().file_name() {
            let dest = target.join(filename);
            if let Err(e) = fs::write(&dest, file.contents()) {
                tracing::warn!("Failed to write {}: {}", dest.display(), e);
            }
        }
    }

    // Recursively extract subdirectories
    for subdir in dir.dirs() {
        if let Some(name) = subdir.path().file_name() {
            extract_embedded_dir(subdir, &target.join(name));
        }
    }
}

/// Initialize Python path to include extracted modules.
/// This only needs to be called once per process.
fn init_python_path(py: Python<'_>) -> PyResult<()> {
    // Check if already initialized
    if PYTHON_PATH_INITIALIZED.load(Ordering::SeqCst) {
        return Ok(());
    }

    // Force extraction by accessing the lazy static
    let lib_path = PYTHON_LIB_PATH.to_string_lossy().to_string();

    let sys = py.import("sys")?;
    let path = sys.getattr("path")?;

    // Insert at beginning so our modules take precedence
    path.call_method1("insert", (0, &lib_path))?;

    PYTHON_PATH_INITIALIZED.store(true, Ordering::SeqCst);
    tracing::debug!("Python path initialized with: {}", lib_path);

    Ok(())
}

/// Run the plugin-based download for a DOI.
///
/// This is the PyO3 equivalent of calling `plugin_runner.py --doi <doi> --output-dir <output_dir>`.
pub fn run_plugin_download(
    doi: &str,
    output_dir: &Path,
    filename: Option<&str>,
    wait_time: f64,
    headless: bool,
) -> Result<PluginDownloadResult, String> {
    Python::with_gil(|py| {
        // Initialize Python path
        init_python_path(py).map_err(|e| format!("Failed to init Python path: {e}"))?;

        // Import using standard Python import mechanism
        let plugins_utils = py
            .import("plugins.utils")
            .map_err(|e| format!("Failed to import plugins.utils: {e}"))?;

        // Import asyncio to run async function
        let asyncio = py
            .import("asyncio")
            .map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Prepare arguments
        let output_dir_str = output_dir.to_string_lossy().to_string();
        let plugin_options = PyDict::new(py);
        let opts = PyDict::new(py);
        opts.set_item("headless", headless)
            .map_err(|e| format!("Failed to set headless: {e}"))?;

        // Set options for all plugins
        for plugin_name in &[
            "wiley",
            "oxford",
            "springer",
            "biorxiv",
            "mdpi",
            "frontiers",
            "pnas",
            "plos",
            "hindawi",
            "nature",
        ] {
            plugin_options
                .set_item(*plugin_name, opts.clone())
                .map_err(|e| format!("Failed to set {plugin_name} opts: {e}"))?;
        }

        // Build kwargs
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("doi", doi)
            .map_err(|e| format!("Failed to set doi: {e}"))?;
        kwargs
            .set_item("output_dir", &output_dir_str)
            .map_err(|e| format!("Failed to set output_dir: {e}"))?;
        if let Some(fname) = filename {
            kwargs
                .set_item("filename", fname)
                .map_err(|e| format!("Failed to set filename: {e}"))?;
        }
        kwargs
            .set_item("wait_time", wait_time)
            .map_err(|e| format!("Failed to set wait_time: {e}"))?;
        kwargs
            .set_item("plugin_options", plugin_options)
            .map_err(|e| format!("Failed to set plugin_options: {e}"))?;

        // Call the async function via asyncio.run()
        let download_fn = plugins_utils
            .getattr("download_with_detected_plugin")
            .map_err(|e| format!("Failed to get download_with_detected_plugin: {e}"))?;

        let coro = download_fn
            .call((), Some(&kwargs))
            .map_err(|e| format!("Failed to create coroutine: {e}"))?;

        let result = asyncio
            .call_method1("run", (coro,))
            .map_err(|e| format!("Failed to run coroutine: {e}"))?;

        // Extract result fields
        extract_plugin_result(&result, doi)
    })
}

/// Run pdf2text extraction on a PDF file.
///
/// This is the PyO3 equivalent of calling `pdf2text pdf --pdf-file <path> --output-dir <dir> ...`.
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
        // Initialize Python path
        init_python_path(py).map_err(|e| format!("Failed to init Python path: {e}"))?;

        // Import using standard Python import mechanism
        let pdf2text_module = py
            .import("pdf2text.pdf2text")
            .map_err(|e| format!("Failed to import pdf2text.pdf2text: {e}"))?;

        // Prepare arguments
        let pdf_file_str = pdf_file.to_string_lossy().to_string();
        let output_dir_str = output_dir.to_string_lossy().to_string();

        // Build kwargs for extract_fulltext
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("pdf_file", &pdf_file_str)
            .map_err(|e| format!("Failed to set pdf_file: {e}"))?;
        kwargs
            .set_item("output_dir", &output_dir_str)
            .map_err(|e| format!("Failed to set output_dir: {e}"))?;

        if let Some(url) = grobid_url {
            kwargs
                .set_item("grobid_url", url)
                .map_err(|e| format!("Failed to set grobid_url: {e}"))?;
        }
        kwargs
            .set_item("auto_start_grobid", !no_auto_start)
            .map_err(|e| format!("Failed to set auto_start_grobid: {e}"))?;
        kwargs
            .set_item("overwrite", overwrite)
            .map_err(|e| format!("Failed to set overwrite: {e}"))?;
        kwargs
            .set_item("generate_markdown", !no_markdown)
            .map_err(|e| format!("Failed to set generate_markdown: {e}"))?;
        kwargs
            .set_item("copy_pdf", copy_pdf)
            .map_err(|e| format!("Failed to set copy_pdf: {e}"))?;
        kwargs
            .set_item("extract_figures", !no_figures)
            .map_err(|e| format!("Failed to set extract_figures: {e}"))?;
        kwargs
            .set_item("extract_tables", !no_tables)
            .map_err(|e| format!("Failed to set extract_tables: {e}"))?;

        // Call extract_fulltext
        let extract_fn = pdf2text_module
            .getattr("extract_fulltext")
            .map_err(|e| format!("Failed to get extract_fulltext: {e}"))?;

        let result = extract_fn.call((), Some(&kwargs));

        match result {
            Ok(json_path_obj) => {
                if json_path_obj.is_none() {
                    Ok(Pdf2TextResult {
                        success: false,
                        json_path: None,
                        markdown_path: None,
                        error: Some("extract_fulltext returned None".to_string()),
                    })
                } else {
                    let json_path: String = json_path_obj
                        .str()
                        .map_err(|e| format!("Failed to convert path to str: {e}"))?
                        .to_string();

                    // Derive markdown path from json path
                    let markdown_path = if !no_markdown {
                        Some(json_path.replace(".json", ".md"))
                    } else {
                        None
                    };

                    Ok(Pdf2TextResult {
                        success: true,
                        json_path: Some(json_path),
                        markdown_path,
                        error: None,
                    })
                }
            }
            Err(e) => Ok(Pdf2TextResult {
                success: false,
                json_path: None,
                markdown_path: None,
                error: Some(format!("extract_fulltext failed: {e}")),
            }),
        }
    })
}

/// Run CDP-based download using Playwright.
///
/// This function calls the Python `execute_download_by_cdp` function from plugins.utils.
pub fn run_cdp_download(
    url: &str,
    output_file: &Path,
    headless: bool,
    timeout: u64,
    wait_time: f64,
) -> Result<PluginDownloadResult, String> {
    Python::with_gil(|py| {
        // Initialize Python path
        init_python_path(py).map_err(|e| format!("Failed to init Python path: {e}"))?;

        // Import using standard Python import mechanism
        let plugins_utils = py
            .import("plugins.utils")
            .map_err(|e| format!("Failed to import plugins.utils: {e}"))?;

        // Import asyncio to run async function
        let asyncio = py
            .import("asyncio")
            .map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Prepare arguments
        let output_file_str = output_file.to_string_lossy().to_string();

        // Build kwargs
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("url", url)
            .map_err(|e| format!("Failed to set url: {e}"))?;
        kwargs
            .set_item("output_file", &output_file_str)
            .map_err(|e| format!("Failed to set output_file: {e}"))?;
        kwargs
            .set_item("headless", headless)
            .map_err(|e| format!("Failed to set headless: {e}"))?;
        kwargs
            .set_item("timeout", timeout)
            .map_err(|e| format!("Failed to set timeout: {e}"))?;
        kwargs
            .set_item("wait_time", wait_time)
            .map_err(|e| format!("Failed to set wait_time: {e}"))?;

        // Call the async function via asyncio.run()
        let download_fn = plugins_utils
            .getattr("execute_download_by_cdp")
            .map_err(|e| format!("Failed to get execute_download_by_cdp: {e}"))?;

        let coro = download_fn
            .call((), Some(&kwargs))
            .map_err(|e| format!("Failed to create coroutine: {e}"))?;

        let result = asyncio
            .call_method1("run", (coro,))
            .map_err(|e| format!("Failed to run coroutine: {e}"))?;

        // Extract result fields
        let mut plugin_result = extract_plugin_result(&result, "")?;
        plugin_result.publisher = None;
        plugin_result.doi = String::new();
        Ok(plugin_result)
    })
}

/// Extract plugin download result from Python object.
fn extract_plugin_result(
    result: &Bound<'_, PyAny>,
    doi: &str,
) -> Result<PluginDownloadResult, String> {
    let success: bool = result
        .getattr("success")
        .and_then(|v| v.extract())
        .unwrap_or(false);

    let file_path: Option<String> = result.getattr("file_path").ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            v.str().ok().map(|s| s.to_string())
        }
    });

    let file_size: u64 = result
        .getattr("file_size")
        .and_then(|v| v.extract())
        .unwrap_or(0);

    let error: Option<String> =
        result.getattr("error").ok().and_then(
            |v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            },
        );

    let publisher: Option<String> = result.getattr("publisher").ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            v.extract().ok()
        }
    });

    let doi_out: String = result
        .getattr("doi")
        .and_then(|v| v.extract())
        .unwrap_or_else(|_| doi.to_string());

    Ok(PluginDownloadResult {
        success,
        file_path,
        file_size,
        error,
        publisher,
        doi: doi_out,
    })
}

/// Check if Python is available and properly configured.
pub fn check_python_available() -> Result<String, String> {
    Python::with_gil(|py| {
        let sys = py
            .import("sys")
            .map_err(|e| format!("Failed to import sys: {e}"))?;
        let version: String = sys
            .getattr("version")
            .and_then(|v| v.extract())
            .map_err(|e| format!("Failed to get Python version: {e}"))?;
        Ok(version)
    })
}

/// Get the path where Python modules are extracted.
/// Useful for debugging.
pub fn get_python_lib_path() -> PathBuf {
    PYTHON_LIB_PATH.clone()
}

/// Run text2table server (vLLM)
///
/// This runs the text2table.server module using subprocess to ensure clean environment processing.
/// It wraps the python command execution.
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
    // Ensure python lib is extracted
    let lib_path = get_python_lib_path();
    if !lib_path.exists() {
        // Force extraction if not already done
        Python::with_gil(|py| {
            let _ = init_python_path(py);
        });
    }

    // Determine python executable
    // We try to use the same python that is running this process if possible,
    // or fallback to "python" or "python3"
    let python_exe = std::env::var("PYTHON_EXECUTABLE").unwrap_or_else(|_| "python".to_string());

    // Build command
    let mut cmd = std::process::Command::new(&python_exe);

    // Set PYTHONPATH to include our extracted lib
    let current_path = std::env::var("PYTHONPATH").unwrap_or_default();
    let new_path = if current_path.is_empty() {
        lib_path.to_string_lossy().to_string()
    } else {
        format!("{}:{}", lib_path.to_string_lossy(), current_path)
    };
    cmd.env("PYTHONPATH", new_path);

    // Module to run
    cmd.arg("-m").arg("text2table.server");

    // Arguments
    cmd.arg("--model").arg(model);
    cmd.arg("--host").arg(host);
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--tensor-parallel-size")
        .arg(tensor_parallel_size.to_string());
    cmd.arg("--gpu-memory-utilization")
        .arg(gpu_memory_utilization.to_string());

    if let Some(len) = max_model_len {
        cmd.arg("--max-model-len").arg(len.to_string());
    }

    if trust_remote_code {
        cmd.arg("--trust-remote-code");
    }

    if let Some(dir) = cache_dir {
        cmd.arg("--cache-dir").arg(dir);
    }

    // Connect stdout/stderr
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());

    tracing::info!("Starting text2table server: {:?}", cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn server process: {e}"))?;

    // Wait for child
    let status = child
        .wait()
        .map_err(|e| format!("Failed to wait for server: {e}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("Server exited with status: {}", status))
    }
}

/// Run text2table extraction pipeline.
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
        // Initialize Python path
        init_python_path(py).map_err(|e| format!("Failed to init Python path: {e}"))?;

        // Import
        let t2t_module = py
            .import("text2table.text2table")
            .map_err(|e| format!("Failed to import text2table.text2table: {e}"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_available() {
        let result = check_python_available();
        assert!(result.is_ok(), "Python should be available: {:?}", result);
        let version = result.unwrap();
        assert!(
            version.starts_with("3."),
            "Python 3.x expected, got: {}",
            version
        );
    }

    #[test]
    fn test_python_lib_path_exists() {
        let path = get_python_lib_path();
        // Force extraction
        Python::with_gil(|py| {
            init_python_path(py).expect("Failed to init Python path");
        });
        assert!(path.exists(), "Python lib path should exist: {:?}", path);
        assert!(
            path.join("plugins").exists(),
            "plugins directory should exist"
        );
        assert!(
            path.join("pdf2text").exists(),
            "pdf2text directory should exist"
        );
    }

    #[test]
    fn test_can_import_plugins() {
        Python::with_gil(|py| {
            init_python_path(py).expect("Failed to init Python path");

            // Test that we can import the plugins module
            let result = py.import("plugins");
            assert!(
                result.is_ok(),
                "Should be able to import plugins: {:?}",
                result.err()
            );

            // Test that we can import plugins.utils
            let result = py.import("plugins.utils");
            assert!(
                result.is_ok(),
                "Should be able to import plugins.utils: {:?}",
                result.err()
            );
        });
    }
}
