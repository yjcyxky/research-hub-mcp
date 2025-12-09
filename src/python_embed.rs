//! Python embedding module for executing embedded Python code via PyO3.
//!
//! This module embeds the `plugins/` and `pdf2text/` Python sources at compile time
//! and provides functions to execute them using `Python::with_gil()` and `PyModule::from_code()`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::ffi::CString;
use std::path::Path;
use tracing::{debug, error, info, warn};

/// Embedded Python source for plugins/common.py
const PLUGINS_COMMON_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/plugins/common.py"));

/// Embedded Python source for plugins/utils.py
const PLUGINS_UTILS_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/plugins/utils.py"));

/// Embedded Python source for plugins/wiley_pdf_downloader.py
const PLUGINS_WILEY_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/plugins/wiley_pdf_downloader.py"));

/// Embedded Python source for plugins/plugin_runner.py
const PLUGINS_RUNNER_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/plugins/plugin_runner.py"));

/// Embedded Python source for pdf2text/grobid.py
const PDF2TEXT_GROBID_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/pdf2text/grobid.py"));

/// Embedded Python source for pdf2text/models.py
const PDF2TEXT_MODELS_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/pdf2text/models.py"));

/// Embedded Python source for pdf2text/pdf2text.py
const PDF2TEXT_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/pdf2text/pdf2text.py"));

/// Embedded Python source for pdf2text/cli.py
const PDF2TEXT_CLI_PY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/pdf2text/cli.py"));

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

/// Initialize the embedded Python modules in the correct dependency order.
///
/// This function registers the embedded Python sources as importable modules
/// so they can be used by subsequent code.
fn init_plugin_modules(py: Python<'_>) -> PyResult<()> {
    // We need to add the embedded modules to sys.modules so imports work
    let sys = py.import("sys")?;
    let modules_attr = sys.getattr("modules")?;
    let modules = modules_attr.downcast::<PyDict>()?;

    // Create plugins package
    let plugins_init = PyModule::from_code(
        py,
        c"",
        c"plugins/__init__.py",
        c"plugins",
    )?;
    modules.set_item("plugins", plugins_init)?;

    // Register plugins.common
    let common = PyModule::from_code(
        py,
        &CString::new(PLUGINS_COMMON_PY).unwrap(),
        c"plugins/common.py",
        c"plugins.common",
    )?;
    modules.set_item("plugins.common", common)?;

    // Register plugins.wiley_pdf_downloader
    let wiley = PyModule::from_code(
        py,
        &CString::new(PLUGINS_WILEY_PY).unwrap(),
        c"plugins/wiley_pdf_downloader.py",
        c"plugins.wiley_pdf_downloader",
    )?;
    modules.set_item("plugins.wiley_pdf_downloader", wiley)?;

    // Register plugins.utils (depends on common and wiley)
    let utils = PyModule::from_code(
        py,
        &CString::new(PLUGINS_UTILS_PY).unwrap(),
        c"plugins/utils.py",
        c"plugins.utils",
    )?;
    modules.set_item("plugins.utils", utils)?;

    Ok(())
}

/// Initialize the pdf2text Python modules.
fn init_pdf2text_modules(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules_attr = sys.getattr("modules")?;
    let modules = modules_attr.downcast::<PyDict>()?;

    // Create pdf2text package
    let pdf2text_init = PyModule::from_code(
        py,
        c"",
        c"pdf2text/__init__.py",
        c"pdf2text",
    )?;
    modules.set_item("pdf2text", pdf2text_init)?;

    // Register pdf2text.grobid
    let grobid = PyModule::from_code(
        py,
        &CString::new(PDF2TEXT_GROBID_PY).unwrap(),
        c"pdf2text/grobid.py",
        c"pdf2text.grobid",
    )?;
    modules.set_item("pdf2text.grobid", grobid)?;

    // Register pdf2text.models
    let models = PyModule::from_code(
        py,
        &CString::new(PDF2TEXT_MODELS_PY).unwrap(),
        c"pdf2text/models.py",
        c"pdf2text.models",
    )?;
    modules.set_item("pdf2text.models", models)?;

    // Register pdf2text.pdf2text
    let pdf2text = PyModule::from_code(
        py,
        &CString::new(PDF2TEXT_PY).unwrap(),
        c"pdf2text/pdf2text.py",
        c"pdf2text.pdf2text",
    )?;
    modules.set_item("pdf2text.pdf2text", pdf2text)?;

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
        // Initialize modules
        init_plugin_modules(py).map_err(|e| format!("Failed to init plugin modules: {e}"))?;

        // Get the utils module
        let sys = py.import("sys").map_err(|e| format!("Failed to import sys: {e}"))?;
        let modules = sys.getattr("modules").map_err(|e| format!("Failed to get modules: {e}"))?;
        let modules = modules.downcast::<PyDict>().map_err(|e| format!("modules is not a dict: {e}"))?;
        
        let utils = modules.get_item("plugins.utils")
            .map_err(|e| format!("Failed to get plugins.utils: {e}"))?
            .ok_or_else(|| "plugins.utils not found in sys.modules".to_string())?;

        // Import asyncio to run async function
        let asyncio = py.import("asyncio").map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Prepare arguments
        let output_dir_str = output_dir.to_string_lossy().to_string();
        let plugin_options = PyDict::new(py);
        let wiley_opts = PyDict::new(py);
        wiley_opts.set_item("headless", headless).map_err(|e| format!("Failed to set headless: {e}"))?;
        plugin_options.set_item("wiley", wiley_opts).map_err(|e| format!("Failed to set wiley opts: {e}"))?;

        // Build kwargs
        let kwargs = PyDict::new(py);
        kwargs.set_item("doi", doi).map_err(|e| format!("Failed to set doi: {e}"))?;
        kwargs.set_item("output_dir", &output_dir_str).map_err(|e| format!("Failed to set output_dir: {e}"))?;
        if let Some(fname) = filename {
            kwargs.set_item("filename", fname).map_err(|e| format!("Failed to set filename: {e}"))?;
        }
        kwargs.set_item("wait_time", wait_time).map_err(|e| format!("Failed to set wait_time: {e}"))?;
        kwargs.set_item("plugin_options", plugin_options).map_err(|e| format!("Failed to set plugin_options: {e}"))?;

        // Call the async function via asyncio.run()
        let download_fn = utils.getattr("download_with_detected_plugin")
            .map_err(|e| format!("Failed to get download_with_detected_plugin: {e}"))?;
        
        let coro = download_fn.call((), Some(&kwargs))
            .map_err(|e| format!("Failed to create coroutine: {e}"))?;

        let result = asyncio.call_method1("run", (coro,))
            .map_err(|e| format!("Failed to run coroutine: {e}"))?;

        // Extract result fields
        let success: bool = result.getattr("success")
            .and_then(|v| v.extract())
            .unwrap_or(false);
        
        let file_path: Option<String> = result.getattr("file_path")
            .ok()
            .and_then(|v| {
                if v.is_none() {
                    None
                } else {
                    v.str().ok().map(|s| s.to_string())
                }
            });

        let file_size: u64 = result.getattr("file_size")
            .and_then(|v| v.extract())
            .unwrap_or(0);

        let error: Option<String> = result.getattr("error")
            .ok()
            .and_then(|v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            });

        let publisher: Option<String> = result.getattr("publisher")
            .ok()
            .and_then(|v| {
                if v.is_none() {
                    None
                } else {
                    v.extract().ok()
                }
            });

        let doi_out: String = result.getattr("doi")
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
        // Initialize modules
        init_pdf2text_modules(py).map_err(|e| format!("Failed to init pdf2text modules: {e}"))?;

        // Get the pdf2text module
        let sys = py.import("sys").map_err(|e| format!("Failed to import sys: {e}"))?;
        let modules = sys.getattr("modules").map_err(|e| format!("Failed to get modules: {e}"))?;
        let modules = modules.downcast::<PyDict>().map_err(|e| format!("modules is not a dict: {e}"))?;

        let pdf2text = modules.get_item("pdf2text.pdf2text")
            .map_err(|e| format!("Failed to get pdf2text.pdf2text: {e}"))?
            .ok_or_else(|| "pdf2text.pdf2text not found in sys.modules".to_string())?;

        // Prepare arguments
        let pdf_file_str = pdf_file.to_string_lossy().to_string();
        let output_dir_str = output_dir.to_string_lossy().to_string();

        // Build kwargs for extract_fulltext
        let kwargs = PyDict::new(py);
        kwargs.set_item("pdf_file", &pdf_file_str).map_err(|e| format!("Failed to set pdf_file: {e}"))?;
        kwargs.set_item("output_dir", &output_dir_str).map_err(|e| format!("Failed to set output_dir: {e}"))?;
        
        if let Some(url) = grobid_url {
            kwargs.set_item("grobid_url", url).map_err(|e| format!("Failed to set grobid_url: {e}"))?;
        }
        kwargs.set_item("auto_start_grobid", !no_auto_start).map_err(|e| format!("Failed to set auto_start_grobid: {e}"))?;
        kwargs.set_item("overwrite", overwrite).map_err(|e| format!("Failed to set overwrite: {e}"))?;
        kwargs.set_item("generate_markdown", !no_markdown).map_err(|e| format!("Failed to set generate_markdown: {e}"))?;
        kwargs.set_item("copy_pdf", copy_pdf).map_err(|e| format!("Failed to set copy_pdf: {e}"))?;
        kwargs.set_item("extract_figures", !no_figures).map_err(|e| format!("Failed to set extract_figures: {e}"))?;
        kwargs.set_item("extract_tables", !no_tables).map_err(|e| format!("Failed to set extract_tables: {e}"))?;

        // Call extract_fulltext
        let extract_fn = pdf2text.getattr("extract_fulltext")
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
                    let json_path: String = json_path_obj.str()
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

/// Check if Python is available and properly configured.
pub fn check_python_available() -> Result<String, String> {
    Python::with_gil(|py| {
        let sys = py.import("sys").map_err(|e| format!("Failed to import sys: {e}"))?;
        let version: String = sys.getattr("version")
            .and_then(|v| v.extract())
            .map_err(|e| format!("Failed to get Python version: {e}"))?;
        Ok(version)
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
        assert!(version.starts_with("3."), "Python 3.x expected, got: {}", version);
    }
}
