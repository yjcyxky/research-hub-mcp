//! Python embedding module for executing embedded Python code via PyO3.
//!
//! This module embeds the `plugins/` and `pdf2text/` Python sources at compile time
//! using `include_dir!` and provides functions to execute them using `Python::with_gil()`
//! and `PyModule::from_code()`.
//!
//! Python files are automatically discovered and loaded in dependency order.

use include_dir::{include_dir, Dir};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::path::Path;
use tracing::warn;

/// Embedded plugins directory - all .py files are included at compile time
static PLUGINS_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/plugins");

/// Embedded pdf2text directory - all .py files are included at compile time
static PDF2TEXT_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/pdf2text");

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

/// Extract module dependencies from Python source code.
/// Looks for `from package.module import ...` and `import package.module` patterns.
fn extract_dependencies(source: &str, package_name: &str) -> Vec<String> {
    let mut deps = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();

        // Handle: from .module import ... or from package.module import ...
        if trimmed.starts_with("from ") {
            if let Some(rest) = trimmed.strip_prefix("from ") {
                // Handle relative imports: from .module import ...
                if rest.starts_with('.') {
                    if let Some(module_part) = rest.strip_prefix('.') {
                        if let Some(module) = module_part.split_whitespace().next() {
                            if !module.is_empty() && module != "import" {
                                deps.push(format!("{}.{}", package_name, module));
                            }
                        }
                    }
                }
                // Handle absolute imports: from package.module import ...
                else if rest.starts_with(package_name) {
                    if let Some(import_part) = rest.split_whitespace().next() {
                        if import_part.contains('.') {
                            // Extract just the module path before 'import'
                            let module_path =
                                import_part.split(" import").next().unwrap_or(import_part);
                            deps.push(module_path.to_string());
                        }
                    }
                }
            }
        }
        // Handle: import package.module
        else if trimmed.starts_with("import ") {
            if let Some(rest) = trimmed.strip_prefix("import ") {
                for module in rest.split(',') {
                    let module = module.trim().split_whitespace().next().unwrap_or("");
                    if module.starts_with(package_name) && module.contains('.') {
                        deps.push(module.to_string());
                    }
                }
            }
        }
    }

    deps
}

/// Topological sort for module loading order.
/// Returns modules in order where dependencies come before dependents.
fn topological_sort(modules: &HashMap<String, String>, package_name: &str) -> Vec<String> {
    let mut deps_map: HashMap<String, Vec<String>> = HashMap::new();

    // Build dependency graph
    for (name, source) in modules {
        let deps = extract_dependencies(source, package_name);
        deps_map.insert(name.clone(), deps);
    }

    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut temp_visited = HashSet::new();

    fn visit(
        node: &str,
        deps_map: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        temp_visited: &mut HashSet<String>,
        result: &mut Vec<String>,
        modules: &HashMap<String, String>,
    ) {
        if visited.contains(node) {
            return;
        }
        if temp_visited.contains(node) {
            // Circular dependency - just skip
            return;
        }

        temp_visited.insert(node.to_string());

        if let Some(deps) = deps_map.get(node) {
            for dep in deps {
                // Only visit if it's a module we're loading
                if modules.contains_key(dep) {
                    visit(dep, deps_map, visited, temp_visited, result, modules);
                }
            }
        }

        temp_visited.remove(node);
        visited.insert(node.to_string());
        result.push(node.to_string());
    }

    for name in modules.keys() {
        visit(
            name,
            &deps_map,
            &mut visited,
            &mut temp_visited,
            &mut result,
            modules,
        );
    }

    result
}

/// Load all Python files from an embedded directory into a package.
fn load_package_modules(
    py: Python<'_>,
    dir: &Dir<'_>,
    package_name: &str,
    modules: &pyo3::Bound<'_, PyDict>,
) -> PyResult<()> {
    // Collect all .py files with their contents
    let mut py_modules: HashMap<String, String> = HashMap::new();

    for file in dir.files() {
        let path = file.path();
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if !filename.ends_with(".py") {
            continue;
        }

        let content = file.contents_utf8().unwrap_or("");
        let module_name = filename.strip_suffix(".py").unwrap_or(filename);

        let full_module_name = if module_name == "__init__" {
            package_name.to_string()
        } else {
            format!("{}.{}", package_name, module_name)
        };

        py_modules.insert(full_module_name, content.to_string());
    }

    // Create the package first with proper __path__ attribute
    // This is critical for relative imports to work
    let package_init = py_modules.get(package_name).cloned().unwrap_or_default();
    let init_code = format!(
        "__path__ = []\n__package__ = '{}'\n{}",
        package_name, package_init
    );

    // We need to register non-__init__ modules first (in dependency order),
    // then execute __init__.py which may import them

    // Get modules sorted by dependencies (excluding __init__)
    let mut non_init_modules: HashMap<String, String> = py_modules
        .iter()
        .filter(|(k, _)| *k != package_name)
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let sorted_modules = topological_sort(&non_init_modules, package_name);

    // First, create a minimal package entry so submodule imports can find the package
    let minimal_init = PyModule::from_code(
        py,
        &CString::new(format!("__path__ = []\n__package__ = '{}'", package_name)).unwrap(),
        &CString::new(format!("{}/__init__.py", package_name)).unwrap(),
        &CString::new(package_name).unwrap(),
    )?;
    modules.set_item(package_name, &minimal_init)?;

    // Load modules in dependency order
    for module_name in &sorted_modules {
        if let Some(source) = non_init_modules.remove(module_name) {
            let file_path = format!(
                "{}/{}.py",
                package_name,
                module_name
                    .strip_prefix(&format!("{}.", package_name))
                    .unwrap_or(module_name)
            );

            let module = PyModule::from_code(
                py,
                &CString::new(source).unwrap(),
                &CString::new(file_path).unwrap(),
                &CString::new(module_name.as_str()).unwrap(),
            )?;
            modules.set_item(module_name.as_str(), &module)?;

            // Also set as attribute on package for attribute access
            let attr_name = module_name
                .strip_prefix(&format!("{}.", package_name))
                .unwrap_or(module_name);
            minimal_init.setattr(attr_name, &module)?;
        }
    }

    // Now load the full __init__.py with all imports
    // This will find all the submodules already registered
    if py_modules.contains_key(package_name) {
        let full_init = PyModule::from_code(
            py,
            &CString::new(init_code).unwrap(),
            &CString::new(format!("{}/__init__.py", package_name)).unwrap(),
            &CString::new(package_name).unwrap(),
        )?;
        modules.set_item(package_name, &full_init)?;
    }

    Ok(())
}

/// Initialize the embedded plugins Python modules.
///
/// This function registers all Python sources from the plugins/ directory
/// as importable modules so they can be used by subsequent code.
fn init_plugin_modules(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules_attr = sys.getattr("modules")?;
    let modules = modules_attr.downcast::<PyDict>()?;

    load_package_modules(py, &PLUGINS_DIR, "plugins", modules)?;

    Ok(())
}

/// Initialize the pdf2text Python modules.
fn init_pdf2text_modules(py: Python<'_>) -> PyResult<()> {
    let sys = py.import("sys")?;
    let modules_attr = sys.getattr("modules")?;
    let modules = modules_attr.downcast::<PyDict>()?;

    load_package_modules(py, &PDF2TEXT_DIR, "pdf2text", modules)?;

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
        let sys = py
            .import("sys")
            .map_err(|e| format!("Failed to import sys: {e}"))?;
        let modules = sys
            .getattr("modules")
            .map_err(|e| format!("Failed to get modules: {e}"))?;
        let modules = modules
            .downcast::<PyDict>()
            .map_err(|e| format!("modules is not a dict: {e}"))?;

        let utils = modules
            .get_item("plugins.utils")
            .map_err(|e| format!("Failed to get plugins.utils: {e}"))?
            .ok_or_else(|| "plugins.utils not found in sys.modules".to_string())?;

        // Import asyncio to run async function
        let asyncio = py
            .import("asyncio")
            .map_err(|e| format!("Failed to import asyncio: {e}"))?;

        // Prepare arguments
        let output_dir_str = output_dir.to_string_lossy().to_string();
        let plugin_options = PyDict::new(py);
        let wiley_opts = PyDict::new(py);
        wiley_opts
            .set_item("headless", headless)
            .map_err(|e| format!("Failed to set headless: {e}"))?;
        plugin_options
            .set_item("wiley", wiley_opts)
            .map_err(|e| format!("Failed to set wiley opts: {e}"))?;

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
        let download_fn = utils
            .getattr("download_with_detected_plugin")
            .map_err(|e| format!("Failed to get download_with_detected_plugin: {e}"))?;

        let coro = download_fn
            .call((), Some(&kwargs))
            .map_err(|e| format!("Failed to create coroutine: {e}"))?;

        let result = asyncio
            .call_method1("run", (coro,))
            .map_err(|e| format!("Failed to run coroutine: {e}"))?;

        // Extract result fields
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

        let error: Option<String> = result.getattr("error").ok().and_then(|v| {
            if v.is_none() {
                None
            } else {
                v.extract().ok()
            }
        });

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
        let sys = py
            .import("sys")
            .map_err(|e| format!("Failed to import sys: {e}"))?;
        let modules = sys
            .getattr("modules")
            .map_err(|e| format!("Failed to get modules: {e}"))?;
        let modules = modules
            .downcast::<PyDict>()
            .map_err(|e| format!("modules is not a dict: {e}"))?;

        let pdf2text = modules
            .get_item("pdf2text.pdf2text")
            .map_err(|e| format!("Failed to get pdf2text.pdf2text: {e}"))?
            .ok_or_else(|| "pdf2text.pdf2text not found in sys.modules".to_string())?;

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
        let extract_fn = pdf2text
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
    fn test_extract_dependencies() {
        let source = r#"
from plugins.common import BasePlugin
from .utils import something
import plugins.wiley_pdf_downloader
"#;
        let deps = extract_dependencies(source, "plugins");
        assert!(deps.contains(&"plugins.common".to_string()));
        assert!(deps.contains(&"plugins.utils".to_string()));
        assert!(deps.contains(&"plugins.wiley_pdf_downloader".to_string()));
    }
}
