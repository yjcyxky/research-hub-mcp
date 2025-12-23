//! PDF to text conversion tool using embedded Python via PyO3.
//!
//! This module provides a standalone tool for converting PDF files to
//! structured JSON and Markdown using the embedded pdf2text Python code.

use crate::python_embed::run_pdf2text;
use crate::{Config, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::spawn_blocking;
use tracing::{debug, info, instrument, warn};

/// Input parameters for the pdf2text tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Pdf2TextInput {
    /// Path to a single PDF file to process
    #[schemars(description = "Path to a single PDF file")]
    pub pdf_file: Option<String>,

    /// Path to a directory containing PDF files
    #[schemars(description = "Directory containing PDF files")]
    pub pdf_dir: Option<String>,

    /// Output directory for extracted content
    #[schemars(description = "Output directory for extracted content (required)")]
    pub output_dir: String,

    /// GROBID server URL (optional, uses public endpoint if not provided)
    #[schemars(
        description = "GROBID server URL. Uses public endpoint by default: https://kermitt2-grobid.hf.space"
    )]
    pub grobid_url: Option<String>,

    /// Don't auto-start a local GROBID server
    #[serde(default)]
    pub no_auto_start: bool,

    /// Don't extract figures
    #[serde(default)]
    pub no_figures: bool,

    /// Don't extract tables
    #[serde(default)]
    pub no_tables: bool,

    /// Copy the source PDF into the output bundle
    #[serde(default)]
    pub copy_pdf: bool,

    /// Overwrite existing outputs
    #[serde(default)]
    pub overwrite: bool,

    /// Don't generate Markdown alongside JSON
    #[serde(default)]
    pub no_markdown: bool,
}

/// Result of a pdf2text conversion
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Pdf2TextOutput {
    /// Whether the conversion succeeded
    pub success: bool,

    /// Path to the generated JSON file
    pub json_path: Option<String>,

    /// Path to the generated Markdown file
    pub markdown_path: Option<String>,

    /// Number of files processed (for directory input)
    pub files_processed: usize,

    /// Error message if failed
    pub error: Option<String>,
}

/// PDF to text conversion tool
#[derive(Clone)]
pub struct Pdf2TextTool {
    #[allow(dead_code)]
    config: Arc<Config>,
}

impl std::fmt::Debug for Pdf2TextTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pdf2TextTool")
            .field("config", &"Config")
            .finish()
    }
}

impl Pdf2TextTool {
    /// Create a new pdf2text tool
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing pdf2text tool");
        Ok(Self { config })
    }

    /// Convert a PDF file or directory to structured JSON and Markdown
    #[instrument(skip(self), fields(pdf_file = ?input.pdf_file, pdf_dir = ?input.pdf_dir, output_dir = %input.output_dir))]
    pub async fn convert(&self, input: Pdf2TextInput) -> Result<Pdf2TextOutput> {
        // Validate input
        if input.pdf_file.is_none() && input.pdf_dir.is_none() {
            return Err(crate::Error::InvalidInput {
                field: "pdf_file/pdf_dir".to_string(),
                reason: "Either pdf_file or pdf_dir must be provided".to_string(),
            });
        }

        if input.pdf_file.is_some() && input.pdf_dir.is_some() {
            return Err(crate::Error::InvalidInput {
                field: "pdf_file/pdf_dir".to_string(),
                reason: "Cannot specify both pdf_file and pdf_dir".to_string(),
            });
        }

        // Check Python availability
        if let Err(e) = crate::python_embed::check_python_available() {
            return Err(crate::Error::Service(format!(
                "Python runtime not available: {e}"
            )));
        }

        let output_dir = PathBuf::from(&input.output_dir);
        tokio::fs::create_dir_all(&output_dir).await?;

        if let Some(pdf_file) = &input.pdf_file {
            // Single file processing
            self.convert_single_file(pdf_file, &input).await
        } else if let Some(pdf_dir) = &input.pdf_dir {
            // Directory processing
            self.convert_directory(pdf_dir, &input).await
        } else {
            unreachable!()
        }
    }

    async fn convert_single_file(
        &self,
        pdf_file: &str,
        input: &Pdf2TextInput,
    ) -> Result<Pdf2TextOutput> {
        let pdf_path = PathBuf::from(pdf_file);
        if !pdf_path.exists() {
            return Err(crate::Error::InvalidInput {
                field: "pdf_file".to_string(),
                reason: format!("PDF file not found: {pdf_file}"),
            });
        }

        let output_dir = PathBuf::from(&input.output_dir);
        let grobid_url = input.grobid_url.clone();
        let no_auto_start = input.no_auto_start;
        let no_figures = input.no_figures;
        let no_tables = input.no_tables;
        let copy_pdf = input.copy_pdf;
        let overwrite = input.overwrite;
        let no_markdown = input.no_markdown;

        debug!("Converting single PDF: {:?}", pdf_path);

        let result = spawn_blocking(move || {
            run_pdf2text(
                &pdf_path,
                &output_dir,
                grobid_url.as_deref(),
                no_auto_start,
                no_figures,
                no_tables,
                copy_pdf,
                overwrite,
                no_markdown,
            )
        })
        .await
        .map_err(|e| crate::Error::Service(format!("pdf2text task panicked: {e}")))?;

        match result {
            Ok(pyo3_result) => Ok(Pdf2TextOutput {
                success: pyo3_result.success,
                json_path: pyo3_result.json_path,
                markdown_path: pyo3_result.markdown_path,
                files_processed: usize::from(pyo3_result.success),
                error: pyo3_result.error,
            }),
            Err(e) => Ok(Pdf2TextOutput {
                success: false,
                json_path: None,
                markdown_path: None,
                files_processed: 0,
                error: Some(e),
            }),
        }
    }

    async fn convert_directory(
        &self,
        pdf_dir: &str,
        input: &Pdf2TextInput,
    ) -> Result<Pdf2TextOutput> {
        let dir_path = PathBuf::from(pdf_dir);
        if !dir_path.exists() || !dir_path.is_dir() {
            return Err(crate::Error::InvalidInput {
                field: "pdf_dir".to_string(),
                reason: format!("Directory not found: {pdf_dir}"),
            });
        }

        // List PDF files
        let mut pdf_files = Vec::new();
        let mut entries = tokio::fs::read_dir(&dir_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "pdf") {
                pdf_files.push(path);
            }
        }

        if pdf_files.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "pdf_dir".to_string(),
                reason: format!("No PDF files found in: {pdf_dir}"),
            });
        }

        info!("Found {} PDF files to process", pdf_files.len());

        let mut successful = 0;
        let mut last_json_path = None;
        let mut last_md_path = None;
        let mut last_error = None;

        for pdf_path in pdf_files {
            let output_dir = PathBuf::from(&input.output_dir);
            let grobid_url = input.grobid_url.clone();
            let no_auto_start = input.no_auto_start;
            let no_figures = input.no_figures;
            let no_tables = input.no_tables;
            let copy_pdf = input.copy_pdf;
            let overwrite = input.overwrite;
            let no_markdown = input.no_markdown;

            debug!("Converting PDF: {:?}", pdf_path);

            let result = spawn_blocking(move || {
                run_pdf2text(
                    &pdf_path,
                    &output_dir,
                    grobid_url.as_deref(),
                    no_auto_start,
                    no_figures,
                    no_tables,
                    copy_pdf,
                    overwrite,
                    no_markdown,
                )
            })
            .await;

            match result {
                Ok(Ok(pyo3_result)) if pyo3_result.success => {
                    successful += 1;
                    last_json_path = pyo3_result.json_path;
                    last_md_path = pyo3_result.markdown_path;
                }
                Ok(Ok(pyo3_result)) => {
                    last_error = pyo3_result.error;
                }
                Ok(Err(e)) => {
                    warn!(error = %e, "PDF conversion failed");
                    last_error = Some(e);
                }
                Err(e) => {
                    warn!(error = %e, "PDF conversion task panicked");
                    last_error = Some(e.to_string());
                }
            }
        }

        Ok(Pdf2TextOutput {
            success: successful > 0,
            json_path: last_json_path,
            markdown_path: last_md_path,
            files_processed: successful,
            error: last_error,
        })
    }
}
