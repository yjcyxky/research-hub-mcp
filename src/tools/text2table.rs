//! Text2Table tool using embedded Python.

use crate::python_embed::{run_text2table, run_text2table_batch};
use crate::{Config, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::task::spawn_blocking;
use tracing::{info, instrument};

/// Input parameters for text2table generation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Text2TableInput {
    /// Text to process
    #[schemars(description = "Raw text to process")]
    pub text: Option<String>,

    /// Path to text file
    #[schemars(description = "Path to text file")]
    pub text_file: Option<String>,

    /// Labels to extract
    #[schemars(description = "Labels to extract")]
    pub labels: Vec<String>,

    /// Path to labels file
    #[schemars(description = "Path to labels file")]
    pub labels_file: Option<String>,

    /// Custom prompt
    pub prompt: Option<String>,

    /// Entity extraction threshold (0.0-1.0)
    #[serde(default = "default_threshold")]
    pub threshold: f64,

    /// GLiNER model
    #[serde(default = "default_gliner_model")]
    pub gliner_model: String,

    /// GLiNER soft threshold for low confidence hints
    pub gliner_soft_threshold: Option<f64>,

    /// vLLM Model name
    pub model: Option<String>,

    /// Enable thinking mode
    #[serde(default)]
    pub enable_thinking: bool,

    /// Server URL
    pub server_url: Option<String>,

    /// GLiNER URL
    pub gliner_url: Option<String>,

    /// Disable GLiNER
    #[serde(default)]
    pub disable_gliner: bool,

    /// Enable row validation
    #[serde(default)]
    pub enable_row_validation: bool,

    /// Row validation mode ("substring" or "llm")
    #[serde(default = "default_row_validation_mode")]
    pub row_validation_mode: String,

    /// API Key for vLLM
    pub api_key: Option<String>,

    /// API Key for GLiNER
    pub gliner_api_key: Option<String>,
}

fn default_threshold() -> f64 {
    0.5
}
fn default_gliner_model() -> String {
    "Ihor/gliner-biomed-large-v1.0".to_string()
}
fn default_row_validation_mode() -> String {
    "substring".to_string()
}

/// Input parameters for batch processing
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Text2TableBatchInput {
    /// Path to input TSV/CSV file
    pub input_file: String,

    /// Path to output file
    pub output_file: Option<String>,

    /// Output format ("jsonl" or "tsv")
    #[serde(default = "default_output_format")]
    pub output_format: String,

    /// Concurrency limit
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,

    /// Common configuration for generation
    #[serde(flatten)]
    pub config: Text2TableInput,
}

fn default_concurrency() -> usize {
    4
}

fn default_output_format() -> String {
    "jsonl".to_string()
}

/// Output of text2table generation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Text2TableOutput {
    pub success: bool,
    pub table: Option<String>,
    pub entities: Option<serde_json::Value>,
    pub thinking: Option<String>,
    pub error: Option<String>,
}

/// Tool wrapper
#[derive(Clone)]
pub struct Text2TableTool {
    #[allow(dead_code)]
    config: Arc<Config>,
}

impl Text2TableTool {
    pub fn new(config: Arc<Config>) -> Result<Self> {
        Ok(Self { config })
    }

    /// Run single text generation
    #[instrument(skip(self, input), fields(text_len = input.text.as_ref().map(|t| t.len()).unwrap_or(0)))]
    pub async fn generate(&self, input: Text2TableInput) -> Result<Text2TableOutput> {
        // Resolve text
        // Resolve text
        let text = if let Some(t) = &input.text {
            t.clone()
        } else if let Some(path) = &input.text_file {
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| crate::Error::InvalidInput {
                    field: "text_file".to_string(),
                    reason: format!("Failed to read file: {e}"),
                })?
        } else {
            return Err(crate::Error::InvalidInput {
                field: "text".to_string(),
                reason: "Either text or text_file must be provided".to_string(),
            });
        };

        // Resolve labels
        let mut labels = input.labels.clone();
        if let Some(path) = &input.labels_file {
            let content =
                tokio::fs::read_to_string(path)
                    .await
                    .map_err(|e| crate::Error::InvalidInput {
                        field: "labels_file".to_string(),
                        reason: format!("Failed to read labels file: {e}"),
                    })?;
            for line in content.lines() {
                if !line.trim().is_empty() {
                    labels.push(line.trim().to_string());
                }
            }
        }

        if labels.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "labels".to_string(),
                reason: "At least one label is required".to_string(),
            });
        }

        // Check Python
        if let Err(e) = crate::python_embed::check_python_available() {
            return Err(crate::Error::Service(format!(
                "Python runtime not available: {e}"
            )));
        }

        // Run in blocking thread
        let server_url = input.server_url.clone().unwrap_or_else(|| "".to_string()); // Should be handled by CLI checking or default from env
        if server_url.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "server_url".to_string(),
                reason: "Server URL is required".to_string(),
            });
        }

        let input_clone = input.clone();

        let result = spawn_blocking(move || {
            run_text2table(
                &text,
                &labels,
                &server_url,
                input_clone.model.as_deref(),
                input_clone.enable_thinking,
                input_clone.prompt.as_deref(),
                input_clone.threshold,
                !input_clone.disable_gliner,
                input_clone.gliner_url.as_deref(),
                &input_clone.gliner_model,
                input_clone.gliner_soft_threshold,
                input_clone.enable_row_validation,
                &input_clone.row_validation_mode,
                input_clone.api_key.as_deref(),
                input_clone.gliner_api_key.as_deref(),
            )
        })
        .await
        .map_err(|e| crate::Error::Service(format!("Task panicked: {e}")))?;

        match result {
            Ok(output) => {
                let entities_json: Option<serde_json::Value> =
                    if let Some(json_str) = output.entities {
                        serde_json::from_str(&json_str).ok()
                    } else {
                        None
                    };

                Ok(Text2TableOutput {
                    success: output.success,
                    table: output.table,
                    entities: entities_json,
                    thinking: output.thinking,
                    error: output.error,
                })
            }
            Err(e) => Ok(Text2TableOutput {
                success: false,
                table: None,
                entities: None,
                thinking: None,
                error: Some(e),
            }),
        }
    }

    /// Process batch input
    #[instrument(skip(self, input))]
    pub async fn process_batch(&self, input: Text2TableBatchInput) -> Result<()> {
        let input_path = PathBuf::from(&input.input_file);
        if !input_path.exists() {
            return Err(crate::Error::InvalidInput {
                field: "input_file".to_string(),
                reason: format!("File not found: {}", input.input_file),
            });
        }

        let output_format = input.output_format.trim().to_lowercase();
        if output_format != "jsonl" && output_format != "tsv" {
            return Err(crate::Error::InvalidInput {
                field: "output_format".to_string(),
                reason: "Output format must be 'jsonl' or 'tsv'".to_string(),
            });
        }
        let output_file = input.output_file.clone().unwrap_or_else(|| {
            if output_format == "tsv" {
                "output.tsv".to_string()
            } else {
                "output.jsonl".to_string()
            }
        });
        if let Err(e) = crate::python_embed::check_python_available() {
            return Err(crate::Error::Service(format!(
                "Python runtime not available: {e}"
            )));
        }

        let server_url = input
            .config
            .server_url
            .clone()
            .unwrap_or_else(|| "".to_string());
        if server_url.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "server_url".to_string(),
                reason: "Server URL is required".to_string(),
            });
        }

        if input.concurrency == 0 {
            return Err(crate::Error::InvalidInput {
                field: "concurrency".to_string(),
                reason: "Concurrency must be a positive integer".to_string(),
            });
        }

        // Ensure output dir exists
        if let Some(parent) = Path::new(&output_file).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let output_path = PathBuf::from(&output_file);
        let input_config = input.config.clone();
        let labels_file = input_config.labels_file.as_ref().map(PathBuf::from);
        let gliner_url = input_config.gliner_url.clone();
        let concurrency = input.concurrency;

        info!("Delegating batch processing to Python (concurrency={})", concurrency);

        let processed = spawn_blocking(move || {
            run_text2table_batch(
                &input_path,
                &output_path,
                &output_format,
                &input_config.labels,
                labels_file.as_deref(),
                input_config.prompt.as_deref(),
                input_config.threshold,
                &input_config.gliner_model,
                input_config.gliner_soft_threshold,
                input_config.model.as_deref(),
                input_config.enable_thinking,
                &server_url,
                gliner_url.as_deref(),
                input_config.disable_gliner,
                input_config.enable_row_validation,
                &input_config.row_validation_mode,
                input_config.api_key.as_deref(),
                input_config.gliner_api_key.as_deref(),
                concurrency,
            )
        })
        .await
        .map_err(|e| crate::Error::Service(format!("Task panicked: {e}")))?;

        match processed {
            Ok(count) => {
                info!(
                    "Batch processing complete ({} rows). Output saved to {}",
                    count, output_file
                );
            }
            Err(e) => {
                return Err(crate::Error::Service(format!(
                    "Batch processing failed: {e}"
                )));
            }
        }

        Ok(())
    }
}
