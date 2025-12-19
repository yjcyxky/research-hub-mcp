//! Text2Table tool using embedded Python.

use crate::python_embed::{run_text2table, Text2TableResult};
use crate::{Config, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::task::spawn_blocking;
use tracing::{debug, info, instrument, warn};

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

        info!("Reading batch file: {:?}", input_path);

        // Read CSV/TSV
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t') // Assume TSV by default, or maybe check extension?
            .from_path(&input_path)
            .or_else(|_| csv::ReaderBuilder::new().from_path(&input_path)) // Fallback to comma?
            .map_err(|e| crate::Error::InvalidInput {
                field: "input_file".to_string(),
                reason: format!("Failed to open CSV/TSV: {e}"),
            })?;

        let headers = rdr.headers().cloned().unwrap_or_default();

        let output_file = input
            .output_file
            .clone()
            .unwrap_or_else(|| "output.jsonl".to_string());
        // For now, let's just write JSONL output
        use std::fs::OpenOptions;
        use std::io::Write;

        // Ensure output dir exists
        if let Some(parent) = Path::new(&output_file).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let semaphore = Arc::new(tokio::sync::Semaphore::new(input.concurrency));
        let mut tasks = Vec::new();

        for result in rdr.records() {
            let record = result.map_err(|e| crate::Error::InvalidInput {
                field: "input_file".to_string(),
                reason: format!("Failed to read record: {e}"),
            })?;

            // Format record as "Key: Value"
            let mut text_parts = Vec::new();
            for (i, field) in record.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    text_parts.push(format!("{}: {}", header, field));
                }
            }
            let formatted_text = text_parts.join("\n");

            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let tool = self.clone();
            let row_config = input.config.clone();

            // Create a task
            tasks.push(tokio::spawn(async move {
                // Determine ID if possible? For now rely on index or content?
                // Let's create a "row_input"
                let mut row_input = row_config;
                row_input.text = Some(formatted_text.clone());
                row_input.text_file = None;

                debug!("Processing row: {:.50}...", formatted_text);

                let res = tool.generate(row_input).await;

                // Write result (simple lock-less append might be issues if concurrent? standard file append on unix is atomic for small writes usually, but better use a channel or mutex)
                // For simplicity in this step, let's just print or format

                drop(permit);
                (formatted_text, res)
            }));
        }

        // Await all and write sequentially to avoid race conditions on file write
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_file)
            .map_err(|e| crate::Error::Service(format!("Failed to open output file: {e}")))?;

        for task in tasks {
            let (original_text, result) = task.await.unwrap(); // Handle join error?

            let output_obj = match result {
                Ok(out) => serde_json::json!({
                    "original_text": original_text,
                    "success": out.success,
                    "table": out.table,
                    "entities": out.entities,
                    "thinking": out.thinking,
                    "error": out.error
                }),
                Err(e) => serde_json::json!({
                    "original_text": original_text,
                    "success": false,
                    "error": e.to_string()
                }),
            };

            if let Err(e) = writeln!(file, "{}", output_obj.to_string()) {
                warn!("Failed to write result: {e}");
            }
        }

        info!("Batch processing complete. Output saved to {}", output_file);

        Ok(())
    }
}
