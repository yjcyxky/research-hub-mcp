//! Text2Table tool using embedded Python.
//!
//! All processing is unified batch processing (1 to N records).
//! Input records are converted to text via key:value format for LLM processing.
//! Output tables use user-specified labels as headers.

use crate::python_embed::run_text2table_cli;
use crate::{Config, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::spawn_blocking;
use tracing::{info, instrument};

/// Input parameters for text2table processing (batch mode from file)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Text2TableInput {
    /// Input file path (.tsv, .csv, .jsonl)
    #[schemars(description = "Input file path (batch processing)")]
    pub input_file: String,

    /// Output file path (format detected from extension)
    #[schemars(description = "Output file path (.tsv, .csv, .jsonl)")]
    pub output: Option<String>,

    /// Labels to extract
    #[schemars(description = "Labels to extract")]
    pub labels: Vec<String>,

    /// Path to labels file
    #[schemars(description = "Path to labels file")]
    pub labels_file: Option<String>,

    /// Column containing text (if not specified, all columns are used as key:value pairs)
    pub text_column: Option<String>,

    /// Column for record ID
    pub id_column: Option<String>,

    /// Concurrency limit
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,

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
fn default_concurrency() -> usize {
    4
}

/// Output of text2table processing
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Text2TableOutput {
    pub success: bool,
    pub message: Option<String>,
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

    /// Run text2table extraction (batch processing from file)
    #[instrument(skip(self, input))]
    pub async fn run(&self, input: Text2TableInput) -> Result<Text2TableOutput> {
        // Validate labels
        if input.labels.is_empty() && input.labels_file.is_none() {
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

        let server_url = input.server_url.clone().ok_or_else(|| {
            crate::Error::InvalidInput {
                field: "server_url".to_string(),
                reason: "Server URL is required".to_string(),
            }
        })?;

        let input_file = PathBuf::from(&input.input_file);
        let output = input.output.clone().map(PathBuf::from);
        let labels_file = input.labels_file.clone().map(PathBuf::from);
        let labels = input.labels.clone();
        let text_column = input.text_column.clone();
        let id_column = input.id_column.clone();
        let concurrency = input.concurrency;
        let prompt = input.prompt.clone();
        let threshold = input.threshold;
        let gliner_model = input.gliner_model.clone();
        let gliner_soft_threshold = input.gliner_soft_threshold;
        let model = input.model.clone();
        let enable_thinking = input.enable_thinking;
        let gliner_url = input.gliner_url.clone();
        let disable_gliner = input.disable_gliner;
        let enable_row_validation = input.enable_row_validation;
        let row_validation_mode = input.row_validation_mode.clone();
        let api_key = input.api_key.clone();
        let gliner_api_key = input.gliner_api_key.clone();

        info!("Running text2table extraction on {:?}...", input_file);

        let result = spawn_blocking(move || {
            run_text2table_cli(
                &input_file,
                output.as_deref(),
                &labels,
                labels_file.as_deref(),
                text_column.as_deref(),
                id_column.as_deref(),
                concurrency,
                prompt.as_deref(),
                threshold,
                &gliner_model,
                gliner_soft_threshold,
                model.as_deref(),
                enable_thinking,
                &server_url,
                gliner_url.as_deref(),
                disable_gliner,
                enable_row_validation,
                &row_validation_mode,
                api_key.as_deref(),
                gliner_api_key.as_deref(),
            )
        })
        .await
        .map_err(|e| crate::Error::Service(format!("Task panicked: {e}")))?;

        match result {
            Ok(()) => Ok(Text2TableOutput {
                success: true,
                message: Some("Text2table processing completed successfully".to_string()),
                error: None,
            }),
            Err(e) => Ok(Text2TableOutput {
                success: false,
                message: None,
                error: Some(e),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text2table_input_defaults() {
        let input = Text2TableInput {
            input_file: "input.jsonl".to_string(),
            output: None,
            labels: vec!["Person".to_string(), "Location".to_string()],
            labels_file: None,
            text_column: None,
            id_column: None,
            concurrency: default_concurrency(),
            prompt: None,
            threshold: default_threshold(),
            gliner_model: default_gliner_model(),
            gliner_soft_threshold: None,
            model: None,
            enable_thinking: false,
            server_url: None,
            gliner_url: None,
            disable_gliner: false,
            enable_row_validation: false,
            row_validation_mode: default_row_validation_mode(),
            api_key: None,
            gliner_api_key: None,
        };

        assert_eq!(input.threshold, 0.5);
        assert_eq!(input.gliner_model, "Ihor/gliner-biomed-large-v1.0");
        assert_eq!(input.row_validation_mode, "substring");
        assert_eq!(input.concurrency, 4);
        assert!(!input.enable_thinking);
        assert!(!input.disable_gliner);
    }

    #[test]
    fn test_text2table_input_serialization() {
        let input = Text2TableInput {
            input_file: "input.jsonl".to_string(),
            output: Some("output.jsonl".to_string()),
            labels: vec!["Gene".to_string()],
            labels_file: None,
            text_column: Some("abstract".to_string()),
            id_column: Some("id".to_string()),
            concurrency: 8,
            prompt: Some("Custom prompt".to_string()),
            threshold: 0.7,
            gliner_model: "custom-model".to_string(),
            gliner_soft_threshold: Some(0.3),
            model: Some("llama3".to_string()),
            enable_thinking: true,
            server_url: Some("http://localhost:8000".to_string()),
            gliner_url: Some("http://localhost:8001".to_string()),
            disable_gliner: true,
            enable_row_validation: true,
            row_validation_mode: "llm".to_string(),
            api_key: Some("api-key".to_string()),
            gliner_api_key: Some("gliner-key".to_string()),
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: Text2TableInput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.input_file, "input.jsonl");
        assert_eq!(deserialized.output, Some("output.jsonl".to_string()));
        assert_eq!(deserialized.labels, vec!["Gene".to_string()]);
        assert_eq!(deserialized.text_column, Some("abstract".to_string()));
        assert_eq!(deserialized.concurrency, 8);
        assert_eq!(deserialized.threshold, 0.7);
        assert!(deserialized.enable_thinking);
        assert!(deserialized.disable_gliner);
        assert!(deserialized.enable_row_validation);
        assert_eq!(deserialized.row_validation_mode, "llm");
    }

    #[test]
    fn test_text2table_output_success() {
        let output = Text2TableOutput {
            success: true,
            message: Some("Processing completed".to_string()),
            error: None,
        };

        assert!(output.success);
        assert!(output.message.is_some());
        assert!(output.error.is_none());

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_text2table_output_error() {
        let output = Text2TableOutput {
            success: false,
            message: None,
            error: Some("Failed to connect to server".to_string()),
        };

        assert!(!output.success);
        assert!(output.message.is_none());
        assert!(output.error.is_some());
    }

    #[test]
    fn test_default_functions() {
        assert_eq!(default_threshold(), 0.5);
        assert_eq!(default_gliner_model(), "Ihor/gliner-biomed-large-v1.0");
        assert_eq!(default_row_validation_mode(), "substring");
        assert_eq!(default_concurrency(), 4);
    }

    #[tokio::test]
    async fn test_run_missing_labels() {
        let config = Config::default();
        let tool = Text2TableTool::new(Arc::new(config)).unwrap();

        let input = Text2TableInput {
            input_file: "input.jsonl".to_string(),
            output: None,
            labels: vec![], // Empty labels
            labels_file: None,
            text_column: None,
            id_column: None,
            concurrency: default_concurrency(),
            prompt: None,
            threshold: default_threshold(),
            gliner_model: default_gliner_model(),
            gliner_soft_threshold: None,
            model: None,
            enable_thinking: false,
            server_url: Some("http://localhost:8000".to_string()),
            gliner_url: None,
            disable_gliner: false,
            enable_row_validation: false,
            row_validation_mode: default_row_validation_mode(),
            api_key: None,
            gliner_api_key: None,
        };

        let result = tool.run(input).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("label"),
            "Error should mention missing labels: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_run_missing_server_url() {
        let config = Config::default();
        let tool = Text2TableTool::new(Arc::new(config)).unwrap();

        let input = Text2TableInput {
            input_file: "input.jsonl".to_string(),
            output: None,
            labels: vec!["Person".to_string()],
            labels_file: None,
            text_column: None,
            id_column: None,
            concurrency: default_concurrency(),
            prompt: None,
            threshold: default_threshold(),
            gliner_model: default_gliner_model(),
            gliner_soft_threshold: None,
            model: None,
            enable_thinking: false,
            server_url: None, // Missing server URL
            gliner_url: None,
            disable_gliner: false,
            enable_row_validation: false,
            row_validation_mode: default_row_validation_mode(),
            api_key: None,
            gliner_api_key: None,
        };

        let result = tool.run(input).await;
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Server URL") || err_msg.contains("server_url"),
            "Error should mention missing server URL: {}",
            err_msg
        );
    }

    #[test]
    fn test_tool_creation() {
        let config = Config::default();
        let tool = Text2TableTool::new(Arc::new(config));
        assert!(tool.is_ok());
    }

    #[test]
    fn test_input_schema_generation() {
        let schema = schemars::schema_for!(Text2TableInput);
        let schema_json = serde_json::to_string_pretty(&schema).unwrap();

        assert!(schema_json.contains("input_file"));
        assert!(schema_json.contains("output"));
        assert!(schema_json.contains("labels"));
        assert!(schema_json.contains("text_column"));
        assert!(schema_json.contains("concurrency"));
        assert!(schema_json.contains("threshold"));
    }

    #[test]
    fn test_output_schema_generation() {
        let schema = schemars::schema_for!(Text2TableOutput);
        let schema_json = serde_json::to_string_pretty(&schema).unwrap();

        assert!(schema_json.contains("success"));
        assert!(schema_json.contains("message"));
        assert!(schema_json.contains("error"));
    }
}
