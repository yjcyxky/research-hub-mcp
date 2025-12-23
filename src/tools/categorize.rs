use crate::client::PaperMetadata;
use crate::services::CategorizationService;
use crate::{Config, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// Input parameters for the paper categorization tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CategorizeInput {
    /// Search query that generated the papers
    pub query: String,
    /// List of papers to categorize
    pub papers: Vec<PaperMetadata>,
    /// Maximum number of abstracts to analyze (optional, defaults to service config)
    pub max_abstracts: Option<usize>,
}

/// Result of paper categorization
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CategorizeResult {
    /// Generated category name
    pub category: String,
    /// Whether the category was generated or is a fallback
    pub is_fallback: bool,
    /// Number of papers analyzed
    pub papers_analyzed: usize,
    /// Number of abstracts used for categorization
    pub abstracts_used: usize,
    /// Sanitized category (ready for filesystem use)
    pub sanitized_category: String,
    /// Prompt length used for categorization
    pub prompt_length: usize,
}

/// Paper categorization tool implementation
#[derive(Clone)]
pub struct CategorizeTool {
    categorization_service: CategorizationService,
    #[allow(dead_code)] // Will be used for future features
    config: Arc<Config>,
}

impl std::fmt::Debug for CategorizeTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CategorizeTool")
            .field("categorization_service", &"CategorizationService")
            .field("config", &"Config")
            .finish()
    }
}

impl CategorizeTool {
    /// Create a new categorization tool
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing paper categorization tool");

        let categorization_service = CategorizationService::new(config.categorization.clone())
            .map_err(|e| {
                crate::Error::Service(format!("Failed to create categorization service: {e}"))
            })?;

        Ok(Self {
            categorization_service,
            config,
        })
    }

    /// Categorize papers based on query and abstracts
    // #[tool] // Will be enabled when rmcp integration is complete
    #[instrument(skip(self), fields(query = %input.query, paper_count = input.papers.len()))]
    pub async fn categorize_papers(&self, input: CategorizeInput) -> Result<CategorizeResult> {
        info!(
            "Categorizing {} papers for query: '{}'",
            input.papers.len(),
            input.query
        );

        // Validate input
        Self::validate_input(&input)?;

        if !self.categorization_service.is_enabled() {
            return Ok(CategorizeResult {
                category: self.categorization_service.default_category().to_string(),
                is_fallback: true,
                papers_analyzed: 0,
                abstracts_used: 0,
                sanitized_category: self.categorization_service.default_category().to_string(),
                prompt_length: 0,
            });
        }

        // Limit papers based on service configuration or input override
        let max_abstracts = input
            .max_abstracts
            .unwrap_or(self.categorization_service.max_abstracts());
        let papers_to_analyze: Vec<&PaperMetadata> =
            input.papers.iter().take(max_abstracts).collect();

        // Count abstracts available
        let abstracts_used = papers_to_analyze
            .iter()
            .filter(|paper| paper.abstract_text.is_some())
            .count();

        debug!(
            "Analyzing {} papers ({} with abstracts)",
            papers_to_analyze.len(),
            abstracts_used
        );

        let papers_analyzed_count = papers_to_analyze.len();

        // Generate categorization prompt
        let prompt = self.categorization_service.generate_category_prompt(
            &input.query,
            &papers_to_analyze.into_iter().cloned().collect::<Vec<_>>(),
        );

        let prompt_length = prompt.len();
        debug!("Generated categorization prompt ({} chars)", prompt_length);

        // For now, use heuristic categorization since this is an MCP server
        // In the future, this could be replaced with an actual LLM call
        let category_response = self.simple_heuristic_categorization(&input.query, &input.papers);

        // Sanitize the category
        let sanitized_category = self
            .categorization_service
            .sanitize_category(&category_response);

        let is_fallback = sanitized_category == self.categorization_service.default_category();

        info!(
            "Categorization complete: '{}' (fallback: {})",
            sanitized_category, is_fallback
        );

        Ok(CategorizeResult {
            category: category_response,
            is_fallback,
            papers_analyzed: papers_analyzed_count,
            abstracts_used,
            sanitized_category,
            prompt_length,
        })
    }

    /// Validate categorization input
    fn validate_input(input: &CategorizeInput) -> Result<()> {
        if input.query.trim().is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "query".to_string(),
                reason: "Query cannot be empty".to_string(),
            });
        }

        if input.query.len() > 1000 {
            return Err(crate::Error::InvalidInput {
                field: "query".to_string(),
                reason: "Query too long (max 1000 characters)".to_string(),
            });
        }

        if input.papers.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "papers".to_string(),
                reason: "At least one paper is required for categorization".to_string(),
            });
        }

        if input.papers.len() > 100 {
            return Err(crate::Error::InvalidInput {
                field: "papers".to_string(),
                reason: "Too many papers (max 100)".to_string(),
            });
        }

        Ok(())
    }

    /// Simple heuristic categorization (fallback when no LLM available)
    fn simple_heuristic_categorization(&self, query: &str, papers: &[PaperMetadata]) -> String {
        let query_lower = query.to_lowercase();

        // Collect keywords from query and paper titles/abstracts
        let mut keywords = vec![query_lower.clone()];

        for paper in papers.iter().take(3) {
            // Analyze first 3 papers
            if let Some(title) = &paper.title {
                keywords.push(title.to_lowercase());
            }
            if let Some(abstract_text) = &paper.abstract_text {
                keywords.push(abstract_text[..abstract_text.len().min(200)].to_lowercase());
            }
        }

        let all_text = keywords.join(" ");

        // Simple keyword-based categorization
        if all_text.contains("machine learning")
            || all_text.contains("neural network")
            || all_text.contains("deep learning")
        {
            "machine_learning".to_string()
        } else if all_text.contains("quantum") || all_text.contains("physics") {
            "quantum_physics".to_string()
        } else if all_text.contains("biology")
            || all_text.contains("genetics")
            || all_text.contains("biomedical")
        {
            "biology_genetics".to_string()
        } else if all_text.contains("computer")
            || all_text.contains("algorithm")
            || all_text.contains("software")
        {
            "computer_science".to_string()
        } else if all_text.contains("climate")
            || all_text.contains("environment")
            || all_text.contains("sustainability")
        {
            "environmental_science".to_string()
        } else if all_text.contains("medicine")
            || all_text.contains("medical")
            || all_text.contains("health")
        {
            "medical_research".to_string()
        } else if all_text.contains("chemistry") || all_text.contains("chemical") {
            "chemistry".to_string()
        } else if all_text.contains("mathematics")
            || all_text.contains("mathematical")
            || all_text.contains("math")
        {
            "mathematics".to_string()
        } else if all_text.contains("ohat")
            || all_text.contains("systematic review")
            || all_text.contains("literature review")
        {
            "systematic_review".to_string()
        } else if all_text.contains("agent")
            || all_text.contains("multi-agent")
            || all_text.contains("agentic")
        {
            "agentic_systems".to_string()
        } else {
            // Extract first meaningful words from query
            let words: Vec<&str> = query_lower
                .split_whitespace()
                .filter(|w| {
                    w.len() > 2 && !["the", "and", "for", "with", "from", "into"].contains(w)
                })
                .take(3)
                .collect();

            if words.is_empty() {
                self.categorization_service.default_category().to_string()
            } else {
                words.join("_")
            }
        }
    }

    /// Check if categorization is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.categorization_service.is_enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_test_categorize_tool() -> Result<CategorizeTool> {
        let config = Arc::new(Config::default());
        CategorizeTool::new(config)
    }

    fn create_test_papers() -> Vec<PaperMetadata> {
        vec![
            PaperMetadata {
                doi: "10.1000/test1".to_string(),
                title: Some("Machine Learning in Healthcare".to_string()),
                authors: vec!["Smith, J.".to_string()],
                journal: None,
                year: Some(2023),
                abstract_text: Some(
                    "This paper explores machine learning applications in medical diagnosis."
                        .to_string(),
                ),
                pdf_url: None,
                file_size: None,
                pmid: None,
                keywords: vec!()
            },
            PaperMetadata {
                doi: "10.1000/test2".to_string(),
                title: Some("Deep Neural Networks for Image Classification".to_string()),
                authors: vec!["Doe, J.".to_string()],
                journal: None,
                year: Some(2023),
                abstract_text: Some(
                    "We present a novel deep learning approach for image classification."
                        .to_string(),
                ),
                pdf_url: None,
                file_size: None,
                pmid: None,
                keywords: vec!()
            },
        ]
    }

    #[test]
    fn test_categorize_input_validation() {
        let _tool = create_test_categorize_tool().unwrap();

        // Empty query should fail
        let empty_query = CategorizeInput {
            query: "".to_string(),
            papers: create_test_papers(),
            max_abstracts: None,
        };
        assert!(CategorizeTool::validate_input(&empty_query).is_err());

        // Too long query should fail
        let long_query = CategorizeInput {
            query: "a".repeat(1001),
            papers: create_test_papers(),
            max_abstracts: None,
        };
        assert!(CategorizeTool::validate_input(&long_query).is_err());

        // Empty papers should fail
        let empty_papers = CategorizeInput {
            query: "machine learning".to_string(),
            papers: vec![],
            max_abstracts: None,
        };
        assert!(CategorizeTool::validate_input(&empty_papers).is_err());

        // Valid input should pass
        let valid_input = CategorizeInput {
            query: "machine learning".to_string(),
            papers: create_test_papers(),
            max_abstracts: None,
        };
        assert!(CategorizeTool::validate_input(&valid_input).is_ok());
    }

    #[tokio::test]
    async fn test_categorize_papers() {
        let tool = create_test_categorize_tool().unwrap();

        let input = CategorizeInput {
            query: "machine learning in healthcare".to_string(),
            papers: create_test_papers(),
            max_abstracts: Some(2),
        };

        let result = tool.categorize_papers(input).await.unwrap();

        assert!(!result.category.is_empty());
        assert!(!result.sanitized_category.is_empty());
        assert_eq!(result.papers_analyzed, 2);
        assert_eq!(result.abstracts_used, 2);
        assert!(result.prompt_length > 0);
    }

    #[tokio::test]
    async fn test_heuristic_categorization() {
        let tool = create_test_categorize_tool().unwrap();
        let papers = create_test_papers();

        // Test machine learning categorization
        let ml_result = tool.simple_heuristic_categorization("machine learning", &papers);
        assert_eq!(ml_result, "machine_learning");

        // Test quantum physics categorization
        let quantum_papers = vec![PaperMetadata {
            doi: "10.1000/quantum1".to_string(),
            title: Some("Quantum Computing Theory".to_string()),
            authors: vec!["Einstein, A.".to_string()],
            journal: None,
            year: Some(2023),
            abstract_text: Some(
                "This paper discusses quantum mechanics and quantum computing.".to_string(),
            ),
            pdf_url: None,
            file_size: None,
            pmid: None,
            keywords: vec!()
        }];
        let quantum_result =
            tool.simple_heuristic_categorization("quantum physics", &quantum_papers);
        assert_eq!(quantum_result, "quantum_physics");

        // Test default fallback
        let generic_result = tool.simple_heuristic_categorization("unknown topic", &[]);
        assert_eq!(generic_result, "unknown_topic");
    }

    #[test]
    fn test_categorize_tool_creation() {
        let config = Arc::new(Config::default());
        let tool = CategorizeTool::new(config);
        assert!(tool.is_ok());

        let tool = tool.unwrap();
        assert!(tool.is_enabled()); // Should be enabled by default
    }
}
