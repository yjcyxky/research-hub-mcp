use crate::client::PaperMetadata;
use crate::Result;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, warn};

/// Configuration for paper categorization
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct CategorizationConfig {
    /// Whether categorization is enabled
    pub enabled: bool,
    /// Maximum length of prompt sent to LLM (in characters)
    pub max_prompt_length: usize,
    /// Default category when categorization fails
    pub default_category: String,
    /// Maximum number of abstracts to include in categorization
    pub max_abstracts: usize,
}

impl Default for CategorizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_prompt_length: 4000,
            default_category: "research_papers".to_string(),
            max_abstracts: 5,
        }
    }
}

/// Service for categorizing papers using LLM
#[derive(Clone)]
pub struct CategorizationService {
    config: CategorizationConfig,
    snake_case_regex: Regex,
}

impl CategorizationService {
    /// Create a new categorization service
    pub fn new(config: CategorizationConfig) -> Result<Self> {
        let snake_case_regex = Regex::new(r"[^a-zA-Z0-9_]")
            .map_err(|e| crate::Error::Service(format!("Failed to compile regex: {e}")))?;

        Ok(Self {
            config,
            snake_case_regex,
        })
    }

    /// Generate category from search query and paper metadata
    pub fn generate_category_prompt(&self, query: &str, papers: &[PaperMetadata]) -> String {
        // Collect abstracts, limiting to configured maximum
        let abstracts: Vec<String> = papers
            .iter()
            .filter_map(|paper| paper.abstract_text.as_ref())
            .take(self.config.max_abstracts)
            .map(|abstract_text| {
                // Truncate long abstracts to prevent token overflow (respecting Unicode boundaries)
                if abstract_text.len() > 500 {
                    // Find the last valid UTF-8 boundary within 500 bytes
                    let mut end = 500.min(abstract_text.len());
                    while end > 0 && !abstract_text.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}...", &abstract_text[..end])
                } else {
                    abstract_text.clone()
                }
            })
            .collect();

        // Create the prompt
        let mut prompt = format!(
            "Based on this search query and paper abstracts, suggest a folder name for organizing these research papers.\n\n\
            Query: \"{query}\"\n\n"
        );

        // Add abstracts if available
        if abstracts.is_empty() {
            prompt.push_str("No abstracts available - categorize based on query only.\n\n");
        } else {
            prompt.push_str("Paper abstracts:\n");
            for (i, abstract_text) in abstracts.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n\n", i + 1, abstract_text));
            }
        }

        prompt.push_str(
            "Requirements:\n\
            - 3-5 words maximum (keep it abstract and broad)\n\
            - Use snake_case format (lowercase with underscores)\n\
            - Filesystem safe (no special characters)\n\
            - Use general research domains, not specific techniques or methods\n\
            - Examples: machine_learning, quantum_computing, biology, chemistry, physics\n\
            - Avoid overly specific terms like 'nlp_transformers' or 'crispr_cas9'\n\n\
            Return only the folder name, nothing else.",
        );

        // Truncate if too long
        if prompt.len() > self.config.max_prompt_length {
            let requirements_text = "Requirements:\n\
                - 3-5 words maximum (keep it abstract and broad)\n\
                - Use snake_case format (lowercase with underscores)\n\
                - Filesystem safe (no special characters)\n\
                - Use general research domains, not specific techniques\n\n\
                Return only the folder name, nothing else.";
            let truncation_notice = "\n\n[Content truncated to fit prompt limits]\n\n";
            let available_length =
                self.config.max_prompt_length - requirements_text.len() - truncation_notice.len();

            if available_length > 0 {
                prompt = format!(
                    "{}{}{}",
                    &prompt[..available_length.min(prompt.len())],
                    truncation_notice,
                    requirements_text
                );
            } else {
                // If even the requirements don't fit, just use requirements
                prompt = requirements_text.to_string();
            }
        }

        debug!("Generated categorization prompt ({} chars)", prompt.len());
        prompt
    }

    /// Sanitize and validate category name from LLM response
    pub fn sanitize_category(&self, category_response: &str) -> String {
        let lowercased = category_response.trim().to_lowercase();

        let category = lowercased
            .lines()
            .next() // Take only first line
            .unwrap_or("")
            .trim();

        // Remove quotes if present
        let category = category.trim_matches('"').trim_matches('\'');

        // Convert to snake_case and remove invalid characters
        let sanitized = self.snake_case_regex.replace_all(category, "_").to_string();

        // Remove multiple consecutive underscores
        let sanitized = Regex::new(r"_+")
            .unwrap()
            .replace_all(&sanitized, "_")
            .to_string();

        // Remove leading/trailing underscores
        let sanitized = sanitized.trim_matches('_').to_string();

        // Validate length (3-5 words roughly = 6-30 characters)
        if sanitized.is_empty() || sanitized.len() < 3 || sanitized.len() > 30 {
            warn!(
                "Category '{}' invalid (length: {}), using default",
                sanitized,
                sanitized.len()
            );
            return self.config.default_category.clone();
        }

        // Validate word count (approximate)
        let word_count = sanitized.split('_').filter(|s| !s.is_empty()).count();
        if word_count > 5 {
            warn!(
                "Category '{}' too many words ({} > 5), using default",
                sanitized, word_count
            );
            return self.config.default_category.clone();
        }

        info!(
            "Sanitized category: '{}' -> '{}'",
            category_response.trim(),
            sanitized
        );
        sanitized
    }

    /// Resolve category conflicts by adding numbers
    pub fn resolve_category_conflict<P: AsRef<Path>>(&self, base_dir: P, category: &str) -> String {
        let base_path = base_dir.as_ref();
        let original_category = category.to_string();
        let mut current_category = original_category.clone();
        let mut counter = 2;

        // Check if the directory already exists
        while base_path.join(&current_category).exists() {
            // If it exists, check if it's actually a directory
            let path = base_path.join(&current_category);
            if path.is_dir() {
                // Directory exists, this is fine - use the existing category
                debug!("Using existing category directory: {}", current_category);
                return current_category;
            }
            // File exists with same name, need to resolve conflict
            current_category = format!("{original_category}_{counter}");
            counter += 1;
        }

        if current_category != original_category {
            info!(
                "Resolved category conflict: '{}' -> '{}'",
                original_category, current_category
            );
        }

        current_category
    }

    /// Check if categorization is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get default category
    #[must_use]
    pub fn default_category(&self) -> &str {
        &self.config.default_category
    }

    /// Get max abstracts configuration
    #[must_use]
    pub const fn max_abstracts(&self) -> usize {
        self.config.max_abstracts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_service() -> CategorizationService {
        CategorizationService::new(CategorizationConfig::default()).unwrap()
    }

    #[test]
    fn test_sanitize_category() {
        let service = create_test_service();

        // Valid categories
        assert_eq!(
            service.sanitize_category("machine_learning"),
            "machine_learning"
        );
        assert_eq!(
            service.sanitize_category("quantum computing"),
            "quantum_computing"
        );
        assert_eq!(
            service.sanitize_category("Biology & Genetics"),
            "biology_genetics"
        );

        // Invalid characters
        assert_eq!(
            service.sanitize_category("ML/AI Research!"),
            "ml_ai_research"
        );
        assert_eq!(
            service.sanitize_category("Computer-Science"),
            "computer_science"
        );

        // Multiple underscores
        assert_eq!(
            service.sanitize_category("machine___learning"),
            "machine_learning"
        );

        // Too short or empty
        assert_eq!(service.sanitize_category(""), "research_papers");
        assert_eq!(service.sanitize_category("ai"), "research_papers");

        // Too many words (>5)
        let long_category = "very_long_category_name_with_too_many_words";
        assert_eq!(service.sanitize_category(long_category), "research_papers");

        // Quoted responses
        assert_eq!(
            service.sanitize_category("\"machine_learning\""),
            "machine_learning"
        );
        assert_eq!(
            service.sanitize_category("'quantum_physics'"),
            "quantum_physics"
        );
    }

    #[test]
    fn test_resolve_category_conflict() {
        let service = create_test_service();
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // No conflict
        assert_eq!(
            service.resolve_category_conflict(base_path, "machine_learning"),
            "machine_learning"
        );

        // Create directory and test existing directory (should use existing)
        fs::create_dir_all(base_path.join("machine_learning")).unwrap();
        assert_eq!(
            service.resolve_category_conflict(base_path, "machine_learning"),
            "machine_learning"
        );

        // Create file with same name (should create numbered variant)
        fs::write(base_path.join("quantum_physics"), "test").unwrap();
        assert_eq!(
            service.resolve_category_conflict(base_path, "quantum_physics"),
            "quantum_physics_2"
        );
    }

    #[test]
    fn test_generate_category_prompt() {
        let service = create_test_service();

        let papers = vec![PaperMetadata {
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
        }];

        let prompt = service.generate_category_prompt("machine learning", &papers);

        assert!(prompt.contains("machine learning"));
        assert!(prompt.contains("machine learning applications in medical diagnosis"));
        assert!(prompt.contains("snake_case format"));
        assert!(prompt.contains("Return only the folder name"));
    }

    #[test]
    fn test_prompt_truncation() {
        let mut config = CategorizationConfig::default();
        config.max_prompt_length = 500; // Very short for testing

        let service = CategorizationService::new(config).unwrap();

        let papers = vec![PaperMetadata {
            doi: "10.1000/test1".to_string(),
            title: Some("Very Long Title".to_string()),
            authors: vec!["Author".to_string()],
            journal: None,
            year: Some(2023),
            abstract_text: Some("A".repeat(1000)), // Very long abstract
            pdf_url: None,
            file_size: None,
            pmid: None,
            keywords: vec!()
        }];

        let prompt = service.generate_category_prompt("test query", &papers);

        assert!(prompt.len() <= 500);
        assert!(prompt.contains("Content truncated"));
    }
}
