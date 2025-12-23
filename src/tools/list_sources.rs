use crate::client::providers::{SearchContext, SourceProvider};
use crate::Result;
use futures::future::join_all;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

/// Input for listing available sources
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListSourcesInput {
    /// Filter by capability (full_text, metadata_only)
    #[serde(default)]
    #[schemars(description = "Filter sources: all, full_text, metadata_only")]
    pub filter: Option<String>,

    /// Include query examples in output
    #[serde(default = "default_include_examples")]
    #[schemars(description = "Include query examples (default: true)")]
    pub include_examples: bool,

    /// Include health status (requires API calls)
    #[serde(default)]
    #[schemars(description = "Include health status check (slower)")]
    pub include_health: bool,
}

fn default_include_examples() -> bool {
    true
}

impl Default for ListSourcesInput {
    fn default() -> Self {
        Self {
            filter: None,
            include_examples: true,
            include_health: false,
        }
    }
}

/// Information about a single source
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceDetails {
    /// Source identifier
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Provider priority (higher = preferred)
    pub priority: u8,
    /// Whether source provides full text/PDF access
    pub supports_full_text: bool,
    /// Supported search types
    pub supported_search_types: Vec<String>,
    /// Query format help
    pub query_format_help: String,
    /// Example queries with descriptions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_examples: Option<Vec<QueryExample>>,
    /// Link to native query syntax documentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_query_syntax: Option<String>,
    /// Health status (if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub healthy: Option<bool>,
}

/// Query example
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryExample {
    /// Example query string
    pub query: String,
    /// Description of what this query finds
    pub description: String,
}

/// Result of listing sources
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListSourcesResult {
    /// Total number of sources
    pub total: usize,
    /// Sources matching the filter
    pub sources: Vec<SourceDetails>,
    /// Available filters
    pub available_filters: Vec<String>,
}

/// Tool for listing available academic sources
pub struct ListSourcesTool {
    providers: HashMap<String, Arc<dyn SourceProvider>>,
}

impl Default for ListSourcesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ListSourcesTool {
    /// Create a new list sources tool
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider
    pub fn register_provider(&mut self, provider: Box<dyn SourceProvider>) {
        let name = provider.name().to_string();
        self.providers.insert(name, Arc::from(provider));
    }

    /// List all available sources (async version with optional health checks)
    pub async fn list(&self, input: ListSourcesInput) -> Result<ListSourcesResult> {
        info!("Listing sources with filter: {:?}", input.filter);

        // Filter providers
        let filtered_providers: Vec<_> = self
            .providers
            .values()
            .filter(|p| match input.filter.as_deref() {
                Some("full_text") => p.supports_full_text(),
                Some("metadata_only") => !p.supports_full_text(),
                _ => true, // "all" or None
            })
            .collect();

        // Perform health checks concurrently if requested
        let health_status: HashMap<String, bool> = if input.include_health {
            info!("Performing health checks on {} providers", filtered_providers.len());
            let context = SearchContext {
                timeout: Duration::from_secs(10),
                user_agent: "rust-research-mcp/1.0".to_string(),
                rate_limit: None,
                headers: HashMap::new(),
            };

            let health_futures: Vec<_> = filtered_providers
                .iter()
                .map(|p| {
                    let provider = Arc::clone(p);
                    let ctx = context.clone();
                    async move {
                        let name = provider.name().to_string();
                        let healthy = provider.health_check(&ctx).await.unwrap_or(false);
                        (name, healthy)
                    }
                })
                .collect();

            join_all(health_futures).await.into_iter().collect()
        } else {
            HashMap::new()
        };

        // Build source details
        let mut sources: Vec<SourceDetails> = filtered_providers
            .iter()
            .map(|p| {
                let query_examples = if input.include_examples {
                    Some(
                        p.query_examples()
                            .into_iter()
                            .map(|(query, desc)| QueryExample {
                                query: query.to_string(),
                                description: desc.to_string(),
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                let name = p.name().to_string();
                let healthy = health_status.get(&name).copied();

                SourceDetails {
                    name,
                    description: p.description().to_string(),
                    priority: p.priority(),
                    supports_full_text: p.supports_full_text(),
                    supported_search_types: p
                        .supported_search_types()
                        .into_iter()
                        .map(|t| format!("{:?}", t).to_lowercase())
                        .collect(),
                    query_format_help: p.query_format_help().to_string(),
                    query_examples,
                    native_query_syntax: p.native_query_syntax().map(String::from),
                    healthy,
                }
            })
            .collect();

        // Sort by priority (highest first)
        sources.sort_by(|a, b| b.priority.cmp(&a.priority));

        let total = sources.len();

        info!("Listed {} sources", total);

        Ok(ListSourcesResult {
            total,
            sources,
            available_filters: vec![
                "all".to_string(),
                "full_text".to_string(),
                "metadata_only".to_string(),
            ],
        })
    }

    /// Get detailed info for a specific source
    pub fn get_source_details(&self, source_name: &str) -> Option<SourceDetails> {
        let provider = self.providers.get(source_name)?;

        Some(SourceDetails {
            name: provider.name().to_string(),
            description: provider.description().to_string(),
            priority: provider.priority(),
            supports_full_text: provider.supports_full_text(),
            supported_search_types: provider
                .supported_search_types()
                .into_iter()
                .map(|t| format!("{:?}", t).to_lowercase())
                .collect(),
            query_format_help: provider.query_format_help().to_string(),
            query_examples: Some(
                provider
                    .query_examples()
                    .into_iter()
                    .map(|(query, desc)| QueryExample {
                        query: query.to_string(),
                        description: desc.to_string(),
                    })
                    .collect(),
            ),
            native_query_syntax: provider.native_query_syntax().map(String::from),
            healthy: None,
        })
    }

    /// Get detailed info for a specific source with health check
    pub async fn get_source_details_with_health(&self, source_name: &str) -> Option<SourceDetails> {
        let provider = self.providers.get(source_name)?;

        let context = SearchContext {
            timeout: Duration::from_secs(10),
            user_agent: "rust-research-mcp/1.0".to_string(),
            rate_limit: None,
            headers: HashMap::new(),
        };

        let healthy = provider.health_check(&context).await.ok();

        Some(SourceDetails {
            name: provider.name().to_string(),
            description: provider.description().to_string(),
            priority: provider.priority(),
            supports_full_text: provider.supports_full_text(),
            supported_search_types: provider
                .supported_search_types()
                .into_iter()
                .map(|t| format!("{:?}", t).to_lowercase())
                .collect(),
            query_format_help: provider.query_format_help().to_string(),
            query_examples: Some(
                provider
                    .query_examples()
                    .into_iter()
                    .map(|(query, desc)| QueryExample {
                        query: query.to_string(),
                        description: desc.to_string(),
                    })
                    .collect(),
            ),
            native_query_syntax: provider.native_query_syntax().map(String::from),
            healthy,
        })
    }
}

impl std::fmt::Debug for ListSourcesTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ListSourcesTool")
            .field("provider_count", &self.providers.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_sources_input_defaults() {
        let input = ListSourcesInput::default();
        assert!(input.filter.is_none());
        assert!(input.include_examples);
        assert!(!input.include_health);
    }

    #[test]
    fn test_list_sources_input_serialization() {
        let input = ListSourcesInput {
            filter: Some("full_text".to_string()),
            include_examples: false,
            include_health: true,
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: ListSourcesInput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.filter, Some("full_text".to_string()));
        assert!(!deserialized.include_examples);
        assert!(deserialized.include_health);
    }

    #[test]
    fn test_source_details_serialization() {
        let details = SourceDetails {
            name: "arxiv".to_string(),
            description: "ArXiv preprint server".to_string(),
            priority: 90,
            supports_full_text: true,
            supported_search_types: vec!["doi".to_string(), "title".to_string()],
            query_format_help: "Use ti: for title search".to_string(),
            query_examples: Some(vec![QueryExample {
                query: "ti:neural networks".to_string(),
                description: "Search in title".to_string(),
            }]),
            native_query_syntax: Some("https://arxiv.org/help/api".to_string()),
            healthy: Some(true),
        };

        let json = serde_json::to_string(&details).unwrap();
        assert!(json.contains("arxiv"));
        assert!(json.contains("priority"));
        assert!(json.contains("healthy"));
    }

    #[test]
    fn test_list_sources_result_structure() {
        let result = ListSourcesResult {
            total: 2,
            sources: vec![
                SourceDetails {
                    name: "arxiv".to_string(),
                    description: "ArXiv".to_string(),
                    priority: 90,
                    supports_full_text: true,
                    supported_search_types: vec!["doi".to_string()],
                    query_format_help: "Help".to_string(),
                    query_examples: None,
                    native_query_syntax: None,
                    healthy: None,
                },
                SourceDetails {
                    name: "crossref".to_string(),
                    description: "CrossRef".to_string(),
                    priority: 85,
                    supports_full_text: false,
                    supported_search_types: vec!["doi".to_string()],
                    query_format_help: "Help".to_string(),
                    query_examples: None,
                    native_query_syntax: None,
                    healthy: None,
                },
            ],
            available_filters: vec!["all".to_string(), "full_text".to_string()],
        };

        assert_eq!(result.total, 2);
        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.sources[0].name, "arxiv");
    }

    #[test]
    fn test_tool_creation() {
        let tool = ListSourcesTool::new();
        assert_eq!(tool.providers.len(), 0);
    }

    #[test]
    fn test_tool_debug() {
        let tool = ListSourcesTool::new();
        let debug_str = format!("{:?}", tool);
        assert!(debug_str.contains("ListSourcesTool"));
        assert!(debug_str.contains("provider_count"));
    }

    #[tokio::test]
    async fn test_empty_list() {
        let tool = ListSourcesTool::new();
        let input = ListSourcesInput::default();
        let result = tool.list(input).await.unwrap();

        assert_eq!(result.total, 0);
        assert!(result.sources.is_empty());
        assert!(!result.available_filters.is_empty());
    }

    #[test]
    fn test_query_example_structure() {
        let example = QueryExample {
            query: "machine learning".to_string(),
            description: "Basic search".to_string(),
        };

        let json = serde_json::to_string(&example).unwrap();
        let deserialized: QueryExample = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.query, "machine learning");
        assert_eq!(deserialized.description, "Basic search");
    }

    #[test]
    fn test_get_nonexistent_source() {
        let tool = ListSourcesTool::new();
        let result = tool.get_source_details("nonexistent");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_filter_all_returns_all_providers() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let input = ListSourcesInput {
            filter: Some("all".to_string()),
            include_examples: false,
            include_health: false,
        };
        let result = tool.list(input).await.unwrap();

        assert_eq!(result.total, 1);
        assert_eq!(result.sources[0].name, "arxiv");
    }

    #[tokio::test]
    async fn test_filter_full_text_only() {
        use crate::client::providers::{ArxivProvider, CrossRefProvider};

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap())); // supports full text
        tool.register_provider(Box::new(CrossRefProvider::new(None).unwrap())); // metadata only

        let input = ListSourcesInput {
            filter: Some("full_text".to_string()),
            include_examples: false,
            include_health: false,
        };
        let result = tool.list(input).await.unwrap();

        // Only ArXiv supports full text
        assert_eq!(result.total, 1);
        assert_eq!(result.sources[0].name, "arxiv");
        assert!(result.sources[0].supports_full_text);
    }

    #[tokio::test]
    async fn test_filter_metadata_only() {
        use crate::client::providers::{ArxivProvider, CrossRefProvider};

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap())); // supports full text
        tool.register_provider(Box::new(CrossRefProvider::new(None).unwrap())); // metadata only

        let input = ListSourcesInput {
            filter: Some("metadata_only".to_string()),
            include_examples: false,
            include_health: false,
        };
        let result = tool.list(input).await.unwrap();

        // Only CrossRef is metadata only
        assert_eq!(result.total, 1);
        assert_eq!(result.sources[0].name, "crossref");
        assert!(!result.sources[0].supports_full_text);
    }

    #[tokio::test]
    async fn test_include_examples_true() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let input = ListSourcesInput {
            filter: None,
            include_examples: true,
            include_health: false,
        };
        let result = tool.list(input).await.unwrap();

        assert!(result.sources[0].query_examples.is_some());
        let examples = result.sources[0].query_examples.as_ref().unwrap();
        assert!(!examples.is_empty());
    }

    #[tokio::test]
    async fn test_include_examples_false() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let input = ListSourcesInput {
            filter: None,
            include_examples: false,
            include_health: false,
        };
        let result = tool.list(input).await.unwrap();

        assert!(result.sources[0].query_examples.is_none());
    }

    #[tokio::test]
    async fn test_sources_sorted_by_priority() {
        use crate::client::providers::{ArxivProvider, SemanticScholarProvider};

        let mut tool = ListSourcesTool::new();
        // ArXiv has priority 80
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));
        // Semantic Scholar has priority 90 (higher)
        tool.register_provider(Box::new(SemanticScholarProvider::new(None).unwrap()));

        let input = ListSourcesInput::default();
        let result = tool.list(input).await.unwrap();

        // Higher priority should come first
        assert_eq!(result.total, 2);
        assert!(result.sources[0].priority >= result.sources[1].priority);
    }

    #[test]
    fn test_get_source_details_success() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let details = tool.get_source_details("arxiv");
        assert!(details.is_some());

        let details = details.unwrap();
        assert_eq!(details.name, "arxiv");
        assert!(details.supports_full_text);
        assert!(details.query_examples.is_some());
    }

    #[tokio::test]
    async fn test_available_filters_always_present() {
        let tool = ListSourcesTool::new();
        let input = ListSourcesInput::default();
        let result = tool.list(input).await.unwrap();

        assert!(result.available_filters.contains(&"all".to_string()));
        assert!(result.available_filters.contains(&"full_text".to_string()));
        assert!(result.available_filters.contains(&"metadata_only".to_string()));
    }

    #[tokio::test]
    async fn test_register_multiple_providers() {
        use crate::client::providers::{
            ArxivProvider, BiorxivProvider, CrossRefProvider, MedrxivProvider,
        };

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));
        tool.register_provider(Box::new(BiorxivProvider::new().unwrap()));
        tool.register_provider(Box::new(CrossRefProvider::new(None).unwrap()));
        tool.register_provider(Box::new(MedrxivProvider::new().unwrap()));

        let input = ListSourcesInput::default();
        let result = tool.list(input).await.unwrap();

        assert_eq!(result.total, 4);
    }

    #[test]
    fn test_source_details_includes_search_types() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let details = tool.get_source_details("arxiv").unwrap();
        assert!(!details.supported_search_types.is_empty());
    }

    #[test]
    fn test_source_details_includes_query_format_help() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let details = tool.get_source_details("arxiv").unwrap();
        assert!(!details.query_format_help.is_empty());
    }

    #[test]
    fn test_provider_with_native_syntax_url() {
        use crate::client::providers::ArxivProvider;

        let mut tool = ListSourcesTool::new();
        tool.register_provider(Box::new(ArxivProvider::new().unwrap()));

        let details = tool.get_source_details("arxiv").unwrap();
        assert!(details.native_query_syntax.is_some());
        let url = details.native_query_syntax.unwrap();
        assert!(url.starts_with("http"));
    }
}
