use crate::client::providers::{SearchQuery, SearchType as ProviderSearchType};
use crate::client::{MetaSearchClient, MetaSearchConfig, MetaSearchResult, PaperMetadata};
use crate::services::CategorizationService;
// use crate::tools::command::{Command, CommandResult, ExecutionContext};
use crate::{Config, Result};
// use rmcp::tool; // Will be enabled when rmcp integration is complete
// use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
// use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};

/// Input parameters for the paper search tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchInput {
    /// Query string - can be DOI, title, or author name
    pub query: String,
    /// Type of search to perform
    #[serde(default)]
    pub search_type: SearchType,
    /// Maximum number of results to return (default: 10)
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Offset for pagination (default: 0)
    #[serde(default)]
    pub offset: u32,
    /// Optional list of primary search sources to use (provider ids)
    #[serde(default)]
    pub sources: Option<Vec<String>>,
    /// Optional list of metadata-only sources for validation/enrichment
    #[serde(default)]
    pub metadata_sources: Option<Vec<String>>,
}

/// Type of search to perform
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchType {
    /// Automatic detection based on query format
    #[default]
    Auto,
    /// Search by DOI
    Doi,
    /// Search by paper title
    Title,
    /// Search by author name
    Author,
    /// Search by combination of author and year
    AuthorYear,
    /// Search by title and abstract combined
    TitleAbstract,
}

/// Result of a paper search operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    /// Search query that was executed
    pub query: String,
    /// Type of search that was performed
    pub search_type: SearchType,
    /// List of papers found
    pub papers: Vec<PaperResult>,
    /// Total number of results available
    pub total_count: u32,
    /// Number of results returned in this response
    pub returned_count: u32,
    /// Offset used for this search
    pub offset: u32,
    /// Whether there are more results available
    pub has_more: bool,
    /// Time taken to execute the search in milliseconds
    pub search_time_ms: u64,
    /// Source mirror that provided the results
    pub source_mirror: Option<String>,
    /// Suggested category for organizing downloaded papers
    pub category: Option<String>,
    /// List of providers that were successfully queried
    pub successful_providers: Vec<String>,
    /// List of providers that failed during the search
    pub failed_providers: Vec<String>,
    /// List of metadata-only providers used for validation
    #[serde(default)]
    pub metadata_providers: Vec<String>,
    /// Details about provider errors that occurred
    pub provider_errors: HashMap<String, String>,
    /// Number of papers found per provider
    pub papers_per_provider: HashMap<String, u32>,
}

/// Individual paper result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PaperResult {
    /// Paper metadata
    #[serde(flatten)]
    pub metadata: PaperMetadata,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f64,
    /// Whether the full paper is available for download
    pub available: bool,
    /// Source where this result came from
    pub source: String,
    /// Suggested category for organizing this paper
    pub category: Option<String>,
}

/// Cache entry for search results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: SearchResult,
    timestamp: SystemTime,
    ttl: Duration,
}

impl CacheEntry {
    fn new(result: SearchResult, ttl: Duration) -> Self {
        Self {
            result,
            timestamp: SystemTime::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.timestamp.elapsed().unwrap_or(Duration::MAX) > self.ttl
    }
}

/// Paper search tool implementation
#[derive(Clone)]
pub struct SearchTool {
    meta_client: Arc<MetaSearchClient>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    config: Arc<Config>,
    categorization_service: CategorizationService,
}

impl std::fmt::Debug for SearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchTool")
            .field("meta_client", &"MetaSearchClient")
            .field("cache", &"RwLock<HashMap>")
            .field("config", &"Config")
            .field("categorization_service", &"CategorizationService")
            .finish()
    }
}

impl SearchTool {
    /// Create a new search tool with meta-search capabilities
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing paper search tool with meta-search");

        // Create meta-search client
        let meta_config = MetaSearchConfig::from_config(&config);
        let meta_client = MetaSearchClient::new((*config).clone(), meta_config).map_err(|e| {
            crate::Error::Service(format!("Failed to create meta-search client: {e}"))
        })?;

        // Create categorization service
        let categorization_service = CategorizationService::new(config.categorization.clone())
            .map_err(|e| {
                crate::Error::Service(format!("Failed to create categorization service: {e}"))
            })?;

        Ok(Self {
            meta_client: Arc::new(meta_client),
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            categorization_service,
        })
    }

    /// Execute a paper search using meta-search across multiple providers
    // #[tool] // Will be enabled when rmcp integration is complete
    #[instrument(skip(self), fields(query = %input.query, search_type = ?input.search_type))]
    pub async fn search_papers(&self, input: SearchInput) -> Result<SearchResult> {
        info!(
            "Executing meta-search: query='{}', type={:?}",
            input.query, input.search_type
        );

        // Validate input
        Self::validate_input(&input)?;

        // Check cache first
        let cache_key = Self::generate_cache_key(&input);
        if let Some(cached_result) = self.get_from_cache(&cache_key).await {
            debug!("Returning cached search result for query: {}", input.query);
            return Ok(cached_result);
        }

        // Convert our SearchType to ProviderSearchType
        let provider_search_type = Self::convert_search_type(&input.search_type);

        // Create search query for meta-search
        let search_query = SearchQuery {
            query: input.query.clone(),
            search_type: provider_search_type,
            max_results: input.limit,
            offset: input.offset,
            params: HashMap::new(),
            sources: input.sources.clone(),
            metadata_sources: input.metadata_sources.clone(),
        };

        // Execute meta-search
        let meta_result = self
            .meta_client
            .search(&search_query)
            .await
            .map_err(|e| crate::Error::Service(format!("Meta-search failed: {e}")))?;

        // Convert to our SearchResult format
        let mut result = Self::convert_meta_result_to_search_result(
            input.query.clone(),
            input.search_type.clone(),
            meta_result,
            &input,
        );

        // Add categorization if enabled and papers were found
        if self.categorization_service.is_enabled() && !result.papers.is_empty() {
            let category = self.categorize_papers(&input.query, &result.papers);
            result.category = Some(category.clone());

            // Set category for each paper result
            for paper in &mut result.papers {
                paper.category = Some(category.clone());
            }
        }

        // Cache the result
        self.cache_result(&cache_key, &result).await;

        // Enhanced logging with provider details
        info!(
            "Meta-search completed in {}ms, found {} results across {} successful providers",
            result.search_time_ms,
            result.returned_count,
            result.successful_providers.len()
        );

        if !result.successful_providers.is_empty() {
            info!(
                "✅ Successful providers: {} | Papers per provider: {:?}",
                result.successful_providers.join(", "),
                result.papers_per_provider
            );
        }

        if !result.failed_providers.is_empty() {
            warn!(
                "❌ Failed providers: {} | Errors: {:?}",
                result.failed_providers.join(", "),
                result.provider_errors
            );
        }

        Ok(result)
    }

    /// Validate search input parameters
    fn validate_input(input: &SearchInput) -> Result<()> {
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

        if input.limit == 0 || input.limit > 100 {
            return Err(crate::Error::InvalidInput {
                field: "limit".to_string(),
                reason: "Limit must be between 1 and 100".to_string(),
            });
        }

        if let Some(sources) = &input.sources {
            if sources.iter().any(|s| s.trim().is_empty()) {
                return Err(crate::Error::InvalidInput {
                    field: "sources".to_string(),
                    reason: "Source identifiers cannot be empty".to_string(),
                });
            }
        }

        if let Some(metadata_sources) = &input.metadata_sources {
            if metadata_sources.iter().any(|s| s.trim().is_empty()) {
                return Err(crate::Error::InvalidInput {
                    field: "metadata_sources".to_string(),
                    reason: "Metadata source identifiers cannot be empty".to_string(),
                });
            }
        }

        // Enhanced security validation - reject potentially malicious input
        let query_lower = input.query.to_lowercase();
        let suspicious_patterns = [
            // SQL Injection patterns
            "' or 1=1",
            "or 1=1 --",
            "' union select",
            "'; insert into",
            "' or 'x'='x",
            "'; exec ",
            "'; drop table",
            "; drop table",
            // XSS patterns
            "<script>",
            "<img src=",
            "onerror=",
            "javascript:",
            "<svg onload=",
            "';alert(",
            "<iframe src=",
            "&lt;script&gt;",
            // Command injection patterns
            "; rm -rf",
            "rm -rf /",
            "| cat /etc/passwd",
            "cat /etc/passwd",
            "| ls",
            "; ls",
            "&& rm",
            "; cat",
            "| cat",
            "&& wget",
            "`rm -rf",
            "$(rm -rf",
            "; shutdown",
            "| nc",
            "nc ",
        ];

        if input.query.contains('\0')
            || input.query.contains('\x1b')
            || input.query.contains("$(")
            || input.query.contains('`')
            || input.query.contains('|')
            || suspicious_patterns
                .iter()
                .any(|&pattern| query_lower.contains(pattern))
            || input.query.len() > 10_000_000
        {
            // 10MB limit to prevent DoS
            return Err(crate::Error::InvalidInput {
                field: "query".to_string(),
                reason: "Query contains potentially malicious content or is too large".to_string(),
            });
        }

        Ok(())
    }

    /// Generate cache key for search input
    fn generate_cache_key(input: &SearchInput) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            input.query.to_lowercase(),
            serde_json::to_string(&input.search_type).unwrap_or_default(),
            input.limit,
            input.offset,
            Self::normalize_sources_for_cache(&input.sources),
            Self::normalize_sources_for_cache(&input.metadata_sources)
        )
    }

    fn normalize_sources_for_cache(sources: &Option<Vec<String>>) -> String {
        sources.as_ref().map_or_else(
            || "default".to_string(),
            |list| {
                let mut normalized: Vec<String> = list
                    .iter()
                    .map(|s| s.trim().to_lowercase().replace([' ', '-'], "_"))
                    .collect();
                normalized.sort();
                normalized.join("|")
            },
        )
    }

    /// Get result from cache
    async fn get_from_cache(&self, cache_key: &str) -> Option<SearchResult> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(cache_key) {
            if !entry.is_expired() {
                let result = entry.result.clone();
                drop(cache);
                return Some(result);
            }
        }
        drop(cache);
        None
    }

    /// Cache search result
    async fn cache_result(&self, cache_key: &str, result: &SearchResult) {
        let mut cache = self.cache.write().await;
        let ttl = Duration::from_secs(self.config.research_source.timeout_secs * 10); // Cache for 10x timeout
        cache.insert(cache_key.to_string(), CacheEntry::new(result.clone(), ttl));

        // Simple cache cleanup - remove expired entries
        cache.retain(|_, entry| !entry.is_expired());

        debug!("Cached search result, cache size: {}", cache.len());
    }

    /// Clear cache (useful for testing)
    #[allow(dead_code)]
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
        debug!("Search cache cleared");
    }

    /// Get cache statistics
    #[allow(dead_code)]
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total = cache.len();
        let expired = cache.values().filter(|entry| entry.is_expired()).count();
        drop(cache);
        (total, expired)
    }

    /// Convert our `SearchType` to provider `SearchType`
    const fn convert_search_type(search_type: &SearchType) -> ProviderSearchType {
        match search_type {
            SearchType::Auto => ProviderSearchType::Auto,
            SearchType::Doi => ProviderSearchType::Doi,
            SearchType::Title => ProviderSearchType::Title,
            SearchType::TitleAbstract => ProviderSearchType::TitleAbstract,
            SearchType::Author => ProviderSearchType::Author,
            SearchType::AuthorYear => ProviderSearchType::Keywords, // Fallback to keywords
        }
    }

    /// Convert `MetaSearchResult` to our `SearchResult` format
    fn convert_meta_result_to_search_result(
        query: String,
        search_type: SearchType,
        meta_result: MetaSearchResult,
        input: &SearchInput,
    ) -> SearchResult {
        // Convert papers to PaperResult format
        let papers: Vec<PaperResult> = meta_result
            .papers
            .into_iter()
            .enumerate()
            .map(|(index, paper)| {
                // Determine source (prefer the first source that provided this paper)
                let source = meta_result
                    .by_source
                    .iter()
                    .find(|(_, papers)| {
                        papers
                            .iter()
                            .any(|p| p.doi == paper.doi || p.title == paper.title)
                    })
                    .map_or_else(|| "Unknown".to_string(), |(source, _)| source.clone());

                PaperResult {
                    metadata: paper,
                    #[allow(clippy::cast_precision_loss)]
                    relevance_score: (index as f64).mul_add(-0.01, 1.0), // Simple scoring based on order
                    available: true, // Assume available since providers returned them
                    source,
                    category: None, // Will be set later by categorization
                }
            })
            .collect();

        let returned_count = u32::try_from(papers.len()).unwrap_or(u32::MAX);

        // Create source summary (exclude metadata-only providers)
        let search_sources: Vec<String> = meta_result
            .by_source
            .keys()
            .filter(|s| !meta_result.metadata_only_sources.contains(*s))
            .cloned()
            .collect();

        let source_summary = if search_sources.len() > 1 {
            format!("Multiple sources: {}", search_sources.join(", "))
        } else {
            search_sources
                .get(0)
                .cloned()
                .unwrap_or_else(|| "No sources".to_string())
        };

        // Extract provider information
        let metadata_providers: Vec<String> =
            meta_result.metadata_only_sources.iter().cloned().collect();

        let successful_providers: Vec<String> = meta_result
            .by_source
            .keys()
            .filter(|provider| !meta_result.metadata_only_sources.contains(*provider))
            .cloned()
            .collect();
        let failed_providers: Vec<String> = meta_result.provider_errors.keys().cloned().collect();
        let papers_per_provider: HashMap<String, u32> = meta_result
            .by_source
            .iter()
            .filter(|(provider, _)| !meta_result.metadata_only_sources.contains(*provider))
            .map(|(provider, papers)| (provider.clone(), u32::try_from(papers.len()).unwrap_or(0)))
            .collect();

        SearchResult {
            query,
            search_type,
            papers,
            total_count: returned_count, // We don't have true total from meta-search
            returned_count,
            offset: input.offset,
            has_more: returned_count >= input.limit, // Estimate based on limit
            search_time_ms: u64::try_from(meta_result.total_search_time.as_millis())
                .unwrap_or(u64::MAX),
            source_mirror: Some(source_summary),
            category: None, // Will be set later by categorization
            successful_providers,
            failed_providers,
            metadata_providers,
            provider_errors: meta_result.provider_errors,
            papers_per_provider,
        }
    }

    /// Categorize papers using LLM based on query and abstracts
    fn categorize_papers(&self, query: &str, papers: &[PaperResult]) -> String {
        info!("Categorizing papers for query: '{}'", query);

        // Extract paper metadata for categorization
        let paper_metadata: Vec<PaperMetadata> =
            papers.iter().map(|paper| paper.metadata.clone()).collect();

        // Generate categorization prompt
        let prompt = self
            .categorization_service
            .generate_category_prompt(query, &paper_metadata);

        debug!("Categorization prompt generated ({} chars)", prompt.len());

        // Since this is an MCP server, the LLM calling us would need to categorize
        // For now, we'll use a simple heuristic categorization based on the query
        // In the future, this could be replaced with an actual LLM call
        let category_response = self.simple_heuristic_categorization(query, &paper_metadata);

        // Sanitize the category
        let category = self
            .categorization_service
            .sanitize_category(&category_response);

        info!("Categorized papers as: '{}'", category);
        category
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
                // Safe string truncation respecting character boundaries
                let truncated = if abstract_text.len() <= 200 {
                    abstract_text.clone()
                } else {
                    let mut end = 200;
                    while end > 0 && !abstract_text.is_char_boundary(end) {
                        end -= 1;
                    }
                    abstract_text[..end].to_string()
                };
                keywords.push(truncated.to_lowercase());
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
}

/// Default limit for search results
const fn default_limit() -> u32 {
    10
}

// Command trait implementation for SearchTool (temporarily disabled)
/*
#[async_trait]
impl Command for SearchTool {
    fn name(&self) -> &'static str {
        "search_papers"
    }

    fn description(&self) -> &'static str {
        "Search for academic papers using DOI, title, author, or keywords across multiple providers"
    }

    fn input_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        let schema = schema_for!(SearchInput);
        serde_json::to_value(schema).unwrap_or_else(|e| {
            tracing::error!("Failed to serialize SearchInput schema: {}", e);
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query string - DOI, title, or author"},
                    "search_type": {"type": "string", "enum": ["auto", "doi", "title", "author", "author_year"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                    "offset": {"type": "integer", "minimum": 0, "default": 0}
                },
                "required": ["query"]
            })
        })
    }

    fn output_schema(&self) -> serde_json::Value {
        use schemars::schema_for;
        let schema = schema_for!(SearchResult);
        serde_json::to_value(schema).unwrap_or_else(|e| {
            tracing::error!("Failed to serialize SearchResult schema: {}", e);
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "papers": {"type": "array"},
                    "total_count": {"type": "integer"},
                    "returned_count": {"type": "integer"},
                    "has_more": {"type": "boolean"}
                }
            })
        })
    }

    async fn execute(
        &self,
        input: serde_json::Value,
        context: ExecutionContext,
    ) -> Result<CommandResult> {
        let start_time = SystemTime::now();

        // Deserialize input
        let search_input: SearchInput =
            serde_json::from_value(input).map_err(|e| crate::Error::InvalidInput {
                field: "input".to_string(),
                reason: format!("Invalid search input: {e}"),
            })?;

        // Execute the search
        let search_result = self.search_papers(search_input).await?;

        let duration = start_time.elapsed().unwrap_or(Duration::ZERO);

        // Create successful command result
        CommandResult::success(
            context.request_id,
            self.name().to_string(),
            search_result,
            duration,
        )
    }

    async fn validate_input(&self, input: &serde_json::Value) -> Result<()> {
        // Try to deserialize to check basic structure
        let search_input: SearchInput =
            serde_json::from_value(input.clone()).map_err(|e| crate::Error::InvalidInput {
                field: "input".to_string(),
                reason: format!("Invalid input structure: {e}"),
            })?;

        // Use existing validation logic
        Self::validate_input(&search_input)
    }

    fn estimated_duration(&self) -> Duration {
        Duration::from_secs(5) // Search typically takes 1-5 seconds
    }

    fn is_concurrent_safe(&self) -> bool {
        true // Search operations are safe to run concurrently
    }

    fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "validation" | "timeout" | "metadata" | "caching" => true,
            _ => false,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::providers::SearchType as ProviderSearchType;
    use crate::client::{MetaSearchResult, PaperMetadata};
    use crate::config::{Config, ResearchSourceConfig};
    use std::collections::{HashMap, HashSet};

    fn create_test_config() -> Arc<Config> {
        let mut config = Config::default();
        config.research_source = ResearchSourceConfig {
            endpoints: vec!["https://sci-hub.se".to_string()],
            rate_limit_per_sec: 1,
            timeout_secs: 30,
            provider_timeout_secs: 60,
            max_retries: 2,
        };
        Arc::new(config)
    }

    fn create_test_search_tool() -> Result<SearchTool> {
        let config = create_test_config();
        SearchTool::new(config)
    }

    #[test]
    fn test_search_input_validation() {
        // Empty query should fail
        let empty_input = SearchInput {
            query: "".to_string(),
            search_type: SearchType::Auto,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };
        assert!(SearchTool::validate_input(&empty_input).is_err());

        // Too long query should fail
        let long_input = SearchInput {
            query: "a".repeat(1001),
            search_type: SearchType::Auto,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };
        assert!(SearchTool::validate_input(&long_input).is_err());

        // Invalid limit should fail
        let invalid_limit = SearchInput {
            query: "test".to_string(),
            search_type: SearchType::Auto,
            limit: 0,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };
        assert!(SearchTool::validate_input(&invalid_limit).is_err());

        // Valid input should pass
        let valid_input = SearchInput {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Auto,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };
        assert!(SearchTool::validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_search_type_conversion() {
        // Test conversion from our SearchType to ProviderSearchType
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::Auto),
            ProviderSearchType::Auto
        ));
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::Doi),
            ProviderSearchType::Doi
        ));
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::Title),
            ProviderSearchType::Title
        ));
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::Author),
            ProviderSearchType::Author
        ));
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::AuthorYear),
            ProviderSearchType::Keywords
        ));
        assert!(matches!(
            SearchTool::convert_search_type(&SearchType::TitleAbstract),
            ProviderSearchType::TitleAbstract
        ));
    }

    #[test]
    fn test_cache_key_generation() {
        let input = SearchInput {
            query: "Test Query".to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };

        let key1 = SearchTool::generate_cache_key(&input);
        let key2 = SearchTool::generate_cache_key(&input);
        assert_eq!(key1, key2);

        // Different queries should generate different keys
        let mut input2 = input.clone();
        input2.query = "Different Query".to_string();
        let key3 = SearchTool::generate_cache_key(&input2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_convert_meta_result_to_search_result() {
        let metadata = PaperMetadata {
            doi: "10.1038/test".to_string(),
            title: Some("Test Paper".to_string()),
            authors: vec!["Author 1".to_string()],
            journal: Some("Test Journal".to_string()),
            year: Some(2023),
            abstract_text: None,
            pdf_url: None,
            file_size: None,
        };

        let mut by_source = HashMap::new();
        by_source.insert("test_source".to_string(), vec![metadata.clone()]);

        let meta_result = MetaSearchResult {
            papers: vec![metadata],
            by_source,
            metadata_only_sources: HashSet::new(),
            total_search_time: Duration::from_millis(100),
            successful_providers: 1,
            failed_providers: 0,
            provider_errors: HashMap::new(),
            provider_metadata: HashMap::new(),
        };

        let input = SearchInput {
            query: "test query".to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };

        let result = SearchTool::convert_meta_result_to_search_result(
            "test query".to_string(),
            SearchType::Title,
            meta_result,
            &input,
        );

        assert_eq!(result.query, "test query");
        assert!(matches!(result.search_type, SearchType::Title));
        assert_eq!(result.returned_count, 1);
        assert_eq!(result.search_time_ms, 100);
        assert!(!result.has_more);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let tool = create_test_search_tool().unwrap();

        let input = SearchInput {
            query: "test".to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };

        let result = SearchResult {
            query: "test".to_string(),
            search_type: SearchType::Title,
            papers: vec![],
            total_count: 0,
            returned_count: 0,
            offset: 0,
            has_more: false,
            search_time_ms: 100,
            source_mirror: None,
            category: None,
            successful_providers: vec![],
            failed_providers: vec![],
            metadata_providers: vec![],
            provider_errors: HashMap::new(),
            papers_per_provider: HashMap::new(),
        };

        let cache_key = SearchTool::generate_cache_key(&input);

        // Initially should be empty
        assert!(tool.get_from_cache(&cache_key).await.is_none());

        // After caching should be available
        tool.cache_result(&cache_key, &result).await;
        let cached = tool.get_from_cache(&cache_key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().query, "test");

        // Clear cache
        tool.clear_cache().await;
        assert!(tool.get_from_cache(&cache_key).await.is_none());
    }
}
