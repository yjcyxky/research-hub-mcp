//! # Provider Traits Module
//!
//! This module defines the core traits and types for academic source providers.
//! It provides a unified interface for searching across different academic databases
//! and repositories such as ArXiv, CrossRef, PubMed, and others.
//!
//! ## Key Components
//!
//! - [`SourceProvider`]: Main trait that all providers must implement
//! - [`SearchQuery`]: Represents a search request with parameters
//! - [`ProviderResult`]: Standardized result format from all providers
//! - [`ProviderError`]: Comprehensive error handling for provider operations
//!
//! ## Provider Implementation Guide
//!
//! To implement a new academic source provider:
//!
//! ```no_run
//! use async_trait::async_trait;
//! use knowledge_accumulator_mcp::client::providers::{SourceProvider, SearchQuery, SearchContext, ProviderResult, ProviderError, SearchType};
//! use std::time::Duration;
//!
//! struct MyProvider {
//!     client: reqwest::Client,
//! }
//!
//! #[async_trait]
//! impl SourceProvider for MyProvider {
//!     fn name(&self) -> &'static str { "my_provider" }
//!     fn priority(&self) -> u8 { 50 }
//!     fn base_delay(&self) -> Duration { Duration::from_millis(500) }
//!
//!     fn supported_search_types(&self) -> Vec<SearchType> {
//!         vec![SearchType::Title, SearchType::Keywords]
//!     }
//!
//!     async fn search(&self, query: &SearchQuery, context: &SearchContext) -> Result<ProviderResult, ProviderError> {
//!         // Implementation here
//!         todo!()
//!     }
//!
//!     async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
//!         // Health check implementation
//!         Ok(true)
//!     }
//! }
//! ```

use crate::client::PaperMetadata;
use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Represents a search query with parameters for academic source providers.
///
/// This structure encapsulates all information needed to perform a search across
/// different academic databases and repositories. Providers should interpret
/// these parameters according to their capabilities and API requirements.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Query string
    pub query: String,
    /// Search type hint
    pub search_type: SearchType,
    /// Maximum results to return
    pub max_results: u32,
    /// Search offset for pagination
    pub offset: u32,
    /// Additional provider-specific parameters
    pub params: HashMap<String, String>,
    /// Explicit list of search-capable sources to use (provider ids). `None` => defaults.
    pub sources: Option<Vec<String>>,
    /// Metadata-only sources to validate/enrich results. `None` => defaults.
    pub metadata_sources: Option<Vec<String>>,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            query: String::new(),
            search_type: SearchType::Auto,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        }
    }
}

/// Specifies the type of search to perform across academic sources.
///
/// Different providers may support different search types. The meta-search client
/// uses this information to route queries to appropriate providers and optimize
/// search strategies.
///
/// # Provider Support
///
/// Not all providers support all search types. Common provider capabilities:
/// - **ArXiv**: Title, Author, Keywords, Subject
/// - **CrossRef**: DOI, Title, Author
/// - **PubMed**: Title, Author, Keywords, Subject
/// - **Semantic Scholar**: Title, Author, Keywords
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchType {
    /// Automatic detection based on query characteristics.
    ///
    /// The provider or meta-search client will analyze the query string
    /// to determine the most appropriate search type.
    Auto,

    /// Search by Digital Object Identifier (DOI).
    ///
    /// Example: "10.1038/nature12373"
    /// This is the most precise search type for finding specific papers.
    Doi,

    /// Search by paper title.
    ///
    /// Example: "Attention Is All You Need"
    /// Useful for finding papers when you know the exact or partial title.
    Title,

    /// Search by author name(s).
    ///
    /// Example: "Geoffrey Hinton" or "Hinton, G."
    /// May return multiple papers by the same author.
    Author,

    /// Search by title and abstract combined.
    ///
    /// This is useful for broader semantic searches when exact title is unknown.
    TitleAbstract,

    /// Search by keywords or free text.
    ///
    /// Example: "machine learning transformer neural networks"
    /// Broad search across paper content, abstracts, and metadata.
    Keywords,

    /// Search by subject classification or category.
    ///
    /// Example: "cs.AI" (Computer Science - Artificial Intelligence)
    /// Uses provider-specific subject taxonomies.
    Subject,
}

/// Context for search operations
#[derive(Debug, Clone)]
pub struct SearchContext {
    /// Timeout for the search operation
    pub timeout: Duration,
    /// User agent string
    pub user_agent: String,
    /// Rate limit constraints
    pub rate_limit: Option<Duration>,
    /// Additional headers
    pub headers: HashMap<String, String>,
}

/// Result from a source provider
#[derive(Debug, Clone)]
pub struct ProviderResult {
    /// Papers found by the provider
    pub papers: Vec<PaperMetadata>,
    /// Source that provided the results
    pub source: String,
    /// Total number of results available (if known)
    pub total_available: Option<u32>,
    /// Time taken to execute the search
    pub search_time: Duration,
    /// Whether there are more results available
    pub has_more: bool,
    /// Provider-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Errors that can occur during provider operations
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout occurred")]
    Timeout,

    #[error("Provider error: {0}")]
    Other(String),
}

/// Core trait for academic source providers.
///
/// This trait defines the interface that all academic source providers must implement
/// to participate in the meta-search system. Providers can be academic databases,
/// repositories, search engines, or any other source of academic papers.
///
/// # Implementation Requirements
///
/// Implementors must:
/// - Be thread-safe (`Send + Sync`)
/// - Handle errors gracefully and return appropriate `ProviderError` variants
/// - Respect rate limits and implement appropriate delays
/// - Validate input parameters and return meaningful error messages
/// - Support health checking for service monitoring
///
/// # Performance Considerations
///
/// - Use connection pooling for HTTP clients
/// - Implement proper timeout handling
/// - Cache responses when appropriate
/// - Use circuit breakers for external service calls
/// - Handle partial failures gracefully
///
/// # Example Implementation
///
/// ```no_run
/// use async_trait::async_trait;
/// use knowledge_accumulator_mcp::client::providers::{SourceProvider, SearchQuery, SearchContext, ProviderResult, ProviderError, SearchType};
/// use std::time::Duration;
///
/// struct ExampleProvider {
///     client: reqwest::Client,
///     api_key: Option<String>,
/// }
///
/// #[async_trait]
/// impl SourceProvider for ExampleProvider {
///     fn name(&self) -> &'static str { "example" }
///     fn priority(&self) -> u8 { 50 }
///     fn base_delay(&self) -> Duration { Duration::from_millis(500) }
///
///     fn supported_search_types(&self) -> Vec<SearchType> {
///         vec![SearchType::Title, SearchType::Keywords, SearchType::Author]
///     }
///
///     async fn search(&self, query: &SearchQuery, context: &SearchContext) -> Result<ProviderResult, ProviderError> {
///         // Validate query
///         if query.query.is_empty() {
///             return Err(ProviderError::InvalidQuery("Empty query".to_string()));
///         }
///
///         // Build API URL (example implementation)
///         let url = format!("https://api.example.com/search?q={}",
///                          urlencoding::encode(&query.query));
///
///         // Perform search with timeout
///         let response = tokio::time::timeout(context.timeout,
///             self.client.get(&url).send()).await
///             .map_err(|_| ProviderError::Timeout)?
///             .map_err(|e| ProviderError::Network(e.to_string()))?;
///
///         // Parse and return results (simplified example)
///         if response.status().is_success() {
///             Ok(ProviderResult {
///                 papers: vec![], // Would parse actual response
///                 source: self.name().to_string(),
///                 total_available: Some(0),
///                 search_time: Duration::from_millis(100),
///                 has_more: false,
///                 metadata: std::collections::HashMap::new(),
///             })
///         } else {
///             Err(ProviderError::ServiceUnavailable(response.status().to_string()))
///         }
///     }
///
///     async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
///         // Implement health check logic
///         Ok(true)
///     }
/// }
/// ```
#[async_trait]
pub trait SourceProvider: Send + Sync {
    /// Returns the unique identifier for this provider.
    ///
    /// This should be a short, lowercase string that uniquely identifies the provider
    /// across the system. Used for logging, configuration, and result attribution.
    ///
    /// # Examples
    /// - "arxiv" for ArXiv.org
    /// - "crossref" for CrossRef API
    /// - "pubmed" for PubMed Central
    fn name(&self) -> &'static str;

    /// Returns the priority level for this provider (0-255, higher = more priority).
    ///
    /// The meta-search client uses this to determine the order in which providers
    /// are queried. Higher priority providers are queried first and may influence
    /// result ranking.
    ///
    /// # Priority Guidelines
    /// - 200+: Authoritative sources (CrossRef, official databases)
    /// - 150-199: High-quality aggregators (Semantic Scholar)
    /// - 100-149: Specialized repositories (ArXiv, PubMed)
    /// - 50-99: General search engines
    /// - 0-49: Fallback sources
    fn priority(&self) -> u8 {
        50 // Default medium priority
    }

    /// Returns the base delay between requests to this provider.
    ///
    /// This is used for rate limiting to ensure respectful usage of external APIs.
    /// The actual delay may be adjusted based on recent response times and errors.
    fn base_delay(&self) -> Duration {
        Duration::from_millis(1000) // Default 1 second
    }

    /// Returns the search types supported by this provider.
    ///
    /// The meta-search client uses this information to route queries to appropriate
    /// providers and optimize search strategies.
    fn supported_search_types(&self) -> Vec<SearchType>;

    /// Returns a human-readable description of this provider.
    ///
    /// Used for logging, debugging, and user interfaces to explain what this
    /// provider offers and its capabilities.
    fn description(&self) -> &'static str {
        "Academic source provider"
    }

    /// Returns whether this provider supports full-text PDF access.
    ///
    /// Used by the meta-search client to prioritize providers when PDF access
    /// is specifically requested or needed.
    fn supports_full_text(&self) -> bool {
        false
    }

    /// Performs a search using this provider.
    ///
    /// This is the core method that executes a search against the provider's API
    /// or database and returns structured results.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query with parameters
    /// * `context` - Additional context like timeouts and headers
    ///
    /// # Returns
    ///
    /// A `ProviderResult` containing papers and metadata, or a `ProviderError`.
    ///
    /// # Error Handling
    ///
    /// Implementations should map provider-specific errors to appropriate
    /// `ProviderError` variants:
    /// - Network failures → `ProviderError::Network`
    /// - Parse errors → `ProviderError::Parse`
    /// - Rate limits → `ProviderError::RateLimit`
    /// - Invalid queries → `ProviderError::InvalidQuery`
    ///
    /// # Performance
    ///
    /// - Respect the timeout specified in `context.timeout`
    /// - Use connection pooling for HTTP requests
    /// - Implement retries for transient failures
    /// - Cache responses when appropriate
    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError>;

    /// Get paper metadata by DOI (if supported)
    async fn get_by_doi(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        let query = SearchQuery {
            query: doi.to_string(),
            search_type: SearchType::Doi,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let result = self.search(&query, context).await?;
        Ok(result.papers.into_iter().next())
    }

    /// Health check for the provider
    async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
        // Default implementation: try a simple search
        let query = SearchQuery {
            query: "test".to_string(),
            search_type: SearchType::Keywords,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        match self.search(&query, context).await {
            Ok(_) => Ok(true),
            Err(ProviderError::RateLimit) => Ok(true), // Rate limit means service is up
            Err(_) => Ok(false),
        }
    }

    /// Attempt to get a direct PDF download URL for a DOI
    /// This is called when standard search doesn't return a PDF URL
    /// Providers can override this to implement specialized PDF retrieval
    async fn get_pdf_url(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<String>, ProviderError> {
        // Default implementation: try search and extract PDF URL
        let result = self.get_by_doi(doi, context).await?;
        Ok(result.and_then(|paper| paper.pdf_url))
    }
}
