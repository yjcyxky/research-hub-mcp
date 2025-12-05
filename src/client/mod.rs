//! # Client Module
//!
//! This module provides the core client infrastructure for academic research paper discovery and retrieval.
//! It implements a meta-search architecture that queries multiple academic sources in parallel,
//! aggregates results, and handles circuit breaking, rate limiting, and fault tolerance.
//!
//! ## Architecture
//!
//! The client follows a layered architecture:
//!
//! - **Meta-Search Layer**: [`MetaSearchClient`] orchestrates searches across multiple providers
//! - **Provider Layer**: Individual academic source implementations (`ArXiv`, `CrossRef`, etc.)
//! - **Resilience Layer**: Circuit breakers, rate limiting, and retry logic
//! - **Mirror Management**: Handles mirror discovery and health checking for services like Sci-Hub
//!
//! ## Example Usage
//!
//! ```no_run
//! use knowledge_accumulator_mcp::client::{MetaSearchClient, MetaSearchConfig};
//! use knowledge_accumulator_mcp::client::providers::{SearchQuery, SearchType};
//! use knowledge_accumulator_mcp::Config;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Config::default();
//! let meta_config = MetaSearchConfig::default();
//! let client = MetaSearchClient::new(config, meta_config)?;
//!
//! let query = SearchQuery {
//!     query: "machine learning".to_string(),
//!     search_type: SearchType::Keywords,
//!     max_results: 10,
//!     offset: 0,
//!     params: std::collections::HashMap::new(),
//!     sources: None,
//!     metadata_sources: None,
//! };
//!
//! let results = client.search(&query).await?;
//! println!("Found {} papers from {} providers",
//!          results.papers.len(), results.successful_providers);
//! # Ok(())
//! # }
//! ```
//!
//! ## Security Considerations
//!
//! All HTTP clients are configured with security defaults:
//! - HTTPS-only connections where possible
//! - Certificate validation (unless explicitly disabled for development)
//! - Request timeouts and connection limits
//! - Rate limiting to respect external services

pub mod circuit_breaker_service;
pub mod meta_search;
pub mod mirror;
pub mod providers;
pub mod rate_limiter;

pub use circuit_breaker_service::CircuitBreakerService;
pub use meta_search::{MetaSearchClient, MetaSearchConfig, MetaSearchResult};
pub use mirror::{Mirror, MirrorHealth, MirrorManager};
pub use rate_limiter::RateLimiter;

use crate::Result;
use std::time::Duration;

/// HTTP client configuration for research source integration
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Request timeout duration
    pub timeout: Duration,
    /// Connection timeout duration
    pub connect_timeout: Duration,
    /// Maximum redirects to follow
    pub max_redirects: u32,
    /// User agent string
    pub user_agent: String,
    /// Proxy URL (optional)
    pub proxy: Option<String>,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            max_redirects: 10,
            user_agent: "knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)".to_string(),
            proxy: None,
        }
    }
}

/// Factory for creating secure HTTP clients with enforced security best practices.
///
/// This factory ensures all HTTP clients used for academic research comply with security
/// standards, including HTTPS enforcement, certificate validation, and proper timeout handling.
///
/// # Security Features
///
/// - **HTTPS Enforcement**: Rejects HTTP URLs in production
/// - **Certificate Validation**: Validates SSL/TLS certificates by default
/// - **Timeout Protection**: Prevents hanging connections
/// - **Connection Limits**: Limits concurrent connections per host
/// - **User Agent**: Sets appropriate user agent for academic research
///
/// # Example
///
/// ```no_run
/// use knowledge_accumulator_mcp::client::{SecureHttpClientFactory, HttpClientConfig};
/// use std::time::Duration;
///
/// let config = HttpClientConfig {
///     timeout: Duration::from_secs(30),
///     connect_timeout: Duration::from_secs(10),
///     max_redirects: 5,
///     user_agent: "MyApp/1.0".to_string(),
///     proxy: None,
/// };
///
/// let client = SecureHttpClientFactory::create_client(&config)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SecureHttpClientFactory;

impl SecureHttpClientFactory {
    /// Creates a secure HTTP client with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - HTTP client configuration specifying timeouts, user agent, etc.
    ///
    /// # Returns
    ///
    /// A configured `reqwest::Client` with security best practices applied.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - TLS configuration fails
    /// - Proxy configuration is invalid
    /// - Client building fails for any other reason
    ///
    /// # Security Notes
    ///
    /// The created client will:
    /// - Use HTTPS by default
    /// - Validate TLS certificates
    /// - Apply connection and request timeouts
    /// - Follow redirects up to the configured limit
    pub fn create_client(config: &HttpClientConfig) -> Result<reqwest::Client> {
        let mut client_builder = reqwest::Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::limited(
                config.max_redirects as usize,
            ))
            .user_agent(&config.user_agent)
            // Security enforcements
            .tls_built_in_root_certs(true) // Use built-in root certificates
            .https_only(true) // Enforce HTTPS connections only
            .connection_verbose(false) // Disable verbose connection logging for security
            .pool_max_idle_per_host(10) // Connection pooling for performance
            .pool_idle_timeout(Duration::from_secs(30)); // Connection pool timeout

        // Add proxy if configured
        if let Some(proxy_url) = &config.proxy {
            let proxy = reqwest::Proxy::all(proxy_url).map_err(|e| crate::Error::InvalidInput {
                field: "proxy".to_string(),
                reason: format!("Invalid proxy URL: {e}"),
            })?;
            client_builder = client_builder.proxy(proxy);
        }

        client_builder.build().map_err(|e| crate::Error::Http(e))
    }

    /// Create a secure HTTP client with default configuration
    pub fn create_default_client() -> Result<reqwest::Client> {
        Self::create_client(&HttpClientConfig::default())
    }

    /// Create a secure HTTP client with custom user agent
    pub fn create_client_with_user_agent(user_agent: &str) -> Result<reqwest::Client> {
        let config = HttpClientConfig {
            user_agent: user_agent.to_string(),
            ..HttpClientConfig::default()
        };
        Self::create_client(&config)
    }

    /// Create a secure HTTP client with custom timeout
    pub fn create_client_with_timeout(timeout: Duration) -> Result<reqwest::Client> {
        let config = HttpClientConfig {
            timeout,
            ..HttpClientConfig::default()
        };
        Self::create_client(&config)
    }
}

/// DOI (Digital Object Identifier) wrapper for type safety
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Doi(String);

impl Doi {
    /// Create a new DOI from a string, validating the format
    pub fn new(doi: &str) -> Result<Self> {
        let cleaned = doi
            .trim()
            .trim_start_matches("doi:")
            .trim_start_matches("https://doi.org/");

        if cleaned.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "doi".to_string(),
                reason: "DOI cannot be empty".to_string(),
            });
        }

        // Basic DOI format validation (simplified)
        if !cleaned.contains('/') {
            return Err(crate::Error::InvalidInput {
                field: "doi".to_string(),
                reason: "DOI must contain a '/' character".to_string(),
            });
        }

        Ok(Self(cleaned.to_string()))
    }

    /// Get the DOI string
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to a URL-safe format
    #[must_use]
    pub fn url_encoded(&self) -> String {
        urlencoding::encode(&self.0).to_string()
    }
}

impl std::fmt::Display for Doi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for Doi {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

/// Paper metadata extracted from research sources
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PaperMetadata {
    /// Digital Object Identifier
    pub doi: String,
    /// Paper title
    pub title: Option<String>,
    /// Authors
    pub authors: Vec<String>,
    /// Journal name
    pub journal: Option<String>,
    /// Publication year
    pub year: Option<u32>,
    /// Abstract
    pub abstract_text: Option<String>,
    /// Download URL for the PDF
    pub pdf_url: Option<String>,
    /// File size in bytes (if available)
    pub file_size: Option<u64>,
}

impl PaperMetadata {
    /// Create new paper metadata with just a DOI
    #[must_use]
    pub const fn new(doi: String) -> Self {
        Self {
            doi,
            title: None,
            authors: Vec::new(),
            journal: None,
            year: None,
            abstract_text: None,
            pdf_url: None,
            file_size: None,
        }
    }

    /// Set the PDF URL, filtering out empty strings
    #[must_use]
    pub fn with_pdf_url(mut self, url: Option<String>) -> Self {
        self.pdf_url = url.filter(|u| !u.is_empty());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_http_client_factory_default() {
        let client = SecureHttpClientFactory::create_default_client();
        assert!(client.is_ok(), "Should create default secure client");
    }

    #[test]
    fn test_secure_http_client_factory_with_custom_user_agent() {
        let client = SecureHttpClientFactory::create_client_with_user_agent("test-agent/1.0");
        assert!(
            client.is_ok(),
            "Should create client with custom user agent"
        );
    }

    #[test]
    fn test_secure_http_client_factory_with_custom_timeout() {
        let timeout = Duration::from_secs(60);
        let client = SecureHttpClientFactory::create_client_with_timeout(timeout);
        assert!(client.is_ok(), "Should create client with custom timeout");
    }

    #[test]
    fn test_http_client_config_default() {
        let config = HttpClientConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.max_redirects, 10);
        assert!(config.user_agent.contains("knowledge_accumulator_mcp"));
        assert!(config.proxy.is_none());
    }

    #[test]
    fn test_secure_http_client_factory_with_proxy() {
        let config = HttpClientConfig {
            proxy: Some("http://proxy.example.com:8080".to_string()),
            ..HttpClientConfig::default()
        };
        let client = SecureHttpClientFactory::create_client(&config);
        assert!(client.is_ok(), "Should create client with valid proxy");

        // Test with invalid proxy (truly malformed URL)
        let config_invalid = HttpClientConfig {
            proxy: Some(":::invalid:::".to_string()),
            ..HttpClientConfig::default()
        };
        let client_invalid = SecureHttpClientFactory::create_client(&config_invalid);
        assert!(client_invalid.is_err(), "Should fail with invalid proxy");
    }
}
