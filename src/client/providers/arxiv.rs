use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::circuit_breaker_service::CircuitBreakerService;
use crate::client::rate_limiter::ProviderRateLimiter;
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use url::Url;

/// arXiv API provider for academic papers
pub struct ArxivProvider {
    client: Client,
    base_url: String,
    rate_limiter: Arc<Mutex<Option<ProviderRateLimiter>>>,
    circuit_breaker_service: Arc<CircuitBreakerService>,
}

impl ArxivProvider {
    /// Create a new arXiv provider
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Other(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://export.arxiv.org/api/query".to_string(),
            rate_limiter: Arc::new(Mutex::new(None)),
            circuit_breaker_service: Arc::new(CircuitBreakerService::new()),
        })
    }

    /// Initialize rate limiter with configuration
    pub async fn init_rate_limiter(&self, config: &crate::config::RateLimitingConfig) {
        let limiter = ProviderRateLimiter::new("arxiv".to_string(), config);
        *self.rate_limiter.lock().await = Some(limiter);
        debug!("Initialized rate limiter for arXiv provider");
    }

    /// Resolve relative URLs to absolute URLs
    fn resolve_pdf_url(href: &str) -> Result<String, ProviderError> {
        // If already absolute, return as-is
        if href.starts_with("http://") || href.starts_with("https://") {
            return Ok(href.to_string());
        }

        // If relative, resolve against arXiv base URL
        if href.starts_with('/') {
            return Ok(format!("https://arxiv.org{href}"));
        }

        // If it's a relative path without leading slash, assume it's relative to base
        if !href.contains("://") && href.contains(".pdf") {
            return Ok(format!("https://arxiv.org/{href}"));
        }

        // If all else fails, validate it's a proper URL
        Url::parse(href)
            .map(|u| u.to_string())
            .map_err(|e| ProviderError::Parse(format!("Invalid PDF URL '{href}': {e}")))
    }

    /// Build arXiv API URL for search
    fn build_search_url(&self, query: &SearchQuery) -> Result<String, ProviderError> {
        let mut url = Url::parse(&self.base_url)
            .map_err(|e| ProviderError::Other(format!("Invalid base URL: {e}")))?;

        // Build search terms based on query type
        let search_query = match query.search_type {
            SearchType::Doi => format!("doi:{query}", query = query.query),
            SearchType::Title => format!("ti:\"{query}\"", query = query.query),
            SearchType::TitleAbstract => {
                // Title/abstract combined: use all fields
                format!("ti:\"{query}\" OR abs:\"{query}\"", query = query.query)
            }
            SearchType::Author => format!("au:\"{query}\"", query = query.query),
            SearchType::Keywords | SearchType::Auto => {
                // For auto/keywords, search in title, abstract, and comments
                format!("all:\"{query}\"", query = query.query)
            }
            SearchType::Subject => format!("cat:{query}", query = query.query),
        };

        url.query_pairs_mut()
            .append_pair("search_query", &search_query)
            .append_pair("start", &query.offset.to_string())
            .append_pair("max_results", &query.max_results.to_string())
            .append_pair("sortBy", "relevance")
            .append_pair("sortOrder", "descending");

        Ok(url.to_string())
    }

    /// Parse arXiv Atom feed response
    fn parse_response(&self, response_text: &str) -> Result<Vec<PaperMetadata>, ProviderError> {
        use roxmltree::Document;

        let doc = Document::parse(response_text)
            .map_err(|e| ProviderError::Parse(format!("Failed to parse XML: {e}")))?;

        let mut papers = Vec::new();

        // Find all entry elements
        for entry in doc.descendants().filter(|n| n.has_tag_name("entry")) {
            let mut paper = PaperMetadata {
                doi: String::new(),
                title: None,
                authors: Vec::new(),
                journal: Some("arXiv".to_string()),
                year: None,
                abstract_text: None,
                pdf_url: None,
                file_size: None,
            };

            // Extract metadata from entry
            for child in entry.children().filter(roxmltree::Node::is_element) {
                match child.tag_name().name() {
                    "id" => {
                        if let Some(id) = child.text() {
                            // Extract arXiv ID from URL
                            if let Some(arxiv_id) = id.split('/').next_back() {
                                paper.doi = format!("arXiv:{arxiv_id}");
                            }
                        }
                    }
                    "title" => {
                        if let Some(title) = child.text() {
                            paper.title = Some(title.trim().replace('\n', " ").replace("  ", " "));
                        }
                    }
                    "summary" => {
                        if let Some(summary) = child.text() {
                            paper.abstract_text =
                                Some(summary.trim().replace('\n', " ").replace("  ", " "));
                        }
                    }
                    "published" => {
                        if let Some(published) = child.text() {
                            // Parse date (format: YYYY-MM-DDTHH:MM:SSZ)
                            if let Some(year_str) = published.split('-').next() {
                                if let Ok(year) = year_str.parse::<u32>() {
                                    paper.year = Some(year);
                                }
                            }
                        }
                    }
                    "author" => {
                        // Extract author name
                        for name_elem in child.descendants().filter(|n| n.has_tag_name("name")) {
                            if let Some(author_name) = name_elem.text() {
                                paper.authors.push(author_name.trim().to_string());
                            }
                        }
                    }
                    "link" => {
                        // Look for PDF links
                        if let Some(href) = child.attribute("href") {
                            if let Some(link_type) = child.attribute("type") {
                                if link_type == "application/pdf" {
                                    // Resolve relative URLs to absolute URLs
                                    let pdf_url = Self::resolve_pdf_url(href)?;
                                    paper.pdf_url = Some(pdf_url);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            if paper.title.is_some() {
                papers.push(paper);
            }
        }

        debug!("Parsed {} papers from arXiv response", papers.len());
        Ok(papers)
    }
}

impl Default for ArxivProvider {
    fn default() -> Self {
        match Self::new() {
            Ok(provider) => provider,
            Err(_) => {
                // Fallback to a minimal client with very basic configuration
                // This should never fail under normal circumstances
                let client = Client::new();
                Self {
                    client,
                    base_url: "https://export.arxiv.org/api/query".to_string(),
                    rate_limiter: Arc::new(Mutex::new(None)),
                    circuit_breaker_service: Arc::new(CircuitBreakerService::new()),
                }
            }
        }
    }
}

#[async_trait]
impl SourceProvider for ArxivProvider {
    fn name(&self) -> &'static str {
        "arxiv"
    }

    fn description(&self) -> &'static str {
        "arXiv.org - Open access e-prints in physics, mathematics, computer science, and more"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Auto,
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Keywords,
            SearchType::Subject,
            SearchType::Doi,
        ]
    }

    fn supports_full_text(&self) -> bool {
        true // arXiv provides free PDF access
    }

    fn priority(&self) -> u8 {
        84 // High priority for CS/physics/math (after medRxiv)
    }

    fn base_delay(&self) -> Duration {
        // Deprecated: Now using configurable ProviderRateLimiter
        Duration::from_millis(500) // Fallback for legacy compatibility
    }

    fn query_format_help(&self) -> &'static str {
        r#"ArXiv supports field-specific search using prefixes:
- ti:term - Search in title
- au:name - Search by author
- abs:term - Search in abstract
- cat:cs.AI - Search by category
- all:term - Search all fields

Boolean operators: AND, OR, ANDNOT (must be uppercase)
Exact phrases: "quoted text"

Categories: cs.AI, cs.LG, cs.CL, physics.*, math.*, stat.ML, etc."#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("ti:transformer AND au:vaswani", "Title contains 'transformer' by author Vaswani"),
            ("cat:cs.LG AND abs:reinforcement", "Machine learning papers about reinforcement"),
            ("all:\"attention mechanism\"", "Exact phrase search across all fields"),
            ("au:hinton AND ti:deep", "Papers by Hinton with 'deep' in title"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("ArXiv API query format: https://arxiv.org/help/api/user-manual#query_details")
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching arXiv for: {} (type: {:?})",
            query.query, query.search_type
        );

        // Apply rate limiting
        let rate_limit_result = {
            let mut guard = self.rate_limiter.lock().await;

            if guard.is_none() {
                // Create with default config if not initialized
                let default_config = crate::config::RateLimitingConfig::default();
                *guard = Some(ProviderRateLimiter::new(
                    "arxiv".to_string(),
                    &default_config,
                ));
                debug!("Created default rate limiter for arXiv provider");
            }

            guard.as_mut().unwrap().acquire().await
        };

        rate_limit_result.map_err(|e| ProviderError::Other(format!("Rate limiting error: {e}")))?;

        // Build the search URL
        let url = self.build_search_url(query)?;
        debug!("arXiv search URL: {}", url);

        // Make the request with circuit breaker protection
        let response = self
            .circuit_breaker_service
            .call_http("arxiv", || async {
                let mut request = self.client.get(&url);

                // Add custom headers
                for (key, value) in &context.headers {
                    request = request.header(key, value);
                }

                request.timeout(context.timeout).send().await
            })
            .await
            .map_err(|e| {
                error!("arXiv request failed: {}", e);
                match e {
                    crate::Error::CircuitBreakerOpen { service } => {
                        ProviderError::ServiceUnavailable(format!(
                            "Circuit breaker open for {service}"
                        ))
                    }
                    crate::Error::NetworkTimeout { .. } => ProviderError::Timeout,
                    crate::Error::ConnectionRefused { .. } => {
                        ProviderError::Network("Connection failed".to_string())
                    }
                    crate::Error::Http(http_err) => {
                        if http_err.is_timeout() {
                            ProviderError::Timeout
                        } else if http_err.is_connect() {
                            ProviderError::Network(format!("Connection failed: {http_err}"))
                        } else {
                            ProviderError::Network(format!("Request failed: {http_err}"))
                        }
                    }
                    _ => ProviderError::Network(format!("Request failed: {e}")),
                }
            })?;

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Err(match status.as_u16() {
                429 => ProviderError::RateLimit,
                503 => ProviderError::ServiceUnavailable(
                    "arXiv service temporarily unavailable".to_string(),
                ),
                _ => ProviderError::Network(format!("HTTP {status}: {error_text}")),
            });
        }

        // Parse the response
        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        let papers = self.parse_response(&response_text)?;
        let search_time = start_time.elapsed();

        // Check if there might be more results
        let has_more = u32::try_from(papers.len()).unwrap_or(u32::MAX) >= query.max_results;

        let mut metadata = HashMap::new();
        metadata.insert("api_url".to_string(), url);
        metadata.insert("response_size".to_string(), response_text.len().to_string());

        info!(
            "arXiv search completed: {} papers found in {:?}",
            papers.len(),
            search_time
        );

        // Record response time for adaptive rate limiting
        {
            let mut guard = self.rate_limiter.lock().await;
            if let Some(limiter) = guard.as_mut() {
                limiter.record_response_time(search_time);
            }
        }

        Ok(ProviderResult {
            papers,
            source: "arXiv".to_string(),
            total_available: None, // arXiv doesn't provide total count
            search_time,
            has_more,
            metadata,
        })
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing arXiv health check");

        // Simplified health check that just tries to connect to the API
        let response = self
            .circuit_breaker_service
            .call_http("arxiv", || async {
                self.client
                    .get(&self.base_url)
                    .timeout(Duration::from_secs(10))
                    .send()
                    .await
            })
            .await;

        match response {
            Ok(resp) => {
                let healthy = resp.status().is_success();
                info!(
                    "arXiv health check: {}",
                    if healthy { "OK" } else { "Service error" }
                );
                Ok(healthy)
            }
            Err(crate::Error::CircuitBreakerOpen { .. }) => {
                warn!("arXiv health check: Circuit breaker open");
                Ok(false)
            }
            Err(e) => {
                warn!("arXiv health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_arxiv_provider_creation() {
        let provider = ArxivProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_arxiv_search_url_building() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "quantum computing".to_string(),
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(
            url.contains("all:%22quantum%20computing%22")
                || url.contains("all:quantum")
                || url.contains("quantum")
        );
        assert!(url.contains("max_results=10"));
        assert!(url.contains("start=0"));
    }

    #[test]
    fn test_arxiv_doi_search_url() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "10.1103/PhysRevA.52.R2493".to_string(),
            search_type: SearchType::Doi,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(
            url.contains("search_query=doi%3A10.1103")
                || url.contains("search_query") && url.contains("doi")
        );
    }

    #[test]
    fn test_pdf_url_resolution() {
        // Test absolute URL (should remain unchanged)
        let absolute_url = "https://arxiv.org/pdf/2409.10516v1.pdf";
        let result = ArxivProvider::resolve_pdf_url(absolute_url).unwrap();
        assert_eq!(result, absolute_url);

        // Test relative URL with leading slash
        let relative_slash = "/pdf/2409.10516v1.pdf";
        let result = ArxivProvider::resolve_pdf_url(relative_slash).unwrap();
        assert_eq!(result, "https://arxiv.org/pdf/2409.10516v1.pdf");

        // Test relative URL without leading slash
        let relative_no_slash = "pdf/2409.10516v1.pdf";
        let result = ArxivProvider::resolve_pdf_url(relative_no_slash).unwrap();
        assert_eq!(result, "https://arxiv.org/pdf/2409.10516v1.pdf");

        // Test HTTP absolute URL
        let http_url = "http://arxiv.org/pdf/2409.10516v1.pdf";
        let result = ArxivProvider::resolve_pdf_url(http_url).unwrap();
        assert_eq!(result, http_url);

        // Test invalid URL (should return error)
        let invalid_url = "not-a-valid-url";
        let result = ArxivProvider::resolve_pdf_url(invalid_url);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_format_help() {
        let provider = ArxivProvider::new().unwrap();
        let help = provider.query_format_help();

        // Should contain key prefixes
        assert!(help.contains("ti:"));
        assert!(help.contains("au:"));
        assert!(help.contains("abs:"));
        assert!(help.contains("cat:"));

        // Should contain boolean operators
        assert!(help.contains("AND"));
        assert!(help.contains("OR"));
        assert!(help.contains("NOT") || help.contains("ANDNOT"));
    }

    #[test]
    fn test_query_examples() {
        let provider = ArxivProvider::new().unwrap();
        let examples = provider.query_examples();

        // Should have at least some examples
        assert!(!examples.is_empty());

        // Each example should have query and description
        for (query, description) in &examples {
            assert!(!query.is_empty());
            assert!(!description.is_empty());
        }

        // Should include typical arXiv prefixes in examples
        let all_queries: String = examples.iter().map(|(q, _)| q.to_string()).collect();
        assert!(all_queries.contains("ti:") || all_queries.contains("au:") || all_queries.contains("cat:"));
    }

    #[test]
    fn test_native_query_syntax() {
        let provider = ArxivProvider::new().unwrap();
        let syntax = provider.native_query_syntax();

        // ArXiv should have native query syntax documentation
        assert!(syntax.is_some());

        let syntax_text = syntax.unwrap();
        // Should mention arXiv's query API
        assert!(syntax_text.contains("arXiv") || syntax_text.contains("arxiv"));
    }

    #[test]
    fn test_supported_search_types() {
        let provider = ArxivProvider::new().unwrap();
        let types = provider.supported_search_types();

        // Should support common search types
        assert!(types.contains(&SearchType::Auto));
        assert!(types.contains(&SearchType::Title));
        assert!(types.contains(&SearchType::Author));
        assert!(types.contains(&SearchType::Keywords));
    }

    #[test]
    fn test_provider_name() {
        let provider = ArxivProvider::new().unwrap();
        assert_eq!(provider.name(), "arxiv");
    }

    #[test]
    fn test_provider_description() {
        let provider = ArxivProvider::new().unwrap();
        let desc = provider.description();
        assert!(!desc.is_empty());
        assert!(desc.to_lowercase().contains("arxiv") || desc.to_lowercase().contains("preprint"));
    }

    #[test]
    fn test_provider_priority() {
        let provider = ArxivProvider::new().unwrap();
        let priority = provider.priority();
        // Priority should be in reasonable range (1-100)
        assert!(priority >= 1 && priority <= 100);
    }

    #[test]
    fn test_title_search_url_building() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "neural networks".to_string(),
            search_type: SearchType::Title,
            max_results: 5,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("ti:") || url.contains("ti%3A"));
        assert!(url.contains("neural"));
    }

    #[test]
    fn test_author_search_url_building() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "Hinton".to_string(),
            search_type: SearchType::Author,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("au:") || url.contains("au%3A"));
        assert!(url.contains("Hinton"));
    }

    #[test]
    fn test_subject_search_url_building() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "cs.AI".to_string(),
            search_type: SearchType::Subject,
            max_results: 20,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("cat:") || url.contains("cat%3A"));
        assert!(url.contains("cs.AI") || url.contains("cs%2EAI"));
    }

    #[test]
    fn test_title_abstract_search_url_building() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "transformer attention".to_string(),
            search_type: SearchType::TitleAbstract,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        // Should search in both title and abstract
        assert!(url.contains("ti:") || url.contains("ti%3A") || url.contains("abs:") || url.contains("abs%3A") || url.contains("OR"));
    }

    #[test]
    fn test_pagination_in_url() {
        let provider = ArxivProvider::new().unwrap();

        let query = SearchQuery {
            query: "test".to_string(),
            search_type: SearchType::Auto,
            max_results: 25,
            offset: 50,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("max_results=25"));
        assert!(url.contains("start=50"));
    }
}
