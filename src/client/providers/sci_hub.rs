use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::circuit_breaker_service::CircuitBreakerService;
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Sci-Hub provider for academic paper access
pub struct SciHubProvider {
    client: Client,
    mirrors: Vec<String>,
    current_mirror_index: std::sync::atomic::AtomicUsize,
    user_agents: Vec<String>,
    current_user_agent_index: std::sync::atomic::AtomicUsize,
    circuit_breaker_service: Arc<CircuitBreakerService>,
}

impl SciHubProvider {
    /// Create a new Sci-Hub provider with known mirrors
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            // Don't set user agent here - we'll rotate them per request
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        // Updated Sci-Hub mirrors (as of 2024)
        let mirrors = vec![
            "https://sci-hub.se".to_string(),
            "https://sci-hub.st".to_string(),
            "https://sci-hub.ru".to_string(),
            "https://sci-hub.tw".to_string(),
            "https://sci-hub.ren".to_string(),
            "https://sci-hub.cat".to_string(),
            "https://sci-hub.ee".to_string(),
        ];

        // Pool of realistic user agents to rotate through
        let user_agents = vec![
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36".to_string(),
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36".to_string(),
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15".to_string(),
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0".to_string(),
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36".to_string(),
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0".to_string(),
        ];

        Ok(Self {
            client,
            mirrors,
            current_mirror_index: std::sync::atomic::AtomicUsize::new(0),
            user_agents,
            current_user_agent_index: std::sync::atomic::AtomicUsize::new(0),
            circuit_breaker_service: Arc::new(CircuitBreakerService::new()),
        })
    }

    /// Get the next mirror to try (optimized with fetch_add for better performance)
    fn get_next_mirror(&self) -> String {
        let index = self
            .current_mirror_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.mirrors[index % self.mirrors.len()].clone()
    }

    /// Get the next user agent to use (optimized with fetch_add for better performance)
    fn get_next_user_agent(&self) -> String {
        let index = self
            .current_user_agent_index
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.user_agents[index % self.user_agents.len()].clone()
    }

    /// Clean DOI format for Sci-Hub
    fn clean_doi(doi: &str) -> String {
        doi.trim()
            .trim_start_matches("doi:")
            .trim_start_matches("https://doi.org/")
            .trim_start_matches("http://dx.doi.org/")
            .to_string()
    }

    /// Try to fetch paper from Sci-Hub
    async fn fetch_from_scihub(
        &self,
        identifier: &str,
        search_type: &SearchType,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        let query = match search_type {
            SearchType::Doi => Self::clean_doi(identifier),
            _ => identifier.to_string(),
        };

        // Try each mirror until one works
        for _ in 0..self.mirrors.len() {
            let mirror = self.get_next_mirror();

            match self.try_mirror(&mirror, &query).await {
                Ok(Some(metadata)) => {
                    info!("Successfully found paper on Sci-Hub mirror: {}", mirror);
                    return Ok(Some(metadata));
                }
                Ok(None) => {
                    debug!("Paper not found on mirror: {}", mirror);
                }
                Err(e) => {
                    warn!("Failed to query mirror {}: {}", mirror, e);
                }
            }
        }

        Ok(None)
    }

    /// Try to fetch from a specific mirror
    async fn try_mirror(
        &self,
        mirror: &str,
        query: &str,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = format!("{}/{}", mirror, urlencoding::encode(query));
        let user_agent = self.get_next_user_agent();

        debug!(
            "Trying Sci-Hub URL: {} with user agent: {}",
            url, user_agent
        );

        let response = self
            .circuit_breaker_service
            .call_http("sci_hub", || async {
                self
                .client
                .get(&url)
                .header("User-Agent", user_agent)
                .header(
                    "Accept",
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                )
                .header("Accept-Language", "en-US,en;q=0.5")
                .header("Accept-Encoding", "gzip, deflate")
                .header("DNT", "1")
                .header("Connection", "keep-alive")
                .header("Upgrade-Insecure-Requests", "1")
                .send()
                .await
            })
            .await
            .map_err(|e| match e {
                crate::Error::CircuitBreakerOpen { service } => {
                    ProviderError::ServiceUnavailable(format!("Circuit breaker open for {service}"))
                }
                crate::Error::NetworkTimeout { .. } => ProviderError::Timeout,
                crate::Error::ConnectionRefused { .. } => {
                    ProviderError::Network("Connection failed".to_string())
                }
                crate::Error::Http(http_err) => {
                    ProviderError::Network(format!("Request failed: {http_err}"))
                }
                _ => ProviderError::Network(format!("Request failed: {e}")),
            })?;

        if !response.status().is_success() {
            // For 403 errors, provide more context about potential solutions
            if response.status() == 403 {
                return Err(ProviderError::Network(format!(
                    "HTTP 403 Forbidden from mirror {mirror}: Access denied, possibly due to geographic blocking or rate limiting. Will try other mirrors."
                )));
            }
            return Err(ProviderError::Network(format!(
                "HTTP {} from mirror {}",
                response.status(),
                mirror
            )));
        }

        let html_content = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        self.parse_scihub_response(&html_content, query)
    }

    /// Parse Sci-Hub HTML response to extract paper metadata
    fn parse_scihub_response(
        &self,
        html: &str,
        original_query: &str,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        let document = Html::parse_document(html);

        // Check if we got an error page
        if html.contains("article not found") || html.contains("no fulltext") {
            return Ok(None);
        }

        // Look for PDF download link
        let pdf_selector = Selector::parse("embed[src], iframe[src], a[href*='.pdf']")
            .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let mut pdf_url = None;
        for element in document.select(&pdf_selector) {
            if let Some(src) = element
                .value()
                .attr("src")
                .or_else(|| element.value().attr("href"))
            {
                // Skip empty attributes
                if src.is_empty() {
                    continue;
                }

                if src.contains(".pdf") || src.starts_with("//") {
                    pdf_url = Some(if src.starts_with("//") {
                        format!("https:{src}")
                    } else if src.starts_with('/') {
                        // Properly resolve relative URLs
                        match url::Url::parse("https://sci-hub.se") {
                            Ok(base) => match base.join(src) {
                                Ok(absolute_url) => absolute_url.to_string(),
                                Err(e) => {
                                    warn!("Failed to resolve relative URL '{}': {}", src, e);
                                    continue;
                                }
                            },
                            Err(e) => {
                                warn!("Invalid base URL: {}", e);
                                continue;
                            }
                        }
                    } else if src.starts_with("http") {
                        src.to_string()
                    } else {
                        // Handle other relative URLs
                        match url::Url::parse("https://sci-hub.se") {
                            Ok(base) => match base.join(src) {
                                Ok(absolute_url) => absolute_url.to_string(),
                                Err(e) => {
                                    warn!("Failed to resolve relative URL '{}': {}", src, e);
                                    continue;
                                }
                            },
                            Err(e) => {
                                warn!("Invalid base URL: {}", e);
                                continue;
                            }
                        }
                    });
                    break;
                }
            }
        }

        // Extract title if available
        let title_selector = Selector::parse("title, h1, .article-title")
            .map_err(|e| ProviderError::Parse(format!("Invalid title selector: {e}")))?;

        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|t| !t.is_empty() && !t.to_lowercase().contains("sci-hub"));

        // If we found a PDF URL, we consider it a success
        if pdf_url.is_some() {
            let metadata = PaperMetadata {
                doi: original_query.to_string(),
                pmid: None,
                title,
                authors: Vec::new(), // Sci-Hub doesn't provide detailed metadata
                journal: None,
                year: None,
                abstract_text: None,
                keywords: Vec::new(),
                pdf_url,
                file_size: None,
            };

            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl SourceProvider for SciHubProvider {
    fn name(&self) -> &'static str {
        "sci_hub"
    }

    fn description(&self) -> &'static str {
        "Sci-Hub - Free access to academic papers"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![SearchType::Doi, SearchType::Title, SearchType::Auto]
    }

    fn query_format_help(&self) -> &'static str {
        r#"Sci-Hub provides access to papers via DOI:
- Best used with DOI for direct PDF access
- Title search has limited accuracy
- Used as fallback when other sources unavailable
- Lower priority in search ordering"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("10.1038/nature12373", "DOI lookup"),
            ("10.1126/science.1157996", "Science article DOI"),
            ("attention is all you need", "Title search (limited)"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        None // Sci-Hub doesn't have public API documentation
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn priority(&self) -> u8 {
        10 // Lower priority, use as fallback for full-text access
    }

    fn base_delay(&self) -> Duration {
        Duration::from_secs(2) // Respectful delay
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching Sci-Hub for: {} (type: {:?})",
            query.query, query.search_type
        );

        let paper = self
            .fetch_from_scihub(&query.query, &query.search_type)
            .await?;

        let papers = if let Some(metadata) = paper {
            vec![metadata]
        } else {
            Vec::new()
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();
        let mut metadata = HashMap::new();
        metadata.insert("mirrors_tried".to_string(), self.mirrors.len().to_string());

        let result = ProviderResult {
            papers,
            source: "Sci-Hub".to_string(),
            total_available: if papers_count == 0 { Some(0) } else { Some(1) },
            search_time,
            has_more: false,
            metadata,
        };

        info!(
            "Sci-Hub search completed: {} papers found in {:?}",
            result.papers.len(),
            search_time
        );

        Ok(result)
    }

    async fn get_by_doi(
        &self,
        doi: &str,
        _context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        info!("Getting paper by DOI from Sci-Hub: {}", doi);
        self.fetch_from_scihub(doi, &SearchType::Doi).await
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing Sci-Hub health check");

        // Try to access the main page of the first mirror
        let mirror = &self.mirrors[0];

        let response = self
            .circuit_breaker_service
            .call_http("sci_hub", || async { self.client.get(mirror).send().await })
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                info!("Sci-Hub health check: OK");
                Ok(true)
            }
            Ok(resp) => {
                warn!("Sci-Hub health check failed with status: {}", resp.status());
                Ok(false)
            }
            Err(crate::Error::CircuitBreakerOpen { .. }) => {
                warn!("Sci-Hub health check: Circuit breaker open");
                Ok(false)
            }
            Err(e) => {
                warn!("Sci-Hub health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

impl Default for SciHubProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create SciHubProvider")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sci_hub_provider_creation() {
        let provider = SciHubProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_clean_doi() {
        assert_eq!(
            SciHubProvider::clean_doi("10.1038/nature12373"),
            "10.1038/nature12373"
        );
        assert_eq!(
            SciHubProvider::clean_doi("doi:10.1038/nature12373"),
            "10.1038/nature12373"
        );
        assert_eq!(
            SciHubProvider::clean_doi("https://doi.org/10.1038/nature12373"),
            "10.1038/nature12373"
        );
    }

    #[test]
    fn test_provider_interface() {
        let provider = SciHubProvider::new().unwrap();

        assert_eq!(provider.name(), "sci_hub");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 10);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
    }
}
