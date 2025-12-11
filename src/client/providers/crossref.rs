use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::circuit_breaker_service::CircuitBreakerService;
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use url::Url;

/// `CrossRef` API response structures
#[derive(Debug, Deserialize)]
struct CrossRefResponse {
    status: String,
    message: CrossRefMessage,
}

#[derive(Debug, Deserialize)]
struct CrossRefMessage {
    #[serde(default)]
    items: Vec<CrossRefWork>,
    #[serde(rename = "total-results")]
    total_results: Option<u64>,
    #[serde(rename = "items-per-page")]
    #[allow(dead_code)]
    items_per_page: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CrossRefWork {
    #[serde(rename = "DOI")]
    doi: Option<String>,
    title: Option<Vec<String>>,
    author: Option<Vec<CrossRefAuthor>>,
    #[serde(rename = "container-title")]
    container_title: Option<Vec<String>>,
    published: Option<CrossRefDate>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    #[serde(rename = "URL")]
    url: Option<String>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    work_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CrossRefAuthor {
    given: Option<String>,
    family: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CrossRefDate {
    #[serde(rename = "date-parts")]
    date_parts: Option<Vec<Vec<u32>>>,
}

/// `CrossRef` API provider
pub struct CrossRefProvider {
    client: Client,
    base_url: String,
    email: Option<String>,
    circuit_breaker_service: Arc<CircuitBreakerService>,
}

impl CrossRefProvider {
    /// Create a new `CrossRef` provider
    pub fn new(email: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Other(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.crossref.org/works".to_string(),
            email,
            circuit_breaker_service: Arc::new(CircuitBreakerService::new()),
        })
    }

    /// Build `CrossRef` API URL for search
    fn build_search_url(&self, query: &SearchQuery) -> Result<String, ProviderError> {
        let mut url = Url::parse(&self.base_url)
            .map_err(|e| ProviderError::Other(format!("Invalid base URL: {e}")))?;

        let mut params = vec![
            ("rows", query.max_results.to_string()),
            ("offset", query.offset.to_string()),
        ];

        // Add email for polite pool if provided
        if let Some(email) = &self.email {
            params.push(("mailto", email.clone()));
        }

        // Build query based on search type
        match query.search_type {
            SearchType::Doi => {
                // For DOI search, use the works/{doi} endpoint
                let clean_doi = query
                    .query
                    .trim_start_matches("doi:")
                    .trim_start_matches("https://doi.org/");
                let doi_url = format!("{}/{}", self.base_url, clean_doi);
                return Ok(doi_url);
            }
            SearchType::Title => {
                params.push(("query.title", query.query.clone()));
            }
            SearchType::TitleAbstract => {
                params.push(("query.bibliographic", query.query.clone()));
            }
            SearchType::Author => {
                params.push(("query.author", query.query.clone()));
            }
            SearchType::Keywords | SearchType::Auto => {
                params.push(("query", query.query.clone()));
            }
            SearchType::Subject => {
                params.push(("query.subject", query.query.clone()));
            }
        }

        // Add query parameters
        for (key, value) in params {
            url.query_pairs_mut().append_pair(key, &value);
        }

        Ok(url.to_string())
    }

    /// Convert `CrossRef` work to `PaperMetadata`
    fn convert_work(&self, work: CrossRefWork) -> PaperMetadata {
        let title = work
            .title
            .and_then(|titles| titles.into_iter().next())
            .map(|title| title.trim().to_string());

        let authors = work
            .author
            .unwrap_or_default()
            .into_iter()
            .map(|author| match (author.given, author.family) {
                (Some(given), Some(family)) => format!("{given} {family}"),
                (None, Some(family)) => family,
                (Some(given), None) => given,
                (None, None) => "Unknown".to_string(),
            })
            .collect();

        let journal = work
            .container_title
            .and_then(|titles| titles.into_iter().next());

        let year = work
            .published
            .and_then(|date| date.date_parts)
            .and_then(|parts| parts.into_iter().next())
            .and_then(|part| part.into_iter().next());

        let doi = work.doi.unwrap_or_default();

        // Filter out empty or invalid PDF URLs - CrossRef URLs are typically landing pages, not PDFs
        let pdf_url = work
            .url
            .filter(|url| !url.is_empty() && url.contains(".pdf"));

        PaperMetadata {
            doi,
            title,
            authors,
            journal,
            year,
            abstract_text: work.abstract_text,
            pdf_url,
            file_size: None,
        }
    }
}

impl Default for CrossRefProvider {
    fn default() -> Self {
        match Self::new(None) {
            Ok(provider) => provider,
            Err(_) => {
                // Fallback to a minimal client with very basic configuration
                // This should never fail under normal circumstances
                let client = Client::new();
                Self {
                    client,
                    base_url: "https://api.crossref.org/works".to_string(),
                    email: None,
                    circuit_breaker_service: Arc::new(CircuitBreakerService::new()),
                }
            }
        }
    }
}

#[async_trait]
impl SourceProvider for CrossRefProvider {
    fn name(&self) -> &'static str {
        "crossref"
    }

    fn description(&self) -> &'static str {
        "CrossRef API - Comprehensive metadata for academic publications with DOI"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Auto,
            SearchType::Doi,
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Keywords,
        ]
    }

    fn supports_full_text(&self) -> bool {
        false // CrossRef provides metadata, not full text
    }

    fn priority(&self) -> u8 {
        90 // Very high priority for metadata
    }

    fn base_delay(&self) -> Duration {
        if self.email.is_some() {
            Duration::from_millis(100) // Polite pool - faster access
        } else {
            Duration::from_millis(1000) // Regular pool
        }
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching CrossRef for: {} (type: {:?})",
            query.query, query.search_type
        );

        // Build the search URL
        let url = self.build_search_url(query)?;
        debug!("CrossRef search URL: {}", url);

        // Make the request with circuit breaker protection
        let response = self
            .circuit_breaker_service
            .call_http("crossref", || async {
                let mut request = self.client.get(&url);

                // Add custom headers
                for (key, value) in &context.headers {
                    request = request.header(key, value);
                }

                request.timeout(context.timeout).send().await
            })
            .await
            .map_err(|e| {
                error!("CrossRef request failed: {}", e);
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
                404 if query.search_type == SearchType::Doi => {
                    // DOI not found is not an error, just return empty results
                    let search_time = start_time.elapsed();
                    return Ok(ProviderResult {
                        papers: Vec::new(),
                        source: "CrossRef".to_string(),
                        total_available: Some(0),
                        search_time,
                        has_more: false,
                        metadata: HashMap::new(),
                    });
                }
                429 => ProviderError::RateLimit,
                503 => ProviderError::ServiceUnavailable(
                    "CrossRef service temporarily unavailable".to_string(),
                ),
                _ => ProviderError::Network(format!("HTTP {status}: {error_text}")),
            });
        }

        // Parse the response
        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        let crossref_response: CrossRefResponse = serde_json::from_str(&response_text)
            .map_err(|e| ProviderError::Parse(format!("Failed to parse JSON: {e}")))?;

        if crossref_response.status != "ok" {
            return Err(ProviderError::Other(format!(
                "CrossRef API error: {}",
                crossref_response.status
            )));
        }

        // Convert works to papers
        let papers: Vec<PaperMetadata> = crossref_response
            .message
            .items
            .into_iter()
            .map(|work| self.convert_work(work))
            .collect();

        let search_time = start_time.elapsed();
        let total_available = crossref_response
            .message
            .total_results
            .map(|t| u32::try_from(t).unwrap_or(u32::MAX));
        let has_more = u32::try_from(papers.len()).unwrap_or(u32::MAX) >= query.max_results;

        let mut metadata = HashMap::new();
        metadata.insert("api_url".to_string(), url);
        metadata.insert("response_size".to_string(), response_text.len().to_string());
        if let Some(total) = total_available {
            metadata.insert("total_results".to_string(), total.to_string());
        }

        info!(
            "CrossRef search completed: {} papers found in {:?}",
            papers.len(),
            search_time
        );

        Ok(ProviderResult {
            papers,
            source: "CrossRef".to_string(),
            total_available,
            search_time,
            has_more,
            metadata,
        })
    }

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

    async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing CrossRef health check");

        // Try to get a well-known DOI
        let query = SearchQuery {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Doi,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        match self.search(&query, context).await {
            Ok(result) => {
                let healthy = !result.papers.is_empty();
                info!(
                    "CrossRef health check: {}",
                    if healthy { "OK" } else { "No results" }
                );
                Ok(healthy)
            }
            Err(ProviderError::RateLimit) => {
                info!("CrossRef health check: OK (rate limited but responsive)");
                Ok(true)
            }
            Err(e) => {
                warn!("CrossRef health check failed: {}", e);
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
    async fn test_crossref_provider_creation() {
        let provider = CrossRefProvider::new(None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_crossref_search_url_building() {
        let provider = CrossRefProvider::new(None).unwrap();

        let query = SearchQuery {
            query: "machine learning".to_string(),
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("query=machine") && url.contains("learning"));
        assert!(url.contains("rows=10"));
        assert!(url.contains("offset=0"));
    }

    #[test]
    fn test_crossref_doi_search_url() {
        let provider = CrossRefProvider::new(None).unwrap();

        let query = SearchQuery {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Doi,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("10.1038/nature12373"));
    }
}
