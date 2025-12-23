use crate::client::providers::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info};

/// `ResearchGate` provider for academic papers and research data
///
/// `ResearchGate` is a social networking site for scientists and researchers.
/// This provider implements ethical scraping with rate limiting and respects
/// `ResearchGate`'s terms of service.
pub struct ResearchGateProvider {
    client: Client,
    base_url: String,
    rate_limit: Duration,
}

impl ResearchGateProvider {
    /// Create a new `ResearchGate` provider with ethical scraping settings
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("knowledge_accumulator_mcp/0.3.0 (Academic Research Tool)")
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://www.researchgate.net".to_string(),
            rate_limit: Duration::from_secs(3), // Respectful rate limiting
        })
    }

    /// Check if a URL is a valid `ResearchGate` publication URL
    fn is_researchgate_url(&self, url: &str) -> bool {
        url.contains("researchgate.net/publication/") || url.contains("researchgate.net/profile/")
    }

    /// Extract basic metadata from `ResearchGate` URL (if publicly accessible)
    async fn extract_metadata_from_url(
        &self,
        url: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        if !self.is_researchgate_url(url) {
            return Ok(None);
        }

        // Add rate limiting to be respectful
        tokio::time::sleep(self.rate_limit).await;

        let response = self
            .client
            .get(url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("ResearchGate request failed: {e}")))?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let html = response.text().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to read ResearchGate response: {e}"))
        })?;

        self.parse_publication_page(&html)
    }

    /// Parse a `ResearchGate` publication page for metadata
    fn parse_publication_page(&self, html: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let document = Html::parse_document(html);

        // Basic selectors for publicly available metadata
        let title_selector = Selector::parse("h1[data-testid='publication-title']")
            .map_err(|_| ProviderError::Parse("Invalid title selector".to_string()))?;

        let authors_selector = Selector::parse("a[data-testid='author-name']")
            .map_err(|_| ProviderError::Parse("Invalid authors selector".to_string()))?;

        let doi_selector = Selector::parse("a[href*='doi.org']")
            .map_err(|_| ProviderError::Parse("Invalid DOI selector".to_string()))?;

        // Extract title
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string());

        // Extract authors
        let authors: Vec<String> = document
            .select(&authors_selector)
            .map(|el| el.text().collect::<String>().trim().to_string())
            .filter(|name| !name.is_empty())
            .collect();

        // Extract DOI
        let doi = document
            .select(&doi_selector)
            .filter_map(|el| el.value().attr("href"))
            .find_map(|href| {
                if let Some(doi_match) = Regex::new(r"doi\.org/(.+)$").ok()?.captures(href) {
                    Some(doi_match.get(1)?.as_str().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        // Only return metadata if we found meaningful information
        if title.is_some() || !authors.is_empty() || !doi.is_empty() {
            Ok(Some(PaperMetadata {
                doi,
                pmid: None,
                title,
                authors,
                journal: None,       // Not easily extractable from RG
                year: None,          // Would need more complex parsing
                abstract_text: None, // Requires login for full access
                keywords: Vec::new(),
                pdf_url: None,       // ResearchGate PDFs require authentication
                file_size: None,
            }))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl SourceProvider for ResearchGateProvider {
    fn name(&self) -> &'static str {
        "researchgate"
    }

    fn description(&self) -> &'static str {
        "ResearchGate provider with ethical scraping (limited functionality)"
    }

    fn priority(&self) -> u8 {
        70 // Lower priority due to access limitations
    }

    fn supports_full_text(&self) -> bool {
        false // ResearchGate PDFs require authentication
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![SearchType::Auto, SearchType::Title, SearchType::Author]
    }

    fn query_format_help(&self) -> &'static str {
        r#"ResearchGate has limited search due to ToS:
- URL extraction supported for paper metadata
- General search disabled to respect terms of service
- Use for extracting metadata from ResearchGate URLs
- PDFs require authentication (not available)"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("https://www.researchgate.net/publication/...", "Extract from URL"),
            ("machine learning", "Limited keyword search (may be empty)"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://www.researchgate.net/search")
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        info!("ResearchGate search for: '{}'", query.query);

        // Check if the query is actually a ResearchGate URL
        if self.is_researchgate_url(&query.query) {
            if let Some(metadata) = self
                .extract_metadata_from_url(&query.query, context)
                .await?
            {
                return Ok(ProviderResult {
                    papers: vec![metadata],
                    source: self.name().to_string(),
                    total_available: Some(1),
                    search_time: Duration::from_millis(100),
                    has_more: false,
                    metadata: HashMap::new(),
                });
            }
        }

        // For general searches, we return empty results to respect ToS
        info!("ResearchGate general search disabled to respect terms of service");

        Ok(ProviderResult {
            papers: vec![],
            source: self.name().to_string(),
            total_available: Some(0),
            search_time: Duration::from_millis(50),
            has_more: false,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "note".to_string(),
                    "Limited functionality to respect ToS".to_string(),
                );
                meta
            },
        })
    }

    async fn get_by_doi(
        &self,
        doi: &str,
        _context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        // ResearchGate doesn't have a reliable DOI-based API access
        debug!("ResearchGate DOI lookup not available for: {}", doi);
        Ok(None)
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        // Simple check to see if ResearchGate is accessible
        match self.client.head(&self.base_url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_context() -> SearchContext {
        SearchContext {
            timeout: Duration::from_secs(30),
            user_agent: "test".to_string(),
            rate_limit: None,
            headers: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_researchgate_provider_creation() {
        let provider = ResearchGateProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_researchgate_url_detection() {
        let provider = ResearchGateProvider::new().unwrap();

        assert!(provider.is_researchgate_url("https://www.researchgate.net/publication/123456789"));
        assert!(provider.is_researchgate_url("https://researchgate.net/profile/John-Doe"));
        assert!(!provider.is_researchgate_url("https://example.com/publication/123"));
    }

    #[test]
    fn test_provider_metadata() {
        let provider = ResearchGateProvider::new().unwrap();

        assert_eq!(provider.name(), "researchgate");
        assert_eq!(provider.priority(), 70);
        assert!(!provider.supports_full_text());

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Auto));
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
    }

    #[tokio::test]
    async fn test_ethical_search_behavior() {
        let provider = ResearchGateProvider::new().unwrap();
        let context = create_test_context();

        let query = SearchQuery {
            query: "machine learning".to_string(),
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let result = provider.search(&query, &context).await;
        assert!(result.is_ok());

        let search_result = result.unwrap();
        // Should return empty results to respect ToS
        assert_eq!(search_result.papers.len(), 0);
        assert!(search_result.metadata.contains_key("note"));
    }
}
