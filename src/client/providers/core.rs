use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// CORE API response for search results
#[derive(Debug, Deserialize)]
struct CoreSearchResponse {
    #[serde(rename = "totalHits")]
    #[allow(dead_code)]
    total_hits: Option<u32>,
    data: Vec<CoreArticle>,
    #[serde(rename = "scrollId")]
    #[allow(dead_code)]
    scroll_id: Option<String>,
}

/// Individual article from CORE API
#[derive(Debug, Deserialize)]
struct CoreArticle {
    id: Option<u32>,
    doi: Option<String>,
    title: Option<String>,
    authors: Option<Vec<String>>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    #[serde(rename = "yearPublished")]
    year_published: Option<u32>,
    #[serde(rename = "publishedDate")]
    #[allow(dead_code)]
    published_date: Option<String>,
    journals: Option<Vec<CoreJournal>>,
    #[serde(rename = "downloadUrl")]
    download_url: Option<String>,
    #[serde(rename = "fullTextIdentifier")]
    full_text_identifier: Option<String>,
    #[serde(rename = "oai")]
    oai: Option<String>,
    #[allow(dead_code)]
    language: Option<String>,
    #[allow(dead_code)]
    subjects: Option<Vec<String>>,
    #[serde(rename = "hasFullText")]
    has_full_text: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CoreJournal {
    title: Option<String>,
    #[allow(dead_code)]
    identifiers: Option<Vec<String>>,
}

/// Individual article response for DOI lookup
#[derive(Debug, Deserialize)]
struct CoreArticleResponse {
    status: String,
    data: Option<CoreArticle>,
}

/// CORE provider for open access research papers
pub struct CoreProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl CoreProvider {
    /// Create a new CORE provider
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.core.ac.uk/v3".to_string(),
            api_key,
        })
    }

    /// Build search URL for CORE API
    fn build_search_url(&self, query: &str, limit: u32, offset: u32) -> String {
        format!(
            "{}/search/works?q={}&limit={}&offset={}",
            self.base_url,
            urlencoding::encode(query),
            limit,
            offset
        )
    }

    /// Build DOI lookup URL
    fn build_doi_url(&self, doi: &str) -> String {
        format!("{}/works?doi={}", self.base_url, urlencoding::encode(doi))
    }

    /// Get request headers including API key if available
    fn get_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        if let Some(api_key) = &self.api_key {
            headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
        }
        headers
    }

    /// Convert CORE article to `PaperMetadata`
    fn convert_article(&self, article: CoreArticle) -> PaperMetadata {
        // Use DOI if available, otherwise create a CORE identifier
        let doi = article.doi.clone().unwrap_or_else(|| {
            if let Some(id) = article.id {
                format!("core:{id}")
            } else if let Some(oai) = &article.oai {
                format!("core:oai:{oai}")
            } else {
                "core:unknown".to_string()
            }
        });

        // Extract journal name from the first journal
        let journal = article
            .journals
            .as_ref()
            .and_then(|journals| journals.first())
            .and_then(|journal| journal.title.clone());

        // Convert authors vec directly
        let authors = article.authors.unwrap_or_default();

        // Use download URL as PDF URL if available and has full text
        let pdf_url = if article.has_full_text.unwrap_or(false) {
            // Filter out empty URLs
            article
                .download_url
                .filter(|url| !url.is_empty())
                .or_else(|| article.full_text_identifier.filter(|url| !url.is_empty()))
        } else {
            None
        };

        PaperMetadata {
            doi,
            title: article.title,
            authors,
            journal,
            year: article.year_published,
            abstract_text: article.abstract_text,
            pdf_url,
            file_size: None,
        }
    }

    /// Search papers by query
    async fn search_papers(
        &self,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let url = self.build_search_url(query, limit, offset);
        debug!("Searching CORE: {}", url);

        let mut request = self.client.get(&url);

        // Add API key header if available
        for (key, value) in self.get_headers() {
            request = request.header(&key, &value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        debug!("CORE response: {}", response_text);

        // First try to parse with the expected response structure
        if let Ok(api_response) = serde_json::from_str::<CoreSearchResponse>(&response_text) {
            let papers = api_response
                .data
                .into_iter()
                .map(|article| self.convert_article(article))
                .collect();
            return Ok(papers);
        }

        // If that fails, try parsing as a direct array
        if let Ok(articles) = serde_json::from_str::<Vec<CoreArticle>>(&response_text) {
            let papers = articles
                .into_iter()
                .map(|article| self.convert_article(article))
                .collect();
            return Ok(papers);
        }

        // If both fail, try parsing as a single article
        if let Ok(article) = serde_json::from_str::<CoreArticle>(&response_text) {
            return Ok(vec![self.convert_article(article)]);
        }

        // If all parsing attempts fail, return error
        warn!(
            "Failed to parse CORE response with any known format: {}",
            response_text
        );
        Err(ProviderError::Parse(
            "Failed to parse CORE API response: unknown format".to_string(),
        ))
    }

    /// Get paper by DOI
    async fn get_paper_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = self.build_doi_url(doi);
        debug!("Getting paper by DOI from CORE: {}", url);

        let mut request = self.client.get(&url);

        // Add API key header if available
        for (key, value) in self.get_headers() {
            request = request.header(&key, &value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if response.status().as_u16() == 404 {
            debug!("Paper not found in CORE for DOI: {}", doi);
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        debug!("CORE DOI response: {}", response_text);

        let article_response: CoreArticleResponse = serde_json::from_str(&response_text)
            .map_err(|e| ProviderError::Parse(format!("Failed to parse JSON: {e}")))?;

        if article_response.status == "OK" {
            if let Some(article) = article_response.data {
                Ok(Some(self.convert_article(article)))
            } else {
                Ok(None)
            }
        } else {
            debug!("CORE API returned status: {}", article_response.status);
            Ok(None)
        }
    }
}

#[async_trait]
impl SourceProvider for CoreProvider {
    fn name(&self) -> &'static str {
        "core"
    }

    fn description(&self) -> &'static str {
        "CORE - World's largest collection of open access research papers"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Doi,
            SearchType::Title,
            SearchType::Author,
            SearchType::Keywords,
            SearchType::Auto,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"CORE supports advanced search operators:
- title:term - Search in title field
- author:name - Search by author name
- year:YYYY - Filter by publication year
- doi:value - Search by DOI
- AND, OR, NOT - Boolean operators
- "phrase" - Exact phrase matching
- Parentheses for grouping"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("machine learning", "Basic keyword search"),
            ("title:neural networks AND author:Hinton", "Field-specific with boolean"),
            ("climate change year:2023", "Topic with year filter"),
            ("\"deep learning\" AND (vision OR NLP)", "Phrase with grouping"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://core.ac.uk/documentation/api")
    }

    fn supports_full_text(&self) -> bool {
        true // CORE specifically focuses on open access full-text papers
    }

    fn priority(&self) -> u8 {
        86 // High priority for open access content, between Unpaywall and SSRN
    }

    fn base_delay(&self) -> Duration {
        if self.api_key.is_some() {
            Duration::from_millis(100) // Faster with API key
        } else {
            Duration::from_millis(500) // Moderate delay for public API
        }
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching CORE for: {} (type: {:?})",
            query.query, query.search_type
        );

        let papers = match query.search_type {
            SearchType::Doi => {
                // Try DOI lookup first
                if let Some(paper) = self.get_paper_by_doi(&query.query).await? {
                    vec![paper]
                } else {
                    // Fallback to search if DOI lookup fails
                    self.search_papers(&query.query, query.max_results, query.offset)
                        .await?
                }
            }
            _ => {
                // Use general search for all other types
                self.search_papers(&query.query, query.max_results, query.offset)
                    .await?
            }
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();

        let result = ProviderResult {
            papers,
            source: "CORE".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: papers_count >= query.max_results as usize,
            metadata: HashMap::new(),
        };

        info!(
            "CORE search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from CORE: {}", doi);
        self.get_paper_by_doi(doi).await
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing CORE health check");

        let url = format!("{}/search/works?q=test&limit=1", self.base_url);

        let mut request = self.client.get(&url);

        // Add API key header if available
        for (key, value) in self.get_headers() {
            request = request.header(&key, &value);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                info!("CORE health check: OK");
                Ok(true)
            }
            Ok(response) => {
                warn!(
                    "CORE health check failed with status: {}",
                    response.status()
                );
                Ok(false)
            }
            Err(e) => {
                warn!("CORE health check failed: {}", e);
                Ok(false)
            }
        }
    }

    async fn get_pdf_url(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<String>, ProviderError> {
        // Try to get the paper first, then extract PDF URL
        if let Some(paper) = self.get_by_doi(doi, context).await? {
            Ok(paper.pdf_url)
        } else {
            Ok(None)
        }
    }
}

impl Default for CoreProvider {
    fn default() -> Self {
        Self::new(None).expect("Failed to create CoreProvider")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_provider_creation() {
        let provider = CoreProvider::new(None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_interface() {
        let provider = CoreProvider::new(None).unwrap();

        assert_eq!(provider.name(), "core");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 86);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
        assert!(provider
            .supported_search_types()
            .contains(&SearchType::Title));
    }

    #[test]
    fn test_url_building() {
        let provider = CoreProvider::new(None).unwrap();

        let search_url = provider.build_search_url("machine learning", 10, 0);
        assert!(search_url.contains("q=machine%20learning"));
        assert!(search_url.contains("limit=10"));
        assert!(search_url.contains("offset=0"));

        let doi_url = provider.build_doi_url("10.1038/nature12373");
        assert!(doi_url.contains("doi=10.1038%2Fnature12373"));
    }

    #[test]
    fn test_headers_with_api_key() {
        let provider = CoreProvider::new(Some("test_key".to_string())).unwrap();
        let headers = provider.get_headers();
        assert!(headers.contains_key("Authorization"));
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test_key".to_string())
        );
    }

    #[test]
    fn test_headers_without_api_key() {
        let provider = CoreProvider::new(None).unwrap();
        let headers = provider.get_headers();
        assert!(!headers.contains_key("Authorization"));
    }
}
