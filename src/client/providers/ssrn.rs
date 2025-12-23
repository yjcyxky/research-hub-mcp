use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// SSRN (Social Science Research Network) provider for academic papers
pub struct SsrnProvider {
    client: Client,
    base_url: String,
}

impl SsrnProvider {
    /// Create a new SSRN provider
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://papers.ssrn.com".to_string(),
        })
    }

    /// Extract SSRN ID from DOI
    fn extract_ssrn_id(&self, doi: &str) -> Option<String> {
        // SSRN DOIs have format: 10.2139/ssrn.XXXXXXX
        if doi.contains("10.2139/ssrn.") {
            doi.split("ssrn.").nth(1).map(|id| id.trim().to_string())
        } else {
            None
        }
    }

    /// Build SSRN paper URL from ID
    fn build_paper_url(&self, ssrn_id: &str) -> String {
        format!("{}/sol3/papers.cfm?abstract_id={}", self.base_url, ssrn_id)
    }

    /// Build SSRN search URL
    fn build_search_url(&self, query: &str) -> String {
        format!(
            "{}/sol3/results.cfm?txtKey_Words={}&npage=1",
            self.base_url,
            urlencoding::encode(query)
        )
    }

    /// Fetch paper by SSRN ID
    async fn fetch_by_id(&self, ssrn_id: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = self.build_paper_url(ssrn_id);
        debug!("Fetching SSRN paper from: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let html_content = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        self.parse_paper_page(&html_content, ssrn_id)
    }

    /// Parse SSRN paper page
    fn parse_paper_page(
        &self,
        html: &str,
        ssrn_id: &str,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        let document = Html::parse_document(html);

        // Extract title
        let title_selector = Selector::parse("meta[name='citation_title']")
            .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let title = document
            .select(&title_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(str::to_string);

        // Extract authors
        let author_selector = Selector::parse("meta[name='citation_author']")
            .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let authors: Vec<String> = document
            .select(&author_selector)
            .filter_map(|el| el.value().attr("content"))
            .map(str::to_string)
            .collect();

        // Extract abstract
        let abstract_selector = Selector::parse("meta[name='citation_abstract']")
            .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let abstract_text = document
            .select(&abstract_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(str::to_string);

        // Extract PDF URL
        let pdf_selector = Selector::parse("meta[name='citation_pdf_url']")
            .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let pdf_url = document
            .select(&pdf_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(str::to_string)
            .or_else(|| {
                // Fallback: look for download button
                let download_selector =
                    Selector::parse("a.download-button, a[href*='download']").ok()?;
                document
                    .select(&download_selector)
                    .next()
                    .and_then(|el| el.value().attr("href"))
                    .and_then(|href| {
                        if href.starts_with("http") {
                            Some(href.to_string())
                        } else {
                            // Properly resolve relative URLs
                            match url::Url::parse(&self.base_url) {
                                Ok(base) => match base.join(href) {
                                    Ok(absolute_url) => Some(absolute_url.to_string()),
                                    Err(e) => {
                                        warn!("Failed to resolve relative URL '{}': {}", href, e);
                                        None
                                    }
                                },
                                Err(e) => {
                                    warn!("Invalid base URL '{}': {}", self.base_url, e);
                                    None
                                }
                            }
                        }
                    })
            })
            .filter(|url| !url.is_empty());

        // Extract publication year
        let date_selector =
            Selector::parse("meta[name='citation_publication_date'], meta[name='citation_date']")
                .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

        let year = document
            .select(&date_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .and_then(|date| date.split('-').next())
            .and_then(|year_str| year_str.parse::<u32>().ok());

        if title.is_some() || !authors.is_empty() {
            Ok(Some(PaperMetadata {
                doi: format!("10.2139/ssrn.{ssrn_id}"),
                pmid: None,
                title,
                authors,
                journal: Some("SSRN Electronic Journal".to_string()),
                year,
                abstract_text,
                keywords: Vec::new(),
                pdf_url,
                file_size: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Search SSRN for papers
    async fn search_papers(
        &self,
        query: &str,
        limit: u32,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let url = self.build_search_url(query);
        debug!("Searching SSRN: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Search request failed: {e}")))?;

        if !response.status().is_success() {
            return Ok(Vec::new());
        }

        let html_content = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        self.parse_search_results(&html_content, limit as usize)
            .await
    }

    /// Parse SSRN search results
    async fn parse_search_results(
        &self,
        html: &str,
        limit: usize,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        // Parse HTML in a separate scope to ensure it's dropped before await
        let ssrn_ids = {
            let document = Html::parse_document(html);

            // Look for paper links in search results
            let link_selector = Selector::parse("a[href*='abstract_id=']")
                .map_err(|e| ProviderError::Parse(format!("Invalid selector: {e}")))?;

            let mut ids = Vec::new();
            for element in document.select(&link_selector).take(limit) {
                if let Some(href) = element.value().attr("href") {
                    if let Some(id) = href.split("abstract_id=").nth(1) {
                        let id = id.split('&').next().unwrap_or(id);
                        if !ids.contains(&id.to_string()) {
                            ids.push(id.to_string());
                        }
                    }
                }
            }
            ids
        };

        // Fetch details for each paper
        let mut papers = Vec::new();
        for ssrn_id in ssrn_ids.iter().take(limit) {
            if let Ok(Some(paper)) = self.fetch_by_id(ssrn_id).await {
                papers.push(paper);
            }
        }

        Ok(papers)
    }
}

#[async_trait]
impl SourceProvider for SsrnProvider {
    fn name(&self) -> &'static str {
        "ssrn"
    }

    fn description(&self) -> &'static str {
        "SSRN - Social Science Research Network (Free preprints)"
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
        r#"SSRN hosts social science and economics preprints:
- Search by title, author, or keywords
- Strong in economics, finance, law, social sciences
- Most papers are free to download
- Abstract ID can be used for direct lookup
- Results sorted by relevance or date"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("behavioral economics", "Basic topic search"),
            ("Eugene Fama market efficiency", "Author with topic"),
            ("corporate governance 2023", "Topic with year"),
            ("10.2139/ssrn.1234567", "SSRN DOI lookup"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://www.ssrn.com/index.cfm/en/")
    }

    fn supports_full_text(&self) -> bool {
        true // SSRN usually provides free PDFs
    }

    fn priority(&self) -> u8 {
        85 // High priority for recent papers and preprints
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(500) // Be respectful but SSRN is robust
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching SSRN for: {} (type: {:?})",
            query.query, query.search_type
        );

        let papers = match query.search_type {
            SearchType::Doi => {
                // Check if it's an SSRN DOI
                if let Some(ssrn_id) = self.extract_ssrn_id(&query.query) {
                    if let Some(paper) = self.fetch_by_id(&ssrn_id).await? {
                        vec![paper]
                    } else {
                        Vec::new()
                    }
                } else {
                    // Not an SSRN DOI, search by title
                    self.search_papers(&query.query, query.max_results).await?
                }
            }
            _ => {
                // For all other search types, use the general search
                self.search_papers(&query.query, query.max_results).await?
            }
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();

        let result = ProviderResult {
            papers,
            source: "SSRN".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: papers_count >= query.max_results as usize,
            metadata: HashMap::new(),
        };

        info!(
            "SSRN search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from SSRN: {}", doi);

        if let Some(ssrn_id) = self.extract_ssrn_id(doi) {
            self.fetch_by_id(&ssrn_id).await
        } else {
            Ok(None)
        }
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing SSRN health check");

        match self.client.get(&self.base_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!("SSRN health check: OK");
                Ok(true)
            }
            Ok(response) => {
                warn!(
                    "SSRN health check failed with status: {}",
                    response.status()
                );
                Ok(false)
            }
            Err(e) => {
                warn!("SSRN health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

impl Default for SsrnProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create SsrnProvider")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssrn_provider_creation() {
        let provider = SsrnProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_extract_ssrn_id() {
        let provider = SsrnProvider::new().unwrap();

        assert_eq!(
            provider.extract_ssrn_id("10.2139/ssrn.5290350"),
            Some("5290350".to_string())
        );

        assert_eq!(provider.extract_ssrn_id("10.1038/nature12373"), None);
    }

    #[test]
    fn test_provider_interface() {
        let provider = SsrnProvider::new().unwrap();

        assert_eq!(provider.name(), "ssrn");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 85);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
    }
}
