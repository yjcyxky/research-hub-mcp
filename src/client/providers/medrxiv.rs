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

/// medRxiv API response for paper details
#[derive(Debug, Deserialize)]
struct MedrxivResponse {
    messages: Vec<MedrxivMessage>,
    collection: Vec<MedrxivPaper>,
}

/// Status message from medRxiv API
#[derive(Debug, Deserialize)]
struct MedrxivMessage {
    status: String,
    #[serde(default)]
    text: Option<String>,
}

/// Individual paper from medRxiv API
#[derive(Debug, Deserialize)]
struct MedrxivPaper {
    doi: String,
    title: String,
    authors: String,
    #[allow(dead_code)]
    author_corresponding: Option<String>,
    #[allow(dead_code)]
    author_corresponding_institution: Option<String>,
    date: String, // Format: YYYY-MM-DD
    #[allow(dead_code)]
    version: Option<u32>,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    paper_type: Option<String>,
    #[allow(dead_code)]
    category: Option<String>,
    #[allow(dead_code)]
    jatsxml: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    #[allow(dead_code)]
    published: Option<String>,
    server: String, // Should be "medrxiv"
}

/// medRxiv provider for clinical and health science preprints
pub struct MedrxivProvider {
    client: Client,
    base_url: String,
}

impl MedrxivProvider {
    /// Create a new medRxiv provider
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.biorxiv.org".to_string(),
        })
    }

    /// Build DOI lookup URL for medRxiv API
    fn build_doi_url(&self, doi: &str) -> String {
        format!("{}/details/medrxiv/{}", self.base_url, doi)
    }

    /// Build search URL for medRxiv API (by date range)
    fn build_date_search_url(
        &self,
        start_date: &str,
        end_date: &str,
        query: Option<&str>,
    ) -> String {
        if let Some(q) = query {
            format!(
                "{}/details/medrxiv/{}/{}/{}",
                self.base_url,
                start_date,
                end_date,
                urlencoding::encode(q)
            )
        } else {
            format!(
                "{}/details/medrxiv/{}/{}",
                self.base_url, start_date, end_date
            )
        }
    }

    /// Extract medRxiv DOI from common formats
    fn extract_medrxiv_doi(doi_or_url: &str) -> Option<String> {
        // medRxiv shares the 10.1101 prefix pattern
        if doi_or_url.contains("10.1101/") {
            if let Some(doi_start) = doi_or_url.find("10.1101/") {
                let doi_part = &doi_or_url[doi_start..];
                // Remove version suffix if present (e.g., "v1", "v2")
                if let Some(version_pos) = doi_part.find('v') {
                    if version_pos > 8 {
                        return Some(doi_part[..version_pos].to_string());
                    }
                }
                return Some(doi_part.to_string());
            }
        }
        None
    }

    /// Convert medRxiv paper to `PaperMetadata`
    fn convert_paper(&self, paper: MedrxivPaper) -> PaperMetadata {
        // Parse authors from the comma-separated string
        let authors: Vec<String> = paper
            .authors
            .split(',')
            .map(|author| author.trim().to_string())
            .filter(|author| !author.is_empty())
            .collect();

        // Extract year from date (YYYY-MM-DD format)
        let year = paper
            .date
            .split('-')
            .next()
            .and_then(|year_str| year_str.parse::<u32>().ok());

        let pdf_url = if paper.server.to_lowercase().contains("medrxiv") {
            // medRxiv PDFs are exposed as https://www.medrxiv.org/content/<doi>v1.full.pdf
            Some(format!(
                "https://www.medrxiv.org/content/{}v1.full.pdf",
                paper.doi
            ))
        } else {
            None
        };

        PaperMetadata {
            doi: paper.doi,
            title: Some(paper.title),
            authors,
            journal: Some("medrxiv preprint".to_string()),
            year,
            abstract_text: paper.abstract_text,
            pdf_url,
            file_size: None,
        }
    }

    /// Get paper by DOI from medRxiv
    async fn get_paper_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = self.build_doi_url(doi);
        debug!("Getting paper by DOI from medRxiv: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if response.status().as_u16() == 404 {
            debug!("Paper not found in medRxiv for DOI: {}", doi);
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

        debug!("medRxiv response: {}", response_text);

        let medrxiv_response: MedrxivResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!("Failed to parse medRxiv response: {}", response_text);
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        // Check for error messages
        for message in &medrxiv_response.messages {
            if message.status != "ok" {
                warn!("medRxiv API message: {:?}", message.text);
            }
        }

        // Return the first paper if found
        if let Some(paper) = medrxiv_response.collection.into_iter().next() {
            Ok(Some(self.convert_paper(paper)))
        } else {
            Ok(None)
        }
    }

    /// Search for recent papers (medRxiv doesn't support broad keyword search via the public API)
    async fn search_recent_papers(
        &self,
        days_back: u32,
        limit: u32,
        query: Option<&str>,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        use chrono::{Duration as ChronoDuration, Utc};

        // Calculate date range
        let end_date = Utc::now();
        let start_date = end_date - ChronoDuration::days(i64::from(days_back));

        let start_date_str = start_date.format("%Y-%m-%d").to_string();
        let end_date_str = end_date.format("%Y-%m-%d").to_string();

        let url = self.build_date_search_url(&start_date_str, &end_date_str, query);
        debug!("Searching medRxiv by date range: {}", url);

        let response = self
            .client
            .get(&url)
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

        debug!("medRxiv search response: {}", response_text);

        let medrxiv_response: MedrxivResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!("Failed to parse medRxiv search response: {}", response_text);
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        for message in &medrxiv_response.messages {
            if message.status != "ok" {
                warn!("medRxiv API message: {:?}", message.text);
            }
        }

        // Convert papers and limit results
        let papers: Vec<PaperMetadata> = medrxiv_response
            .collection
            .into_iter()
            .take(limit as usize)
            .map(|paper| self.convert_paper(paper))
            .collect();

        Ok(papers)
    }
}

#[async_trait]
impl SourceProvider for MedrxivProvider {
    fn name(&self) -> &'static str {
        "medrxiv"
    }

    fn description(&self) -> &'static str {
        "medRxiv - Clinical and health science preprint server"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Doi,
            SearchType::Keywords,
            SearchType::TitleAbstract,
            SearchType::Auto,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"medRxiv supports basic keyword search:
- Keywords are searched across title and abstract
- DOI lookup for medRxiv-specific DOIs (10.1101/*)
- Date range search through API (recent papers)
- No advanced field-specific query syntax
- Focused on clinical and health sciences preprints"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("COVID-19 clinical trial", "Basic keyword search"),
            ("10.1101/2023.01.01.23284123", "DOI lookup"),
            ("diabetes treatment outcomes", "Clinical topic search"),
            ("vaccine effectiveness real world", "Public health research"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://www.medrxiv.org/search")
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn priority(&self) -> u8 {
        86 // Preferred after bioRxiv for clinical relevance
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(500)
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching medRxiv for: {} (type: {:?})",
            query.query, query.search_type
        );

        let papers = match query.search_type {
            SearchType::Doi => {
                if let Some(doi) = Self::extract_medrxiv_doi(&query.query) {
                    if let Some(paper) = self.get_paper_by_doi(&doi).await? {
                        vec![paper]
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            }
            SearchType::Keywords | SearchType::TitleAbstract | SearchType::Auto => {
                match self
                    .search_recent_papers(365, query.max_results, Some(&query.query))
                    .await
                {
                    Ok(papers) => papers,
                    Err(e) => {
                        warn!("medRxiv keyword search failed: {}", e);
                        Vec::new()
                    }
                }
            }
            _ => {
                warn!(
                    "medRxiv supports DOI and limited keyword searches, ignoring query: {}",
                    query.query
                );
                Vec::new()
            }
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();

        let result = ProviderResult {
            papers,
            source: "medRxiv".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: false,
            metadata: HashMap::new(),
        };

        info!(
            "medRxiv search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from medRxiv: {}", doi);

        if let Some(medrxiv_doi) = Self::extract_medrxiv_doi(doi) {
            self.get_paper_by_doi(&medrxiv_doi).await
        } else {
            Ok(None)
        }
    }

    async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
        // Use a simple recent-paper query for health checks
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
            Err(ProviderError::RateLimit) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}
