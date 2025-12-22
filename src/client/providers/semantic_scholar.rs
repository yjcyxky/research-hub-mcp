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

/// Semantic Scholar API response for paper search
#[derive(Debug, Deserialize)]
struct SemanticScholarResponse {
    data: Vec<SemanticScholarPaper>,
    #[allow(dead_code)]
    total: Option<u32>,
    #[allow(dead_code)]
    offset: Option<u32>,
    #[serde(rename = "next")]
    #[allow(dead_code)]
    next: Option<u32>,
}

/// Individual paper from Semantic Scholar API
#[derive(Debug, Deserialize)]
struct SemanticScholarPaper {
    #[serde(rename = "paperId")]
    paper_id: Option<String>,
    #[serde(rename = "externalIds")]
    external_ids: Option<ExternalIds>,
    title: Option<String>,
    authors: Option<Vec<Author>>,
    venue: Option<String>,
    year: Option<u32>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    #[serde(rename = "openAccessPdf")]
    open_access_pdf: Option<OpenAccessPdf>,
    #[serde(rename = "publicationDate")]
    #[allow(dead_code)]
    publication_date: Option<String>,
    #[serde(rename = "journal")]
    journal: Option<Journal>,
    #[serde(rename = "citationCount")]
    #[allow(dead_code)]
    citation_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ExternalIds {
    #[serde(rename = "DOI")]
    doi: Option<String>,
    #[serde(rename = "ArXiv")]
    #[allow(dead_code)]
    arxiv: Option<String>,
    #[serde(rename = "PubMed")]
    #[allow(dead_code)]
    pubmed: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Author {
    name: Option<String>,
    #[serde(rename = "authorId")]
    #[allow(dead_code)]
    author_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAccessPdf {
    url: Option<String>,
    #[allow(dead_code)]
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Journal {
    name: Option<String>,
}

/// Semantic Scholar provider for academic papers
pub struct SemanticScholarProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl SemanticScholarProvider {
    /// Create a new Semantic Scholar provider
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.semanticscholar.org/graph/v1".to_string(),
            api_key,
        })
    }

    /// Build search URL for Semantic Scholar API
    fn build_search_url(&self, query: &str, fields: &[&str], limit: u32, offset: u32) -> String {
        let fields_param = fields.join(",");
        format!(
            "{}/paper/search?query={}&fields={}&limit={}&offset={}",
            self.base_url,
            urlencoding::encode(query),
            urlencoding::encode(&fields_param),
            limit,
            offset
        )
    }

    /// Build DOI lookup URL
    fn build_doi_url(&self, doi: &str, fields: &[&str]) -> String {
        let fields_param = fields.join(",");
        format!(
            "{}/paper/DOI:{}?fields={}",
            self.base_url,
            urlencoding::encode(doi),
            urlencoding::encode(&fields_param)
        )
    }

    /// Get request headers including API key if available
    fn get_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        if let Some(api_key) = &self.api_key {
            headers.insert("x-api-key".to_string(), api_key.clone());
        }
        headers
    }

    /// Convert Semantic Scholar paper to `PaperMetadata`
    fn convert_paper(&self, paper: SemanticScholarPaper) -> PaperMetadata {
        let doi = paper
            .external_ids
            .as_ref()
            .and_then(|ids| ids.doi.clone())
            .unwrap_or_else(|| {
                let paper_id = paper.paper_id.unwrap_or_else(|| "unknown".to_string());
                format!("semantic_scholar:{paper_id}")
            });

        let authors = paper
            .authors
            .unwrap_or_default()
            .into_iter()
            .filter_map(|author| author.name)
            .collect();

        let journal = paper.journal.and_then(|j| j.name).or(paper.venue);

        let pdf_url = paper
            .open_access_pdf
            .and_then(|pdf| pdf.url)
            .filter(|url| !url.is_empty());

        PaperMetadata {
            doi,
            title: paper.title,
            authors,
            journal,
            year: paper.year,
            abstract_text: paper.abstract_text,
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
        let fields = [
            "paperId",
            "externalIds",
            "title",
            "authors",
            "venue",
            "year",
            "abstract",
            "openAccessPdf",
            "publicationDate",
            "journal",
            "citationCount",
        ];

        let url = self.build_search_url(query, &fields, limit, offset);
        debug!("Searching Semantic Scholar: {}", url);

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
            let status = response.status();
            return Err(match status.as_u16() {
                429 => ProviderError::RateLimit,
                503 => ProviderError::ServiceUnavailable(
                    "Semantic Scholar service temporarily unavailable".to_string(),
                ),
                _ => ProviderError::Network(format!(
                    "API request failed with status: {}",
                    status
                )),
            });
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        debug!("Semantic Scholar response: {}", response_text);

        let api_response: SemanticScholarResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!(
                    "Failed to parse Semantic Scholar response: {}",
                    response_text
                );
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        let papers = api_response
            .data
            .into_iter()
            .map(|paper| self.convert_paper(paper))
            .collect();

        Ok(papers)
    }

    /// Get paper by DOI
    async fn get_paper_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let fields = [
            "paperId",
            "externalIds",
            "title",
            "authors",
            "venue",
            "year",
            "abstract",
            "openAccessPdf",
            "publicationDate",
            "journal",
            "citationCount",
        ];

        let url = self.build_doi_url(doi, &fields);
        debug!("Getting paper by DOI from Semantic Scholar: {}", url);

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
            debug!("Paper not found in Semantic Scholar for DOI: {}", doi);
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

        debug!("Semantic Scholar DOI response: {}", response_text);

        let paper: SemanticScholarPaper = serde_json::from_str(&response_text)
            .map_err(|e| ProviderError::Parse(format!("Failed to parse JSON: {e}")))?;

        Ok(Some(self.convert_paper(paper)))
    }
}

#[async_trait]
impl SourceProvider for SemanticScholarProvider {
    fn name(&self) -> &'static str {
        "semantic_scholar"
    }

    fn description(&self) -> &'static str {
        "Semantic Scholar - AI-powered research tool with free PDF access"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Doi,
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Keywords,
            SearchType::Auto,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"Semantic Scholar supports natural language queries:
- Simple keyword search works well
- Use quotes for exact phrases
- Author names are automatically detected
- fieldsOfStudy filter for specific domains
- year filter for publication year range
- openAccessPdf filter for open access only"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("attention is all you need", "Search by paper title"),
            ("Geoffrey Hinton deep learning", "Author and topic search"),
            ("transformer neural networks 2017-2023", "Topic with year range"),
            ("BERT language model", "Search for specific model papers"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://api.semanticscholar.org/api-docs/")
    }

    fn supports_full_text(&self) -> bool {
        true // Semantic Scholar provides open access PDFs when available
    }

    fn priority(&self) -> u8 {
        80 // Kept in default set but after MDPI in ordering
    }

    fn base_delay(&self) -> Duration {
        if self.api_key.is_some() {
            Duration::from_millis(100) // Faster with API key
        } else {
            Duration::from_millis(1000) // Respect rate limits for public API
        }
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching Semantic Scholar for: {} (type: {:?})",
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
            source: "Semantic Scholar".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: papers_count >= query.max_results as usize,
            metadata: HashMap::new(),
        };

        info!(
            "Semantic Scholar search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from Semantic Scholar: {}", doi);
        self.get_paper_by_doi(doi).await
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing Semantic Scholar health check");

        let url = format!("{}/paper/search?query=test&limit=1", self.base_url);

        let mut request = self.client.get(&url);

        // Add API key header if available
        for (key, value) in self.get_headers() {
            request = request.header(&key, &value);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                info!("Semantic Scholar health check: OK");
                Ok(true)
            }
            Ok(response) => {
                warn!(
                    "Semantic Scholar health check failed with status: {}",
                    response.status()
                );
                Ok(false)
            }
            Err(e) => {
                warn!("Semantic Scholar health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

impl Default for SemanticScholarProvider {
    fn default() -> Self {
        match Self::new(None) {
            Ok(provider) => provider,
            Err(_) => {
                // Fallback to a minimal client with very basic configuration
                // This should never fail under normal circumstances
                let client = Client::new();
                Self {
                    client,
                    base_url: "https://api.semanticscholar.org/graph/v1".to_string(),
                    api_key: None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_scholar_provider_creation() {
        let provider = SemanticScholarProvider::new(None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_interface() {
        let provider = SemanticScholarProvider::new(None).unwrap();

        assert_eq!(provider.name(), "semantic_scholar");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 80);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
    }

    #[test]
    fn test_url_building() {
        let provider = SemanticScholarProvider::new(None).unwrap();

        let search_url =
            provider.build_search_url("machine learning", &["title", "authors"], 10, 0);
        assert!(search_url.contains("query=machine%20learning"));
        assert!(search_url.contains("fields=title%2Cauthors"));
        assert!(search_url.contains("limit=10"));

        let doi_url = provider.build_doi_url("10.1038/nature12373", &["title"]);
        assert!(doi_url.contains("DOI:10.1038%2Fnature12373"));
        assert!(doi_url.contains("fields=title"));
    }
}
