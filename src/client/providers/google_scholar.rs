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

/// Minimal subset of the SerpAPI Google Scholar response we care about
#[derive(Debug, Deserialize)]
struct SerpApiResult {
    #[serde(default)]
    organic_results: Vec<SerpOrganicResult>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SerpOrganicResult {
    title: Option<String>,
    link: Option<String>,
    #[serde(default)]
    publication_info: PublicationInfo,
    #[serde(default)]
    resources: Vec<Resource>,
}

#[derive(Debug, Default, Deserialize)]
struct PublicationInfo {
    #[serde(default)]
    authors: Vec<Author>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    year: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct Author {
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Resource {
    file_format: Option<String>,
    link: Option<String>,
}

/// Google Scholar provider powered by SerpAPI (requires API key)
pub struct GoogleScholarProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl GoogleScholarProvider {
    /// Create a new Google Scholar provider. Provide `GOOGLE_SCHOLAR_API_KEY` env var for SerpAPI access.
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://serpapi.com".to_string(),
            api_key,
        })
    }

    fn build_search_url(&self, query: &SearchQuery, api_key: &str) -> String {
        format!(
            "{}/search.json?engine=google_scholar&q={}&api_key={}&num={}&start={}",
            self.base_url,
            urlencoding::encode(&query.query),
            api_key,
            query.max_results,
            query.offset
        )
    }

    fn extract_pdf_url(resources: &[Resource]) -> Option<String> {
        resources.iter().find_map(|res| {
            if let Some(fmt) = &res.file_format {
                if fmt.to_lowercase().contains("pdf") {
                    return res.link.clone();
                }
            }
            res.link
                .as_ref()
                .filter(|link| link.to_lowercase().ends_with(".pdf"))
                .cloned()
        })
    }
}

#[async_trait]
impl SourceProvider for GoogleScholarProvider {
    fn name(&self) -> &'static str {
        "google_scholar"
    }

    fn description(&self) -> &'static str {
        "Google Scholar via SerpAPI (metadata-oriented, PDF links when available)"
    }

    fn priority(&self) -> u8 {
        92 // Prioritize just below PubMed Central
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Auto,
            SearchType::Title,
            SearchType::Author,
            SearchType::Keywords,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"Google Scholar supports natural language queries:
- Comprehensive academic search coverage
- author:name - Search by author
- intitle:term - Search in title only
- "phrase" - Exact phrase matching
- site:domain - Limit to specific site
- Requires GOOGLE_SCHOLAR_API_KEY (SerpAPI)"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("machine learning neural networks", "Basic keyword search"),
            ("author:\"Geoffrey Hinton\" deep learning", "Author-specific search"),
            ("intitle:transformer attention", "Title search"),
            ("\"attention is all you need\"", "Exact phrase search"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://scholar.google.com/intl/en/scholar/help.html")
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(1200)
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        // Soft-disable if no API key provided
        let api_key = match &self.api_key {
            Some(key) => key,
            None => {
                let search_time = start_time.elapsed();
                warn!("Google Scholar provider disabled: missing GOOGLE_SCHOLAR_API_KEY");
                return Ok(ProviderResult {
                    papers: Vec::new(),
                    source: "Google Scholar (api-key missing)".to_string(),
                    total_available: Some(0),
                    search_time,
                    has_more: false,
                    metadata: HashMap::from([(
                        "note".to_string(),
                        "Provide GOOGLE_SCHOLAR_API_KEY (SerpAPI) to enable Google Scholar search"
                            .to_string(),
                    )]),
                });
            }
        };

        let url = self.build_search_url(query, api_key);
        debug!("Google Scholar search URL: {}", url);

        let response = self
            .client
            .get(&url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if response.status().as_u16() == 429 {
            return Err(ProviderError::RateLimit);
        }

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "Request failed with status: {}",
                response.status()
            )));
        }

        let parsed: SerpApiResult = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse Google Scholar JSON: {e}"))
        })?;

        if let Some(err) = parsed.error {
            return Err(ProviderError::Other(format!(
                "Google Scholar API error: {err}"
            )));
        }

        let papers: Vec<PaperMetadata> = parsed
            .organic_results
            .iter()
            .take(query.max_results as usize)
            .map(|item| {
                let authors: Vec<String> = item
                    .publication_info
                    .authors
                    .iter()
                    .filter_map(|a| a.name.clone())
                    .collect();

                let journal = item
                    .publication_info
                    .summary
                    .clone()
                    .or_else(|| item.link.clone());

                let pdf_url = Self::extract_pdf_url(&item.resources);

                PaperMetadata {
                    doi: String::new(), // Scholar rarely surfaces DOI directly
                    title: item.title.clone(),
                    authors,
                    journal,
                    year: item.publication_info.year,
                    abstract_text: None,
                    pdf_url,
                    file_size: None,
                }
            })
            .collect();

        let search_time = start_time.elapsed();
        info!(
            "Google Scholar search completed: {} papers found in {:?}",
            papers.len(),
            search_time
        );

        Ok(ProviderResult {
            papers,
            source: "Google Scholar".to_string(),
            total_available: None,
            search_time,
            has_more: false,
            metadata: HashMap::new(),
        })
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        // If an API key is provided, assume healthy; otherwise soft-disabled but healthy
        Ok(true)
    }
}
