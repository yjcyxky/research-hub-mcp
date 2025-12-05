use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::rate_limiter::ProviderRateLimiter;
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use url::Url;

/// OpenAlex API provider for academic papers
///
/// OpenAlex is a fully open catalog of scholarly papers, authors, venues, institutions, and concepts.
/// It provides access to 240+ million scholarly works with no authentication requirements.
///
/// Rate limit: 100,000 requests per day (approximately 1.15 requests per second)
/// API Documentation: https://docs.openalex.org/
pub struct OpenAlexProvider {
    client: Client,
    base_url: String,
    rate_limiter: Arc<Mutex<Option<ProviderRateLimiter>>>,
}

/// OpenAlex Work response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAlexWork {
    id: String,
    doi: Option<String>,
    title: Option<String>,
    #[serde(default)]
    authorships: Vec<Authorship>,
    publication_year: Option<u32>,
    primary_location: Option<Location>,
    best_oa_location: Option<Location>,
    abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct Authorship {
    author: AuthorInfo,
}

#[derive(Debug, Clone, Deserialize)]
struct AuthorInfo {
    display_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Location {
    source: Option<SourceInfo>,
    pdf_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SourceInfo {
    display_name: Option<String>,
}

/// OpenAlex API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAlexResponse {
    results: Vec<OpenAlexWork>,
    meta: MetaInfo,
}

#[derive(Debug, Clone, Deserialize)]
struct MetaInfo {
    count: u32,
    #[serde(default)]
    per_page: u32,
    next_cursor: Option<String>,
}

impl OpenAlexProvider {
    /// Create a new OpenAlex provider
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.6.6 (Academic Research Tool; mailto:cthomasbrittain@hotmail.com)")
            .build()
            .map_err(|e| ProviderError::Other(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.openalex.org/works".to_string(),
            rate_limiter: Arc::new(Mutex::new(None)),
        })
    }

    /// Initialize rate limiter with configuration
    pub async fn init_rate_limiter(&self, config: &crate::config::RateLimitingConfig) {
        let limiter = ProviderRateLimiter::new("openalex".to_string(), config);
        *self.rate_limiter.lock().await = Some(limiter);
        debug!("Initialized rate limiter for OpenAlex provider");
    }

    /// Build OpenAlex API URL for search
    fn build_search_url(&self, query: &SearchQuery) -> Result<String, ProviderError> {
        let mut url = Url::parse(&self.base_url)
            .map_err(|e| ProviderError::Other(format!("Invalid base URL: {e}")))?;

        // Build search filter based on query type
        let filter = match query.search_type {
            SearchType::Doi => {
                // Ensure DOI format is correct for OpenAlex
                let doi = if query.query.starts_with("http") {
                    query.query.clone()
                } else if query.query.starts_with("10.") {
                    format!("https://doi.org/{}", query.query)
                } else {
                    return Err(ProviderError::InvalidQuery(
                        "Invalid DOI format".to_string(),
                    ));
                };
                format!("doi:{}", doi)
            }
            SearchType::Title => {
                if !query.query.trim().is_empty() {
                    format!("title.search:{}", query.query)
                } else {
                    return Err(ProviderError::InvalidQuery("Empty query".to_string()));
                }
            }
            SearchType::Author => {
                if !query.query.trim().is_empty() {
                    format!("authorships.author.display_name.search:{}", query.query)
                } else {
                    return Err(ProviderError::InvalidQuery("Empty query".to_string()));
                }
            }
            SearchType::TitleAbstract | SearchType::Keywords | SearchType::Auto => {
                // For keywords/auto, search in abstract or use default search
                if !query.query.trim().is_empty() {
                    format!("default.search:{}", query.query)
                } else {
                    return Err(ProviderError::InvalidQuery("Empty query".to_string()));
                }
            }
            SearchType::Subject => {
                // OpenAlex uses concepts for subjects
                if !query.query.trim().is_empty() {
                    format!("concepts.display_name.search:{}", query.query)
                } else {
                    return Err(ProviderError::InvalidQuery("Empty query".to_string()));
                }
            }
        };

        url.query_pairs_mut()
            .append_pair("filter", &filter)
            .append_pair("per-page", &query.max_results.min(200).to_string()) // OpenAlex max is 200 per page
            .append_pair("cursor", &if query.offset == 0 { "*".to_string() } else { format!("offset:{}", query.offset) })
            .append_pair("sort", "relevance_score:desc")
            .append_pair("select", "id,doi,title,authorships,publication_year,primary_location,best_oa_location,abstract_inverted_index");

        Ok(url.to_string())
    }

    /// Parse OpenAlex API response into PaperMetadata
    fn parse_response(
        &self,
        response: OpenAlexResponse,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let mut papers = Vec::new();

        for work in response.results {
            // Extract DOI - remove the https://doi.org/ prefix if present
            let doi = work
                .doi
                .as_ref()
                .map(|d| {
                    if d.starts_with("https://doi.org/") {
                        d.strip_prefix("https://doi.org/").unwrap_or(d).to_string()
                    } else {
                        d.clone()
                    }
                })
                .unwrap_or_else(|| {
                    // Generate a pseudo-DOI from OpenAlex ID if no real DOI exists
                    work.id
                        .strip_prefix("https://openalex.org/")
                        .unwrap_or(&work.id)
                        .to_string()
                });

            // Extract authors
            let authors: Vec<String> = work
                .authorships
                .iter()
                .filter_map(|authorship| authorship.author.display_name.clone())
                .collect();

            // Extract journal name from primary location
            let journal = work
                .primary_location
                .as_ref()
                .and_then(|loc| loc.source.as_ref())
                .and_then(|src| src.display_name.clone());

            // Extract PDF URL from best open access location
            let pdf_url = work
                .best_oa_location
                .as_ref()
                .and_then(|loc| loc.pdf_url.clone())
                .filter(|url| !url.is_empty());

            // Reconstruct abstract from inverted index
            let abstract_text = Self::reconstruct_abstract(work.abstract_inverted_index.as_ref());

            let paper = PaperMetadata {
                doi,
                title: work.title,
                authors,
                journal,
                year: work.publication_year,
                abstract_text,
                pdf_url,
                file_size: None, // OpenAlex doesn't provide file size
            };

            papers.push(paper);
        }

        debug!("Parsed {} papers from OpenAlex response", papers.len());
        Ok(papers)
    }

    /// Reconstruct abstract text from OpenAlex's inverted index format
    fn reconstruct_abstract(inverted_index: Option<&HashMap<String, Vec<u32>>>) -> Option<String> {
        let index = inverted_index?;

        if index.is_empty() {
            return None;
        }

        // Create a vector to hold words at their positions
        let max_position = index
            .values()
            .flat_map(|positions| positions.iter())
            .max()
            .copied()? as usize;

        let mut words = vec![None; max_position + 1];

        // Place each word at its positions
        for (word, positions) in index {
            for &pos in positions {
                if let Some(slot) = words.get_mut(pos as usize) {
                    *slot = Some(word.as_str());
                }
            }
        }

        // Join words, filtering out None values
        let abstract_text: String = words.into_iter().flatten().collect::<Vec<&str>>().join(" ");

        if abstract_text.trim().is_empty() {
            None
        } else {
            Some(abstract_text)
        }
    }

    /// Apply rate limiting before making requests
    async fn apply_rate_limit(&self) -> Result<(), ProviderError> {
        let mut rate_limiter = self.rate_limiter.lock().await;
        if let Some(ref mut limiter) = *rate_limiter {
            let _ = limiter.acquire().await;
        } else {
            // Default delay if no rate limiter is configured
            tokio::time::sleep(Duration::from_millis(870)).await; // ~1.15 req/s
        }
        Ok(())
    }
}

#[async_trait]
impl SourceProvider for OpenAlexProvider {
    fn name(&self) -> &'static str {
        "openalex"
    }

    fn priority(&self) -> u8 {
        180 // High priority - authoritative source with good coverage
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(870) // ~1.15 requests per second to stay under 100k/day limit
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Doi,
            SearchType::Keywords,
            SearchType::Subject,
            SearchType::Auto,
        ]
    }

    fn description(&self) -> &'static str {
        "OpenAlex: Open catalog of scholarly papers, authors, venues, and institutions with 240M+ works"
    }

    fn supports_full_text(&self) -> bool {
        true // Many papers have open access PDFs
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let search_start = Instant::now();

        debug!(
            "Starting OpenAlex search: query='{}', type={:?}, max_results={}",
            query.query, query.search_type, query.max_results
        );

        // Validate query
        if query.query.trim().is_empty() {
            return Err(ProviderError::InvalidQuery("Empty query".to_string()));
        }

        // Apply rate limiting
        self.apply_rate_limit().await?;

        // Build search URL
        let search_url = self.build_search_url(query)?;
        debug!("OpenAlex search URL: {}", search_url);

        // Execute search with timeout
        let response = tokio::time::timeout(context.timeout, {
            let client = &self.client;
            let url = search_url.clone();

            async move {
                client
                    .get(&url)
                    .headers({
                        let mut headers = reqwest::header::HeaderMap::new();
                        headers.insert(
                            reqwest::header::USER_AGENT,
                            reqwest::header::HeaderValue::from_static(
                                "knowledge_accumulator_mcp/0.6.6 (Academic Research Tool)",
                            ),
                        );
                        // Add custom headers from context if any
                        for (key, value) in &context.headers {
                            if let (Ok(header_name), Ok(header_value)) = (
                                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                                reqwest::header::HeaderValue::from_str(value),
                            ) {
                                headers.insert(header_name, header_value);
                            }
                        }
                        headers
                    })
                    .send()
                    .await
            }
        })
        .await
        .map_err(|_| ProviderError::Timeout)?
        .map_err(|e| ProviderError::Network(format!("HTTP request failed: {e}")))?;

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            return match status.as_u16() {
                429 => Err(ProviderError::RateLimit),
                401 | 403 => Err(ProviderError::Auth(format!(
                    "Authentication failed: {}",
                    status
                ))),
                400 => Err(ProviderError::InvalidQuery(error_text)),
                _ => Err(ProviderError::ServiceUnavailable(format!(
                    "HTTP {}: {}",
                    status, error_text
                ))),
            };
        }

        // Parse response
        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

        let openalex_response: OpenAlexResponse = serde_json::from_str(&response_text)
            .map_err(|e| ProviderError::Parse(format!("Failed to parse JSON response: {e}")))?;

        // Convert to papers
        let papers = self.parse_response(openalex_response.clone())?;
        let search_time = search_start.elapsed();

        info!(
            "OpenAlex search completed: {} papers found in {:?}",
            papers.len(),
            search_time
        );

        // Prepare metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "total_results".to_string(),
            openalex_response.meta.count.to_string(),
        );
        metadata.insert(
            "per_page".to_string(),
            openalex_response.meta.per_page.to_string(),
        );
        if let Some(ref cursor) = openalex_response.meta.next_cursor {
            metadata.insert("next_cursor".to_string(), cursor.clone());
        }

        Ok(ProviderResult {
            papers,
            source: self.name().to_string(),
            total_available: Some(openalex_response.meta.count),
            search_time,
            has_more: openalex_response.meta.next_cursor.is_some(),
            metadata,
        })
    }

    async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing OpenAlex health check");

        let health_url = "https://api.openalex.org/works?filter=title.search:test&per-page=1";

        let response = tokio::time::timeout(context.timeout, self.client.get(health_url).send())
            .await
            .map_err(|_| ProviderError::Timeout)?
            .map_err(|e| ProviderError::Network(format!("Health check failed: {e}")))?;

        let is_healthy = response.status().is_success();

        if is_healthy {
            debug!("OpenAlex health check passed");
        } else {
            warn!("OpenAlex health check failed: {}", response.status());
        }

        Ok(is_healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::providers::traits::{SearchQuery, SearchType};
    use std::collections::HashMap;
    use std::time::Duration;

    #[tokio::test]
    async fn test_openalex_provider_creation() {
        let provider = OpenAlexProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_openalex_search_url_building() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "machine learning".to_string(),
            search_type: SearchType::Keywords,
            max_results: 50,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=default.search%3Amachine+learning"));
        assert!(url.contains("per-page=50"));
        assert!(url.contains("cursor=*"));
    }

    #[test]
    fn test_openalex_doi_search_url() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Doi,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=doi%3Ahttps%3A%2F%2Fdoi.org%2F10.1038%2Fnature12373"));
    }

    #[test]
    fn test_title_search_url() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "Attention Is All You Need".to_string(),
            search_type: SearchType::Title,
            max_results: 20,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=title.search%3AAttention+Is+All+You+Need"));
        assert!(url.contains("per-page=20"));
        assert!(url.contains("cursor=*"));
    }

    #[test]
    fn test_author_search_url() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "Geoffrey Hinton".to_string(),
            search_type: SearchType::Author,
            max_results: 30,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=authorships.author.display_name.search%3AGeoffrey+Hinton"));
        assert!(url.contains("per-page=30"));
    }

    #[test]
    fn test_subject_search_url() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "Artificial Intelligence".to_string(),
            search_type: SearchType::Subject,
            max_results: 15,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=concepts.display_name.search%3AArtificial+Intelligence"));
        assert!(url.contains("per-page=15"));
    }

    #[test]
    fn test_auto_search_url() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "neural networks deep learning".to_string(),
            search_type: SearchType::Auto,
            max_results: 25,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("filter=default.search%3Aneural+networks+deep+learning"));
        assert!(url.contains("per-page=25"));
    }

    #[test]
    fn test_pagination_offset() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "machine learning".to_string(),
            search_type: SearchType::Keywords,
            max_results: 100,
            offset: 200,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("cursor=offset%3A200"));
        assert!(url.contains("per-page=100"));
    }

    #[test]
    fn test_max_results_clamping() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "test query".to_string(),
            search_type: SearchType::Keywords,
            max_results: 500, // Above OpenAlex's max of 200
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let url = provider.build_search_url(&query).unwrap();
        assert!(url.contains("per-page=200")); // Should be clamped to 200
    }

    #[test]
    fn test_abstract_reconstruction() {
        let mut inverted_index = HashMap::new();
        inverted_index.insert("This".to_string(), vec![0]);
        inverted_index.insert("is".to_string(), vec![1]);
        inverted_index.insert("a".to_string(), vec![2]);
        inverted_index.insert("test".to_string(), vec![3]);
        inverted_index.insert("abstract".to_string(), vec![4]);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        assert_eq!(abstract_text, Some("This is a test abstract".to_string()));
    }

    #[test]
    fn test_empty_abstract_reconstruction() {
        let inverted_index = HashMap::new();
        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        assert_eq!(abstract_text, None);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(None);
        assert_eq!(abstract_text, None);
    }

    #[test]
    fn test_provider_properties() {
        let provider = OpenAlexProvider::new().unwrap();

        assert_eq!(provider.name(), "openalex");
        assert_eq!(provider.priority(), 180);
        assert_eq!(provider.base_delay(), Duration::from_millis(870));
        assert!(provider.supports_full_text());

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
        assert!(supported_types.contains(&SearchType::Doi));
        assert!(supported_types.contains(&SearchType::Keywords));
        assert!(supported_types.contains(&SearchType::Subject));
        assert!(supported_types.contains(&SearchType::Auto));
    }

    #[test]
    fn test_parse_complete_response() {
        let provider = OpenAlexProvider::new().unwrap();

        // Mock OpenAlex response with complete data
        let response = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W2741809807".to_string(),
                doi: Some("https://doi.org/10.1038/nature12373".to_string()),
                title: Some("Attention Is All You Need".to_string()),
                authorships: vec![
                    Authorship {
                        author: AuthorInfo {
                            display_name: Some("Ashish Vaswani".to_string()),
                        },
                    },
                    Authorship {
                        author: AuthorInfo {
                            display_name: Some("Noam Shazeer".to_string()),
                        },
                    },
                ],
                publication_year: Some(2017),
                primary_location: Some(Location {
                    source: Some(SourceInfo {
                        display_name: Some("Nature".to_string()),
                    }),
                    pdf_url: None,
                }),
                best_oa_location: Some(Location {
                    source: None,
                    pdf_url: Some("https://arxiv.org/pdf/1706.03762.pdf".to_string()),
                }),
                abstract_inverted_index: {
                    let mut index = HashMap::new();
                    index.insert("The".to_string(), vec![0]);
                    index.insert("dominant".to_string(), vec![1]);
                    index.insert("sequence".to_string(), vec![2]);
                    index.insert("models".to_string(), vec![3]);
                    Some(index)
                },
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers = provider.parse_response(response).unwrap();

        assert_eq!(papers.len(), 1);
        let paper = &papers[0];

        assert_eq!(paper.doi, "10.1038/nature12373"); // Should strip prefix
        assert_eq!(paper.title, Some("Attention Is All You Need".to_string()));
        assert_eq!(paper.authors, vec!["Ashish Vaswani", "Noam Shazeer"]);
        assert_eq!(paper.journal, Some("Nature".to_string()));
        assert_eq!(paper.year, Some(2017));
        assert_eq!(
            paper.pdf_url,
            Some("https://arxiv.org/pdf/1706.03762.pdf".to_string())
        );
        assert_eq!(
            paper.abstract_text,
            Some("The dominant sequence models".to_string())
        );
    }

    #[test]
    fn test_parse_minimal_response() {
        let provider = OpenAlexProvider::new().unwrap();

        // Mock OpenAlex response with minimal data
        let response = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W12345".to_string(),
                doi: None,
                title: None,
                authorships: vec![],
                publication_year: None,
                primary_location: None,
                best_oa_location: None,
                abstract_inverted_index: None,
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers = provider.parse_response(response).unwrap();

        assert_eq!(papers.len(), 1);
        let paper = &papers[0];

        assert_eq!(paper.doi, "W12345"); // Should use OpenAlex ID as fallback
        assert_eq!(paper.title, None);
        assert!(paper.authors.is_empty());
        assert_eq!(paper.journal, None);
        assert_eq!(paper.year, None);
        assert_eq!(paper.pdf_url, None);
        assert_eq!(paper.abstract_text, None);
    }

    #[test]
    fn test_doi_extraction_variations() {
        let provider = OpenAlexProvider::new().unwrap();

        // Test DOI with https prefix
        let response1 = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W1".to_string(),
                doi: Some("https://doi.org/10.1000/test1".to_string()),
                title: None,
                authorships: vec![],
                publication_year: None,
                primary_location: None,
                best_oa_location: None,
                abstract_inverted_index: None,
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers1 = provider.parse_response(response1).unwrap();
        assert_eq!(papers1[0].doi, "10.1000/test1");

        // Test DOI without prefix
        let response2 = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W2".to_string(),
                doi: Some("10.1000/test2".to_string()),
                title: None,
                authorships: vec![],
                publication_year: None,
                primary_location: None,
                best_oa_location: None,
                abstract_inverted_index: None,
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers2 = provider.parse_response(response2).unwrap();
        assert_eq!(papers2[0].doi, "10.1000/test2");
    }

    #[test]
    fn test_author_extraction_variations() {
        let provider = OpenAlexProvider::new().unwrap();

        let response = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W1".to_string(),
                doi: None,
                title: None,
                authorships: vec![
                    Authorship {
                        author: AuthorInfo {
                            display_name: Some("John Doe".to_string()),
                        },
                    },
                    Authorship {
                        author: AuthorInfo {
                            display_name: None, // Missing name
                        },
                    },
                    Authorship {
                        author: AuthorInfo {
                            display_name: Some("Jane Smith".to_string()),
                        },
                    },
                ],
                publication_year: None,
                primary_location: None,
                best_oa_location: None,
                abstract_inverted_index: None,
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers = provider.parse_response(response).unwrap();
        assert_eq!(papers[0].authors, vec!["John Doe", "Jane Smith"]); // Should skip author without name
    }

    #[test]
    fn test_pdf_url_filtering() {
        let provider = OpenAlexProvider::new().unwrap();

        let response = OpenAlexResponse {
            results: vec![OpenAlexWork {
                id: "https://openalex.org/W1".to_string(),
                doi: None,
                title: None,
                authorships: vec![],
                publication_year: None,
                primary_location: None,
                best_oa_location: Some(Location {
                    source: None,
                    pdf_url: Some("".to_string()), // Empty URL should be filtered
                }),
                abstract_inverted_index: None,
            }],
            meta: MetaInfo {
                count: 1,
                per_page: 1,
                next_cursor: None,
            },
        };

        let papers = provider.parse_response(response).unwrap();
        assert_eq!(papers[0].pdf_url, None); // Empty URL should be filtered out
    }

    // Error Handling Tests
    #[test]
    fn test_empty_query_error() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "".to_string(), // Empty query
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let result = provider.build_search_url(&query);
        assert!(result.is_err());

        if let Err(ProviderError::InvalidQuery(msg)) = result {
            assert!(msg.contains("Empty query"));
        } else {
            panic!("Expected InvalidQuery error for empty query");
        }
    }

    #[test]
    fn test_whitespace_only_query_error() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "   \t\n   ".to_string(), // Whitespace only
            search_type: SearchType::Title,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let result = provider.build_search_url(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_doi_format_error() {
        let provider = OpenAlexProvider::new().unwrap();

        let query = SearchQuery {
            query: "invalid-doi-format".to_string(), // Invalid DOI
            search_type: SearchType::Doi,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        let result = provider.build_search_url(&query);
        assert!(result.is_err());

        if let Err(ProviderError::InvalidQuery(msg)) = result {
            assert!(msg.contains("Invalid DOI format"));
        } else {
            panic!("Expected InvalidQuery error for invalid DOI");
        }
    }

    #[test]
    fn test_valid_doi_formats() {
        let provider = OpenAlexProvider::new().unwrap();

        // Test various valid DOI formats
        let valid_dois = vec![
            "10.1000/test",
            "https://doi.org/10.1000/test",
            "http://doi.org/10.1000/test",
        ];

        for doi in valid_dois {
            let query = SearchQuery {
                query: doi.to_string(),
                search_type: SearchType::Doi,
                max_results: 10,
                offset: 0,
                params: HashMap::new(),
                sources: None,
                metadata_sources: None,
            };

            let result = provider.build_search_url(&query);
            assert!(result.is_ok(), "DOI '{}' should be valid", doi);
        }
    }

    // Complex Abstract Reconstruction Tests
    #[test]
    fn test_complex_abstract_reconstruction() {
        // Test with words appearing multiple times at different positions
        let mut inverted_index = HashMap::new();
        inverted_index.insert("the".to_string(), vec![0, 5, 8]);
        inverted_index.insert("quick".to_string(), vec![1]);
        inverted_index.insert("brown".to_string(), vec![2]);
        inverted_index.insert("fox".to_string(), vec![3]);
        inverted_index.insert("jumps".to_string(), vec![4]);
        inverted_index.insert("over".to_string(), vec![6]);
        inverted_index.insert("lazy".to_string(), vec![7]);
        inverted_index.insert("dog".to_string(), vec![9]);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        assert_eq!(
            abstract_text,
            Some("the quick brown fox jumps the over lazy the dog".to_string())
        );
    }

    #[test]
    fn test_abstract_with_gaps() {
        // Test with missing position indices (gaps)
        let mut inverted_index = HashMap::new();
        inverted_index.insert("This".to_string(), vec![0]);
        inverted_index.insert("has".to_string(), vec![2]); // Position 1 is missing
        inverted_index.insert("gaps".to_string(), vec![4]); // Position 3 is missing

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        // Should handle gaps gracefully by only including available words
        assert_eq!(abstract_text, Some("This has gaps".to_string()));
    }

    #[test]
    fn test_abstract_with_single_word() {
        let mut inverted_index = HashMap::new();
        inverted_index.insert("Word".to_string(), vec![0]);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        assert_eq!(abstract_text, Some("Word".to_string()));
    }

    #[test]
    fn test_abstract_with_out_of_order_positions() {
        // Test that positions are correctly ordered even if HashMap iteration is unordered
        let mut inverted_index = HashMap::new();
        inverted_index.insert("third".to_string(), vec![2]);
        inverted_index.insert("first".to_string(), vec![0]);
        inverted_index.insert("second".to_string(), vec![1]);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        assert_eq!(abstract_text, Some("first second third".to_string()));
    }

    #[test]
    fn test_large_position_indices() {
        // Test with large position numbers (but keep reasonable for test performance)
        let mut inverted_index = HashMap::new();
        inverted_index.insert("start".to_string(), vec![0]);
        inverted_index.insert("middle".to_string(), vec![50]);
        inverted_index.insert("end".to_string(), vec![100]);

        let abstract_text = OpenAlexProvider::reconstruct_abstract(Some(&inverted_index));
        // Should handle large indices without issues
        assert!(abstract_text.is_some());
        let text = abstract_text.unwrap();
        assert!(text.starts_with("start"));
        assert!(text.contains("middle"));
        assert!(text.ends_with("end"));
    }

    #[test]
    fn test_provider_health_check_properties() {
        let provider = OpenAlexProvider::new().unwrap();

        // Test provider description
        assert!(provider.description().contains("OpenAlex"));
        assert!(provider.description().contains("240"));

        // Test supported search types completeness
        let types = provider.supported_search_types();
        assert_eq!(types.len(), 6);
    }
}
