use crate::client::providers::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info};

/// `OpenReview` provider for machine learning conference papers
///
/// `OpenReview` is a venue for ML conference papers, reviews, and conference proceedings.
/// This provider searches `OpenReview` submissions and provides access to papers
/// from venues like `NeurIPS`, `ICLR`, `ICML`, etc.
pub struct OpenReviewProvider {
    client: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct OpenReviewResponse {
    notes: Vec<OpenReviewNote>,
    #[allow(dead_code)]
    count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OpenReviewNote {
    id: String,
    #[serde(default)]
    content: OpenReviewContent,
    #[serde(default)]
    details: Option<OpenReviewDetails>,
    #[allow(dead_code)]
    forum: Option<String>,
    invitation: Option<String>,
    #[allow(dead_code)]
    readers: Option<Vec<String>>,
    #[allow(dead_code)]
    writers: Option<Vec<String>>,
    #[allow(dead_code)]
    signatures: Option<Vec<String>>,
    #[serde(default)]
    tcdate: Option<u64>, // Creation timestamp
    #[serde(default)]
    #[allow(dead_code)]
    tmdate: Option<u64>, // Modification timestamp
}

#[derive(Debug, Deserialize, Default)]
struct OpenReviewContent {
    title: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    authors: Option<Vec<String>>,
    #[allow(dead_code)]
    authorids: Option<Vec<String>>,
    pdf: Option<String>,
    venue: Option<String>,
    #[serde(rename = "_bibtex")]
    #[allow(dead_code)]
    bibtex: Option<String>,
    #[allow(dead_code)]
    keywords: Option<Vec<String>>,
    #[serde(rename = "subject_areas")]
    #[allow(dead_code)]
    subject_areas: Option<Vec<String>>,
    #[serde(rename = "track")]
    #[allow(dead_code)]
    track: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenReviewDetails {
    #[serde(rename = "originalPdf")]
    original_pdf: Option<String>,
    #[serde(rename = "presentation")]
    #[allow(dead_code)]
    presentation: Option<String>,
    #[serde(rename = "directPdfLink")]
    direct_pdf_link: Option<String>,
}

impl OpenReviewProvider {
    /// Create a new `OpenReview` provider
    pub fn new() -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("knowledge_accumulator_mcp/0.3.0 (Academic Research Tool)")
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.openreview.net".to_string(),
        })
    }

    /// Search `OpenReview` using their API
    async fn search_openreview(
        &self,
        query: &str,
        max_results: usize,
        context: &SearchContext,
    ) -> Result<Vec<OpenReviewNote>, ProviderError> {
        let search_url = format!("{}/notes", self.base_url);

        let max_results_str = max_results.to_string();
        let offset_str = "0".to_string();
        let details_str = "directPdfLink,originalPdf".to_string();
        let mut params = vec![
            ("limit", &max_results_str),
            ("offset", &offset_str),
            ("details", &details_str),
        ];

        // Build content search query
        let content_query = format!("content.title:{query} OR content.abstract:{query}");
        params.push(("content", &content_query));

        let response = self
            .client
            .get(&search_url)
            .query(&params)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| {
                ProviderError::Network(format!("OpenReview search request failed: {e}"))
            })?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "OpenReview search failed with status: {}",
                response.status()
            )));
        }

        let search_result: OpenReviewResponse = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse OpenReview response: {e}"))
        })?;

        debug!(
            "OpenReview search found {} results for query: '{}'",
            search_result.notes.len(),
            query
        );

        Ok(search_result.notes)
    }

    /// Build query string for different search types
    fn build_query(query: &SearchQuery) -> String {
        match query.search_type {
            SearchType::Title => {
                // Search specifically in title
                query.query.clone()
            }
            SearchType::Author => {
                // Search in authors field
                query.query.clone()
            }
            SearchType::TitleAbstract
            | SearchType::Keywords
            | SearchType::Auto
            | SearchType::Subject => {
                // General search across title and abstract
                query.query.clone()
            }
            SearchType::Doi => {
                // OpenReview doesn't typically use DOIs, but we can try title search
                query.query.clone()
            }
        }
    }

    /// Convert `OpenReview` note to `PaperMetadata`
    fn convert_to_paper(&self, note: &OpenReviewNote) -> PaperMetadata {
        let authors = note.content.authors.clone().unwrap_or_default();

        // Build OpenReview URL for the paper (commented out as it's not used currently)
        // let _paper_url = format!("https://openreview.net/forum?id={}", note.id);

        // Try to get PDF URL from different sources
        let pdf_url = note
            .content
            .pdf
            .clone()
            .or_else(|| {
                note.details
                    .as_ref()
                    .and_then(|d| d.direct_pdf_link.clone())
            })
            .or_else(|| note.details.as_ref().and_then(|d| d.original_pdf.clone()))
            .map(|url| {
                if url.starts_with("http") {
                    url
                } else if url.starts_with("/pdf/") {
                    format!("https://openreview.net{url}")
                } else {
                    format!("https://openreview.net/pdf?id={}", note.id)
                }
            })
            .filter(|url| !url.is_empty());

        // Extract year from timestamp or venue
        let year = note
            .tcdate
            .map(|ts| {
                // Convert timestamp (milliseconds) to year

                u32::try_from(ts / 1000 / 60 / 60 / 24 / 365).unwrap_or(0) + 1970
            })
            .or_else(|| {
                // Try to extract year from venue string
                note.content.venue.as_ref().and_then(|venue| {
                    Regex::new(r"(\d{4})")
                        .ok()?
                        .captures(venue)?
                        .get(1)?
                        .as_str()
                        .parse::<u32>()
                        .ok()
                })
            });

        // Use venue or derive from invitation
        let journal = note.content.venue.clone().or_else(|| {
            note.invitation.as_ref().and_then(|inv| {
                // Extract conference name from invitation
                inv.split('/').nth(0).map(str::to_uppercase)
            })
        });

        PaperMetadata {
            doi: String::new(), // OpenReview papers typically don't have DOIs
            pmid: None,
            title: note.content.title.clone(),
            authors,
            abstract_text: note.content.abstract_text.clone(),
            keywords: Vec::new(),
            journal,
            year,
            pdf_url,
            file_size: None, // File size not available from OpenReview API
        }
    }

    /// Check if this looks like a machine learning paper
    fn is_ml_paper(&self, title: &str, abstract_text: &str, venue: &str) -> bool {
        let ml_keywords = [
            "machine learning",
            "deep learning",
            "neural network",
            "artificial intelligence",
            "ai",
            "ml",
            "transformer",
            "attention",
            "convolution",
            "gradient",
            "optimization",
            "classification",
            "regression",
            "supervised",
            "unsupervised",
            "reinforcement",
            "pytorch",
            "tensorflow",
            "keras",
            "scikit",
            "algorithm",
            "model",
            "training",
            "inference",
            "prediction",
            "feature",
            "dataset",
            "benchmark",
            "nlp",
            "computer vision",
            "cv",
            "natural language",
            "image recognition",
            "speech",
            "generative",
            "gan",
            "autoencoder",
            "lstm",
            "rnn",
            "cnn",
            "bert",
            "gpt",
            "language model",
        ];

        let ml_venues = [
            "neurips",
            "nips",
            "iclr",
            "icml",
            "aaai",
            "ijcai",
            "acl",
            "emnlp",
            "naacl",
            "iccv",
            "cvpr",
            "eccv",
            "kdd",
            "www",
            "wsdm",
            "recsys",
            "colt",
            "aistats",
            "uai",
            "aamas",
            "coling",
            "interspeech",
            "sigir",
        ];

        let combined_text = format!(
            "{} {} {}",
            title.to_lowercase(),
            abstract_text.to_lowercase(),
            venue.to_lowercase()
        );

        ml_keywords
            .iter()
            .any(|&keyword| combined_text.contains(keyword))
            || ml_venues
                .iter()
                .any(|&venue_name| venue.to_lowercase().contains(venue_name))
    }
}

#[async_trait]
impl SourceProvider for OpenReviewProvider {
    fn name(&self) -> &'static str {
        "openreview"
    }

    fn description(&self) -> &'static str {
        "OpenReview provider for ML conference papers and proceedings"
    }

    fn priority(&self) -> u8 {
        85 // High priority for ML papers
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Keywords,
            SearchType::Auto,
            SearchType::Subject,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"OpenReview hosts ML/AI conference papers:
- Search by title, author, or keywords
- Covers NeurIPS, ICLR, ICML submissions
- Includes reviews and discussion
- venue:NeurIPS - Filter by conference
- year:YYYY - Filter by year"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("transformer attention mechanism", "Topic search"),
            ("Geoffrey Hinton", "Author search"),
            ("NeurIPS 2023 language model", "Conference-specific search"),
            ("reinforcement learning robotics", "Multi-topic search"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://docs.openreview.net/reference/api-v2")
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        info!("Searching OpenReview for: '{}'", query.query);

        let search_query = Self::build_query(query);
        let notes = self
            .search_openreview(&search_query, query.max_results as usize, context)
            .await?;

        if notes.is_empty() {
            info!(
                "No results found in OpenReview for query: '{}'",
                query.query
            );
            return Ok(ProviderResult {
                papers: vec![],
                source: self.name().to_string(),
                total_available: Some(0),
                search_time: Duration::from_millis(0),
                has_more: false,
                metadata: HashMap::new(),
            });
        }

        let start_time = std::time::Instant::now();

        let papers: Vec<PaperMetadata> = notes
            .iter()
            .map(|note| self.convert_to_paper(note))
            .filter(|paper| {
                // Filter for ML-relevant papers
                let title = paper.title.as_deref().unwrap_or("");
                let abstract_text = paper.abstract_text.as_deref().unwrap_or("");
                let venue = paper.journal.as_deref().unwrap_or("");
                self.is_ml_paper(title, abstract_text, venue)
            })
            .collect();

        let search_time = start_time.elapsed();
        let papers_count = u32::try_from(papers.len()).unwrap_or(u32::MAX);

        info!(
            "OpenReview found {} ML papers for query: '{}'",
            papers_count, query.query
        );

        Ok(ProviderResult {
            papers,
            source: self.name().to_string(),
            total_available: Some(papers_count),
            search_time,
            has_more: papers_count >= query.max_results,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("filtered_for_ml".to_string(), "true".to_string());
                meta
            },
        })
    }

    async fn get_by_doi(
        &self,
        _doi: &str,
        _context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        // OpenReview papers typically don't have DOIs
        debug!("OpenReview doesn't typically use DOIs for paper identification");
        Ok(None)
    }

    async fn health_check(&self, context: &SearchContext) -> Result<bool, ProviderError> {
        let health_url = format!("{}/notes?limit=1", self.base_url);

        let response = self
            .client
            .get(&health_url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("OpenReview health check failed: {e}")))?;

        if response.status().is_success() {
            debug!("OpenReview health check passed");
            Ok(true)
        } else {
            debug!(
                "OpenReview health check failed with status: {}",
                response.status()
            );
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[allow(dead_code)]
    fn create_test_context() -> SearchContext {
        SearchContext {
            timeout: Duration::from_secs(30),
            user_agent: "test".to_string(),
            rate_limit: None,
            headers: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_openreview_provider_creation() {
        let provider = OpenReviewProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_ml_paper_detection() {
        let provider = OpenReviewProvider::new().unwrap();

        // Should detect ML papers
        assert!(provider.is_ml_paper(
            "Deep Learning for Image Classification",
            "We propose a neural network approach",
            "NeurIPS 2023"
        ));

        assert!(provider.is_ml_paper(
            "Attention Is All You Need",
            "We propose the Transformer model",
            "NIPS 2017"
        ));

        // Should not detect non-ML papers
        assert!(!provider.is_ml_paper(
            "Analysis of Economic Trends",
            "We study market behavior",
            "Economics Journal"
        ));
    }

    #[test]
    fn test_query_building() {
        let _provider = OpenReviewProvider::new().unwrap();

        let title_query = SearchQuery {
            query: "transformer attention mechanism".to_string(),
            search_type: SearchType::Title,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            OpenReviewProvider::build_query(&title_query),
            "transformer attention mechanism"
        );

        let author_query = SearchQuery {
            query: "Yoshua Bengio".to_string(),
            search_type: SearchType::Author,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            OpenReviewProvider::build_query(&author_query),
            "Yoshua Bengio"
        );
    }

    #[test]
    fn test_provider_metadata() {
        let provider = OpenReviewProvider::new().unwrap();

        assert_eq!(provider.name(), "openreview");
        assert_eq!(provider.priority(), 85);
        assert!(provider.supports_full_text());

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
        assert!(supported_types.contains(&SearchType::Keywords));
        assert!(supported_types.contains(&SearchType::Auto));
        assert!(supported_types.contains(&SearchType::Subject));
        assert!(!supported_types.contains(&SearchType::Doi));
    }

    #[test]
    fn test_paper_conversion() {
        let provider = OpenReviewProvider::new().unwrap();

        let note = OpenReviewNote {
            id: "test123".to_string(),
            content: OpenReviewContent {
                title: Some("Test ML Paper".to_string()),
                abstract_text: Some("This is a test abstract about machine learning".to_string()),
                authors: Some(vec!["John Doe".to_string(), "Jane Smith".to_string()]),
                authorids: None,
                pdf: Some("/pdf/test123.pdf".to_string()),
                venue: Some("NeurIPS 2023".to_string()),
                bibtex: None,
                keywords: None,
                subject_areas: None,
                track: None,
            },
            details: None,
            forum: None,
            invitation: Some("NeurIPS.cc/2023/Conference/-/Blind_Submission".to_string()),
            readers: None,
            writers: None,
            signatures: None,
            tcdate: Some(1_640_995_200_000), // 2022-01-01 timestamp
            tmdate: None,
        };

        let paper = provider.convert_to_paper(&note);

        assert_eq!(paper.title.unwrap(), "Test ML Paper");
        assert_eq!(paper.authors.len(), 2);
        assert_eq!(paper.authors[0], "John Doe");
        assert_eq!(paper.journal.unwrap(), "NeurIPS 2023");
        assert!(paper.pdf_url.is_some());
        assert!(paper.pdf_url.unwrap().contains("openreview.net"));
        assert_eq!(paper.year.unwrap(), 2022);
    }
}
