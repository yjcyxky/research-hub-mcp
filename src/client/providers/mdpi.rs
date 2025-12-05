use crate::client::providers::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use regex::Regex;
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, USER_AGENT},
    Client,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// MDPI provider for open access journals
///
/// MDPI (Multidisciplinary Digital Publishing Institute) is a publisher of
/// open access scientific journals covering various disciplines.
/// This provider searches MDPI publications and provides access to papers
/// from journals like Sensors, Materials, Sustainability, etc.
pub struct MdpiProvider {
    client: Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct MdpiSearchResponse {
    #[serde(rename = "searchResults")]
    search_results: MdpiSearchResults,
}

#[derive(Debug, Deserialize)]
struct MdpiSearchResults {
    articles: Vec<MdpiArticle>,
    #[serde(rename = "totalResults")]
    #[allow(dead_code)]
    total_results: Option<u32>,
    #[serde(rename = "currentPage")]
    #[allow(dead_code)]
    current_page: Option<u32>,
    #[serde(rename = "totalPages")]
    #[allow(dead_code)]
    total_pages: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MdpiArticle {
    title: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    authors: Option<Vec<MdpiAuthor>>,
    journal: Option<MdpiJournal>,
    doi: Option<String>,
    #[serde(rename = "pdfUrl")]
    pdf_url: Option<String>,
    #[serde(rename = "htmlUrl")]
    html_url: Option<String>,
    #[serde(rename = "publicationDate")]
    publication_date: Option<String>,
    #[allow(dead_code)]
    volume: Option<String>,
    #[allow(dead_code)]
    issue: Option<String>,
    #[serde(rename = "pageNumbers")]
    #[allow(dead_code)]
    page_numbers: Option<String>,
    #[allow(dead_code)]
    keywords: Option<Vec<String>>,
    #[serde(rename = "articleNumber")]
    #[allow(dead_code)]
    article_number: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MdpiAuthor {
    #[serde(rename = "firstName")]
    first_name: Option<String>,
    #[serde(rename = "lastName")]
    last_name: Option<String>,
    #[serde(rename = "fullName")]
    full_name: Option<String>,
    #[allow(dead_code)]
    affiliation: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MdpiJournal {
    name: Option<String>,
    #[serde(rename = "shortName")]
    #[allow(dead_code)]
    short_name: Option<String>,
    #[allow(dead_code)]
    issn: Option<String>,
    #[serde(rename = "eIssn")]
    #[allow(dead_code)]
    e_issn: Option<String>,
}

impl MdpiProvider {
    /// Create a new MDPI provider
    pub fn new() -> Result<Self, ProviderError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            ),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static(
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            ),
        );
        headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.9"));

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://www.mdpi.com".to_string(),
        })
    }

    /// Search MDPI using their search API or fallback to web scraping
    async fn search_mdpi(
        &self,
        query: &str,
        max_results: usize,
        context: &SearchContext,
    ) -> Result<Vec<MdpiArticle>, ProviderError> {
        // MDPI doesn't have a public API, so we'll construct search URLs
        // and attempt to parse results from their search page
        let search_url = format!("{}/search", self.base_url);

        let max_results_str = max_results.to_string();
        let params = vec![
            ("q", query),
            ("sort", "relevance"),
            ("per_page", &max_results_str),
            ("page", "1"),
            ("format", "json"), // Try JSON first, fallback to HTML if needed
        ];

        let response = self
            .client
            .get(&search_url)
            .query(&params)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("MDPI search request failed: {e}")))?;

        if !response.status().is_success() {
            // Try alternative search approach using their advanced search
            return self.fallback_search(query, max_results, context).await;
        }

        // Try to parse as JSON first
        let response_text = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read MDPI response: {e}")))?;

        // Since MDPI likely doesn't have a JSON API, we'll parse HTML
        if response_text.contains("<!DOCTYPE html") || response_text.contains("<html") {
            return self.parse_html_results(&response_text);
        }

        // Try to parse as JSON if available
        match serde_json::from_str::<MdpiSearchResponse>(&response_text) {
            Ok(search_result) => {
                debug!(
                    "MDPI search found {} results for query: '{}'",
                    search_result.search_results.articles.len(),
                    query
                );
                Ok(search_result.search_results.articles)
            }
            Err(_) => {
                // Fallback to HTML parsing
                self.parse_html_results(&response_text)
            }
        }
    }

    /// Fallback search method using direct URL construction
    async fn fallback_search(
        &self,
        query: &str,
        _max_results: usize,
        context: &SearchContext,
    ) -> Result<Vec<MdpiArticle>, ProviderError> {
        let search_url = format!(
            "{}/search?q={}&sort=relevance",
            self.base_url,
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(&search_url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("MDPI fallback search failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "MDPI search failed with status: {}",
                response.status()
            )));
        }

        let html_content = response
            .text()
            .await
            .map_err(|e| ProviderError::Network(format!("Failed to read MDPI HTML: {e}")))?;

        self.parse_html_results(&html_content)
    }

    /// Parse HTML search results from MDPI
    fn parse_html_results(&self, html: &str) -> Result<Vec<MdpiArticle>, ProviderError> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(html);

        // MDPI search results are typically in article elements or divs with specific classes
        let article_selector = Selector::parse(
            "article.article-item, .search-result, .article-entry",
        )
        .map_err(|_| ProviderError::Parse("Failed to parse article selector".to_string()))?;

        let title_selector = Selector::parse("h3 a, .title a, .article-title a, h2 a")
            .map_err(|_| ProviderError::Parse("Failed to parse title selector".to_string()))?;

        let author_selector = Selector::parse(".authors, .author-list, .article-authors")
            .map_err(|_| ProviderError::Parse("Failed to parse author selector".to_string()))?;

        let abstract_selector = Selector::parse(".abstract, .article-abstract, p.abstract")
            .map_err(|_| ProviderError::Parse("Failed to parse abstract selector".to_string()))?;

        let doi_selector = Selector::parse(".doi, [data-doi], .article-doi")
            .map_err(|_| ProviderError::Parse("Failed to parse DOI selector".to_string()))?;

        let mut articles = Vec::new();

        for article_element in document.select(&article_selector) {
            let mut article = MdpiArticle {
                title: None,
                abstract_text: None,
                authors: None,
                journal: None,
                doi: None,
                pdf_url: None,
                html_url: None,
                publication_date: None,
                volume: None,
                issue: None,
                page_numbers: None,
                keywords: None,
                article_number: None,
            };

            // Extract title
            if let Some(title_element) = article_element.select(&title_selector).next() {
                article.title = Some(title_element.text().collect::<String>().trim().to_string());

                // Extract URL from title link
                if let Some(href) = title_element.value().attr("href") {
                    let full_url = if href.starts_with("http") {
                        href.to_string()
                    } else {
                        format!("{}{}", self.base_url, href)
                    };
                    article.html_url = Some(full_url.clone());

                    // Construct PDF URL (MDPI typically has /pdf/ URLs)
                    if let Some(article_id) = Self::extract_article_id(&full_url) {
                        article.pdf_url = Some(format!("{}/pdf/{}.pdf", self.base_url, article_id));
                    }
                }
            }

            // Extract authors
            if let Some(authors_element) = article_element.select(&author_selector).next() {
                let authors_text = authors_element.text().collect::<String>();
                let author_names: Vec<String> = authors_text
                    .split(&[',', ';', '&'])
                    .map(|name| name.trim().to_string())
                    .filter(|name| !name.is_empty())
                    .collect();

                if !author_names.is_empty() {
                    let mdpi_authors: Vec<MdpiAuthor> = author_names
                        .into_iter()
                        .map(|name| MdpiAuthor {
                            first_name: None,
                            last_name: None,
                            full_name: Some(name),
                            affiliation: None,
                        })
                        .collect();
                    article.authors = Some(mdpi_authors);
                }
            }

            // Extract abstract
            if let Some(abstract_element) = article_element.select(&abstract_selector).next() {
                article.abstract_text = Some(
                    abstract_element
                        .text()
                        .collect::<String>()
                        .trim()
                        .to_string(),
                );
            }

            // Extract DOI
            if let Some(doi_element) = article_element.select(&doi_selector).next() {
                let doi_text = doi_element.text().collect::<String>();
                if let Some(doi) = Self::extract_doi(&doi_text) {
                    article.doi = Some(doi);
                }
            }

            // Try to extract journal info from URL or metadata
            if let Some(url) = &article.html_url {
                article.journal = self.extract_journal_from_url(url);
            }

            articles.push(article);
        }

        debug!("Parsed {} articles from MDPI HTML", articles.len());
        Ok(articles)
    }

    /// Extract article ID from MDPI URL
    fn extract_article_id(url: &str) -> Option<String> {
        // MDPI URLs typically look like: https://www.mdpi.com/1424-8220/21/1/123
        let re = Regex::new(r"/(\d+)-(\d+)/(\d+)/(\d+)/(\d+)").ok()?;
        if let Some(captures) = re.captures(url) {
            Some(format!(
                "{}-{}-{:05}-{:05}",
                captures.get(1)?.as_str(),
                captures.get(2)?.as_str(),
                captures.get(4)?.as_str().parse::<u32>().ok()?,
                captures.get(5)?.as_str().parse::<u32>().ok()?
            ))
        } else {
            None
        }
    }

    /// Extract DOI from text
    fn extract_doi(text: &str) -> Option<String> {
        let re = Regex::new(r"10\.\d+/[^\s]+").ok()?;
        re.find(text).map(|m| m.as_str().to_string())
    }

    /// Extract journal name from URL
    fn extract_journal_from_url(&self, url: &str) -> Option<MdpiJournal> {
        // Common MDPI journal mappings
        let journal_map = [
            ("sensors", "Sensors"),
            ("materials", "Materials"),
            ("sustainability", "Sustainability"),
            ("applsci", "Applied Sciences"),
            ("molecules", "Molecules"),
            ("ijms", "International Journal of Molecular Sciences"),
            ("remotesensing", "Remote Sensing"),
            ("energies", "Energies"),
            ("water", "Water"),
            ("forests", "Forests"),
            ("buildings", "Buildings"),
            ("electronics", "Electronics"),
            ("polymers", "Polymers"),
            ("nutrients", "Nutrients"),
            ("jcm", "Journal of Clinical Medicine"),
            ("cancers", "Cancers"),
        ];

        for (key, name) in &journal_map {
            if url.contains(key) {
                return Some(MdpiJournal {
                    name: Some((*name).to_string()),
                    short_name: Some(key.to_uppercase()),
                    issn: None,
                    e_issn: None,
                });
            }
        }

        None
    }

    /// Build query string for different search types
    fn build_query(query: &SearchQuery) -> String {
        match query.search_type {
            SearchType::Title => {
                format!("title:({})", query.query)
            }
            SearchType::Author => {
                format!("author:({})", query.query)
            }
            SearchType::Doi => {
                // Direct DOI search
                query.query.clone()
            }
            SearchType::Keywords | SearchType::Auto | SearchType::Subject => {
                // General search
                query.query.clone()
            }
        }
    }

    /// Convert MDPI article to `PaperMetadata`
    fn convert_to_paper(article: &MdpiArticle) -> PaperMetadata {
        let authors = article
            .authors
            .as_ref()
            .map(|authors| {
                authors
                    .iter()
                    .map(|author| {
                        author
                            .full_name
                            .clone()
                            .or_else(|| match (&author.first_name, &author.last_name) {
                                (Some(first), Some(last)) => Some(format!("{first} {last}")),
                                (Some(first), None) => Some(first.clone()),
                                (None, Some(last)) => Some(last.clone()),
                                _ => None,
                            })
                            .unwrap_or_default()
                    })
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        // Extract year from publication date
        let year = article.publication_date.as_ref().and_then(|date| {
            Regex::new(r"(\d{4})")
                .ok()?
                .captures(date)?
                .get(1)?
                .as_str()
                .parse::<u32>()
                .ok()
        });

        let journal_name = article
            .journal
            .as_ref()
            .and_then(|j| j.name.clone())
            .or_else(|| Some("MDPI Journal".to_string()));

        PaperMetadata {
            doi: article.doi.clone().unwrap_or_default(),
            title: article.title.clone(),
            authors,
            abstract_text: article.abstract_text.clone(),
            journal: journal_name,
            year,
            pdf_url: article.pdf_url.clone(),
            file_size: None, // File size not available from MDPI search
        }
    }

    /// Check if this looks like an open access paper
    const fn is_open_access(&self, _article: &MdpiArticle) -> bool {
        // MDPI is primarily an open access publisher
        true
    }
}

#[async_trait]
impl SourceProvider for MdpiProvider {
    fn name(&self) -> &'static str {
        "mdpi"
    }

    fn description(&self) -> &'static str {
        "MDPI provider for open access scientific journals"
    }

    fn priority(&self) -> u8 {
        82 // Preferred after medRxiv/arXiv in default ordering
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Title,
            SearchType::Author,
            SearchType::Doi,
            SearchType::Keywords,
            SearchType::Auto,
            SearchType::Subject,
        ]
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        info!("Searching MDPI for: '{}'", query.query);

        let search_query = Self::build_query(query);
        let articles = match self
            .search_mdpi(&search_query, query.max_results as usize, context)
            .await
        {
            Ok(articles) => articles,
            Err(ProviderError::Network(msg)) if msg.contains("403") => {
                warn!("MDPI returned 403 Forbidden. Returning empty result with notice.");
                return Ok(ProviderResult {
                    papers: vec![],
                    source: self.name().to_string(),
                    total_available: Some(0),
                    search_time: Duration::from_millis(0),
                    has_more: false,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert(
                            "note".to_string(),
                            "MDPI blocked the request (403). Try again later or with different network settings."
                                .to_string(),
                        );
                        meta
                    },
                });
            }
            Err(e) => return Err(e),
        };

        if articles.is_empty() {
            info!("No results found in MDPI for query: '{}'", query.query);
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

        let papers: Vec<PaperMetadata> = articles
            .iter()
            .filter(|article| self.is_open_access(article))
            .map(Self::convert_to_paper)
            .collect();

        let search_time = start_time.elapsed();
        let papers_count = u32::try_from(papers.len()).unwrap_or(u32::MAX);

        info!(
            "MDPI found {} open access papers for query: '{}'",
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
                meta.insert("open_access".to_string(), "true".to_string());
                meta.insert("publisher".to_string(), "MDPI".to_string());
                meta
            },
        })
    }

    async fn get_by_doi(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        // Try direct DOI lookup
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
        let health_url = format!("{}/search?q=test", self.base_url);

        let response = self
            .client
            .get(&health_url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("MDPI health check failed: {e}")))?;

        if response.status().is_success() {
            debug!("MDPI health check passed");
            Ok(true)
        } else {
            debug!(
                "MDPI health check failed with status: {}",
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
    async fn test_mdpi_provider_creation() {
        let provider = MdpiProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_doi_extraction() {
        let _provider = MdpiProvider::new().unwrap();

        assert_eq!(
            MdpiProvider::extract_doi("DOI: 10.3390/s21010123"),
            Some("10.3390/s21010123".to_string())
        );

        assert_eq!(
            MdpiProvider::extract_doi("https://doi.org/10.3390/materials14020456"),
            Some("10.3390/materials14020456".to_string())
        );

        assert_eq!(MdpiProvider::extract_doi("No DOI here"), None);
    }

    #[test]
    fn test_article_id_extraction() {
        let _provider = MdpiProvider::new().unwrap();

        assert_eq!(
            MdpiProvider::extract_article_id("https://www.mdpi.com/1424-8220/21/1/123"),
            Some("1424-8220-00001-00123".to_string())
        );

        assert_eq!(
            MdpiProvider::extract_article_id("https://www.mdpi.com/invalid/url"),
            None
        );
    }

    #[test]
    fn test_journal_extraction() {
        let provider = MdpiProvider::new().unwrap();

        let journal = provider.extract_journal_from_url("https://www.mdpi.com/sensors/");
        assert!(journal.is_some());
        assert_eq!(journal.unwrap().name.unwrap(), "Sensors");

        let journal = provider.extract_journal_from_url("https://www.mdpi.com/materials/");
        assert!(journal.is_some());
        assert_eq!(journal.unwrap().name.unwrap(), "Materials");

        let journal = provider.extract_journal_from_url("https://www.mdpi.com/unknown/");
        assert!(journal.is_none());
    }

    #[test]
    fn test_query_building() {
        let _provider = MdpiProvider::new().unwrap();

        let title_query = SearchQuery {
            query: "machine learning sensors".to_string(),
            search_type: SearchType::Title,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            MdpiProvider::build_query(&title_query),
            "title:(machine learning sensors)"
        );

        let author_query = SearchQuery {
            query: "John Smith".to_string(),
            search_type: SearchType::Author,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            MdpiProvider::build_query(&author_query),
            "author:(John Smith)"
        );

        let doi_query = SearchQuery {
            query: "10.3390/s21010123".to_string(),
            search_type: SearchType::Doi,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(MdpiProvider::build_query(&doi_query), "10.3390/s21010123");
    }

    #[test]
    fn test_provider_metadata() {
        let provider = MdpiProvider::new().unwrap();

        assert_eq!(provider.name(), "mdpi");
        assert_eq!(provider.priority(), 75);
        assert!(provider.supports_full_text());

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
        assert!(supported_types.contains(&SearchType::Doi));
        assert!(supported_types.contains(&SearchType::Keywords));
        assert!(supported_types.contains(&SearchType::Auto));
        assert!(supported_types.contains(&SearchType::Subject));
    }

    #[test]
    fn test_paper_conversion() {
        let _provider = MdpiProvider::new().unwrap();

        let article = MdpiArticle {
            title: Some("Test MDPI Paper".to_string()),
            abstract_text: Some("This is a test abstract".to_string()),
            authors: Some(vec![
                MdpiAuthor {
                    first_name: Some("John".to_string()),
                    last_name: Some("Doe".to_string()),
                    full_name: None,
                    affiliation: None,
                },
                MdpiAuthor {
                    first_name: None,
                    last_name: None,
                    full_name: Some("Jane Smith".to_string()),
                    affiliation: None,
                },
            ]),
            journal: Some(MdpiJournal {
                name: Some("Sensors".to_string()),
                short_name: Some("SENSORS".to_string()),
                issn: None,
                e_issn: None,
            }),
            doi: Some("10.3390/s21010123".to_string()),
            pdf_url: Some("https://www.mdpi.com/pdf/test.pdf".to_string()),
            html_url: None,
            publication_date: Some("2021-01-01".to_string()),
            volume: None,
            issue: None,
            page_numbers: None,
            keywords: None,
            article_number: None,
        };

        let paper = MdpiProvider::convert_to_paper(&article);

        assert_eq!(paper.title.unwrap(), "Test MDPI Paper");
        assert_eq!(paper.authors.len(), 2);
        assert_eq!(paper.authors[0], "John Doe");
        assert_eq!(paper.authors[1], "Jane Smith");
        assert_eq!(paper.journal.unwrap(), "Sensors");
        assert_eq!(paper.doi, "10.3390/s21010123");
        assert!(paper.pdf_url.is_some());
        assert_eq!(paper.year.unwrap(), 2021);
    }
}
