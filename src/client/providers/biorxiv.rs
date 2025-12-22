use super::traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, USER_AGENT},
    Client,
};
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// bioRxiv API response for paper details
#[derive(Debug, Deserialize)]
struct BiorxivResponse {
    messages: Vec<BiorxivMessage>,
    collection: Vec<BiorxivPaper>,
}

/// Status message from bioRxiv API
#[derive(Debug, Deserialize)]
struct BiorxivMessage {
    status: String,
    #[serde(default)]
    text: Option<String>,
}

/// Individual paper from bioRxiv API
#[derive(Debug, Deserialize)]
struct BiorxivPaper {
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
    server: String, // "biorxiv" or "medrxiv"
}

/// Rxivist response subset
#[derive(Debug, Deserialize)]
struct RxivistResponse {
    results: Option<Vec<Value>>,
}

/// bioRxiv provider for biology preprints
pub struct BiorxivProvider {
    client: Client,
    base_url: String,
}

impl BiorxivProvider {
    /// Create a new bioRxiv provider
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
            .timeout(Duration::from_secs(30))
            .default_headers(headers)
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.biorxiv.org".to_string(),
        })
    }

    /// Build DOI lookup URL for bioRxiv API
    fn build_doi_url(&self, doi: &str) -> String {
        format!("{}/details/biorxiv/{}", self.base_url, doi)
    }

    /// Build search URL for bioRxiv API (by date range)
    fn build_date_search_url(
        &self,
        start_date: &str,
        end_date: &str,
        query: Option<&str>,
    ) -> String {
        if let Some(q) = query {
            format!(
                "{}/details/biorxiv/{}/{}/{}",
                self.base_url,
                start_date,
                end_date,
                urlencoding::encode(q)
            )
        } else {
            format!(
                "{}/details/biorxiv/{}/{}",
                self.base_url, start_date, end_date
            )
        }
    }

    /// Build HTML search URL for biorxiv.org
    fn build_html_search_url(&self, query: &str, page: u32) -> String {
        format!(
            "https://www.biorxiv.org/search/{}%20numresults%3A200%20sort%3Arelevance-rank?page={}&format=standard",
            urlencoding::encode(query),
            page
        )
    }

    /// Parse HTML search results from biorxiv.org
    fn parse_html_results(&self, html: &str) -> Vec<PaperMetadata> {
        let document = Html::parse_document(html);

        let citation_selector = Selector::parse("div.highwire-article-citation").ok();
        let title_selector =
            Selector::parse(".highwire-cite-title, .highwire-cite-linked-title").ok();
        let author_selector = Selector::parse(".highwire-citation-author").ok();
        let doi_selector = Selector::parse(".highwire-cite-metadata-doi").ok();
        let doi_link_selector = Selector::parse("a").ok();
        let year_selector = Selector::parse(".highwire-cite-metadata-year").ok();

        let mut papers = Vec::new();

        let Some(citation_sel) = citation_selector else {
            return papers;
        };

        for citation in document.select(&citation_sel) {
            let title = title_selector.as_ref().and_then(|sel| {
                citation
                    .select(sel)
                    .next()
                    .map(|el| el.text().collect::<String>().trim().to_string())
                    .filter(|t| !t.is_empty())
            });

            let authors: Vec<String> = author_selector
                .as_ref()
                .map(|sel| {
                    citation
                        .select(sel)
                        .map(|el| el.text().collect::<String>().trim().to_string())
                        .filter(|a| !a.is_empty())
                        .collect()
                })
                .unwrap_or_default();

            // DOI from explicit metadata or href
            let doi_from_meta = doi_selector.as_ref().and_then(|sel| {
                citation
                    .select(sel)
                    .next()
                    .and_then(|el| el.value().attr("data-doi"))
                    .map(|s| s.to_string())
                    .or_else(|| {
                        citation
                            .select(sel)
                            .next()
                            .map(|el| el.text().collect::<String>())
                    })
            });

            let mut doi = doi_from_meta
                .map(|raw| raw.replace("doi:", "").trim().to_string())
                .unwrap_or_default();

            if doi.is_empty() {
                if let Some(link_sel) = &doi_link_selector {
                    for a in citation.select(link_sel) {
                        if let Some(href) = a.value().attr("href") {
                            if href.contains("/content/10.1101/") {
                                if let Some(start) = href.find("10.1101/") {
                                    let doi_part = &href[start..];
                                    doi = doi_part
                                        .trim_end_matches(|c| c == '/' || c == '#')
                                        .to_string();
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            let year = year_selector.as_ref().and_then(|sel| {
                citation
                    .select(sel)
                    .next()
                    .and_then(|el| el.text().collect::<String>().trim().parse::<u32>().ok())
            });

            if title.is_none() && doi.is_empty() {
                continue;
            }

            let pdf_url = if doi.is_empty() {
                None
            } else {
                Some(format!("https://www.biorxiv.org/content/{}.full.pdf", doi))
            };

            papers.push(PaperMetadata {
                doi,
                title,
                authors,
                journal: Some("biorxiv preprint".to_string()),
                year,
                abstract_text: None,
                pdf_url,
                file_size: None,
            });
        }

        papers
    }

    /// HTML search fallback to get keyword results
    async fn search_html(
        &self,
        query: &str,
        max_results: u32,
        context: &SearchContext,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let mut all_papers = Vec::new();
        let mut page = 0;

        while (all_papers.len() as u32) < max_results && page < 5 {
            let url = self.build_html_search_url(query, page);
            debug!("bioRxiv HTML search page {}: {}", page, url);

            let response = self
                .client
                .get(&url)
                .timeout(context.timeout)
                .send()
                .await
                .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

            if !response.status().is_success() {
                warn!(
                    "bioRxiv HTML search returned status {} on page {}",
                    response.status(),
                    page
                );
                break;
            }

            let html = response
                .text()
                .await
                .map_err(|e| ProviderError::Network(format!("Failed to read response: {e}")))?;

            let parsed = self.parse_html_results(&html);
            if parsed.is_empty() {
                break;
            }

            let before = all_papers.len();
            all_papers.extend(parsed);
            if all_papers.len() == before {
                break;
            }

            page += 1;
        }

        all_papers.truncate(max_results as usize);
        Ok(all_papers)
    }

    /// Rxivist fallback search (community API)
    async fn search_rxivist(
        &self,
        query: &str,
        max_results: u32,
        context: &SearchContext,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let url = format!(
            "https://rxivist.org/api/v1/papers?q={}&source=biorxiv&per_page={}&sort=relevance",
            urlencoding::encode(query),
            max_results
        );

        let response = self
            .client
            .get(&url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Rxivist request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "Rxivist search failed with status: {}",
                response.status()
            )));
        }

        let parsed: RxivistResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Failed to parse Rxivist response: {e}")))?;

        let mut papers = Vec::new();
        if let Some(results) = parsed.results {
            for entry in results.into_iter().take(max_results as usize) {
                let doi = entry
                    .get("doi")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let title = entry
                    .get("title")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let authors: Vec<String> = entry
                    .get("authors")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|a| a.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                let abstract_text = entry
                    .get("abstract")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let year = entry
                    .get("published_date")
                    .and_then(Value::as_str)
                    .and_then(|s| s.split('-').next())
                    .and_then(|y| y.parse::<u32>().ok());

                let pdf_url = if doi.is_empty() {
                    None
                } else {
                    Some(format!("https://www.biorxiv.org/content/{}.full.pdf", doi))
                };

                papers.push(PaperMetadata {
                    doi,
                    title,
                    authors,
                    journal: Some("biorxiv preprint (rxivist)".to_string()),
                    year,
                    abstract_text,
                    pdf_url,
                    file_size: None,
                });
            }
        }

        Ok(papers)
    }

    /// Helper to run API search over a date window
    async fn search_api_within_days(
        &self,
        days_back: u32,
        max_results: u32,
        query: &str,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        self.search_recent_papers(days_back, max_results, Some(query))
            .await
    }

    /// CrossRef fallback filtered to bioRxiv prefix
    async fn search_crossref_fallback(
        &self,
        query: &str,
        max_results: u32,
        context: &SearchContext,
    ) -> Result<Vec<PaperMetadata>, ProviderError> {
        let url = format!(
            "https://api.crossref.org/works?rows={}&query={}&filter=prefix:10.1101",
            max_results,
            urlencoding::encode(query)
        );

        let response = self
            .client
            .get(&url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("CrossRef fallback failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "CrossRef fallback failed with status: {}",
                response.status()
            )));
        }

        let value: Value = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse CrossRef fallback response: {e}"))
        })?;

        let mut papers = Vec::new();
        if let Some(items) = value
            .get("message")
            .and_then(|m| m.get("items"))
            .and_then(Value::as_array)
        {
            for item in items.iter().take(max_results as usize) {
                let doi = item
                    .get("DOI")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let title = item
                    .get("title")
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first())
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let authors: Vec<String> = item
                    .get("author")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|a| {
                                let given =
                                    a.get("given").and_then(Value::as_str).unwrap_or_default();
                                let family =
                                    a.get("family").and_then(Value::as_str).unwrap_or_default();
                                let name = format!("{} {}", given, family).trim().to_string();
                                if name.is_empty() {
                                    None
                                } else {
                                    Some(name)
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let year = item
                    .get("issued")
                    .and_then(|i| i.get("date-parts"))
                    .and_then(Value::as_array)
                    .and_then(|arr| arr.first())
                    .and_then(Value::as_array)
                    .and_then(|part| part.first())
                    .and_then(Value::as_i64)
                    .map(|y| y as u32);
                let abstract_text = item
                    .get("abstract")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                let pdf_url = item.get("link").and_then(Value::as_array).and_then(|arr| {
                    arr.iter().find_map(|l| {
                        let content_type =
                            l.get("content-type").and_then(Value::as_str).unwrap_or("");
                        let url = l.get("URL").and_then(Value::as_str);
                        if content_type.contains("pdf") {
                            url.map(str::to_string)
                        } else {
                            None
                        }
                    })
                });

                papers.push(PaperMetadata {
                    doi,
                    title,
                    authors,
                    journal: Some("biorxiv preprint (crossref)".to_string()),
                    year,
                    abstract_text,
                    pdf_url,
                    file_size: None,
                });
            }
        }

        Ok(papers)
    }

    /// Extract DOI from various bioRxiv formats
    fn extract_biorxiv_doi(doi_or_url: &str) -> Option<String> {
        // Handle various bioRxiv DOI formats:
        // - 10.1101/2023.01.01.000001
        // - https://doi.org/10.1101/2023.01.01.000001
        // - https://www.biorxiv.org/content/10.1101/2023.01.01.000001v1

        if doi_or_url.contains("10.1101/") {
            // Extract the DOI part
            if let Some(doi_start) = doi_or_url.find("10.1101/") {
                let doi_part = &doi_or_url[doi_start..];
                // Remove version suffix if present (e.g., "v1", "v2")
                if let Some(version_pos) = doi_part.find('v') {
                    if version_pos > 8 {
                        // Ensure it's not part of the date
                        return Some(doi_part[..version_pos].to_string());
                    }
                }
                return Some(doi_part.to_string());
            }
        }
        None
    }

    /// Convert bioRxiv paper to `PaperMetadata`
    fn convert_paper(&self, paper: BiorxivPaper) -> PaperMetadata {
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

        // Generate PDF URL based on DOI
        let pdf_url = Some(format!(
            "https://www.biorxiv.org/content/biorxiv/early/{}/{}.full.pdf",
            paper.date.replace('-', "/"),
            paper.doi
        ));

        PaperMetadata {
            doi: paper.doi,
            title: Some(paper.title),
            authors,
            journal: Some(format!("{} preprint", paper.server)), // "biorxiv preprint" or "medrxiv preprint"
            year,
            abstract_text: paper.abstract_text,
            pdf_url,
            file_size: None,
        }
    }

    /// Get paper by DOI from bioRxiv
    async fn get_paper_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = self.build_doi_url(doi);
        debug!("Getting paper by DOI from bioRxiv: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if response.status().as_u16() == 404 {
            debug!("Paper not found in bioRxiv for DOI: {}", doi);
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

        debug!("bioRxiv response: {}", response_text);

        let biorxiv_response: BiorxivResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!("Failed to parse bioRxiv response: {}", response_text);
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        // Check for error messages
        for message in &biorxiv_response.messages {
            if message.status != "ok" {
                warn!("bioRxiv API message: {:?}", message.text);
            }
        }

        // Return the first paper if found
        if let Some(paper) = biorxiv_response.collection.into_iter().next() {
            Ok(Some(self.convert_paper(paper)))
        } else {
            Ok(None)
        }
    }

    /// Search for papers by date range (bioRxiv doesn't support general text search)
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
        debug!("Searching bioRxiv by date range: {}", url);

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

        debug!("bioRxiv search response: {}", response_text);

        let biorxiv_response: BiorxivResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!("Failed to parse bioRxiv search response: {}", response_text);
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        // Check for error messages
        for message in &biorxiv_response.messages {
            if message.status != "ok" {
                warn!("bioRxiv API message: {:?}", message.text);
            }
        }

        // Convert papers and limit results
        let papers: Vec<PaperMetadata> = biorxiv_response
            .collection
            .into_iter()
            .take(limit as usize)
            .map(|paper| self.convert_paper(paper))
            .collect();

        Ok(papers)
    }
}

#[async_trait]
impl SourceProvider for BiorxivProvider {
    fn name(&self) -> &'static str {
        "biorxiv"
    }

    fn description(&self) -> &'static str {
        "bioRxiv - Biology preprint server"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Doi,
            SearchType::Keywords,
            SearchType::TitleAbstract,
            SearchType::Auto,
        ] // Limited search capabilities
    }

    fn query_format_help(&self) -> &'static str {
        r#"bioRxiv supports basic keyword search:
- Keywords are searched across title and abstract
- DOI lookup for bioRxiv-specific DOIs (10.1101/*)
- Date range search through API (recent papers)
- No advanced field-specific query syntax
- Results sorted by relevance or date"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("CRISPR gene editing", "Basic keyword search"),
            ("10.1101/2023.01.01.123456", "DOI lookup"),
            ("single cell RNA sequencing", "Multi-word topic search"),
            ("COVID-19 vaccine efficacy", "Timely research topics"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://www.biorxiv.org/search")
    }

    fn supports_full_text(&self) -> bool {
        true // bioRxiv provides PDF access for all preprints
    }

    fn priority(&self) -> u8 {
        88 // Prefer bioRxiv early after PubMed/Google Scholar
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(500) // Be respectful to the free API
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching bioRxiv for: {} (type: {:?})",
            query.query, query.search_type
        );

        let papers = match query.search_type {
            SearchType::Doi => {
                // Check if this is a bioRxiv DOI
                if let Some(biorxiv_doi) = Self::extract_biorxiv_doi(&query.query) {
                    if let Some(paper) = self.get_paper_by_doi(&biorxiv_doi).await? {
                        vec![paper]
                    } else {
                        Vec::new()
                    }
                } else {
                    // Not a bioRxiv DOI, return empty
                    Vec::new()
                }
            }
            SearchType::Keywords | SearchType::TitleAbstract | SearchType::Auto => {
                // Try HTML search first (more flexible), fallback to API date-window search
                let mut papers = self
                    .search_html(&query.query, query.max_results, context)
                    .await
                    .unwrap_or_else(|e| {
                        warn!("bioRxiv HTML search failed: {}", e);
                        Vec::new()
                    });

                if papers.is_empty() {
                    // Try progressively larger date windows
                    for days in [365_u32, 3650, 10000] {
                        match self
                            .search_api_within_days(days, query.max_results, &query.query)
                            .await
                        {
                            Ok(mut api_papers) => {
                                if !api_papers.is_empty() {
                                    papers.append(&mut api_papers);
                                    break;
                                }
                            }
                            Err(e) => warn!("bioRxiv API search failed ({} days): {}", days, e),
                        }
                    }
                }

                // Final fallback: Rxivist
                if papers.is_empty() {
                    match self
                        .search_rxivist(&query.query, query.max_results, context)
                        .await
                    {
                        Ok(mut rxivist_papers) => {
                            papers.append(&mut rxivist_papers);
                        }
                        Err(e) => warn!("bioRxiv Rxivist fallback failed: {}", e),
                    }
                }

                // CrossRef prefix-filter fallback
                if papers.is_empty() {
                    match self
                        .search_crossref_fallback(&query.query, query.max_results, context)
                        .await
                    {
                        Ok(mut crossref_papers) => {
                            papers.append(&mut crossref_papers);
                        }
                        Err(e) => warn!("bioRxiv CrossRef fallback failed: {}", e),
                    }
                }

                papers
            }
            _ => {
                // bioRxiv doesn't support other search types
                warn!(
                    "bioRxiv only supports DOI and limited keyword searches, ignoring query: {}",
                    query.query
                );
                Vec::new()
            }
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();

        let result = ProviderResult {
            papers,
            source: "bioRxiv".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: false, // bioRxiv API doesn't support pagination in our simple implementation
            metadata: HashMap::new(),
        };

        info!(
            "bioRxiv search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from bioRxiv: {}", doi);

        // Check if this is a bioRxiv DOI first
        if let Some(biorxiv_doi) = Self::extract_biorxiv_doi(doi) {
            self.get_paper_by_doi(&biorxiv_doi).await
        } else {
            // Not a bioRxiv DOI
            Ok(None)
        }
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing bioRxiv health check");

        // Use a known bioRxiv DOI for health check
        let test_url = self.build_doi_url("10.1101/2020.01.01.000001");

        match self.client.get(&test_url).send().await {
            Ok(response) if response.status().is_success() || response.status().as_u16() == 404 => {
                info!("bioRxiv health check: OK");
                Ok(true)
            }
            Ok(response) => {
                warn!(
                    "bioRxiv health check failed with status: {}",
                    response.status()
                );
                Ok(false)
            }
            Err(e) => {
                warn!("bioRxiv health check failed: {}", e);
                Ok(false)
            }
        }
    }

    async fn get_pdf_url(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<String>, ProviderError> {
        // For bioRxiv, if we can get the paper, we can construct the PDF URL
        if let Some(paper) = self.get_by_doi(doi, context).await? {
            Ok(paper.pdf_url)
        } else {
            Ok(None)
        }
    }
}

impl Default for BiorxivProvider {
    fn default() -> Self {
        Self::new().expect("Failed to create BiorxivProvider")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biorxiv_provider_creation() {
        let provider = BiorxivProvider::new();
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_interface() {
        let provider = BiorxivProvider::new().unwrap();

        assert_eq!(provider.name(), "biorxiv");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 75);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
    }

    #[test]
    fn test_biorxiv_doi_extraction() {
        let _provider = BiorxivProvider::new().unwrap();

        let test_cases = vec![
            (
                "10.1101/2023.01.01.000001",
                Some("10.1101/2023.01.01.000001"),
            ),
            (
                "https://doi.org/10.1101/2023.01.01.000001",
                Some("10.1101/2023.01.01.000001"),
            ),
            (
                "https://www.biorxiv.org/content/10.1101/2023.01.01.000001v1",
                Some("10.1101/2023.01.01.000001"),
            ),
            ("10.1038/nature12373", None), // Not a bioRxiv DOI
        ];

        for (input, expected) in test_cases {
            let result = BiorxivProvider::extract_biorxiv_doi(input);
            assert_eq!(result.as_deref(), expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_url_building() {
        let provider = BiorxivProvider::new().unwrap();

        let doi_url = provider.build_doi_url("10.1101/2023.01.01.000001");
        assert!(doi_url.contains("api.biorxiv.org"));
        assert!(doi_url.contains("details/biorxiv"));
        assert!(doi_url.contains("10.1101/2023.01.01.000001"));

        let search_url = provider.build_date_search_url("2023-01-01", "2023-01-31", None);
        assert!(search_url.contains("2023-01-01"));
        assert!(search_url.contains("2023-01-31"));
    }
}
