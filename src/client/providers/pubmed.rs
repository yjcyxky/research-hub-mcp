use crate::client::providers::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
use crate::client::PaperMetadata;
use async_trait::async_trait;
use regex::Regex;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// PubMed provider for biomedical literature citations and abstracts
///
/// PubMed is a free database of biomedical and life science journal citations
/// and abstracts maintained by the National Library of Medicine (NLM).
/// Unlike PubMed Central (PMC), PubMed doesn't provide full-text access directly,
/// but provides comprehensive metadata and links to publisher sites.
pub struct PubMedProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>, // Optional NCBI API key for higher rate limits
}

#[derive(Debug, Deserialize)]
struct PubMedSearchResponse {
    esearchresult: ESearchResult,
}

#[derive(Debug, Deserialize)]
struct ESearchResult {
    count: String,
    #[allow(dead_code)]
    retmax: String,
    #[allow(dead_code)]
    retstart: String,
    #[serde(deserialize_with = "deserialize_idlist")]
    idlist: Vec<String>,
    #[serde(default)]
    errorlist: Option<ErrorList>,
    #[serde(default)]
    #[allow(dead_code)]
    warninglist: Option<WarningList>,
}

#[derive(Debug, Deserialize)]
struct ErrorList {
    phrasesnotfound: Option<Vec<String>>,
    #[allow(dead_code)]
    fieldsnotfound: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct WarningList {
    #[allow(dead_code)]
    phrasesignored: Option<Vec<String>>,
    #[allow(dead_code)]
    quotedphrasenotfound: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct PubMedFetchResponse {
    result: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct PubMedArticle {
    #[allow(dead_code)]
    uid: Option<String>,
    title: Option<String>,
    authors: Option<Vec<PubMedAuthor>>,
    #[serde(rename = "fulljournalname")]
    journal_name: Option<String>,
    #[serde(rename = "pubdate")]
    pub_date: Option<String>,
    #[serde(rename = "elocationid")]
    elocation_id: Option<String>, // Often contains DOI
    #[serde(rename = "articleids")]
    article_ids: Option<Vec<ArticleId>>,
    #[serde(rename = "hasabstract")]
    #[allow(dead_code)]
    has_abstract: Option<String>,
    // Note: esummary doesn't return abstract; need efetch for that
}

#[derive(Debug, Deserialize)]
struct PubMedAuthor {
    name: Option<String>,
    #[allow(dead_code)]
    authtype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArticleId {
    idtype: String,
    value: String,
}

/// Support both array and string forms for idlist
fn deserialize_idlist<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct IdListVisitor;

    impl<'de> serde::de::Visitor<'de> for IdListVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(vec![v.to_string()])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut values = Vec::new();
            while let Some(value) = seq.next_element::<String>()? {
                values.push(value);
            }
            Ok(values)
        }
    }

    deserializer.deserialize_any(IdListVisitor)
}

impl PubMedProvider {
    /// Create a new PubMed provider
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent(
                "rust_research_mcp/0.6.0 (https://github.com/Ladvien/research_hub_mcp)",
            )
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://eutils.ncbi.nlm.nih.gov".to_string(),
            api_key,
        })
    }

    /// Search PubMed using the E-utilities API
    async fn search_pubmed(
        &self,
        query: &str,
        max_results: usize,
        context: &SearchContext,
    ) -> Result<Vec<String>, ProviderError> {
        let search_url = format!("{}/entrez/eutils/esearch.fcgi", self.base_url);

        let max_results_str = max_results.to_string();
        let mut params = vec![
            ("db", "pubmed"), // Key difference: use "pubmed" instead of "pmc"
            ("term", query),
            ("retmode", "json"),
            ("retmax", &max_results_str),
            ("sort", "relevance"),
        ];

        // Add API key if available for higher rate limits
        if let Some(ref api_key) = self.api_key {
            params.push(("api_key", api_key));
        }

        let response = self
            .client
            .get(&search_url)
            .query(&params)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("PubMed search request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "PubMed search failed with status: {}",
                response.status()
            )));
        }

        let search_result: PubMedSearchResponse = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse PubMed search response: {e}"))
        })?;

        // Check for errors in the response
        if let Some(ref error_list) = search_result.esearchresult.errorlist {
            if let Some(ref phrases) = error_list.phrasesnotfound {
                warn!("PubMed search phrases not found: {:?}", phrases);
            }
        }

        debug!(
            "PubMed search found {} results for query: '{}'",
            search_result.esearchresult.count, query
        );

        Ok(search_result.esearchresult.idlist)
    }

    /// Fetch article details for given PubMed IDs (PMIDs)
    async fn fetch_articles(
        &self,
        pmids: &[String],
        context: &SearchContext,
    ) -> Result<Vec<PubMedArticle>, ProviderError> {
        if pmids.is_empty() {
            return Ok(vec![]);
        }

        let fetch_url = format!("{}/entrez/eutils/esummary.fcgi", self.base_url);
        let ids_str = pmids.join(",");

        let mut params = vec![
            ("db", "pubmed"), // Key difference: use "pubmed"
            ("id", &ids_str),
            ("retmode", "json"),
        ];

        if let Some(ref api_key) = self.api_key {
            params.push(("api_key", api_key));
        }

        let response = self
            .client
            .get(&fetch_url)
            .query(&params)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("PubMed fetch request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "PubMed fetch failed with status: {}",
                response.status()
            )));
        }

        let fetch_result: PubMedFetchResponse = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse PubMed fetch response: {e}"))
        })?;

        let mut articles = Vec::new();
        for (key, value) in fetch_result.result {
            if key == "uids" {
                continue;
            }
            if value.is_object() {
                match serde_json::from_value::<PubMedArticle>(value) {
                    Ok(mut article) => {
                        if article.uid.is_none() {
                            article.uid = Some(key.clone());
                        }
                        articles.push(article);
                    }
                    Err(e) => warn!("Failed to parse PubMed article {}: {}", key, e),
                }
            }
        }

        Ok(articles)
    }

    /// Fetch abstracts using efetch API (esummary doesn't return abstracts)
    async fn fetch_abstracts(
        &self,
        pmids: &[String],
        context: &SearchContext,
    ) -> Result<HashMap<String, String>, ProviderError> {
        if pmids.is_empty() {
            return Ok(HashMap::new());
        }

        let fetch_url = format!("{}/entrez/eutils/efetch.fcgi", self.base_url);
        let ids_str = pmids.join(",");

        let mut params = vec![
            ("db", "pubmed"),
            ("id", &ids_str),
            ("retmode", "xml"),
            ("rettype", "abstract"),
        ];

        if let Some(ref api_key) = self.api_key {
            params.push(("api_key", api_key));
        }

        let response = self
            .client
            .get(&fetch_url)
            .query(&params)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| {
                ProviderError::Network(format!("PubMed abstract fetch request failed: {e}"))
            })?;

        if !response.status().is_success() {
            debug!(
                "PubMed abstract fetch failed with status: {}",
                response.status()
            );
            return Ok(HashMap::new()); // Return empty map on failure, don't fail entire request
        }

        let xml_text = response.text().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to read PubMed abstract response: {e}"))
        })?;

        // Parse XML to extract abstracts
        // For simplicity, use regex to extract abstracts (full XML parsing would be more robust)
        let mut abstracts = HashMap::new();

        // Match PMID and AbstractText pairs
        let pmid_regex = Regex::new(r"<PMID[^>]*>(\d+)</PMID>").unwrap();
        let abstract_regex =
            Regex::new(r"(?s)<Abstract>\s*<AbstractText[^>]*>(.*?)</AbstractText>\s*</Abstract>")
                .unwrap();

        // Split by article
        let article_regex = Regex::new(r"(?s)<PubmedArticle>(.*?)</PubmedArticle>").unwrap();
        let tag_cleanup_regex = Regex::new(r"<[^>]+>").unwrap();

        for article_cap in article_regex.captures_iter(&xml_text) {
            let article_xml = &article_cap[1];

            if let Some(pmid_cap) = pmid_regex.captures(article_xml) {
                let pmid = pmid_cap[1].to_string();

                if let Some(abstract_cap) = abstract_regex.captures(article_xml) {
                    // Clean up the abstract text (remove XML tags)
                    let abstract_text = abstract_cap[1]
                        .replace("<AbstractText>", "")
                        .replace("</AbstractText>", " ")
                        .replace('\n', " ")
                        .trim()
                        .to_string();

                    // Remove any remaining XML-like tags
                    let clean_abstract =
                        tag_cleanup_regex.replace_all(&abstract_text, "");
                    if !clean_abstract.is_empty() {
                        abstracts.insert(pmid, clean_abstract.to_string());
                    }
                }
            }
        }

        debug!("Fetched {} abstracts from PubMed", abstracts.len());
        Ok(abstracts)
    }

    /// Build query string for different search types
    fn build_query(query: &SearchQuery) -> String {
        match query.search_type {
            SearchType::Doi => {
                // Search by DOI
                format!("{}[doi]", query.query)
            }
            SearchType::Title => {
                // Search in title field
                format!("{}[Title]", query.query)
            }
            SearchType::TitleAbstract => {
                // Search in title/abstract combined
                format!("{}[Title/Abstract]", query.query)
            }
            SearchType::Author => {
                // Search in author field
                format!("{}[Author]", query.query)
            }
            SearchType::Keywords | SearchType::Auto | SearchType::Subject => {
                // General search across all fields
                query.query.clone()
            }
        }
    }

    /// Convert PubMed article to `PaperMetadata`
    fn convert_to_paper(article: &PubMedArticle, abstract_text: Option<&String>) -> PaperMetadata {
        let authors = article
            .authors
            .as_ref()
            .map(|author_list| author_list.iter().filter_map(|a| a.name.clone()).collect())
            .unwrap_or_default();

        // Extract DOI from article IDs or elocation_id
        let doi = article
            .article_ids
            .as_ref()
            .and_then(|ids| {
                ids.iter()
                    .find(|id| id.idtype == "doi")
                    .map(|id| id.value.clone())
            })
            .or_else(|| {
                // Try to extract from elocation_id (format: "doi: 10.xxxx/yyyy")
                article.elocation_id.as_ref().and_then(|eid| {
                    if eid.starts_with("doi:") {
                        Some(eid.trim_start_matches("doi:").trim().to_string())
                    } else if eid.starts_with("10.") {
                        Some(eid.clone())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_default();

        // Extract PMID for building PubMed URL
        let pmid = article.uid.clone().or_else(|| {
            article.article_ids.as_ref().and_then(|ids| {
                ids.iter()
                    .find(|id| id.idtype == "pubmed")
                    .map(|id| id.value.clone())
            })
        });

        // PubMed doesn't provide direct PDF URLs, but we can link to the PubMed page
        // which often has links to full text from publishers
        let pdf_url = pmid.map(|id| format!("https://pubmed.ncbi.nlm.nih.gov/{}/", id));

        PaperMetadata {
            doi,
            title: article.title.clone(),
            authors,
            abstract_text: abstract_text.cloned(),
            journal: article.journal_name.clone(),
            year: article.pub_date.as_ref().and_then(|date| {
                // Try to extract year from date string (format varies: "2024 Jan 15", "2024", etc.)
                Regex::new(r"(\d{4})")
                    .ok()?
                    .captures(date)?
                    .get(1)?
                    .as_str()
                    .parse::<u32>()
                    .ok()
            }),
            pdf_url,
            file_size: None,
        }
    }
}

#[async_trait]
impl SourceProvider for PubMedProvider {
    fn name(&self) -> &'static str {
        "pubmed"
    }

    fn priority(&self) -> u8 {
        92 // High priority for biomedical literature (slightly lower than PMC since no full text)
    }

    fn supports_full_text(&self) -> bool {
        false // PubMed provides citations/abstracts, not full text
    }

    fn description(&self) -> &'static str {
        "PubMed database for biomedical literature citations and abstracts"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![
            SearchType::Doi,
            SearchType::Title,
            SearchType::TitleAbstract,
            SearchType::Author,
            SearchType::Keywords,
            SearchType::Auto,
            SearchType::Subject,
        ]
    }

    fn query_format_help(&self) -> &'static str {
        r#"PubMed supports comprehensive biomedical literature search:
- [Title] - Search in title only
- [Title/Abstract] - Search in title and abstract
- [Author] - Search by author name (LastName FirstInitial)
- [Journal] - Search by journal name
- [MeSH Terms] - Search using Medical Subject Headings
- [PMID] - Search by PubMed ID
- Use AND, OR, NOT for boolean operations
- Use quotes for exact phrases
- Date filters: 2020:2024[dp] for date range"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "cancer immunotherapy[Title]",
                "Search for cancer immunotherapy in titles",
            ),
            (
                "Smith J[Author] AND diabetes[MeSH]",
                "Author search combined with MeSH term",
            ),
            (
                "\"machine learning\"[Title/Abstract]",
                "Exact phrase in title or abstract",
            ),
            (
                "COVID-19[MeSH Terms] AND vaccine",
                "MeSH term combined with keyword",
            ),
            (
                "Nature[Journal] AND 2024[dp]",
                "Journal search with date filter",
            ),
            ("35000000[PMID]", "Search by specific PubMed ID"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://pubmed.ncbi.nlm.nih.gov/help/#search-tags")
    }

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        info!("Searching PubMed for: '{}'", query.query);

        let search_query = Self::build_query(query);
        let pmids = self
            .search_pubmed(&search_query, query.max_results as usize, context)
            .await?;

        if pmids.is_empty() {
            info!("No results found in PubMed for query: '{}'", query.query);
            return Ok(ProviderResult {
                papers: vec![],
                source: self.name().to_string(),
                total_available: Some(0),
                search_time: std::time::Duration::from_millis(0),
                has_more: false,
                metadata: HashMap::new(),
            });
        }

        let start_time = std::time::Instant::now();

        // Fetch article metadata
        let articles = self.fetch_articles(&pmids, context).await?;

        // Fetch abstracts separately (esummary doesn't include them)
        let abstracts = self.fetch_abstracts(&pmids, context).await.unwrap_or_default();

        let search_time = start_time.elapsed();

        let papers: Vec<PaperMetadata> = articles
            .iter()
            .map(|article| {
                let pmid = article.uid.as_ref();
                let abstract_text = pmid.and_then(|id| abstracts.get(id));
                Self::convert_to_paper(article, abstract_text)
            })
            .collect();

        let papers_count = u32::try_from(papers.len()).unwrap_or(u32::MAX);
        info!(
            "PubMed found {} papers for query: '{}'",
            papers_count, query.query
        );

        Ok(ProviderResult {
            papers,
            source: self.name().to_string(),
            total_available: Some(papers_count),
            search_time,
            has_more: papers_count >= query.max_results,
            metadata: HashMap::new(),
        })
    }

    async fn get_by_doi(
        &self,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
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
        let health_url = format!(
            "{}/entrez/eutils/einfo.fcgi?db=pubmed&retmode=json",
            self.base_url
        );

        let response = self
            .client
            .get(&health_url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("PubMed health check failed: {e}")))?;

        if response.status().is_success() {
            debug!("PubMed health check passed");
            Ok(true)
        } else {
            debug!(
                "PubMed health check failed with status: {}",
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
    async fn test_pubmed_provider_creation() {
        let provider = PubMedProvider::new(None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_metadata() {
        let provider = PubMedProvider::new(None).unwrap();

        assert_eq!(provider.name(), "pubmed");
        assert_eq!(provider.priority(), 92);
        assert!(!provider.supports_full_text()); // PubMed doesn't provide full text

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Doi));
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
        assert!(supported_types.contains(&SearchType::Keywords));
        assert!(supported_types.contains(&SearchType::TitleAbstract));
    }

    #[test]
    fn test_query_building() {
        let doi_query = SearchQuery {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Doi,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };
        assert_eq!(
            PubMedProvider::build_query(&doi_query),
            "10.1038/nature12373[doi]"
        );

        let title_query = SearchQuery {
            query: "cancer immunotherapy".to_string(),
            search_type: SearchType::Title,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };
        assert_eq!(
            PubMedProvider::build_query(&title_query),
            "cancer immunotherapy[Title]"
        );

        let author_query = SearchQuery {
            query: "Smith J".to_string(),
            search_type: SearchType::Author,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };
        assert_eq!(
            PubMedProvider::build_query(&author_query),
            "Smith J[Author]"
        );
    }

    #[test]
    fn test_idlist_deserialization_supports_string_or_array() {
        let json_array = r#"{"esearchresult":{"count":"1","retmax":"1","retstart":"0","idlist":["12345678"]}}"#;
        let parsed_array: PubMedSearchResponse = serde_json::from_str(json_array).unwrap();
        assert_eq!(
            parsed_array.esearchresult.idlist,
            vec!["12345678".to_string()]
        );

        let json_single = r#"{"esearchresult":{"count":"1","retmax":"1","retstart":"0","idlist":"12345678"}}"#;
        let parsed_single: PubMedSearchResponse = serde_json::from_str(json_single).unwrap();
        assert_eq!(
            parsed_single.esearchresult.idlist,
            vec!["12345678".to_string()]
        );
    }

    #[test]
    fn test_query_format_help() {
        let provider = PubMedProvider::new(None).unwrap();
        let help = provider.query_format_help();

        assert!(help.contains("[Title]"));
        assert!(help.contains("[Author]"));
        assert!(help.contains("[MeSH Terms]"));
        assert!(help.contains("AND, OR, NOT"));
    }

    #[test]
    fn test_query_examples() {
        let provider = PubMedProvider::new(None).unwrap();
        let examples = provider.query_examples();

        assert!(!examples.is_empty());
        assert!(examples
            .iter()
            .any(|(q, _)| q.contains("[Title]") || q.contains("[Author]")));
    }

    #[test]
    fn test_native_query_syntax() {
        let provider = PubMedProvider::new(None).unwrap();
        let syntax = provider.native_query_syntax();

        assert!(syntax.is_some());
        assert!(syntax.unwrap().contains("pubmed.ncbi.nlm.nih.gov"));
    }
}
