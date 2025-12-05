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

/// `PubMed Central` provider for biomedical and life science papers
///
/// PMC provides free access to full-text biomedical and life science journal articles
/// that have been deposited in the PMC repository. This provider searches the PMC
/// database and provides access to open access papers.
pub struct PubMedCentralProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>, // Optional NCBI API key for higher rate limits
}

#[derive(Debug, Deserialize)]
struct PmcSearchResponse {
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
struct PmcFetchResponse {
    result: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct PmcArticle {
    #[allow(dead_code)]
    uid: Option<String>,
    title: Option<String>,
    authors: Option<Vec<PmcAuthor>>,
    #[serde(rename = "fulljournalname")]
    journal_name: Option<String>,
    #[serde(rename = "pubdate")]
    pub_date: Option<String>,
    doi: Option<String>,
    #[allow(dead_code)]
    pmid: Option<String>,
    #[serde(rename = "pmcid")]
    pmc_id: Option<String>,
    #[serde(rename = "articleids")]
    article_ids: Option<Vec<ArticleId>>,
    #[serde(rename = "hasabstract")]
    #[allow(dead_code)]
    has_abstract: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PmcAuthor {
    name: Option<String>,
    #[allow(dead_code)]
    authtype: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArticleId {
    idtype: String,
    value: String,
}

/// Support both array and string forms for PMC idlist
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

impl PubMedCentralProvider {
    /// Create a new `PubMed Central` provider
    pub fn new(api_key: Option<String>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent(
                "knowledge_accumulator_mcp/0.3.0 (https://github.com/Ladvien/research_hub_mcp)",
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

    /// Search PMC using the E-utilities API
    async fn search_pmc(
        &self,
        query: &str,
        max_results: usize,
        context: &SearchContext,
    ) -> Result<Vec<String>, ProviderError> {
        let search_url = format!("{}/entrez/eutils/esearch.fcgi", self.base_url);

        let max_results_str = max_results.to_string();
        let mut params = vec![
            ("db", "pmc"),
            ("term", query),
            ("retmode", "json"),
            ("retmax", &max_results_str),
            ("sort", "relevance"), // Sort by relevance for better results
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
            .map_err(|e| ProviderError::Network(format!("PMC search request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "PMC search failed with status: {}",
                response.status()
            )));
        }

        let search_result: PmcSearchResponse = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse PMC search response: {e}"))
        })?;

        // Check for errors in the response
        if let Some(ref error_list) = search_result.esearchresult.errorlist {
            if let Some(ref phrases) = error_list.phrasesnotfound {
                warn!("PMC search phrases not found: {:?}", phrases);
            }
        }

        debug!(
            "PMC search found {} results for query: '{}'",
            search_result.esearchresult.count, query
        );

        Ok(search_result.esearchresult.idlist)
    }

    /// Fetch article details for given PMC IDs
    async fn fetch_articles(
        &self,
        pmc_ids: &[String],
        context: &SearchContext,
    ) -> Result<Vec<PmcArticle>, ProviderError> {
        if pmc_ids.is_empty() {
            return Ok(vec![]);
        }

        let fetch_url = format!("{}/entrez/eutils/esummary.fcgi", self.base_url);
        let ids_str = pmc_ids.join(",");

        let mut params = vec![("db", "pmc"), ("id", &ids_str), ("retmode", "json")];

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
            .map_err(|e| ProviderError::Network(format!("PMC fetch request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ProviderError::Network(format!(
                "PMC fetch failed with status: {}",
                response.status()
            )));
        }

        let fetch_result: PmcFetchResponse = response.json().await.map_err(|e| {
            ProviderError::Parse(format!("Failed to parse PMC fetch response: {e}"))
        })?;

        // The PMC response includes a "uids" array plus one object per UID. Filter and deserialize objects only.
        let mut articles = Vec::new();
        for (key, value) in fetch_result.result {
            if key == "uids" {
                continue;
            }
            if value.is_object() {
                match serde_json::from_value::<PmcArticle>(value) {
                    Ok(article) => articles.push(article),
                    Err(e) => warn!("Failed to parse PMC article {}: {}", key, e),
                }
            }
        }

        Ok(articles)
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
                format!("{}[title]", query.query)
            }
            SearchType::TitleAbstract => {
                // Search in title/abstract combined
                format!("{}[tiab]", query.query)
            }
            SearchType::Author => {
                // Search in author field
                format!("{}[author]", query.query)
            }
            SearchType::Keywords | SearchType::Auto | SearchType::Subject => {
                // General search across all fields
                query.query.clone()
            }
        }
    }

    /// Convert PMC article to `PaperMetadata`
    fn convert_to_paper(article: &PmcArticle) -> PaperMetadata {
        let mut authors = Vec::new();
        if let Some(ref author_list) = article.authors {
            authors = author_list.iter().filter_map(|a| a.name.clone()).collect();
        }

        // Extract DOI from article IDs if not in main DOI field
        let doi = article
            .doi
            .clone()
            .or_else(|| {
                article.article_ids.as_ref().and_then(|ids| {
                    ids.iter()
                        .find(|id| id.idtype == "doi")
                        .map(|id| id.value.clone())
                })
            })
            .unwrap_or_default();

        // Build PMC URL for PDF access
        let pdf_url = article
            .pmc_id
            .as_ref()
            .map(|pmc_id| format!("https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"));

        PaperMetadata {
            doi,
            title: article.title.clone(),
            authors,
            abstract_text: article.abstract_text.clone(),
            journal: article.journal_name.clone(),
            year: article.pub_date.as_ref().and_then(|date| {
                // Try to extract year from date string (format varies)
                Regex::new(r"(\d{4})")
                    .ok()?
                    .captures(date)?
                    .get(1)?
                    .as_str()
                    .parse::<u32>()
                    .ok()
            }),
            pdf_url,
            file_size: None, // File size not available from PMC API
        }
    }

    /// Check if DOI corresponds to a biomedical/life science paper
    fn is_biomedical_doi(&self, doi: &str) -> bool {
        // Common biomedical journal prefixes
        let biomedical_prefixes = [
            "10.1371", // PLOS
            "10.1038", // Nature
            "10.1016", // Elsevier (many biomedical journals)
            "10.1186", // BioMed Central
            "10.3389", // Frontiers
            "10.1172", // JCI
            "10.1158", // AACR
            "10.1084", // Rockefeller University Press
            "10.1073", // PNAS
            "10.1056", // NEJM
            "10.1001", // JAMA
            "10.1126", // Science
        ];

        biomedical_prefixes
            .iter()
            .any(|prefix| doi.starts_with(prefix))
    }
}

#[async_trait]
impl SourceProvider for PubMedCentralProvider {
    fn name(&self) -> &'static str {
        "pubmed_central"
    }

    fn priority(&self) -> u8 {
        95 // Highest priority for biomedical papers
    }

    fn supports_full_text(&self) -> bool {
        true
    }

    fn description(&self) -> &'static str {
        "PubMed Central provider for biomedical and life science papers"
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

    async fn search(
        &self,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        info!("Searching PMC for: '{}'", query.query);

        let search_query = Self::build_query(query);
        let pmc_ids = self
            .search_pmc(&search_query, query.max_results as usize, context)
            .await?;

        if pmc_ids.is_empty() {
            info!("No results found in PMC for query: '{}'", query.query);
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
        let articles = self.fetch_articles(&pmc_ids, context).await?;
        let search_time = start_time.elapsed();

        let papers: Vec<PaperMetadata> = articles.iter().map(Self::convert_to_paper).collect();

        let papers_count = u32::try_from(papers.len()).unwrap_or(u32::MAX);
        info!(
            "PMC found {} papers for query: '{}'",
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
        // Only handle biomedical DOIs for efficiency
        if !self.is_biomedical_doi(doi) {
            debug!("DOI {} doesn't appear to be biomedical, skipping PMC", doi);
            return Ok(None);
        }

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
            "{}/entrez/eutils/einfo.fcgi?db=pmc&retmode=json",
            self.base_url
        );

        let response = self
            .client
            .get(&health_url)
            .timeout(context.timeout)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("PMC health check failed: {e}")))?;

        if response.status().is_success() {
            debug!("PMC health check passed");
            Ok(true)
        } else {
            debug!("PMC health check failed with status: {}", response.status());
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
    async fn test_pmc_provider_creation() {
        let provider = PubMedCentralProvider::new(None);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_idlist_deserialization_supports_string_or_array() {
        let json_array =
            r#"{"esearchresult":{"count":"1","retmax":"1","retstart":"0","idlist":["12345"]}}"#;
        let parsed_array: PmcSearchResponse = serde_json::from_str(json_array).unwrap();
        assert_eq!(parsed_array.esearchresult.idlist, vec!["12345".to_string()]);

        let json_single =
            r#"{"esearchresult":{"count":"1","retmax":"1","retstart":"0","idlist":"6853299"}}"#;
        let parsed_single: PmcSearchResponse = serde_json::from_str(json_single).unwrap();
        assert_eq!(
            parsed_single.esearchresult.idlist,
            vec!["6853299".to_string()]
        );
    }

    #[tokio::test]
    async fn test_biomedical_doi_detection() {
        let provider = PubMedCentralProvider::new(None).unwrap();

        assert!(provider.is_biomedical_doi("10.1371/journal.pone.0000001"));
        assert!(provider.is_biomedical_doi("10.1038/nature12345"));
        assert!(!provider.is_biomedical_doi("10.1109/computer.science.123"));
    }

    #[tokio::test]
    async fn test_query_building() {
        let _provider = PubMedCentralProvider::new(None).unwrap();

        let doi_query = SearchQuery {
            query: "10.1371/journal.pone.0000001".to_string(),
            search_type: SearchType::Doi,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            PubMedCentralProvider::build_query(&doi_query),
            "10.1371/journal.pone.0000001[doi]"
        );

        let title_query = SearchQuery {
            query: "CRISPR gene editing".to_string(),
            search_type: SearchType::Title,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        assert_eq!(
            PubMedCentralProvider::build_query(&title_query),
            "CRISPR gene editing[title]"
        );
    }

    #[test]
    fn test_provider_metadata() {
        let provider = PubMedCentralProvider::new(None).unwrap();

        assert_eq!(provider.name(), "pubmed_central");
        assert_eq!(provider.priority(), 89);
        assert!(provider.supports_full_text());

        let supported_types = provider.supported_search_types();
        assert!(supported_types.contains(&SearchType::Doi));
        assert!(supported_types.contains(&SearchType::Title));
        assert!(supported_types.contains(&SearchType::Author));
        assert!(supported_types.contains(&SearchType::Keywords));
    }
}
