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

/// Unpaywall API response for a DOI lookup
#[derive(Debug, Deserialize)]
struct UnpaywallResponse {
    doi: String,
    title: Option<String>,
    #[serde(rename = "z_authors")]
    authors: Option<Vec<UnpaywallAuthor>>,
    year: Option<u32>,
    #[allow(dead_code)]
    genre: Option<String>,
    journal_name: Option<String>,
    #[serde(rename = "is_oa")]
    is_open_access: bool,
    #[serde(rename = "best_oa_location")]
    best_oa_location: Option<OALocation>,
    #[serde(rename = "oa_locations")]
    oa_locations: Option<Vec<OALocation>>,
    #[serde(rename = "oa_date")]
    #[allow(dead_code)]
    oa_date: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnpaywallAuthor {
    family: Option<String>,
    given: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OALocation {
    #[allow(dead_code)]
    endpoint_id: Option<String>,
    #[allow(dead_code)]
    evidence: Option<String>,
    #[allow(dead_code)]
    has_repository_copy: Option<bool>,
    #[allow(dead_code)]
    is_best: Option<bool>,
    #[allow(dead_code)]
    license: Option<String>,
    #[allow(dead_code)]
    oa_date: Option<String>,
    #[allow(dead_code)]
    pmh_id: Option<String>,
    #[allow(dead_code)]
    repository_institution: Option<String>,
    #[allow(dead_code)]
    updated: Option<String>,
    #[allow(dead_code)]
    url: Option<String>,
    #[allow(dead_code)]
    url_for_landing_page: Option<String>,
    url_for_pdf: Option<String>,
    #[allow(dead_code)]
    version: Option<String>,
}

/// Unpaywall provider for finding open access versions of papers
pub struct UnpaywallProvider {
    client: Client,
    base_url: String,
    email: String,
}

impl UnpaywallProvider {
    /// Create a new Unpaywall provider
    /// Requires an email address as per Unpaywall API terms
    pub fn new(email: String) -> Result<Self, ProviderError> {
        if email.is_empty() || !email.contains('@') {
            return Err(ProviderError::Auth(
                "Valid email address required for Unpaywall API".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)")
            .build()
            .map_err(|e| ProviderError::Network(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: "https://api.unpaywall.org".to_string(),
            email,
        })
    }

    /// Create a new Unpaywall provider with default email
    pub fn new_with_default_email() -> Result<Self, ProviderError> {
        // Use a default email for testing - in production this should come from config
        Self::new("knowledge_accumulator_mcp@academic-tool.org".to_string())
    }

    /// Build Unpaywall DOI lookup URL
    fn build_doi_url(&self, doi: &str) -> String {
        format!(
            "{}/v2/{}?email={}",
            self.base_url,
            urlencoding::encode(doi),
            urlencoding::encode(&self.email)
        )
    }

    /// Convert Unpaywall response to `PaperMetadata`
    fn convert_response(&self, response: UnpaywallResponse) -> PaperMetadata {
        // Build author list from Unpaywall format
        let authors = response
            .authors
            .unwrap_or_default()
            .into_iter()
            .map(|author| {
                let given = author.given.unwrap_or_default();
                let family = author.family.unwrap_or_default();
                if given.is_empty() && family.is_empty() {
                    "Unknown Author".to_string()
                } else if given.is_empty() {
                    family
                } else if family.is_empty() {
                    given
                } else {
                    format!("{given} {family}")
                }
            })
            .collect();

        // Get the best PDF URL from open access locations
        let pdf_url = response
            .best_oa_location
            .as_ref()
            .and_then(|loc| loc.url_for_pdf.clone())
            .or_else(|| {
                // Fallback: look through all OA locations for a PDF
                response
                    .oa_locations
                    .as_ref()
                    .and_then(|locations| locations.iter().find_map(|loc| loc.url_for_pdf.clone()))
            })
            .filter(|url| !url.is_empty());

        PaperMetadata {
            doi: response.doi,
            title: response.title,
            authors,
            journal: response.journal_name,
            year: response.year,
            abstract_text: None, // Unpaywall doesn't provide abstracts
            pdf_url,
            file_size: None,
        }
    }

    /// Get paper by DOI from Unpaywall
    async fn get_paper_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let url = self.build_doi_url(doi);
        debug!("Getting paper by DOI from Unpaywall: {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Request failed: {e}")))?;

        if response.status().as_u16() == 404 {
            debug!("Paper not found in Unpaywall for DOI: {}", doi);
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

        debug!("Unpaywall response: {}", response_text);

        let unpaywall_response: UnpaywallResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                warn!("Failed to parse Unpaywall response: {}", response_text);
                ProviderError::Parse(format!("Failed to parse JSON: {e}"))
            })?;

        // Only return papers that have open access
        if unpaywall_response.is_open_access {
            Ok(Some(self.convert_response(unpaywall_response)))
        } else {
            debug!("Paper found but not open access: {}", doi);
            Ok(None)
        }
    }
}

#[async_trait]
impl SourceProvider for UnpaywallProvider {
    fn name(&self) -> &'static str {
        "unpaywall"
    }

    fn description(&self) -> &'static str {
        "Unpaywall - Database of open access research papers"
    }

    fn supported_search_types(&self) -> Vec<SearchType> {
        vec![SearchType::Doi] // Unpaywall only supports DOI lookups
    }

    fn query_format_help(&self) -> &'static str {
        r#"Unpaywall only supports DOI lookups:
- Provide a valid DOI to find open access versions
- Returns best available open access location
- No keyword or author search supported
- Use other providers for discovery, Unpaywall for OA access"#
    }

    fn query_examples(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("10.1038/nature12373", "Standard DOI lookup"),
            ("10.1126/science.1157996", "Science journal DOI"),
            ("10.1371/journal.pone.0000000", "PLOS ONE DOI"),
        ]
    }

    fn native_query_syntax(&self) -> Option<&'static str> {
        Some("https://unpaywall.org/products/api")
    }

    fn supports_full_text(&self) -> bool {
        true // Unpaywall specifically finds open access PDFs
    }

    fn priority(&self) -> u8 {
        87 // High priority for finding legal free versions
    }

    fn base_delay(&self) -> Duration {
        Duration::from_millis(200) // Be respectful to the free API
    }

    async fn search(
        &self,
        query: &SearchQuery,
        _context: &SearchContext,
    ) -> Result<ProviderResult, ProviderError> {
        let start_time = Instant::now();

        info!(
            "Searching Unpaywall for: {} (type: {:?})",
            query.query, query.search_type
        );

        let papers = if query.search_type == SearchType::Doi {
            // Unpaywall only supports DOI lookups
            if let Some(paper) = self.get_paper_by_doi(&query.query).await? {
                vec![paper]
            } else {
                Vec::new()
            }
        } else {
            // Unpaywall doesn't support other search types
            warn!(
                "Unpaywall only supports DOI searches, ignoring query: {}",
                query.query
            );
            Vec::new()
        };

        let search_time = start_time.elapsed();
        let papers_count = papers.len();

        let result = ProviderResult {
            papers,
            source: "Unpaywall".to_string(),
            total_available: Some(u32::try_from(papers_count).unwrap_or(u32::MAX)),
            search_time,
            has_more: false, // Single DOI lookup can't have more results
            metadata: HashMap::new(),
        };

        info!(
            "Unpaywall search completed: {} papers found in {:?}",
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
        info!("Getting paper by DOI from Unpaywall: {}", doi);
        self.get_paper_by_doi(doi).await
    }

    async fn health_check(&self, _context: &SearchContext) -> Result<bool, ProviderError> {
        debug!("Performing Unpaywall health check");

        // Use a known DOI for health check
        let test_url = self.build_doi_url("10.1038/nature12373");

        match self.client.get(&test_url).send().await {
            Ok(response) if response.status().is_success() || response.status().as_u16() == 404 => {
                info!("Unpaywall health check: OK");
                Ok(true)
            }
            Ok(response) => {
                warn!(
                    "Unpaywall health check failed with status: {}",
                    response.status()
                );
                Ok(false)
            }
            Err(e) => {
                warn!("Unpaywall health check failed: {}", e);
                Ok(false)
            }
        }
    }
}

impl Default for UnpaywallProvider {
    fn default() -> Self {
        Self::new_with_default_email().expect("Failed to create UnpaywallProvider")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpaywall_provider_creation() {
        let provider = UnpaywallProvider::new("test@example.com".to_string());
        assert!(provider.is_ok());
    }

    #[test]
    fn test_unpaywall_provider_invalid_email() {
        let provider = UnpaywallProvider::new("invalid_email".to_string());
        assert!(provider.is_err());
    }

    #[test]
    fn test_provider_interface() {
        let provider = UnpaywallProvider::new("test@example.com".to_string()).unwrap();

        assert_eq!(provider.name(), "unpaywall");
        assert!(provider.supports_full_text());
        assert_eq!(provider.priority(), 87);
        assert!(provider.supported_search_types().contains(&SearchType::Doi));
        assert!(!provider
            .supported_search_types()
            .contains(&SearchType::Title));
    }

    #[test]
    fn test_url_building() {
        let provider = UnpaywallProvider::new("test@example.com".to_string()).unwrap();

        let doi_url = provider.build_doi_url("10.1038/nature12373");
        assert!(doi_url.contains("10.1038%2Fnature12373"));
        assert!(doi_url.contains("email=test%40example.com"));
        assert!(doi_url.contains("api.unpaywall.org"));
    }

    #[test]
    fn test_default_email_provider() {
        let provider = UnpaywallProvider::new_with_default_email();
        assert!(provider.is_ok());
    }
}
