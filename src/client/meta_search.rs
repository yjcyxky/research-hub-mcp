//! Provider Registry and Cascade Lookup Client
//!
//! This module provides a registry of academic source providers and cascade
//! lookup functionality for DOI resolution and PDF URL discovery.
//!
//! For single-source searches, use the `search_source` tool instead.
//! For listing available sources, use the `list_sources` tool.

use crate::client::providers::{
    ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, GoogleScholarProvider,
    MdpiProvider, MedrxivProvider, OpenAlexProvider, OpenReviewProvider, ProviderError,
    PubMedCentralProvider, PubMedProvider, ResearchGateProvider, SciHubProvider, SearchContext,
    SearchType, SemanticScholarProvider, SourceProvider, SsrnProvider, UnpaywallProvider,
};
use crate::client::PaperMetadata;
use crate::Config;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Configuration for provider operations
#[derive(Debug, Clone)]
pub struct MetaSearchConfig {
    /// Timeout for each provider operation
    pub provider_timeout: Duration,
}

impl Default for MetaSearchConfig {
    fn default() -> Self {
        Self {
            provider_timeout: Duration::from_secs(30),
        }
    }
}

impl MetaSearchConfig {
    /// Create a new `MetaSearchConfig` with custom provider timeout
    #[must_use]
    pub const fn with_provider_timeout(provider_timeout: Duration) -> Self {
        Self { provider_timeout }
    }

    /// Create `MetaSearchConfig` from app config
    #[must_use]
    pub const fn from_config(config: &Config) -> Self {
        Self::with_provider_timeout(Duration::from_secs(
            config.research_source.provider_timeout_secs,
        ))
    }
}

/// Provider registry and cascade lookup client
///
/// This client maintains a registry of academic source providers and provides
/// cascade lookup functionality for DOI resolution and PDF URL discovery.
///
/// Note: For search operations, use the `search_source` tool which routes
/// queries directly to individual providers.
pub struct MetaSearchClient {
    providers: Vec<Arc<dyn SourceProvider>>,
    config: MetaSearchConfig,
}

impl MetaSearchClient {
    /// Create a new provider registry client
    pub fn new(_app_config: Config, meta_config: MetaSearchConfig) -> Result<Self, ProviderError> {
        let providers: Vec<Arc<dyn SourceProvider>> = vec![
            // CrossRef provider (highest priority for authoritative metadata)
            Arc::new(CrossRefProvider::new(None)?),
            // Google Scholar (SerpAPI-backed; metadata-first, PDF when available)
            Arc::new(GoogleScholarProvider::new(
                std::env::var("GOOGLE_SCHOLAR_API_KEY").ok(),
            )?),
            // Semantic Scholar provider (very high priority for PDF access + metadata)
            Arc::new(SemanticScholarProvider::new(None)?),
            // OpenAlex provider (high priority for comprehensive academic coverage)
            Arc::new(OpenAlexProvider::new()?),
            // Unpaywall provider (high priority for legal free PDF discovery)
            Arc::new(UnpaywallProvider::new_with_default_email()?),
            // PubMed Central provider (very high priority for biomedical papers with full text)
            Arc::new(PubMedCentralProvider::new(None)?),
            // PubMed provider (high priority for biomedical citations and abstracts)
            Arc::new(PubMedProvider::new(None)?),
            // CORE provider (high priority for open access collection)
            Arc::new(CoreProvider::new(None)?),
            // SSRN provider (high priority for recent papers and preprints)
            Arc::new(SsrnProvider::new()?),
            // arXiv provider (high priority for CS/physics/math)
            Arc::new(ArxivProvider::new()?),
            // bioRxiv provider (biology preprints)
            Arc::new(BiorxivProvider::new()?),
            // medRxiv provider (clinical preprints)
            Arc::new(MedrxivProvider::new()?),
            // OpenReview provider (high priority for ML conference papers)
            Arc::new(OpenReviewProvider::new()?),
            // MDPI provider (good priority for open access journals)
            Arc::new(MdpiProvider::new()?),
            // ResearchGate provider (lower priority due to access limitations)
            Arc::new(ResearchGateProvider::new()?),
            // Sci-Hub provider (lowest priority, for full-text access)
            Arc::new(SciHubProvider::new()?),
        ];

        info!(
            "Initialized provider registry with {} providers",
            providers.len()
        );

        Ok(Self {
            providers,
            config: meta_config,
        })
    }

    /// Get list of available provider names
    #[must_use]
    pub fn providers(&self) -> Vec<String> {
        self.providers
            .iter()
            .map(|p| p.name().to_string())
            .collect()
    }

    /// Get a provider by name
    #[must_use]
    pub fn get_provider(&self, name: &str) -> Option<Arc<dyn SourceProvider>> {
        self.providers.iter().find(|p| p.name() == name).cloned()
    }

    /// Get all providers as Arc references
    #[must_use]
    pub fn all_providers(&self) -> &[Arc<dyn SourceProvider>] {
        &self.providers
    }

    /// Perform health checks on all providers
    pub async fn health_check(&self) -> HashMap<String, bool> {
        let context = self.create_search_context();
        let mut results = HashMap::new();

        for provider in &self.providers {
            let health = provider.health_check(&context).await.unwrap_or(false);
            results.insert(provider.name().to_string(), health);

            if health {
                info!("Provider {} is healthy", provider.name());
            } else {
                warn!("Provider {} is unhealthy", provider.name());
            }
        }

        results
    }

    /// Search for a paper by DOI across providers that support it (cascade)
    ///
    /// This method tries providers in priority order until one returns a result.
    pub async fn get_by_doi(&self, doi: &str) -> Result<Option<PaperMetadata>, ProviderError> {
        let normalized_doi = Self::normalize_doi(doi);
        info!("Searching for DOI: {}", normalized_doi);

        let context = self.create_search_context();
        let doi_providers = self.select_doi_providers();

        for provider in doi_providers {
            if let Some(paper) = self
                .try_provider_for_doi(&provider, &normalized_doi, &context)
                .await?
            {
                return Ok(Some(paper));
            }
        }

        info!("DOI {} not found in any provider", normalized_doi);
        Ok(None)
    }

    /// Try to get a PDF URL from any provider, cascading through them by priority
    ///
    /// This method tries providers in priority order until one returns a valid URL.
    #[allow(clippy::cognitive_complexity)]
    pub async fn get_pdf_url_cascade(&self, doi: &str) -> Result<Option<String>, ProviderError> {
        info!("Attempting cascade PDF retrieval for DOI: {}", doi);

        let context = self.create_search_context();

        // Sort providers by priority (highest first)
        let mut providers: Vec<_> = self.providers.iter().collect();
        providers.sort_by_key(|p| std::cmp::Reverse(p.priority()));

        let mut last_error = None;

        for provider in providers {
            info!(
                "Trying PDF retrieval from provider: {} (priority: {})",
                provider.name(),
                provider.priority()
            );

            // Apply rate limiting
            if let Err(e) = Self::apply_rate_limit(provider).await {
                warn!("Rate limit hit for {}: {}", provider.name(), e);
                last_error = Some(e);
                continue;
            }

            // Try to get PDF URL from this provider
            match provider.get_pdf_url(doi, &context).await {
                Ok(Some(pdf_url)) if !pdf_url.is_empty() => {
                    info!(
                        "Successfully found PDF URL from {}: {}",
                        provider.name(),
                        pdf_url
                    );
                    return Ok(Some(pdf_url));
                }
                Ok(Some(empty_url)) => {
                    warn!(
                        "Data Quality Issue: Provider '{}' returned empty PDF URL for DOI '{}'. \
                        Empty URL value: '{}'",
                        provider.name(),
                        doi,
                        empty_url
                    );
                    debug!(
                        "Provider '{}' should return None instead of empty string for missing PDFs",
                        provider.name()
                    );
                }
                Ok(None) => {
                    debug!("Provider {} has no PDF for DOI: {}", provider.name(), doi);
                }
                Err(e) => {
                    warn!("Provider {} failed to get PDF: {}", provider.name(), e);
                    last_error = Some(e);
                }
            }
        }

        // If we get here, no provider could provide a PDF
        last_error.map_or_else(
            || {
                info!("No provider could find a PDF for DOI: {}", doi);
                Ok(None)
            },
            |error| Err(error),
        )
    }

    /// Deduplicate papers based on DOI and title similarity
    #[must_use]
    pub fn deduplicate_papers(papers: Vec<PaperMetadata>) -> Vec<PaperMetadata> {
        let original_count = papers.len();
        let mut unique_papers = Vec::new();
        let mut seen_dois = HashSet::new();
        let mut seen_titles = HashSet::new();

        for paper in papers {
            let mut is_duplicate = false;

            // Check DOI duplicates
            if !paper.doi.is_empty() {
                if seen_dois.contains(&paper.doi) {
                    is_duplicate = true;
                } else {
                    seen_dois.insert(paper.doi.clone());
                }
            }

            // Check title duplicates (case-insensitive, normalized)
            if !is_duplicate {
                if let Some(title) = &paper.title {
                    let normalized_title = title.to_lowercase().replace([' ', '\t', '\n'], "");
                    if seen_titles.contains(&normalized_title) {
                        is_duplicate = true;
                    } else {
                        seen_titles.insert(normalized_title);
                    }
                }
            }

            if !is_duplicate {
                unique_papers.push(paper);
            }
        }

        debug!(
            "Deduplicated {} papers to {} unique papers",
            original_count,
            unique_papers.len()
        );

        unique_papers
    }

    // --- Private helper methods ---

    /// Create search context with common settings
    fn create_search_context(&self) -> SearchContext {
        SearchContext {
            timeout: self.config.provider_timeout,
            user_agent: "knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)".to_string(),
            rate_limit: Some(Duration::from_millis(1000)),
            headers: HashMap::new(),
        }
    }

    /// Normalize DOI format for consistent processing
    fn normalize_doi(doi: &str) -> String {
        let trimmed = doi.trim();

        // Remove "doi:" prefix if present
        let without_prefix = if trimmed.to_lowercase().starts_with("doi:") {
            &trimmed[4..]
        } else {
            trimmed
        };

        // Remove URL prefixes if present
        let without_url = if without_prefix
            .to_lowercase()
            .starts_with("https://doi.org/")
        {
            &without_prefix[16..]
        } else if without_prefix.to_lowercase().starts_with("http://doi.org/") {
            &without_prefix[15..]
        } else {
            without_prefix
        };

        without_url.trim().to_string()
    }

    /// Select and prioritize providers that support DOI searches
    fn select_doi_providers(&self) -> Vec<Arc<dyn SourceProvider>> {
        let mut providers: Vec<_> = self
            .providers
            .iter()
            .filter(|p| p.supported_search_types().contains(&SearchType::Doi))
            .collect();

        // Sort by priority (highest first)
        providers.sort_by_key(|p| std::cmp::Reverse(p.priority()));

        debug!(
            "Selected {} DOI-capable providers: {:?}",
            providers.len(),
            providers.iter().map(|p| p.name()).collect::<Vec<_>>()
        );

        providers.into_iter().cloned().collect()
    }

    /// Try a single provider for DOI lookup
    async fn try_provider_for_doi(
        &self,
        provider: &Arc<dyn SourceProvider>,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        // Apply rate limiting
        if let Err(e) = Self::apply_rate_limit(provider).await {
            warn!("Rate limit hit for {}: {}", provider.name(), e);
            return Ok(None);
        }

        self.execute_doi_query(provider, doi, context).await
    }

    /// Execute the actual DOI query against a provider
    async fn execute_doi_query(
        &self,
        provider: &Arc<dyn SourceProvider>,
        doi: &str,
        context: &SearchContext,
    ) -> Result<Option<PaperMetadata>, ProviderError> {
        match provider.get_by_doi(doi, context).await {
            Ok(Some(paper)) => {
                info!("Found paper for DOI {} from {}", doi, provider.name());
                Ok(Some(paper))
            }
            Ok(None) => {
                debug!("DOI {} not found in {}", doi, provider.name());
                Ok(None)
            }
            Err(e) => {
                warn!("Error searching {} for DOI {}: {}", provider.name(), doi, e);
                // Continue to the next provider instead of failing completely
                Ok(None)
            }
        }
    }

    /// Apply rate limiting for a provider
    async fn apply_rate_limit(provider: &Arc<dyn SourceProvider>) -> Result<(), ProviderError> {
        let base_delay = provider.base_delay();
        tokio::time::sleep(base_delay).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_meta_search_client_creation() {
        let config = Config::default();
        let meta_config = MetaSearchConfig::from_config(&config);
        let client = MetaSearchClient::new(config, meta_config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_provider_listing() {
        let config = Config::default();
        let meta_config = MetaSearchConfig::from_config(&config);
        let client = MetaSearchClient::new(config, meta_config).unwrap();

        let providers = client.providers();
        assert!(providers.contains(&"arxiv".to_string()));
        assert!(providers.contains(&"biorxiv".to_string()));
        assert!(providers.contains(&"medrxiv".to_string()));
        assert!(providers.contains(&"core".to_string()));
        assert!(providers.contains(&"crossref".to_string()));
        assert!(providers.contains(&"google_scholar".to_string()));
        assert!(providers.contains(&"pubmed_central".to_string()));
        assert!(providers.contains(&"semantic_scholar".to_string()));
        assert!(providers.contains(&"unpaywall".to_string()));
        assert!(providers.contains(&"ssrn".to_string()));
        assert!(providers.contains(&"sci_hub".to_string()));
    }

    #[tokio::test]
    async fn test_get_provider_by_name() {
        let config = Config::default();
        let meta_config = MetaSearchConfig::from_config(&config);
        let client = MetaSearchClient::new(config, meta_config).unwrap();

        // Test getting existing provider
        let arxiv = client.get_provider("arxiv");
        assert!(arxiv.is_some());
        assert_eq!(arxiv.unwrap().name(), "arxiv");

        // Test getting non-existent provider
        let nonexistent = client.get_provider("nonexistent");
        assert!(nonexistent.is_none());
    }

    #[tokio::test]
    async fn test_all_providers() {
        let config = Config::default();
        let meta_config = MetaSearchConfig::from_config(&config);
        let client = MetaSearchClient::new(config, meta_config).unwrap();

        let all = client.all_providers();
        assert_eq!(all.len(), 15); // 15 providers registered
    }

    #[tokio::test]
    async fn test_deduplication() {
        let papers = vec![
            PaperMetadata {
                doi: "10.1038/nature12373".to_string(),
                title: Some("Test Paper".to_string()),
                authors: vec!["Author 1".to_string()],
                journal: Some("Nature".to_string()),
                year: Some(2023),
                abstract_text: None,
                pdf_url: None,
                file_size: None,
                pmid: None,
                keywords: vec!(),
            },
            PaperMetadata {
                doi: "10.1038/nature12373".to_string(), // Same DOI
                title: Some("Test Paper".to_string()),
                authors: vec!["Author 1".to_string()],
                journal: Some("Nature".to_string()),
                year: Some(2023),
                abstract_text: None,
                pdf_url: None,
                file_size: None,
                pmid: None,
                keywords: vec!()
            },
        ];

        let deduplicated = MetaSearchClient::deduplicate_papers(papers);
        assert_eq!(deduplicated.len(), 1);
    }

    #[test]
    fn test_normalize_doi() {
        // Standard DOI
        assert_eq!(
            MetaSearchClient::normalize_doi("10.1038/nature12373"),
            "10.1038/nature12373"
        );

        // With doi: prefix
        assert_eq!(
            MetaSearchClient::normalize_doi("doi:10.1038/nature12373"),
            "10.1038/nature12373"
        );

        // With https URL prefix
        assert_eq!(
            MetaSearchClient::normalize_doi("https://doi.org/10.1038/nature12373"),
            "10.1038/nature12373"
        );

        // With http URL prefix
        assert_eq!(
            MetaSearchClient::normalize_doi("http://doi.org/10.1038/nature12373"),
            "10.1038/nature12373"
        );

        // With whitespace
        assert_eq!(
            MetaSearchClient::normalize_doi("  10.1038/nature12373  "),
            "10.1038/nature12373"
        );
    }

    #[test]
    fn test_meta_search_config_default() {
        let config = MetaSearchConfig::default();
        assert_eq!(config.provider_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_meta_search_config_from_app_config() {
        let mut app_config = Config::default();
        app_config.research_source.provider_timeout_secs = 60;

        let meta_config = MetaSearchConfig::from_config(&app_config);
        assert_eq!(meta_config.provider_timeout, Duration::from_secs(60));
    }
}
