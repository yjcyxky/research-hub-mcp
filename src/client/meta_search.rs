use crate::client::providers::{
    ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, GoogleScholarProvider,
    MdpiProvider, MedrxivProvider, OpenAlexProvider, OpenReviewProvider, ProviderError,
    ProviderResult, PubMedCentralProvider, ResearchGateProvider, SciHubProvider, SearchContext,
    SearchQuery, SearchType, SemanticScholarProvider, SourceProvider, SsrnProvider,
    UnpaywallProvider,
};
use crate::client::PaperMetadata;
use crate::Config;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

const DEFAULT_SEARCH_SOURCES: &[&str] = &[
    "pubmed_central",
    "google_scholar",
    "biorxiv",
    "medrxiv",
    "arxiv",
    "mdpi",
    "semantic_scholar",
];

const DEFAULT_METADATA_SOURCES: &[&str] = &["crossref"];
const METADATA_ONLY_SOURCES: &[&str] = &["crossref"];
#[allow(dead_code)]
const OPT_IN_SOURCES: &[&str] = &[
    "openreview",
    "openalex",
    "core",
    "ssrn",
    "unpaywall",
    "researchgate",
    "sci_hub",
];

#[derive(Debug, Clone)]
struct SourceSelection {
    search_sources: HashSet<String>,
    metadata_sources: HashSet<String>,
}

/// Configuration for meta-search behavior
#[derive(Debug, Clone)]
pub struct MetaSearchConfig {
    /// Maximum number of providers to query in parallel
    pub max_parallel_providers: usize,
    /// Timeout for each provider
    pub provider_timeout: Duration,
    /// Whether to continue searching if some providers fail
    pub continue_on_failure: bool,
    /// Whether to deduplicate results
    pub deduplicate_results: bool,
    /// Minimum relevance score to include results
    pub min_relevance_score: f64,
}

impl Default for MetaSearchConfig {
    fn default() -> Self {
        Self {
            max_parallel_providers: 3,
            provider_timeout: Duration::from_secs(30),
            continue_on_failure: true,
            deduplicate_results: true,
            min_relevance_score: 0.0,
        }
    }
}

impl MetaSearchConfig {
    /// Create a new `MetaSearchConfig` with custom provider timeout
    #[must_use]
    pub const fn with_provider_timeout(provider_timeout: Duration) -> Self {
        Self {
            max_parallel_providers: 3,
            provider_timeout,
            continue_on_failure: true,
            deduplicate_results: true,
            min_relevance_score: 0.0,
        }
    }

    /// Create `MetaSearchConfig` from app config
    #[must_use]
    pub const fn from_config(config: &Config) -> Self {
        Self::with_provider_timeout(Duration::from_secs(
            config.research_source.provider_timeout_secs,
        ))
    }
}

/// Result from meta-search across multiple providers
#[derive(Debug, Clone)]
pub struct MetaSearchResult {
    /// All papers found across providers
    pub papers: Vec<PaperMetadata>,
    /// Results grouped by source
    pub by_source: HashMap<String, Vec<PaperMetadata>>,
    /// Metadata-only providers used for validation
    pub metadata_only_sources: HashSet<String>,
    /// Total search time
    pub total_search_time: Duration,
    /// Number of providers that succeeded
    pub successful_providers: usize,
    /// Number of providers that failed
    pub failed_providers: usize,
    /// Errors from failed providers
    pub provider_errors: HashMap<String, String>,
    /// Metadata from all providers
    pub provider_metadata: HashMap<String, HashMap<String, String>>,
}

/// Client that performs meta-search across multiple academic sources
/// Provider performance statistics for adaptive concurrency
#[derive(Debug, Clone)]
struct ProviderStats {
    /// Average response time in milliseconds
    avg_response_time: f64,
    /// Number of requests made
    request_count: u64,
    /// Last update timestamp
    last_updated: Instant,
}

impl Default for ProviderStats {
    fn default() -> Self {
        Self {
            avg_response_time: 1000.0, // Start with 1 second assumption
            request_count: 0,
            last_updated: Instant::now(),
        }
    }
}

pub struct MetaSearchClient {
    providers: Vec<Arc<dyn SourceProvider>>,
    config: MetaSearchConfig,
    #[allow(dead_code)]
    rate_limiters: Arc<RwLock<HashMap<String, Instant>>>,
    /// Provider performance statistics for adaptive semaphore sizing
    provider_stats: Arc<RwLock<HashMap<String, ProviderStats>>>,
}

impl MetaSearchClient {
    /// Create a new meta-search client
    pub fn new(_app_config: Config, meta_config: MetaSearchConfig) -> Result<Self, ProviderError> {
        let providers: Vec<Arc<dyn SourceProvider>> = vec![
            // CrossRef provider (highest priority for authoritative metadata)
            Arc::new(CrossRefProvider::new(None)?), // TODO: Get email from config
            // Google Scholar (SerpAPI-backed; metadata-first, PDF when available)
            Arc::new(GoogleScholarProvider::new(
                std::env::var("GOOGLE_SCHOLAR_API_KEY").ok(),
            )?),
            // Semantic Scholar provider (very high priority for PDF access + metadata)
            Arc::new(SemanticScholarProvider::new(None)?), // TODO: Get API key from config
            // OpenAlex provider (high priority for comprehensive academic coverage)
            Arc::new(OpenAlexProvider::new()?),
            // Unpaywall provider (high priority for legal free PDF discovery)
            Arc::new(UnpaywallProvider::new_with_default_email()?), // TODO: Get email from config
            // PubMed Central provider (very high priority for biomedical papers)
            Arc::new(PubMedCentralProvider::new(None)?), // TODO: Get API key from config
            // CORE provider (high priority for open access collection)
            Arc::new(CoreProvider::new(None)?), // TODO: Get API key from config
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
            "Initialized meta-search client with {} providers",
            providers.len()
        );

        Ok(Self {
            providers,
            config: meta_config,
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            provider_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Calculate adaptive semaphore size based on provider response times
    async fn calculate_adaptive_semaphore_size(&self, provider_count: usize) -> usize {
        let stats = self.provider_stats.read().await;

        if stats.is_empty() {
            // No stats available, use default
            return std::cmp::min(self.config.max_parallel_providers, provider_count);
        }

        // Calculate average response time across all providers
        let total_response_time: f64 = stats.values().map(|s| s.avg_response_time).sum();
        #[allow(clippy::cast_precision_loss)]
        let avg_response_time = total_response_time / stats.len() as f64;
        drop(stats);

        // Adaptive sizing: faster providers = more concurrency
        let adaptive_size = if avg_response_time < 500.0 {
            // Fast providers (< 500ms): allow more parallelism
            std::cmp::min(self.config.max_parallel_providers * 2, provider_count)
        } else if avg_response_time < 1500.0 {
            // Medium providers (500-1500ms): normal parallelism
            std::cmp::min(self.config.max_parallel_providers, provider_count)
        } else {
            // Slow providers (> 1500ms): reduce parallelism
            std::cmp::min(
                std::cmp::max(1, self.config.max_parallel_providers / 2),
                provider_count,
            )
        };

        debug!(
            "Adaptive semaphore sizing: avg_response_time={:.1}ms, size={}/{}",
            avg_response_time, adaptive_size, provider_count
        );

        adaptive_size
    }

    /// Static version of `update_provider_stats` for use in spawned tasks
    async fn update_provider_stats_static(
        provider_stats: &Arc<RwLock<HashMap<String, ProviderStats>>>,
        provider_name: &str,
        response_time_ms: f64,
    ) {
        let (avg_response_time, request_count) = {
            let mut stats = provider_stats.write().await;
            let provider_stats = stats.entry(provider_name.to_string()).or_default();

            // Update running average using exponential moving average
            let alpha: f64 = 0.2; // Weighting factor for new measurements
            if provider_stats.request_count == 0 {
                provider_stats.avg_response_time = response_time_ms;
            } else {
                provider_stats.avg_response_time = alpha.mul_add(
                    response_time_ms,
                    (1.0 - alpha) * provider_stats.avg_response_time,
                );
            }

            provider_stats.request_count += 1;
            provider_stats.last_updated = Instant::now();

            (
                provider_stats.avg_response_time,
                provider_stats.request_count,
            )
        };

        debug!(
            "Updated provider stats for {}: avg_time={:.1}ms, requests={}",
            provider_name, avg_response_time, request_count
        );
    }

    /// Get list of available providers
    #[must_use]
    pub fn providers(&self) -> Vec<String> {
        self.providers
            .iter()
            .map(|p| p.name().to_string())
            .collect()
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

    fn resolve_source_selection(&self, query: &SearchQuery) -> SourceSelection {
        let default_search: HashSet<String> = DEFAULT_SEARCH_SOURCES
            .iter()
            .map(|s| Self::normalize_source_name(s))
            .collect();
        let default_metadata: HashSet<String> = DEFAULT_METADATA_SOURCES
            .iter()
            .map(|s| Self::normalize_source_name(s))
            .collect();
        let metadata_only: HashSet<String> = METADATA_ONLY_SOURCES
            .iter()
            .map(|s| Self::normalize_source_name(s))
            .collect();

        let mut search_sources = query
            .sources
            .as_ref()
            .map(|sources| Self::normalize_sources(sources))
            .unwrap_or_else(|| default_search.clone());

        if search_sources.is_empty() {
            search_sources = default_search;
        }

        // Remove metadata-only providers from search list
        search_sources.retain(|s| !metadata_only.contains(s));

        let mut metadata_sources = query
            .metadata_sources
            .as_ref()
            .map(|sources| Self::normalize_sources(sources))
            .unwrap_or_else(|| default_metadata.clone());

        if metadata_sources.is_empty() {
            metadata_sources = default_metadata;
        }

        // Always include metadata-only providers
        metadata_sources.extend(metadata_only.iter().cloned());

        let available: HashSet<String> = self
            .providers
            .iter()
            .map(|p| p.name().to_string())
            .collect();
        search_sources.retain(|s| available.contains(s));
        metadata_sources.retain(|s| available.contains(s));

        SourceSelection {
            search_sources,
            metadata_sources,
        }
    }

    fn normalize_sources(sources: &[String]) -> HashSet<String> {
        sources
            .iter()
            .map(|s| Self::normalize_source_name(s))
            .collect()
    }

    fn normalize_source_name(source: &str) -> String {
        let normalized = source.trim().to_lowercase().replace([' ', '-'], "_");

        match normalized.as_str() {
            "pubmed" | "pmc" => "pubmed_central".to_string(),
            "google" | "google_scholar" => "google_scholar".to_string(),
            "semanticscholar" => "semantic_scholar".to_string(),
            "bioxiv" => "biorxiv".to_string(),
            "medrxiv" => "medrxiv".to_string(),
            "scihub" => "sci_hub".to_string(),
            other => other.to_string(),
        }
    }

    /// Search across multiple providers
    pub async fn search(&self, query: &SearchQuery) -> Result<MetaSearchResult, ProviderError> {
        let start_time = Instant::now();
        info!(
            "Starting meta-search for: {} (type: {:?})",
            query.query, query.search_type
        );

        // Create search context
        let context = self.create_search_context();

        let source_selection = self.resolve_source_selection(query);
        info!(
            "Active sources -> search: {:?}, metadata: {:?}",
            source_selection.search_sources, source_selection.metadata_sources
        );

        // Filter providers based on query type and supported features
        let suitable_providers =
            self.filter_providers_for_query(query, &source_selection.search_sources);
        let metadata_providers =
            self.filter_metadata_providers(query, &source_selection.metadata_sources);
        info!(
            "Using {} providers for search: {:?}",
            suitable_providers.len(),
            suitable_providers
                .iter()
                .map(|p| p.name())
                .collect::<Vec<_>>()
        );

        if !metadata_providers.is_empty() {
            info!(
                "Using {} metadata providers: {:?}",
                metadata_providers.len(),
                metadata_providers
                    .iter()
                    .map(|p| p.name())
                    .collect::<Vec<_>>()
            );
        }

        // Search providers in parallel
        let (provider_results, provider_errors) = self
            .execute_parallel_search(suitable_providers, query, &context)
            .await;

        let (metadata_results, metadata_errors) = if metadata_providers.is_empty() {
            (Vec::new(), HashMap::new())
        } else {
            self.execute_parallel_search(metadata_providers, query, &context)
                .await
        };

        let mut combined_errors = provider_errors;
        combined_errors.extend(metadata_errors);

        // Aggregate results
        let meta_result = self.aggregate_results(
            &provider_results,
            &metadata_results,
            combined_errors,
            start_time,
            &source_selection.metadata_sources,
        );

        info!(
            "Meta-search completed: {} total papers from {} providers in {:?}",
            meta_result.papers.len(),
            meta_result.successful_providers,
            meta_result.total_search_time
        );

        Ok(meta_result)
    }

    /// Search for a paper by DOI across providers that support it
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

    /// Normalize DOI format for consistent processing
    fn normalize_doi(doi: &str) -> String {
        // Remove common prefixes and normalize format
        let trimmed = doi.trim();

        // Remove "doi:" prefix if present
        let without_prefix = if trimmed.to_lowercase().starts_with("doi:") {
            &trimmed[4..]
        } else {
            trimmed
        };

        // Remove "https://doi.org/" prefix if present
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

    /// Try a single provider for DOI lookup with retry logic
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
                // For DOI searches, we don't want to fail completely on provider errors
                // Instead, we'll continue to the next provider
                Ok(None)
            }
        }
    }

    /// Create search context with common settings
    fn create_search_context(&self) -> SearchContext {
        SearchContext {
            timeout: self.config.provider_timeout,
            user_agent: "knowledge_accumulator_mcp/0.2.1 (Academic Research Tool)".to_string(),
            rate_limit: Some(Duration::from_millis(1000)),
            headers: HashMap::new(),
        }
    }

    /// Filter providers based on query characteristics
    fn filter_providers_for_query(
        &self,
        query: &SearchQuery,
        allowed_sources: &HashSet<String>,
    ) -> Vec<Arc<dyn SourceProvider>> {
        let mut suitable = Vec::new();

        for provider in &self.providers {
            let provider_name = provider.name().to_string();

            if !allowed_sources.contains(&provider_name) {
                continue;
            }

            // Check if provider supports the search type
            let supports_search_type = provider
                .supported_search_types()
                .contains(&query.search_type)
                || provider
                    .supported_search_types()
                    .contains(&SearchType::Auto);

            // For Auto searches, allow providers that support keyword searches even without explicit Auto support
            let supports_auto_via_keywords = query.search_type == SearchType::Auto
                && provider
                    .supported_search_types()
                    .contains(&SearchType::Keywords);

            if supports_search_type || supports_auto_via_keywords {
                suitable.push(provider.clone());
            }
        }

        // Apply intelligent priority ordering based on query characteristics
        Self::apply_intelligent_priority_ordering(&mut suitable, query);

        suitable
    }

    /// Filter providers that are designated for metadata validation/enrichment
    fn filter_metadata_providers(
        &self,
        query: &SearchQuery,
        metadata_sources: &HashSet<String>,
    ) -> Vec<Arc<dyn SourceProvider>> {
        let mut suitable = self
            .providers
            .iter()
            .filter(|provider| metadata_sources.contains(provider.name()))
            .filter(|provider| {
                provider
                    .supported_search_types()
                    .contains(&query.search_type)
                    || provider
                        .supported_search_types()
                        .contains(&SearchType::Auto)
            })
            .cloned()
            .collect::<Vec<_>>();

        Self::apply_intelligent_priority_ordering(&mut suitable, query);

        suitable
    }

    /// Execute parallel search across multiple providers
    async fn execute_parallel_search(
        &self,
        providers: Vec<Arc<dyn SourceProvider>>,
        query: &SearchQuery,
        context: &SearchContext,
    ) -> (Vec<(String, ProviderResult)>, HashMap<String, String>) {
        let mut provider_results = Vec::new();
        let mut provider_errors = HashMap::new();

        // Use adaptive semaphore sizing based on provider performance
        let adaptive_size = self
            .calculate_adaptive_semaphore_size(providers.len())
            .await;
        let semaphore = Arc::new(tokio::sync::Semaphore::new(adaptive_size));

        let mut tasks = Vec::new();
        for provider in providers {
            let provider = provider.clone();
            let query = query.clone();
            let context = context.clone();
            let semaphore = semaphore.clone();
            let timeout_duration = self.config.provider_timeout;

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let start_time = Instant::now();

                // Apply rate limiting
                if let Err(e) = Self::apply_rate_limit(&provider).await {
                    return (provider.name().to_string(), Err(e), start_time.elapsed());
                }

                // Search with timeout
                let result = timeout(timeout_duration, provider.search(&query, &context)).await;
                let elapsed = start_time.elapsed();

                let provider_name = provider.name().to_string();
                let result = match result {
                    Ok(Ok(provider_result)) => Ok(provider_result),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Err(ProviderError::Timeout),
                };

                (provider_name, result, elapsed)
            });

            tasks.push(task);
        }

        // Collect results and update statistics
        for task in tasks {
            match task.await {
                Ok((provider_name, Ok(result), elapsed)) => {
                    #[allow(clippy::cast_precision_loss)]
                    let response_time_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as f64;
                    info!(
                        "Provider {} returned {} results in {:.1}ms",
                        provider_name,
                        result.papers.len(),
                        response_time_ms
                    );

                    // Update provider statistics asynchronously
                    let provider_stats_clone = self.provider_stats.clone();
                    let provider_name_clone = provider_name.clone();
                    tokio::spawn(async move {
                        Self::update_provider_stats_static(
                            &provider_stats_clone,
                            &provider_name_clone,
                            response_time_ms,
                        )
                        .await;
                    });

                    provider_results.push((provider_name, result));
                }
                Ok((provider_name, Err(error), elapsed)) => {
                    #[allow(clippy::cast_precision_loss)]
                    let response_time_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as f64;
                    warn!(
                        "Provider {} failed after {:.1}ms: {}",
                        provider_name, response_time_ms, error
                    );

                    // Update stats even for failed requests to track response times
                    let provider_stats_clone = self.provider_stats.clone();
                    let provider_name_clone = provider_name.clone();
                    tokio::spawn(async move {
                        Self::update_provider_stats_static(
                            &provider_stats_clone,
                            &provider_name_clone,
                            response_time_ms,
                        )
                        .await;
                    });

                    provider_errors.insert(provider_name, error.to_string());
                }
                Err(e) => {
                    error!("Task failed: {}", e);
                }
            }
        }

        (provider_results, provider_errors)
    }

    /// Apply intelligent priority ordering based on query characteristics
    fn apply_intelligent_priority_ordering(
        providers: &mut Vec<Arc<dyn SourceProvider>>,
        query: &SearchQuery,
    ) {
        let query_lower = query.query.to_lowercase();

        // Create priority adjustments based on query analysis
        let mut provider_scores: Vec<(Arc<dyn SourceProvider>, i32)> = providers
            .iter()
            .map(|provider| {
                let base_priority = i32::from(provider.priority());
                let mut adjusted_priority = base_priority;

                // Domain-specific priority adjustments
                adjusted_priority += Self::calculate_domain_priority_boost(provider, &query_lower);

                // Search type priority adjustments
                adjusted_priority += Self::calculate_search_type_priority_boost(provider, query);

                // Content availability priority adjustments
                adjusted_priority += Self::calculate_content_priority_boost(provider, &query_lower);

                // Time-sensitive priority adjustments
                adjusted_priority +=
                    Self::calculate_temporal_priority_boost(provider, &query_lower);

                (provider.clone(), adjusted_priority)
            })
            .collect();

        // Sort by adjusted priority (highest first)
        provider_scores.sort_by_key(|(_, score)| std::cmp::Reverse(*score));

        // Update the providers vector with the new ordering
        *providers = provider_scores
            .into_iter()
            .map(|(provider, _)| provider)
            .collect();

        debug!(
            "Reordered providers based on query analysis: {:?}",
            providers.iter().map(|p| p.name()).collect::<Vec<_>>()
        );
    }

    /// Calculate domain-specific priority boost
    fn calculate_domain_priority_boost(provider: &Arc<dyn SourceProvider>, query: &str) -> i32 {
        let provider_name = provider.name();

        // Computer Science & Machine Learning
        if Self::contains_cs_ml_keywords(query) {
            match provider_name {
                "arxiv" => 15,           // arXiv is primary for CS papers
                "openreview" => 12,      // OpenReview for ML conference papers
                "semantic_scholar" => 8, // Good for CS papers
                "core" => 5,             // Open access CS papers
                _ => 0,
            }
        }
        // Biomedical & Life Sciences
        else if Self::contains_biomedical_keywords(query) {
            match provider_name {
                "pubmed_central" => 15,  // Primary for biomedical
                "biorxiv" => 12,         // Biology preprints
                "semantic_scholar" => 8, // Good coverage
                "unpaywall" => 5,        // Often has biomedical papers
                _ => 0,
            }
        }
        // Physics & Mathematics
        else if Self::contains_physics_math_keywords(query) {
            match provider_name {
                "arxiv" => 20,   // Primary for physics/math
                "crossref" => 8, // Good metadata
                "semantic_scholar" => 5,
                _ => 0,
            }
        }
        // Social Sciences & Economics
        else if Self::contains_social_science_keywords(query) {
            match provider_name {
                "ssrn" => 15,    // Primary for social sciences
                "crossref" => 8, // Good metadata
                "semantic_scholar" => 5,
                _ => 0,
            }
        }
        // Open Access & General Academic
        else if Self::contains_open_access_keywords(query) {
            match provider_name {
                "unpaywall" => 12,        // Specialized in open access
                "core" => 10,             // Large open access collection
                "mdpi" => 8,              // Open access publisher
                "biorxiv" | "arxiv" => 5, // Open preprints
                _ => 0,
            }
        } else {
            0 // No domain-specific boost
        }
    }

    /// Calculate search type priority boost
    fn calculate_search_type_priority_boost(
        provider: &Arc<dyn SourceProvider>,
        query: &SearchQuery,
    ) -> i32 {
        let provider_name = provider.name();

        match query.search_type {
            SearchType::Doi => {
                // DOI searches work best with metadata providers
                match provider_name {
                    "crossref" => 10, // Best for DOI resolution
                    "unpaywall" => 8, // Good DOI support
                    "semantic_scholar" => 6,
                    "pubmed_central" => 5, // Good for biomedical DOIs
                    _ => 0,
                }
            }
            SearchType::Author => {
                // Author searches work best with comprehensive databases
                match provider_name {
                    "semantic_scholar" => 10, // Excellent author disambiguation
                    "crossref" => 8,          // Good author metadata
                    "pubmed_central" => 6,    // Good for biomedical authors
                    "core" => 5,              // Large author database
                    _ => 0,
                }
            }
            SearchType::Title => {
                // All providers are generally good for title searches
                2 // Small boost for all
            }
            SearchType::Keywords => {
                // Keyword searches benefit from full-text providers
                match provider_name {
                    "semantic_scholar" => 8, // Good semantic search
                    "core" => 6,             // Full-text search
                    "unpaywall" => 4,        // Good coverage
                    _ => 0,
                }
            }
            SearchType::Subject => {
                // Subject searches benefit from specialized providers
                match provider_name {
                    "arxiv" | "pubmed_central" => 8, // Good subject classification
                    "semantic_scholar" => 6,         // AI-powered classification
                    _ => 0,
                }
            }
            SearchType::Auto => {
                0 // No specific boost for auto searches
            }
        }
    }

    /// Calculate content availability priority boost
    fn calculate_content_priority_boost(provider: &Arc<dyn SourceProvider>, query: &str) -> i32 {
        let provider_name = provider.name();

        // If query suggests need for full-text/PDF access
        if query.contains("pdf") || query.contains("full text") || query.contains("download") {
            match provider_name {
                "arxiv" | "biorxiv" => 12,                           // Always has PDFs
                "unpaywall" => 10,                                   // Specialized in free PDFs
                "semantic_scholar" | "pubmed_central" | "mdpi" => 8, // Often has PDF links/full text
                "ssrn" | "core" => 6,                                // Often has PDFs/full text
                "sci_hub" => 15, // Always tries for PDFs (but lowest base priority)
                _ => 0,
            }
        }
        // If query suggests need for recent/preprint content
        else if query.contains("recent")
            || query.contains("preprint")
            || query.contains("2024")
            || query.contains("2023")
        {
            match provider_name {
                "arxiv" | "biorxiv" => 10, // Latest preprints
                "ssrn" => 8,               // Recent working papers
                "openreview" => 6,         // Recent ML papers
                _ => 0,
            }
        } else {
            0
        }
    }

    /// Calculate temporal priority boost for time-sensitive queries
    fn calculate_temporal_priority_boost(provider: &Arc<dyn SourceProvider>, query: &str) -> i32 {
        let provider_name = provider.name();

        // Boost for recent year mentions
        if query.contains("2024") || query.contains("2023") {
            match provider_name {
                "arxiv" | "biorxiv" => 8,   // Best for recent preprints
                "ssrn" | "openreview" => 6, // Recent working papers/ML conference papers
                "semantic_scholar" => 4,    // Good recent coverage
                _ => 0,
            }
        }
        // Boost for historical content
        else if query.contains("historical")
            || query.contains("classic")
            || query.contains("1990")
            || query.contains("2000")
        {
            match provider_name {
                "crossref" => 8,         // Comprehensive historical metadata
                "pubmed_central" => 6,   // Long history for biomedical
                "semantic_scholar" => 4, // Good historical coverage
                _ => 0,
            }
        } else {
            0
        }
    }

    /// Check if query contains computer science/ML keywords
    fn contains_cs_ml_keywords(query: &str) -> bool {
        let cs_keywords = [
            "computer science",
            "machine learning",
            "deep learning",
            "neural network",
            "artificial intelligence",
            "ai",
            "ml",
            "algorithm",
            "data structure",
            "programming",
            "software",
            "computer vision",
            "nlp",
            "natural language",
            "database",
            "distributed system",
            "security",
            "cryptography",
            "compiler",
            "operating system",
            "network",
            "internet",
            "web",
            "mobile",
            "app",
            "tensorflow",
            "pytorch",
            "keras",
            "python",
            "java",
            "c++",
            "javascript",
            "transformer",
            "bert",
            "gpt",
            "lstm",
            "cnn",
            "gan",
            "reinforcement learning",
        ];

        cs_keywords.iter().any(|&keyword| query.contains(keyword))
    }

    /// Check if query contains biomedical keywords
    fn contains_biomedical_keywords(query: &str) -> bool {
        let bio_keywords = [
            "medicine",
            "medical",
            "biology",
            "biomedical",
            "clinical",
            "patient",
            "disease",
            "cancer",
            "drug",
            "therapy",
            "treatment",
            "diagnosis",
            "gene",
            "genome",
            "protein",
            "dna",
            "rna",
            "cell",
            "molecular",
            "pharmaceutical",
            "clinical trial",
            "epidemiology",
            "public health",
            "neuroscience",
            "cardiology",
            "oncology",
            "immunology",
            "microbiology",
            "biochemistry",
            "genetics",
            "pathology",
            "pharmacology",
            "physiology",
        ];

        bio_keywords.iter().any(|&keyword| query.contains(keyword))
    }

    /// Check if query contains physics/math keywords
    fn contains_physics_math_keywords(query: &str) -> bool {
        let physics_math_keywords = [
            "physics",
            "quantum",
            "relativity",
            "mechanics",
            "thermodynamics",
            "electromagnetism",
            "optics",
            "astronomy",
            "astrophysics",
            "cosmology",
            "mathematics",
            "algebra",
            "calculus",
            "geometry",
            "topology",
            "statistics",
            "probability",
            "number theory",
            "differential equation",
            "linear algebra",
            "mathematical",
            "theorem",
            "proof",
            "formula",
            "equation",
        ];

        physics_math_keywords
            .iter()
            .any(|&keyword| query.contains(keyword))
    }

    /// Check if query contains social science keywords
    fn contains_social_science_keywords(query: &str) -> bool {
        let social_keywords = [
            "economics",
            "economic",
            "finance",
            "financial",
            "business",
            "management",
            "psychology",
            "sociology",
            "political science",
            "anthropology",
            "education",
            "law",
            "legal",
            "policy",
            "social",
            "society",
            "culture",
            "history",
            "literature",
            "philosophy",
            "linguistics",
            "communication",
            "media",
            "marketing",
            "accounting",
            "organization",
            "leadership",
            "strategy",
        ];

        social_keywords
            .iter()
            .any(|&keyword| query.contains(keyword))
    }

    /// Check if query contains open access keywords
    fn contains_open_access_keywords(query: &str) -> bool {
        let oa_keywords = [
            "open access",
            "free",
            "libre",
            "creative commons",
            "cc by",
            "cc0",
            "public domain",
            "open source",
            "preprint",
            "repository",
            "institutional",
            "self-archived",
            "green oa",
            "gold oa",
            "hybrid",
            "subscription",
        ];

        oa_keywords.iter().any(|&keyword| query.contains(keyword))
    }

    /// Apply rate limiting for a provider
    async fn apply_rate_limit(provider: &Arc<dyn SourceProvider>) -> Result<(), ProviderError> {
        // Simple rate limiting - wait for base delay since last request
        let base_delay = provider.base_delay();

        // For now, just wait the base delay
        // In a more sophisticated implementation, we'd track per-provider timing
        tokio::time::sleep(base_delay).await;

        Ok(())
    }

    /// Aggregate results from multiple providers
    fn aggregate_results(
        &self,
        provider_results: &[(String, ProviderResult)],
        metadata_results: &[(String, ProviderResult)],
        provider_errors: HashMap<String, String>,
        start_time: Instant,
        metadata_only_sources: &HashSet<String>,
    ) -> MetaSearchResult {
        let mut all_papers = Vec::new();
        let mut by_source = HashMap::new();
        let mut provider_metadata = HashMap::new();

        // Collect all papers and organize by source (including metadata providers)
        for (source, result) in provider_results.iter().chain(metadata_results.iter()) {
            if !metadata_only_sources.contains(source) {
                all_papers.extend(result.papers.clone());
            }
            by_source.insert(source.clone(), result.papers.clone());
            provider_metadata.insert(source.clone(), result.metadata.clone());
        }

        // Deduplicate if requested
        if self.config.deduplicate_results {
            all_papers = Self::deduplicate_papers(all_papers);
        }

        // Filter by relevance score if needed
        if self.config.min_relevance_score > 0.0 {
            // Note: We'd need to add relevance scoring to PaperMetadata
            // For now, include all papers
        }

        // Sort by source priority and then by some relevance metric
        // For now, just keep the order

        MetaSearchResult {
            papers: all_papers,
            by_source,
            total_search_time: start_time.elapsed(),
            successful_providers: provider_results.len() + metadata_results.len(),
            failed_providers: provider_errors.len(),
            provider_errors,
            provider_metadata,
            metadata_only_sources: metadata_only_sources.clone(),
        }
    }

    /// Deduplicate papers based on DOI and title similarity
    fn deduplicate_papers(papers: Vec<PaperMetadata>) -> Vec<PaperMetadata> {
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

    /// Try to get a PDF URL from any provider, cascading through them by priority
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
                        "⚠️ Data Quality Issue: Provider '{}' returned empty PDF URL for DOI '{}'. \
                        This suggests the provider found the paper but has no PDF link available. \
                        Empty URL value: '{}'",
                        provider.name(),
                        doi,
                        empty_url
                    );
                    debug!(
                        "📊 Provider '{}' should return None instead of empty string for missing PDFs",
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
    async fn test_deduplication() {
        let config = Config::default();
        let meta_config = MetaSearchConfig::from_config(&config);
        let _client = MetaSearchClient::new(config, meta_config).unwrap();

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
            },
        ];

        let deduplicated = MetaSearchClient::deduplicate_papers(papers);
        assert_eq!(deduplicated.len(), 1);
    }
}
