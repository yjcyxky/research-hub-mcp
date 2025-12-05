//! # Meta Search Adapter
//!
//! Concrete implementation of the SearchServicePort that uses the MetaSearchClient
//! to search across multiple academic providers.

use crate::client::{MetaSearchClient, MetaSearchConfig};
use crate::ports::search_service::{
    CircuitBreakerState, HealthStatus, ProviderHealth, SearchServicePort, ServiceHealth,
};
use crate::services::CategorizationService;
use crate::tools::search::{SearchInput, SearchResult};
use crate::{Config, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

/// Meta search adapter that implements SearchServicePort
///
/// This adapter wraps the existing MetaSearchClient and provides the
/// hexagonal architecture interface. It maintains compatibility with
/// the existing implementation while providing the new port interface.
#[derive(Clone)]
pub struct MetaSearchAdapter {
    /// Underlying meta search client
    meta_client: Arc<MetaSearchClient>,
    /// Configuration reference
    config: Arc<Config>,
    /// Categorization service for organizing results
    categorization_service: CategorizationService,
    /// Cache for search results
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Service start time for uptime calculation
    start_time: SystemTime,
    /// Metrics tracking
    metrics: Arc<RwLock<ServiceMetrics>>,
}

/// Cache entry for search results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: SearchResult,
    timestamp: SystemTime,
    ttl: Duration,
}

impl CacheEntry {
    fn new(result: SearchResult, ttl: Duration) -> Self {
        Self {
            result,
            timestamp: SystemTime::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.timestamp.elapsed().unwrap_or(Duration::MAX) > self.ttl
    }
}

/// Service metrics for monitoring
#[derive(Debug, Default)]
struct ServiceMetrics {
    total_searches: u64,
    successful_searches: u64,
    failed_searches: u64,
    cache_hits: u64,
    total_search_time_ms: u64,
    provider_stats: HashMap<String, ProviderMetrics>,
}

#[derive(Debug, Default)]
struct ProviderMetrics {
    queries: u64,
    successes: u64,
    failures: u64,
    total_response_time_ms: u64,
    last_success: Option<SystemTime>,
    last_failure: Option<SystemTime>,
}

impl std::fmt::Debug for MetaSearchAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaSearchAdapter")
            .field("meta_client", &"MetaSearchClient")
            .field("config", &"Config")
            .field("categorization_service", &"CategorizationService")
            .field("cache", &"RwLock<HashMap>")
            .field("start_time", &self.start_time)
            .finish()
    }
}

impl MetaSearchAdapter {
    /// Create a new meta search adapter
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing MetaSearchAdapter");

        // Create meta-search client
        let meta_config = MetaSearchConfig::from_config(&config);
        let meta_client = MetaSearchClient::new((*config).clone(), meta_config).map_err(|e| {
            crate::Error::Service(format!("Failed to create meta-search client: {e}"))
        })?;

        // Create categorization service
        let categorization_service = CategorizationService::new(config.categorization.clone())
            .map_err(|e| {
                crate::Error::Service(format!("Failed to create categorization service: {e}"))
            })?;

        Ok(Self {
            meta_client: Arc::new(meta_client),
            config,
            categorization_service,
            cache: Arc::new(RwLock::new(HashMap::new())),
            start_time: SystemTime::now(),
            metrics: Arc::new(RwLock::new(ServiceMetrics::default())),
        })
    }

    /// Generate cache key for search input
    fn generate_cache_key(input: &SearchInput) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            input.query.to_lowercase(),
            serde_json::to_string(&input.search_type).unwrap_or_default(),
            input.limit,
            input.offset,
            Self::normalize_sources_for_cache(&input.sources),
            Self::normalize_sources_for_cache(&input.metadata_sources)
        )
    }

    fn normalize_sources_for_cache(sources: &Option<Vec<String>>) -> String {
        sources.as_ref().map_or_else(
            || "default".to_string(),
            |list| {
                let mut normalized: Vec<String> = list
                    .iter()
                    .map(|s| s.trim().to_lowercase().replace([' ', '-'], "_"))
                    .collect();
                normalized.sort();
                normalized.join("|")
            },
        )
    }

    /// Get result from cache
    async fn get_from_cache(&self, cache_key: &str) -> Option<SearchResult> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(cache_key) {
            if !entry.is_expired() {
                let result = entry.result.clone();
                drop(cache);

                // Update cache hit metrics
                let mut metrics = self.metrics.write().await;
                metrics.cache_hits += 1;

                return Some(result);
            }
        }
        None
    }

    /// Cache search result
    async fn cache_result(&self, cache_key: &str, result: &SearchResult) {
        let mut cache = self.cache.write().await;
        let ttl = Duration::from_secs(self.config.research_source.timeout_secs * 10);
        cache.insert(cache_key.to_string(), CacheEntry::new(result.clone(), ttl));

        // Simple cache cleanup - remove expired entries
        cache.retain(|_, entry| !entry.is_expired());

        debug!("Cached search result, cache size: {}", cache.len());
    }

    /// Update service metrics
    async fn update_metrics(&self, success: bool, duration: Duration, cache_hit: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_searches += 1;
        if success {
            metrics.successful_searches += 1;
        } else {
            metrics.failed_searches += 1;
        }
        if !cache_hit {
            metrics.total_search_time_ms += duration.as_millis() as u64;
        }
    }

    /// Convert HealthStatus from string or other representations
    fn convert_health_status(&self, status: &str) -> HealthStatus {
        match status.to_lowercase().as_str() {
            "healthy" | "ok" | "up" => HealthStatus::Healthy,
            "degraded" | "warning" => HealthStatus::Degraded,
            "unhealthy" | "down" | "error" => HealthStatus::Unhealthy,
            _ => HealthStatus::Unknown,
        }
    }

    /// Get provider health from meta client
    async fn get_provider_health(&self) -> HashMap<String, ProviderHealth> {
        let mut provider_health = HashMap::new();
        let metrics = self.metrics.read().await;

        // For each provider, create health information
        for (provider_name, provider_metrics) in &metrics.provider_stats {
            let success_rate = if provider_metrics.queries > 0 {
                (provider_metrics.successes as f64 / provider_metrics.queries as f64) * 100.0
            } else {
                100.0
            };

            let status = if success_rate >= 80.0 {
                HealthStatus::Healthy
            } else if success_rate >= 50.0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unhealthy
            };

            let avg_response_time = if provider_metrics.successes > 0 {
                Some(provider_metrics.total_response_time_ms / provider_metrics.successes)
            } else {
                None
            };

            let circuit_breaker_state = if success_rate < 30.0 {
                CircuitBreakerState::Open
            } else if success_rate < 70.0 {
                CircuitBreakerState::HalfOpen
            } else {
                CircuitBreakerState::Closed
            };

            provider_health.insert(
                provider_name.clone(),
                ProviderHealth {
                    status,
                    last_success: provider_metrics.last_success,
                    last_failure: provider_metrics.last_failure,
                    response_time_ms: avg_response_time,
                    error_message: None,
                    circuit_breaker_state,
                },
            );
        }

        provider_health
    }
}

#[async_trait]
impl SearchServicePort for MetaSearchAdapter {
    #[instrument(skip(self), fields(query = %input.query, search_type = ?input.search_type))]
    async fn search_papers(&self, input: SearchInput) -> Result<SearchResult> {
        let start_time = SystemTime::now();
        info!(
            "Executing search: query='{}', type={:?}",
            input.query, input.search_type
        );

        // Check cache first
        let cache_key = Self::generate_cache_key(&input);
        if let Some(cached_result) = self.get_from_cache(&cache_key).await {
            debug!("Returning cached search result for query: {}", input.query);
            self.update_metrics(true, Duration::ZERO, true).await;
            return Ok(cached_result);
        }

        // Convert search input to provider format and execute search
        let provider_search_type = match input.search_type {
            crate::tools::search::SearchType::Auto => crate::client::providers::SearchType::Auto,
            crate::tools::search::SearchType::Doi => crate::client::providers::SearchType::Doi,
            crate::tools::search::SearchType::Title => crate::client::providers::SearchType::Title,
            crate::tools::search::SearchType::Author => {
                crate::client::providers::SearchType::Author
            }
            crate::tools::search::SearchType::AuthorYear => {
                crate::client::providers::SearchType::Keywords
            }
        };

        let search_query = crate::client::providers::SearchQuery {
            query: input.query.clone(),
            search_type: provider_search_type,
            max_results: input.limit,
            offset: input.offset,
            params: HashMap::new(),
            sources: input.sources.clone(),
            metadata_sources: input.metadata_sources.clone(),
        };

        // Execute meta-search
        let meta_result = match self.meta_client.search(&search_query).await {
            Ok(result) => result,
            Err(e) => {
                let duration = start_time.elapsed().unwrap_or(Duration::ZERO);
                self.update_metrics(false, duration, false).await;
                return Err(crate::Error::Service(format!("Meta-search failed: {e}")));
            }
        };

        // Convert to SearchResult format (reusing existing logic)
        let mut result = crate::tools::search::SearchTool::convert_meta_result_to_search_result(
            input.query.clone(),
            input.search_type.clone(),
            meta_result,
            &input,
        );

        // Add categorization if enabled
        if self.categorization_service.is_enabled() && !result.papers.is_empty() {
            let paper_metadata: Vec<crate::client::PaperMetadata> = result
                .papers
                .iter()
                .map(|paper| paper.metadata.clone())
                .collect();

            let category = self
                .categorization_service
                .sanitize_category(&format!("research_{}", input.query.replace(' ', "_")));

            result.category = Some(category.clone());
            for paper in &mut result.papers {
                paper.category = Some(category.clone());
            }
        }

        // Cache the result
        self.cache_result(&cache_key, &result).await;

        let duration = start_time.elapsed().unwrap_or(Duration::ZERO);
        self.update_metrics(true, duration, false).await;

        info!(
            "Search completed in {:?}, found {} results",
            duration, result.returned_count
        );

        Ok(result)
    }

    async fn health_check(&self) -> Result<ServiceHealth> {
        let provider_health = self.get_provider_health().await;
        let uptime = self.start_time.elapsed().unwrap_or(Duration::ZERO);
        let metrics = self.metrics.read().await;

        let error_rate = if metrics.total_searches > 0 {
            (metrics.failed_searches as f64 / metrics.total_searches as f64) * 100.0
        } else {
            0.0
        };

        let overall_status = if error_rate < 10.0 {
            HealthStatus::Healthy
        } else if error_rate < 30.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        Ok(ServiceHealth {
            status: overall_status,
            providers: provider_health,
            checked_at: SystemTime::now(),
            uptime_seconds: uptime.as_secs(),
            error_rate_percent: error_rate,
        })
    }

    async fn get_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let metrics = self.metrics.read().await;
        let mut result = HashMap::new();

        result.insert("total_searches".to_string(), metrics.total_searches.into());
        result.insert(
            "successful_searches".to_string(),
            metrics.successful_searches.into(),
        );
        result.insert(
            "failed_searches".to_string(),
            metrics.failed_searches.into(),
        );
        result.insert("cache_hits".to_string(), metrics.cache_hits.into());
        result.insert(
            "total_search_time_ms".to_string(),
            metrics.total_search_time_ms.into(),
        );

        if metrics.total_searches > 0 {
            let avg_search_time = metrics.total_search_time_ms / metrics.total_searches;
            result.insert("avg_search_time_ms".to_string(), avg_search_time.into());

            let cache_hit_rate =
                (metrics.cache_hits as f64 / metrics.total_searches as f64) * 100.0;
            result.insert("cache_hit_rate_percent".to_string(), cache_hit_rate.into());
        }

        let cache_size = self.cache.read().await.len();
        result.insert("cache_size".to_string(), cache_size.into());

        Ok(result)
    }

    async fn clear_cache(&self) -> Result<()> {
        self.cache.write().await.clear();
        info!("Search cache cleared");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ResearchSourceConfig};

    fn create_test_config() -> Arc<Config> {
        let mut config = Config::default();
        config.research_source = ResearchSourceConfig {
            endpoints: vec!["https://sci-hub.se".to_string()],
            rate_limit_per_sec: 1,
            timeout_secs: 30,
            max_retries: 2,
        };
        Arc::new(config)
    }

    #[test]
    fn test_adapter_creation() {
        let config = create_test_config();
        let adapter = MetaSearchAdapter::new(config);
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_cache_key_generation() {
        let input = SearchInput {
            query: "test query".to_string(),
            search_type: crate::tools::search::SearchType::Title,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };

        let key1 = MetaSearchAdapter::generate_cache_key(&input);
        let key2 = MetaSearchAdapter::generate_cache_key(&input);
        assert_eq!(key1, key2);

        let mut input2 = input.clone();
        input2.query = "different query".to_string();
        let key3 = MetaSearchAdapter::generate_cache_key(&input2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = create_test_config();
        let adapter = MetaSearchAdapter::new(config).unwrap();

        let input = SearchInput {
            query: "test".to_string(),
            search_type: crate::tools::search::SearchType::Title,
            limit: 10,
            offset: 0,
            sources: None,
            metadata_sources: None,
        };

        let result = SearchResult {
            query: "test".to_string(),
            search_type: crate::tools::search::SearchType::Title,
            papers: vec![],
            total_count: 0,
            returned_count: 0,
            offset: 0,
            has_more: false,
            search_time_ms: 100,
            source_mirror: None,
            category: None,
            successful_providers: Vec::new(),
            failed_providers: Vec::new(),
            metadata_providers: Vec::new(),
            provider_errors: HashMap::new(),
            papers_per_provider: HashMap::new(),
        };

        let cache_key = MetaSearchAdapter::generate_cache_key(&input);

        // Initially should be empty
        assert!(adapter.get_from_cache(&cache_key).await.is_none());

        // After caching should be available
        adapter.cache_result(&cache_key, &result).await;
        let cached = adapter.get_from_cache(&cache_key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().query, "test");

        // Clear cache
        adapter.clear_cache().await.unwrap();
        assert!(adapter.get_from_cache(&cache_key).await.is_none());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = create_test_config();
        let adapter = MetaSearchAdapter::new(config).unwrap();

        let health = adapter.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Healthy));
        assert!(health.uptime_seconds >= 0);
        assert_eq!(health.error_rate_percent, 0.0); // No searches yet
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = create_test_config();
        let adapter = MetaSearchAdapter::new(config).unwrap();

        let metrics = adapter.get_metrics().await.unwrap();
        assert!(metrics.contains_key("total_searches"));
        assert!(metrics.contains_key("successful_searches"));
        assert!(metrics.contains_key("failed_searches"));
        assert!(metrics.contains_key("cache_hits"));

        // Initially all should be 0
        assert_eq!(metrics.get("total_searches").unwrap().as_u64(), Some(0));
        assert_eq!(
            metrics.get("successful_searches").unwrap().as_u64(),
            Some(0)
        );
        assert_eq!(metrics.get("failed_searches").unwrap().as_u64(), Some(0));
        assert_eq!(metrics.get("cache_hits").unwrap().as_u64(), Some(0));
    }
}
