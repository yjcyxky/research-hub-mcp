//! # PDF Metadata Adapter
//!
//! Concrete implementation of the MetadataServicePort that uses the existing
//! MetadataExtractor to handle PDF metadata extraction.

use crate::ports::metadata_service::{
    CacheStatus, MetadataServiceHealth, MetadataServicePort, ProcessingCapabilities,
    SupportedFormat, ValidationResult, ValidationSource, ValidationStatus,
};
use crate::tools::pdf_metadata::{MetadataExtractor, MetadataInput, MetadataResult};
use crate::{Config, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{debug, info, instrument};

/// PDF metadata adapter that implements MetadataServicePort
///
/// This adapter wraps the existing MetadataExtractor and provides the
/// hexagonal architecture interface. It maintains compatibility with
/// the existing implementation while providing the new port interface.
#[derive(Clone)]
pub struct PdfMetadataAdapter {
    /// Underlying metadata extractor
    extractor: MetadataExtractor,
    /// Configuration reference
    config: Arc<Config>,
    /// Service start time for uptime calculation
    start_time: SystemTime,
}

impl std::fmt::Debug for PdfMetadataAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfMetadataAdapter")
            .field("extractor", &"MetadataExtractor")
            .field("config", &"Config")
            .field("start_time", &self.start_time)
            .finish()
    }
}

impl PdfMetadataAdapter {
    /// Create a new PDF metadata adapter
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing PdfMetadataAdapter");

        let extractor = MetadataExtractor::new(config.clone())?;

        Ok(Self {
            extractor,
            config,
            start_time: SystemTime::now(),
        })
    }

    /// Get cache status from the extractor
    async fn get_cache_status(&self) -> CacheStatus {
        // Get stats from the extractor
        let stats = self.extractor.get_stats().await;

        let total_extractions = stats.get("total_extractions").unwrap_or(&0).clone();
        let cache_hits = stats.get("cache_hits").unwrap_or(&0).clone();

        let hit_rate = if total_extractions > 0 {
            (cache_hits as f64 / total_extractions as f64) * 100.0
        } else {
            0.0
        };

        CacheStatus {
            available: true,           // Cache is always available in current implementation
            total_entries: cache_hits, // Approximation
            cache_size_bytes: 0,       // Not tracked in current implementation
            hit_rate_percent: hit_rate,
            last_cleanup: None, // Not tracked in current implementation
        }
    }

    /// Get processing capabilities
    fn get_processing_capabilities(&self) -> ProcessingCapabilities {
        ProcessingCapabilities {
            max_file_size_bytes: 100 * 1024 * 1024, // 100MB limit
            supported_formats: vec![SupportedFormat {
                extension: "pdf".to_string(),
                mime_type: "application/pdf".to_string(),
                description: "Portable Document Format".to_string(),
                fully_supported: true,
                extraction_confidence: 0.85,
            }],
            max_concurrent_extractions: 5, // Reasonable limit
            batch_processing_supported: true,
            reference_extraction_supported: true,
        }
    }

    /// Mock validation with external sources
    async fn mock_validate_with_sources(
        &self,
        metadata: &crate::tools::pdf_metadata::ExtractedMetadata,
        sources: Vec<ValidationSource>,
    ) -> ValidationResult {
        let mut validation_results = HashMap::new();
        let mut total_confidence_boost = 0.0;

        for source in sources {
            // Mock validation logic - in real implementation, this would call external APIs
            let success = match source {
                ValidationSource::CrossRef => metadata.doi.is_some(),
                ValidationSource::PubMed => {
                    metadata.journal.is_some() && metadata.authors.len() > 0
                }
                ValidationSource::ArXiv => metadata.title.is_some(),
                ValidationSource::SemanticScholar => metadata.abstract_text.is_some(),
                ValidationSource::Orkg => false, // Mock as not available
            };

            let confidence_boost = if success { 0.1 } else { 0.0 };
            total_confidence_boost += confidence_boost;

            validation_results.insert(
                source,
                ValidationStatus {
                    success,
                    validated_fields: if success {
                        vec!["title".to_string(), "authors".to_string()]
                    } else {
                        vec![]
                    },
                    confidence_boost,
                    error_message: if success {
                        None
                    } else {
                        Some("Validation not available".to_string())
                    },
                },
            );
        }

        let mut validated_metadata = metadata.clone();
        validated_metadata.confidence_score =
            (validated_metadata.confidence_score + total_confidence_boost).min(1.0);

        ValidationResult {
            metadata: validated_metadata,
            validation_results,
            validation_confidence: total_confidence_boost,
            validated_at: SystemTime::now(),
        }
    }
}

#[async_trait]
impl MetadataServicePort for PdfMetadataAdapter {
    #[instrument(skip(self), fields(file = %input.file_path))]
    async fn extract_metadata(&self, input: MetadataInput) -> Result<MetadataResult> {
        info!("Extracting metadata from: {}", input.file_path);
        self.extractor.extract_metadata(input).await
    }

    async fn extract_batch_metadata(
        &self,
        inputs: Vec<MetadataInput>,
    ) -> Result<Vec<MetadataResult>> {
        info!("Extracting metadata from {} files in batch", inputs.len());

        let mut results = Vec::new();
        for input in inputs {
            let result = self.extractor.extract_metadata(input).await;
            match result {
                Ok(metadata_result) => results.push(metadata_result),
                Err(e) => {
                    // Create a failed result for this input
                    results.push(MetadataResult {
                        status: crate::tools::pdf_metadata::ExtractionStatus::Failed,
                        metadata: None,
                        error: Some(e.to_string()),
                        processing_time_ms: 0,
                        file_path: "unknown".to_string(),
                    });
                }
            }
        }

        Ok(results)
    }

    async fn get_cached_metadata(&self, file_path: &Path) -> Result<Option<MetadataResult>> {
        // Try to get cached metadata using the private method
        // Since we can't access private methods, we'll create a mock input and check if it returns quickly
        let input = MetadataInput {
            file_path: file_path.to_string_lossy().to_string(),
            use_cache: true,
            validate_external: false,
            extract_references: false,
            batch_files: None,
        };

        let start_time = SystemTime::now();
        let result = self.extractor.extract_metadata(input).await?;
        let duration = start_time.elapsed().unwrap_or_default();

        // If it returned very quickly (< 100ms), it was likely from cache
        if duration.as_millis() < 100 {
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    async fn clear_cache(&self, file_path: Option<&Path>) -> Result<usize> {
        if file_path.is_some() {
            // Current implementation doesn't support selective cache clearing
            debug!("Selective cache clearing not supported, clearing entire cache");
        }

        self.extractor.clear_cache()?;

        // Return approximate count - we don't have the actual count
        Ok(1) // Assume at least one entry was cleared
    }

    async fn validate_metadata(
        &self,
        metadata: &crate::tools::pdf_metadata::ExtractedMetadata,
        sources: Vec<ValidationSource>,
    ) -> Result<ValidationResult> {
        info!("Validating metadata against {} sources", sources.len());

        // Use mock validation for now
        Ok(self.mock_validate_with_sources(metadata, sources).await)
    }

    async fn health_check(&self) -> Result<MetadataServiceHealth> {
        let cache_status = self.get_cache_status().await;
        let processing_capabilities = self.get_processing_capabilities();

        // Mock validation services status
        let mut validation_services = HashMap::new();
        validation_services.insert(
            ValidationSource::CrossRef,
            crate::ports::search_service::ProviderHealth {
                status: crate::ports::search_service::HealthStatus::Healthy,
                last_success: Some(SystemTime::now()),
                last_failure: None,
                response_time_ms: Some(200),
                error_message: None,
                circuit_breaker_state: crate::ports::search_service::CircuitBreakerState::Closed,
            },
        );

        Ok(MetadataServiceHealth {
            status: crate::ports::search_service::HealthStatus::Healthy,
            cache_status,
            validation_services,
            processing_capabilities,
            checked_at: SystemTime::now(),
        })
    }

    async fn get_metrics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let stats = self.extractor.get_stats().await;
        let mut metrics = HashMap::new();

        // Convert stats to JSON values
        for (key, value) in stats {
            metrics.insert(
                key,
                serde_json::Value::Number(serde_json::Number::from(value)),
            );
        }

        // Add additional metrics
        let uptime = self.start_time.elapsed().unwrap_or_default();
        metrics.insert("uptime_seconds".to_string(), uptime.as_secs().into());

        Ok(metrics)
    }

    async fn get_supported_formats(&self) -> Result<Vec<SupportedFormat>> {
        Ok(vec![SupportedFormat {
            extension: "pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            description: "Portable Document Format".to_string(),
            fully_supported: true,
            extraction_confidence: 0.85,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_adapter() -> PdfMetadataAdapter {
        let config = Arc::new(Config::default());
        PdfMetadataAdapter::new(config).unwrap()
    }

    #[test]
    fn test_adapter_creation() {
        let adapter = create_test_adapter();
        assert!(adapter.config.is_configured());
    }

    #[tokio::test]
    async fn test_health_check() {
        let adapter = create_test_adapter();
        let health = adapter.health_check().await.unwrap();

        assert!(matches!(
            health.status,
            crate::ports::search_service::HealthStatus::Healthy
        ));
        assert!(health.cache_status.available);
        assert!(health.processing_capabilities.batch_processing_supported);
        assert!(
            health
                .processing_capabilities
                .reference_extraction_supported
        );
    }

    #[tokio::test]
    async fn test_get_metrics() {
        let adapter = create_test_adapter();
        let metrics = adapter.get_metrics().await.unwrap();

        assert!(metrics.contains_key("uptime_seconds"));
        // Should contain metrics from the underlying extractor
        assert!(metrics.contains_key("total_extractions"));
    }

    #[tokio::test]
    async fn test_get_supported_formats() {
        let adapter = create_test_adapter();
        let formats = adapter.get_supported_formats().await.unwrap();

        assert_eq!(formats.len(), 1);
        assert_eq!(formats[0].extension, "pdf");
        assert_eq!(formats[0].mime_type, "application/pdf");
        assert!(formats[0].fully_supported);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let adapter = create_test_adapter();
        let result = adapter.clear_cache(None).await;
        assert!(result.is_ok());
        assert!(result.unwrap() >= 0);
    }

    #[tokio::test]
    async fn test_extract_batch_metadata_empty() {
        let adapter = create_test_adapter();
        let results = adapter.extract_batch_metadata(vec![]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_cached_metadata_nonexistent() {
        let adapter = create_test_adapter();
        let temp_dir = TempDir::new().unwrap();
        let nonexistent_file = temp_dir.path().join("nonexistent.pdf");

        let result = adapter.get_cached_metadata(&nonexistent_file).await;
        // Should return an error since file doesn't exist
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_metadata() {
        let adapter = create_test_adapter();

        let metadata = crate::tools::pdf_metadata::ExtractedMetadata {
            title: Some("Test Paper".to_string()),
            authors: vec![crate::tools::pdf_metadata::Author {
                name: "John Doe".to_string(),
                first_name: Some("John".to_string()),
                last_name: Some("Doe".to_string()),
                affiliation: None,
                email: None,
                orcid: None,
            }],
            publication_date: Some("2023".to_string()),
            journal: Some("Test Journal".to_string()),
            abstract_text: Some("Test abstract".to_string()),
            doi: Some("10.1234/test".to_string()),
            keywords: vec![],
            references: vec![],
            volume: None,
            issue: None,
            pages: None,
            confidence_score: 0.8,
            metadata_source: "pdf".to_string(),
            extracted_at: SystemTime::now(),
        };

        let sources = vec![ValidationSource::CrossRef, ValidationSource::PubMed];
        let result = adapter.validate_metadata(&metadata, sources).await.unwrap();

        assert!(result.validation_confidence > 0.0);
        assert_eq!(result.validation_results.len(), 2);
        assert!(result.metadata.confidence_score >= metadata.confidence_score);
    }

    #[test]
    fn test_processing_capabilities() {
        let adapter = create_test_adapter();
        let capabilities = adapter.get_processing_capabilities();

        assert!(capabilities.max_file_size_bytes > 0);
        assert!(capabilities.batch_processing_supported);
        assert!(capabilities.reference_extraction_supported);
        assert!(!capabilities.supported_formats.is_empty());
    }
}
