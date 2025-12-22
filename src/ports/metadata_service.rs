//! # Metadata Service Port
//!
//! Defines the port interface for metadata extraction services.
//! This interface abstracts the metadata extraction functionality, allowing different
//! extraction implementations to be used interchangeably.

use crate::tools::pdf_metadata::{MetadataInput, MetadataResult};
use crate::Result;
use async_trait::async_trait;
use std::fmt::Debug;

/// Port interface for metadata extraction services
///
/// This trait defines the contract for extracting metadata from academic papers.
/// Implementations should handle:
/// - PDF text and metadata extraction
/// - External validation (CrossRef, etc.)
/// - Caching for performance
/// - Batch processing capabilities
/// - Reference/citation extraction
///
/// # Design Principles
///
/// - **Format Agnostic**: Should work with various document formats
/// - **Cacheable**: Results should be cached to avoid re-processing
/// - **Extensible**: Support for different extraction methods
/// - **Robust**: Handle malformed or encrypted documents gracefully
/// - **Observable**: Provide metrics and progress tracking
///
/// # Example Implementation Structure
///
/// ```rust
/// use async_trait::async_trait;
/// use crate::ports::MetadataServicePort;
/// use crate::tools::pdf_metadata::{MetadataInput, MetadataResult};
/// use crate::Result;
///
/// pub struct PdfMetadataAdapter {
///     // Implementation details...
/// }
///
/// #[async_trait]
/// impl MetadataServicePort for PdfMetadataAdapter {
///     async fn extract_metadata(&self, input: MetadataInput) -> Result<MetadataResult> {
///         // 1. Validate file and input
///         // 2. Extract text and embedded metadata
///         // 3. Parse structured information
///         // 4. Validate with external sources if requested
///         // 5. Cache results
///         // 6. Return formatted metadata
///         todo!()
///     }
/// }
/// ```
#[async_trait]
pub trait MetadataServicePort: Send + Sync + Debug {
    /// Extract metadata from a document based on the provided input
    ///
    /// # Arguments
    ///
    /// * `input` - Extraction parameters including file path, options, etc.
    ///
    /// # Returns
    ///
    /// A `MetadataResult` containing:
    /// - Extracted metadata (title, authors, abstract, etc.)
    /// - Extraction confidence score
    /// - Processing statistics
    /// - Source information
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be accessed or read
    /// - File format is not supported
    /// - Critical extraction errors occur
    ///
    /// Partial extraction failures should be handled gracefully
    /// and reflected in the confidence score.
    async fn extract_metadata(&self, input: MetadataInput) -> Result<MetadataResult>;

    /// Extract metadata from multiple files in batch
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of extraction inputs for batch processing
    ///
    /// # Returns
    ///
    /// A vector of `MetadataResult` for each input file.
    /// Failed extractions are included with error information.
    ///
    /// # Errors
    ///
    /// Returns an error only for critical system failures.
    /// Individual file failures are reported in the results.
    async fn extract_batch_metadata(
        &self,
        inputs: Vec<MetadataInput>,
    ) -> Result<Vec<MetadataResult>>;

    /// Get cached metadata for a file if available
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file to check for cached metadata
    ///
    /// # Returns
    ///
    /// Cached metadata result if available and valid, None otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if cache access fails.
    async fn get_cached_metadata(
        &self,
        file_path: &std::path::Path,
    ) -> Result<Option<MetadataResult>>;

    /// Clear cached metadata entries
    ///
    /// # Arguments
    ///
    /// * `file_path` - Optional specific file to clear, or None to clear all
    ///
    /// # Returns
    ///
    /// Number of cache entries cleared.
    ///
    /// # Errors
    ///
    /// Returns an error if cache operations fail.
    async fn clear_cache(&self, file_path: Option<&std::path::Path>) -> Result<usize>;

    /// Validate extracted metadata against external sources
    ///
    /// # Arguments
    ///
    /// * `metadata` - Metadata to validate
    /// * `sources` - External sources to use for validation (CrossRef, etc.)
    ///
    /// # Returns
    ///
    /// Validation result with updated metadata and confidence scores.
    ///
    /// # Errors
    ///
    /// Returns an error if validation services are unavailable.
    /// Individual source failures should be handled gracefully.
    async fn validate_metadata(
        &self,
        metadata: &crate::tools::pdf_metadata::ExtractedMetadata,
        sources: Vec<ValidationSource>,
    ) -> Result<ValidationResult>;

    /// Get metadata service health and status
    ///
    /// # Returns
    ///
    /// Health information including:
    /// - Service operational status
    /// - Cache status and statistics
    /// - External validation service status
    /// - Processing capabilities
    async fn health_check(&self) -> Result<MetadataServiceHealth>;

    /// Get metadata service metrics
    ///
    /// # Returns
    ///
    /// A map of metric names to values, including:
    /// - Total extractions performed
    /// - Success/failure rates
    /// - Average processing times
    /// - Cache hit rates
    /// - Confidence score distributions
    async fn get_metrics(&self) -> Result<std::collections::HashMap<String, serde_json::Value>>;

    /// Get supported file formats for metadata extraction
    ///
    /// # Returns
    ///
    /// A vector of supported file extensions and MIME types.
    async fn get_supported_formats(&self) -> Result<Vec<SupportedFormat>>;
}

/// External sources for metadata validation
#[derive(
    Debug, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ValidationSource {
    /// CrossRef DOI database
    CrossRef,
    /// PubMed/MEDLINE database
    PubMed,
    /// arXiv preprint server
    ArXiv,
    /// Semantic Scholar API
    SemanticScholar,
    /// Open Research Knowledge Graph
    Orkg,
}

/// Result of metadata validation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ValidationResult {
    /// Updated metadata with validated information
    pub metadata: crate::tools::pdf_metadata::ExtractedMetadata,
    /// Validation success by source
    pub validation_results: std::collections::HashMap<ValidationSource, ValidationStatus>,
    /// Overall validation confidence score
    pub validation_confidence: f64,
    /// Validation timestamp
    pub validated_at: std::time::SystemTime,
}

/// Status of validation against a specific source
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ValidationStatus {
    /// Whether validation was successful
    pub success: bool,
    /// Fields that were validated/updated
    pub validated_fields: Vec<String>,
    /// Confidence boost from this validation
    pub confidence_boost: f64,
    /// Error message if validation failed
    pub error_message: Option<String>,
}

/// Health status of the metadata service
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MetadataServiceHealth {
    /// Overall service status
    pub status: super::search_service::HealthStatus,
    /// Cache status and statistics
    pub cache_status: CacheStatus,
    /// External validation services status
    pub validation_services:
        std::collections::HashMap<ValidationSource, super::search_service::ProviderHealth>,
    /// Processing capabilities
    pub processing_capabilities: ProcessingCapabilities,
    /// Last health check timestamp
    pub checked_at: std::time::SystemTime,
}

/// Cache status information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CacheStatus {
    /// Whether cache is available
    pub available: bool,
    /// Total cache entries
    pub total_entries: u64,
    /// Cache size in bytes
    pub cache_size_bytes: u64,
    /// Cache hit rate percentage
    pub hit_rate_percent: f64,
    /// Last cache cleanup timestamp
    pub last_cleanup: Option<std::time::SystemTime>,
}

/// Processing capabilities of the service
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ProcessingCapabilities {
    /// Maximum file size that can be processed (bytes)
    pub max_file_size_bytes: u64,
    /// Supported document formats
    pub supported_formats: Vec<SupportedFormat>,
    /// Maximum concurrent extractions
    pub max_concurrent_extractions: usize,
    /// Whether batch processing is supported
    pub batch_processing_supported: bool,
    /// Whether reference extraction is supported
    pub reference_extraction_supported: bool,
}

/// Supported file format information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SupportedFormat {
    /// File extension (e.g., "pdf", "docx")
    pub extension: String,
    /// MIME type
    pub mime_type: String,
    /// Format description
    pub description: String,
    /// Whether metadata extraction is fully supported
    pub fully_supported: bool,
    /// Extraction confidence for this format
    pub extraction_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_source_serialization() {
        let source = ValidationSource::CrossRef;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, "\"cross_ref\"");

        let source = ValidationSource::PubMed;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, "\"pub_med\"");
    }

    #[test]
    fn test_supported_format_creation() {
        let format = SupportedFormat {
            extension: "pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            description: "Portable Document Format".to_string(),
            fully_supported: true,
            extraction_confidence: 0.95,
        };

        assert_eq!(format.extension, "pdf");
        assert!(format.fully_supported);
        assert!((format.extraction_confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_status_creation() {
        let cache_status = CacheStatus {
            available: true,
            total_entries: 1000,
            cache_size_bytes: 1024 * 1024, // 1MB
            hit_rate_percent: 85.5,
            last_cleanup: Some(std::time::SystemTime::now()),
        };

        assert!(cache_status.available);
        assert_eq!(cache_status.total_entries, 1000);
        assert!((cache_status.hit_rate_percent - 85.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validation_status_creation() {
        let status = ValidationStatus {
            success: true,
            validated_fields: vec!["title".to_string(), "authors".to_string()],
            confidence_boost: 0.1,
            error_message: None,
        };

        assert!(status.success);
        assert_eq!(status.validated_fields.len(), 2);
        assert!((status.confidence_boost - 0.1).abs() < f64::EPSILON);
        assert!(status.error_message.is_none());
    }
}
