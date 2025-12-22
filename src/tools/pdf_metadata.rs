use crate::{Config, Result};
use futures::StreamExt;
use lopdf::{Document, Object};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::io::AsyncReadExt;
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

/// Input parameters for PDF metadata extraction
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataInput {
    /// Path to single PDF file
    #[schemars(
        description = "Path to PDF file for single extraction. Use batch_files for multiple PDFs."
    )]
    pub file_path: String,
    /// Whether to use cached results if available
    #[schemars(description = "Use cached metadata if available (default: true)")]
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    /// Whether to extract references/citations
    #[schemars(description = "Extract references and citations from the PDF (default: false)")]
    #[serde(default = "default_extract_refs")]
    pub extract_references: bool,
    /// Batch processing file list
    #[schemars(
        description = "Array of PDF file paths for batch processing. Processes up to 12 files concurrently. No limit on total files."
    )]
    pub batch_files: Option<Vec<String>>,
}

/// Extracted metadata structure
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExtractedMetadata {
    /// Paper title
    pub title: Option<String>,
    /// List of authors
    pub authors: Vec<Author>,
    /// Publication date
    pub publication_date: Option<String>,
    /// Journal or conference name
    pub journal: Option<String>,
    /// Paper abstract
    pub abstract_text: Option<String>,
    /// DOI if found
    pub doi: Option<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// References/citations
    pub references: Vec<Reference>,
    /// Volume number
    pub volume: Option<String>,
    /// Issue number
    pub issue: Option<String>,
    /// Page range
    pub pages: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence_score: f64,
    /// Source of metadata (pdf, crossref, etc.)
    pub metadata_source: String,
    /// Extraction timestamp
    pub extracted_at: SystemTime,
}

/// Author information
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Author {
    /// Full name
    pub name: String,
    /// First name if parsed
    pub first_name: Option<String>,
    /// Last name if parsed
    pub last_name: Option<String>,
    /// Affiliation/institution
    pub affiliation: Option<String>,
    /// Email if found
    pub email: Option<String>,
    /// ORCID if found
    pub orcid: Option<String>,
}

/// Reference/citation information
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Reference {
    /// Reference ID/number
    pub id: String,
    /// Raw reference text
    pub raw_text: String,
    /// Parsed title if available
    pub title: Option<String>,
    /// Parsed authors if available
    pub authors: Option<String>,
    /// Parsed year if available
    pub year: Option<String>,
    /// Parsed journal if available
    pub journal: Option<String>,
    /// DOI if found
    pub doi: Option<String>,
}

/// Result of metadata extraction
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataResult {
    /// Extraction status
    pub status: ExtractionStatus,
    /// Extracted metadata
    pub metadata: Option<ExtractedMetadata>,
    /// Error message if failed
    pub error: Option<String>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// File path processed
    pub file_path: String,
}

/// Batch metadata extraction result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BatchMetadataResult {
    /// Individual results for each file
    pub results: Vec<MetadataResult>,
    /// Total processing time
    pub total_time_ms: u64,
    /// Number of successful extractions
    pub success_count: usize,
    /// Number of failed extractions
    pub failure_count: usize,
}

/// Extraction status
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionStatus {
    /// Successfully extracted
    Success,
    /// Partially extracted (some fields missing)
    Partial,
    /// Failed to extract
    Failed,
    /// Cached result returned
    Cached,
}

/// Default for `use_cache`
const fn default_use_cache() -> bool {
    true
}

/// Default for `extract_references`
const fn default_extract_refs() -> bool {
    false
}

/// Cache entry for metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    metadata: ExtractedMetadata,
    file_hash: String,
    cached_at: SystemTime,
}

/// PDF Metadata extraction tool
#[derive(Clone)]
pub struct MetadataExtractor {
    #[allow(dead_code)] // Will be used for configuration access
    config: Arc<Config>,
    cache_db: Option<sled::Db>,
    cache_ttl: Duration,
    extraction_patterns: ExtractionPatterns,
    stats: Arc<RwLock<ExtractionStats>>,
}

/// Statistics for extraction operations
#[derive(Debug, Default)]
struct ExtractionStats {
    total_extractions: u64,
    successful_extractions: u64,
    failed_extractions: u64,
    cache_hits: u64,
    total_processing_time_ms: u64,
}

/// Regex patterns for metadata extraction
#[allow(clippy::struct_field_names)] // Pattern suffix is intentional for clarity
#[derive(Clone)]
struct ExtractionPatterns {
    #[allow(dead_code)] // May be used in future enhancements
    title_pattern: Regex,
    author_pattern: Regex,
    doi_pattern: Regex,
    date_pattern: Regex,
    #[allow(dead_code)] // May be used in future enhancements
    email_pattern: Regex,
    abstract_pattern: Regex,
    reference_pattern: Regex,
    journal_pattern: Regex,
    volume_issue_pattern: Regex,
}

impl Default for ExtractionPatterns {
    fn default() -> Self {
        Self {
            // Title patterns - look for large fonts, first non-author text
            title_pattern: Regex::new(r"(?i)^([A-Z][^.!?]*[.!?]?)$").unwrap(),

            // Author patterns - name formats
            author_pattern: Regex::new(r"(?i)([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*\s+[A-Z][a-z]+)").unwrap(),

            // DOI pattern
            doi_pattern: Regex::new(r"(?i)(?:doi:?\s*|https?://doi\.org/|https?://dx\.doi\.org/)?(10\.\d{4,}/[-._;()/:\w]+)").unwrap(),

            // Date patterns
            date_pattern: Regex::new(r"(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2})[,\s]+\d{4}|\d{4}").unwrap(),

            // Email pattern
            email_pattern: Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),

            // Abstract pattern
            abstract_pattern: Regex::new(r"(?is)abstract[:\s]*(.+?)(?:keywords|introduction|1\.?\s*introduction|\n\n)").unwrap(),

            // Reference pattern
            reference_pattern: Regex::new(r"(?i)^\[?\d+\]?\.?\s*(.+)$").unwrap(),

            // Journal pattern
            journal_pattern: Regex::new(r"(?i)(?:journal|proceedings|conference|trans(?:actions)?\.?)\s+(?:of\s+)?(.+?)(?:\s+vol|\s+\d{4}|$)").unwrap(),

            // Volume/Issue pattern
            volume_issue_pattern: Regex::new(r"(?i)vol(?:ume)?\.?\s*(\d+)(?:.*?(?:no|issue)\.?\s*(\d+))?").unwrap(),
        }
    }
}

impl MetadataExtractor {
    /// Validate PDF file before attempting to parse
    async fn validate_pdf_file(file_path: &Path) -> Result<()> {
        // Check file exists and has valid size
        let metadata =
            tokio::fs::metadata(file_path)
                .await
                .map_err(|e| crate::Error::InvalidInput {
                    field: "file_path".to_string(),
                    reason: format!("Cannot access file: {e}"),
                })?;

        if metadata.len() == 0 {
            return Err(crate::Error::Parse {
                context: "PDF validation".to_string(),
                message: "File is empty (0 bytes)".to_string(),
            });
        }

        if metadata.len() < 1024 {
            return Err(crate::Error::Parse {
                context: "PDF validation".to_string(),
                message: format!(
                    "File is too small ({} bytes) to be a valid PDF",
                    metadata.len()
                ),
            });
        }

        // Check for PDF magic bytes
        let mut file = tokio::fs::File::open(file_path)
            .await
            .map_err(crate::Error::Io)?;

        let mut header = [0u8; 8];
        file.read_exact(&mut header)
            .await
            .map_err(|e| crate::Error::Parse {
                context: "PDF validation".to_string(),
                message: format!("Cannot read file header: {e}"),
            })?;

        // Check for PDF signature
        if !header.starts_with(b"%PDF-") {
            return Err(crate::Error::Parse {
                context: "PDF validation".to_string(),
                message: format!(
                    "Invalid file header - not a PDF file. Got: {:?}",
                    String::from_utf8_lossy(&header[..5])
                ),
            });
        }

        Ok(())
    }

    /// Create a new metadata extractor
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing metadata extraction tool");

        // Initialize cache database
        let cache_db = {
            match sled::open(".metadata_cache") {
                Ok(db) => {
                    info!("Opened metadata cache database");
                    Some(db)
                }
                Err(e) => {
                    warn!(
                        "Failed to open cache database: {}, continuing without cache",
                        e
                    );
                    None
                }
            }
        };

        Ok(Self {
            config,
            cache_db,
            cache_ttl: Duration::from_secs(86400 * 7), // 7 days
            extraction_patterns: ExtractionPatterns::default(),
            stats: Arc::new(RwLock::new(ExtractionStats::default())),
        })
    }

    /// Extract metadata from a PDF file
    #[instrument(skip(self), fields(file = %input.file_path))]
    pub async fn extract_metadata(&self, input: MetadataInput) -> Result<MetadataResult> {
        let start_time = SystemTime::now();
        info!("Starting metadata extraction for: {}", input.file_path);

        // Check for batch processing
        if let Some(batch_files) = input.batch_files {
            // Use Box::pin to avoid recursion issue
            return Box::pin(self.extract_batch(batch_files, input.use_cache)).await;
        }

        let file_path = PathBuf::from(&input.file_path);

        // Validate file exists
        if !file_path.exists() {
            return Ok(MetadataResult {
                status: ExtractionStatus::Failed,
                metadata: None,
                error: Some(format!(
                    "File not found: {file_path}",
                    file_path = input.file_path
                )),
                processing_time_ms: 0,
                file_path: input.file_path,
            });
        }

        // Check cache if enabled
        if input.use_cache {
            if let Some(cached) = self.get_cached_metadata(&file_path).await? {
                info!("Returning cached metadata for: {}", input.file_path);
                let processing_time = start_time.elapsed().unwrap_or_default();

                self.update_stats(
                    true,
                    true,
                    processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                )
                .await;

                return Ok(MetadataResult {
                    status: ExtractionStatus::Cached,
                    metadata: Some(cached),
                    error: None,
                    processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                    file_path: input.file_path,
                });
            }
        }

        // Extract metadata from PDF
        let metadata = match self
            .extract_from_pdf(&file_path, input.extract_references)
            .await
        {
            Ok(meta) => {
                // Cache the result
                if input.use_cache {
                    self.cache_metadata(&file_path, &meta).await?;
                }

                meta
            }
            Err(e) => {
                error!("Failed to extract metadata: {}", e);
                let processing_time = start_time.elapsed().unwrap_or_default();

                self.update_stats(
                    false,
                    false,
                    processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                )
                .await;

                return Ok(MetadataResult {
                    status: ExtractionStatus::Failed,
                    metadata: None,
                    error: Some(e.to_string()),
                    processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                    file_path: input.file_path,
                });
            }
        };

        let processing_time = start_time.elapsed().unwrap_or_default();
        self.update_stats(
            true,
            false,
            processing_time.as_millis().try_into().unwrap_or(u64::MAX),
        )
        .await;

        info!(
            "Metadata extraction completed in {}ms",
            processing_time.as_millis()
        );

        Ok(MetadataResult {
            status: if metadata.confidence_score > 0.7 {
                ExtractionStatus::Success
            } else {
                ExtractionStatus::Partial
            },
            metadata: Some(metadata),
            error: None,
            processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
            file_path: input.file_path,
        })
    }

    /// Extract metadata from PDF file
    async fn extract_from_pdf(
        &self,
        file_path: &Path,
        extract_refs: bool,
    ) -> Result<ExtractedMetadata> {
        debug!("Loading PDF document: {:?}", file_path);

        // Validate PDF file before attempting to parse
        Self::validate_pdf_file(file_path).await?;

        // Load PDF document with better error handling
        let doc = tokio::task::spawn_blocking({
            let path = file_path.to_path_buf();
            move || {
                match Document::load(&path) {
                    Ok(doc) => Ok(doc),
                    Err(e) => {
                        // Try to provide more specific error information
                        if let Ok(metadata) = std::fs::metadata(&path) {
                            if metadata.len() == 0 {
                                return Err("PDF file is empty (0 bytes)".to_string());
                            } else if metadata.len() < 1024 {
                                return Err(format!(
                                    "PDF file is too small ({} bytes), likely corrupted",
                                    metadata.len()
                                ));
                            }
                        }
                        Err(format!("PDF parsing failed: {e}"))
                    }
                }
            }
        })
        .await
        .map_err(|e| crate::Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
        .map_err(|e| crate::Error::Parse {
            context: "PDF loading".to_string(),
            message: e,
        })?;

        // Extract text from PDF
        let text = Self::extract_text_from_pdf(&doc)?;

        // Parse metadata from text
        let mut metadata = self.parse_metadata_from_text(&text, extract_refs);

        // Try to extract metadata from PDF info dictionary
        Self::extract_pdf_info(&doc, &mut metadata);

        // Calculate confidence score
        metadata.confidence_score = Self::calculate_confidence(&metadata);
        metadata.metadata_source = "pdf".to_string();
        metadata.extracted_at = SystemTime::now();

        Ok(metadata)
    }

    /// Extract text content from PDF
    fn extract_text_from_pdf(doc: &Document) -> Result<String> {
        let mut all_text = String::new();

        // Iterate through pages
        for page_id in doc.get_pages().values() {
            if let Ok(content) = doc.get_page_content(*page_id) {
                // Convert content to string (simplified - real implementation would need proper PDF text extraction)
                let text = String::from_utf8_lossy(&content);
                all_text.push_str(&text);
                all_text.push('\n');
            }
        }

        if all_text.is_empty() {
            return Err(crate::Error::Parse {
                context: "PDF text extraction".to_string(),
                message: "No text content found in PDF".to_string(),
            });
        }

        Ok(all_text)
    }

    /// Parse metadata from extracted text
    fn parse_metadata_from_text(&self, text: &str, extract_refs: bool) -> ExtractedMetadata {
        let mut metadata = ExtractedMetadata {
            title: None,
            authors: Vec::new(),
            publication_date: None,
            journal: None,
            abstract_text: None,
            doi: None,
            keywords: Vec::new(),
            references: Vec::new(),
            volume: None,
            issue: None,
            pages: None,
            confidence_score: 0.0,
            metadata_source: String::new(),
            extracted_at: SystemTime::now(),
        };

        // Extract DOI
        if let Some(captures) = self.extraction_patterns.doi_pattern.captures(text) {
            metadata.doi = captures.get(1).map(|m| m.as_str().to_string());
            debug!("Found DOI: {:?}", metadata.doi);
        }

        // Extract title (usually in the first few lines with larger font)
        let lines: Vec<&str> = text.lines().take(20).collect();
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.len() > 10
                && trimmed.len() < 300
                && !self.extraction_patterns.author_pattern.is_match(trimmed)
            {
                metadata.title = Some(trimmed.to_string());
                debug!("Found title: {:?}", metadata.title);
                break;
            }
        }

        // Extract authors
        for captures in self
            .extraction_patterns
            .author_pattern
            .captures_iter(text)
            .take(20)
        {
            if let Some(name) = captures.get(1) {
                let author_name = name.as_str().to_string();
                if !author_name.contains("University") && !author_name.contains("Department") {
                    metadata.authors.push(Author {
                        name: author_name.clone(),
                        first_name: None,
                        last_name: None,
                        affiliation: None,
                        email: None,
                        orcid: None,
                    });
                }
            }
        }
        debug!("Found {} authors", metadata.authors.len());

        // Extract abstract
        if let Some(captures) = self.extraction_patterns.abstract_pattern.captures(text) {
            if let Some(abstract_match) = captures.get(1) {
                let abstract_text = abstract_match.as_str().trim().to_string();
                if abstract_text.len() > 50 && abstract_text.len() < 5000 {
                    metadata.abstract_text = Some(abstract_text);
                    debug!("Found abstract");
                }
            }
        }

        // Extract publication date
        if let Some(captures) = self.extraction_patterns.date_pattern.captures(text) {
            metadata.publication_date = captures.get(0).map(|m| m.as_str().to_string());
            debug!("Found publication date: {:?}", metadata.publication_date);
        }

        // Extract journal
        if let Some(captures) = self.extraction_patterns.journal_pattern.captures(text) {
            metadata.journal = captures.get(1).map(|m| m.as_str().trim().to_string());
            debug!("Found journal: {:?}", metadata.journal);
        }

        // Extract volume and issue
        if let Some(captures) = self.extraction_patterns.volume_issue_pattern.captures(text) {
            metadata.volume = captures.get(1).map(|m| m.as_str().to_string());
            metadata.issue = captures.get(2).map(|m| m.as_str().to_string());
            debug!(
                "Found volume: {:?}, issue: {:?}",
                metadata.volume, metadata.issue
            );
        }

        // Extract references if requested
        if extract_refs {
            metadata.references = self.extract_references(text);
            debug!("Found {} references", metadata.references.len());
        }

        metadata
    }

    /// Extract metadata from PDF info dictionary
    fn extract_pdf_info(doc: &Document, metadata: &mut ExtractedMetadata) {
        if let Ok(Object::Dictionary(info)) = doc.trailer.get(b"Info") {
            // Title
            if metadata.title.is_none() {
                if let Ok(Object::String(title, _)) = info.get(b"Title") {
                    let title_str = String::from_utf8_lossy(title.as_ref()).to_string();
                    if !title_str.is_empty() {
                        metadata.title = Some(title_str);
                    }
                }
            }

            // Author
            if metadata.authors.is_empty() {
                if let Ok(Object::String(author, _)) = info.get(b"Author") {
                    let author_str = String::from_utf8_lossy(author.as_ref()).to_string();
                    if !author_str.is_empty() {
                        metadata.authors.push(Author {
                            name: author_str,
                            first_name: None,
                            last_name: None,
                            affiliation: None,
                            email: None,
                            orcid: None,
                        });
                    }
                }
            }

            // Keywords
            if let Ok(Object::String(keywords, _)) = info.get(b"Keywords") {
                let keywords_str = String::from_utf8_lossy(keywords.as_ref());
                metadata.keywords = keywords_str
                    .split(',')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .collect();
            }
        }
    }

    /// Extract references from text
    fn extract_references(&self, text: &str) -> Vec<Reference> {
        let mut references = Vec::new();
        let mut in_references = false;
        let mut ref_counter = 1;
        let year_regex = Regex::new(r"\b(19|20)\d{2}\b").unwrap();

        for line in text.lines() {
            let line_lower = line.to_lowercase();

            // Check if we're entering references section
            if line_lower.contains("references") || line_lower.contains("bibliography") {
                in_references = true;
                continue;
            }

            if in_references {
                // Stop at next major section
                if line_lower.starts_with("appendix") || line_lower.starts_with("acknowledgment") {
                    break;
                }

                // Extract reference if line matches pattern
                if self.extraction_patterns.reference_pattern.is_match(line) {
                    let mut reference = Reference {
                        id: ref_counter.to_string(),
                        raw_text: line.trim().to_string(),
                        title: None,
                        authors: None,
                        year: None,
                        journal: None,
                        doi: None,
                    };

                    // Try to extract DOI from reference
                    if let Some(captures) = self.extraction_patterns.doi_pattern.captures(line) {
                        reference.doi = captures.get(1).map(|m| m.as_str().to_string());
                    }

                    // Extract year
                    if let Some(captures) = year_regex.captures(line) {
                        reference.year = captures.get(0).map(|m| m.as_str().to_string());
                    }

                    references.push(reference);
                    ref_counter += 1;

                    if references.len() >= 100 {
                        break; // Limit references to prevent excessive processing
                    }
                }
            }
        }

        references
    }

    /// Calculate confidence score for extracted metadata
    fn calculate_confidence(metadata: &ExtractedMetadata) -> f64 {
        let mut score = 0.0;
        let mut weight = 0.0;

        // Title (weight: 0.3)
        if metadata.title.is_some() {
            score += 0.3;
        }
        weight += 0.3;

        // Authors (weight: 0.2)
        if !metadata.authors.is_empty() {
            score += 0.2;
        }
        weight += 0.2;

        // Abstract (weight: 0.2)
        if metadata.abstract_text.is_some() {
            score += 0.2;
        }
        weight += 0.2;

        // DOI (weight: 0.15)
        if metadata.doi.is_some() {
            score += 0.15;
        }
        weight += 0.15;

        // Publication info (weight: 0.15)
        if metadata.journal.is_some() || metadata.publication_date.is_some() {
            score += 0.15;
        }
        weight += 0.15;

        if weight > 0.0 {
            score / weight
        } else {
            0.0
        }
    }

    /// Get cached metadata if available
    async fn get_cached_metadata(&self, file_path: &Path) -> Result<Option<ExtractedMetadata>> {
        if let Some(db) = &self.cache_db {
            let file_hash = self.calculate_file_hash(file_path).await?;
            let cache_key = format!("metadata:{file_hash}");

            match db.get(cache_key.as_bytes()) {
                Ok(Some(data)) => {
                    match bincode::deserialize::<CacheEntry>(&data) {
                        Ok(entry) => {
                            // Check if cache is still valid
                            if let Ok(elapsed) = SystemTime::now().duration_since(entry.cached_at) {
                                if elapsed < self.cache_ttl {
                                    return Ok(Some(entry.metadata));
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to deserialize cached metadata: {}", e);
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("Failed to get cached metadata: {}", e);
                }
            }
        }

        Ok(None)
    }

    /// Cache extracted metadata
    async fn cache_metadata(&self, file_path: &Path, metadata: &ExtractedMetadata) -> Result<()> {
        if let Some(db) = &self.cache_db {
            let file_hash = self.calculate_file_hash(file_path).await?;
            let cache_key = format!("metadata:{file_hash}");

            let entry = CacheEntry {
                metadata: metadata.clone(),
                file_hash,
                cached_at: SystemTime::now(),
            };

            match bincode::serialize(&entry) {
                Ok(data) => {
                    if let Err(e) = db.insert(cache_key.as_bytes(), data) {
                        warn!("Failed to cache metadata: {}", e);
                    }
                }
                Err(e) => {
                    warn!("Failed to serialize metadata for cache: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Calculate file hash for cache key
    async fn calculate_file_hash(&self, file_path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};
        use tokio::fs::File;
        use tokio::io::AsyncReadExt;

        let mut file = File::open(file_path).await.map_err(crate::Error::Io)?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer).await.map_err(crate::Error::Io)?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Extract metadata from a single file (internal method for batch processing)
    async fn extract_single_file_internal(&self, input: MetadataInput) -> Result<MetadataResult> {
        let start_time = SystemTime::now();
        debug!("Starting metadata extraction for: {}", input.file_path);

        let file_path = PathBuf::from(&input.file_path);

        // Validate file exists
        if !file_path.exists() {
            return Ok(MetadataResult {
                status: ExtractionStatus::Failed,
                metadata: None,
                error: Some(format!(
                    "File not found: {file_path}",
                    file_path = input.file_path
                )),
                processing_time_ms: 0,
                file_path: input.file_path,
            });
        }

        // Check cache if enabled
        if input.use_cache {
            if let Some(cached) = self.get_cached_metadata(&file_path).await? {
                debug!("Returning cached metadata for: {}", input.file_path);
                let processing_time = start_time.elapsed().unwrap_or_default();

                return Ok(MetadataResult {
                    status: ExtractionStatus::Cached,
                    metadata: Some(cached),
                    error: None,
                    processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                    file_path: input.file_path,
                });
            }
        }

        // Extract metadata from PDF
        let metadata = match self
            .extract_from_pdf(&file_path, input.extract_references)
            .await
        {
            Ok(meta) => {
                // Cache the result
                if input.use_cache {
                    self.cache_metadata(&file_path, &meta).await?;
                }

                meta
            }
            Err(e) => {
                error!("Failed to extract metadata: {}", e);
                let processing_time = start_time.elapsed().unwrap_or_default();

                return Ok(MetadataResult {
                    status: ExtractionStatus::Failed,
                    metadata: None,
                    error: Some(e.to_string()),
                    processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
                    file_path: input.file_path,
                });
            }
        };

        let processing_time = start_time.elapsed().unwrap_or_default();

        // Return successful result
        Ok(MetadataResult {
            status: ExtractionStatus::Success,
            metadata: Some(metadata),
            error: None,
            processing_time_ms: processing_time.as_millis().try_into().unwrap_or(u64::MAX),
            file_path: input.file_path,
        })
    }

    /// Extract metadata from multiple files
    async fn extract_batch(&self, files: Vec<String>, use_cache: bool) -> Result<MetadataResult> {
        let start_time = SystemTime::now();
        let num_files = files.len();

        info!(
            "Starting batch metadata extraction for {} files with parallel processing",
            num_files
        );

        // Use semaphore to limit concurrent extractions (tripled to 12 for CPU-bound work)
        let semaphore = Arc::new(tokio::sync::Semaphore::new(12));

        // Process files in parallel using futures::stream
        let results: Vec<MetadataResult> = futures::stream::iter(files.into_iter())
            .map(|file_path| {
                let semaphore = semaphore.clone();
                let extractor_config = self.config.clone();
                let cache_db = self.cache_db.clone();
                let patterns = self.extraction_patterns.clone();

                async move {
                    let _permit = semaphore.acquire().await.map_err(|e| {
                        crate::Error::Service(format!("Failed to acquire metadata semaphore: {e}"))
                    })?;

                    debug!("Starting metadata extraction for: {}", file_path);

                    // Create a temporary extractor for this task to avoid Send issues
                    let temp_extractor = Self {
                        config: extractor_config,
                        cache_db,
                        cache_ttl: self.cache_ttl,
                        extraction_patterns: patterns,
                        stats: Arc::new(RwLock::new(ExtractionStats::default())), // Temp stats
                    };

                    let input = MetadataInput {
                        file_path: file_path.clone(),
                        use_cache,
                        extract_references: false,
                        batch_files: None,
                    };

                    // Process single file directly without recursion
                    let result = temp_extractor.extract_single_file_internal(input).await;
                    debug!(
                        "Completed metadata extraction for: {} - result: {:?}",
                        file_path,
                        result.as_ref().map(|r| &r.status)
                    );

                    result
                }
            })
            .buffer_unordered(12) // Process up to 12 files concurrently
            .collect::<Vec<Result<MetadataResult>>>()
            .await
            .into_iter()
            .map(|result| match result {
                Ok(metadata_result) => metadata_result,
                Err(e) => MetadataResult {
                    status: ExtractionStatus::Failed,
                    metadata: None,
                    error: Some(e.to_string()),
                    processing_time_ms: 0,
                    file_path: "unknown".to_string(),
                },
            })
            .collect();

        // Calculate statistics
        let mut success_count = 0;

        for result in &results {
            if matches!(
                result.status,
                ExtractionStatus::Success | ExtractionStatus::Cached
            ) {
                success_count += 1;
            }
        }
        let failure_count: usize = results.len() - success_count;

        let total_time = start_time.elapsed().unwrap_or_default();

        info!(
            "Batch metadata extraction completed: {}/{} successful in {:.2}s",
            success_count,
            num_files,
            total_time.as_secs_f64()
        );

        // Return as JSON-serialized batch result in a single MetadataResult
        let batch_result = BatchMetadataResult {
            results,
            total_time_ms: total_time.as_millis().try_into().unwrap_or(u64::MAX),
            success_count,
            failure_count,
        };

        // Serialize batch result to JSON
        let num_files = batch_result.results.len();
        let batch_json = serde_json::to_string(&batch_result)
            .map_err(|e| crate::Error::Service(format!("Serialization error: {e}")))?;

        Ok(MetadataResult {
            status: if failure_count == 0 {
                ExtractionStatus::Success
            } else if success_count > 0 {
                ExtractionStatus::Partial
            } else {
                ExtractionStatus::Failed
            },
            metadata: None,
            error: Some(batch_json), // Store batch result in error field as JSON
            processing_time_ms: total_time.as_millis().try_into().unwrap_or(u64::MAX),
            file_path: format!("batch:{num_files} files"),
        })
    }

    /// Update extraction statistics
    async fn update_stats(&self, success: bool, cache_hit: bool, time_ms: u64) {
        let mut stats = self.stats.write().await;
        stats.total_extractions += 1;
        if success {
            stats.successful_extractions += 1;
        } else {
            stats.failed_extractions += 1;
        }
        if cache_hit {
            stats.cache_hits += 1;
        }
        stats.total_processing_time_ms += time_ms;
    }

    /// Get extraction statistics
    pub async fn get_stats(&self) -> HashMap<String, u64> {
        let stats = self.stats.read().await;
        let mut map = HashMap::new();
        map.insert("total_extractions".to_string(), stats.total_extractions);
        map.insert(
            "successful_extractions".to_string(),
            stats.successful_extractions,
        );
        map.insert("failed_extractions".to_string(), stats.failed_extractions);
        map.insert("cache_hits".to_string(), stats.cache_hits);
        map.insert(
            "total_processing_time_ms".to_string(),
            stats.total_processing_time_ms,
        );
        if stats.total_extractions > 0 {
            map.insert(
                "avg_processing_time_ms".to_string(),
                stats.total_processing_time_ms / stats.total_extractions,
            );
        }
        map
    }

    /// Clear the metadata cache
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(db) = &self.cache_db {
            db.clear().map_err(|e| crate::Error::Cache {
                operation: "clear".to_string(),
                reason: format!("Failed to clear cache: {e}"),
            })?;
            info!("Metadata cache cleared");
        }
        Ok(())
    }
}

impl std::fmt::Debug for MetadataExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetadataExtractor")
            .field("config", &"Config")
            .field("cache_enabled", &self.cache_db.is_some())
            .field("cache_ttl", &self.cache_ttl)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_extractor() -> MetadataExtractor {
        let config = Arc::new(Config::default());
        MetadataExtractor::new(config).unwrap()
    }

    #[test]
    fn test_metadata_input_defaults() {
        let input = MetadataInput {
            file_path: "test.pdf".to_string(),
            use_cache: true,
            extract_references: false,
            batch_files: None,
        };

        assert!(input.use_cache);
        assert!(!input.extract_references);
    }

    #[test]
    fn test_confidence_calculation() {
        let _extractor = create_test_extractor();

        // Full metadata
        let full_metadata = ExtractedMetadata {
            title: Some("Test Title".to_string()),
            authors: vec![Author {
                name: "John Doe".to_string(),
                first_name: None,
                last_name: None,
                affiliation: None,
                email: None,
                orcid: None,
            }],
            publication_date: Some("2024".to_string()),
            journal: Some("Test Journal".to_string()),
            abstract_text: Some("Test abstract".to_string()),
            doi: Some("10.1234/test".to_string()),
            keywords: vec![],
            references: vec![],
            volume: None,
            issue: None,
            pages: None,
            confidence_score: 0.0,
            metadata_source: "pdf".to_string(),
            extracted_at: SystemTime::now(),
        };

        let score = MetadataExtractor::calculate_confidence(&full_metadata);
        assert!(score > 0.9);

        // Minimal metadata
        let minimal_metadata = ExtractedMetadata {
            title: Some("Test Title".to_string()),
            authors: vec![],
            publication_date: None,
            journal: None,
            abstract_text: None,
            doi: None,
            keywords: vec![],
            references: vec![],
            volume: None,
            issue: None,
            pages: None,
            confidence_score: 0.0,
            metadata_source: "pdf".to_string(),
            extracted_at: SystemTime::now(),
        };

        let score = MetadataExtractor::calculate_confidence(&minimal_metadata);
        assert!(score < 0.5);
    }

    #[test]
    fn test_extraction_patterns() {
        let patterns = ExtractionPatterns::default();

        // Test DOI pattern
        let text = "DOI: 10.1038/nature12373";
        assert!(patterns.doi_pattern.is_match(text));

        // Test email pattern
        let text = "Contact: john.doe@example.com";
        assert!(patterns.email_pattern.is_match(text));

        // Test date pattern
        let text = "Published: March 2024";
        assert!(patterns.date_pattern.is_match(text));
    }

    #[tokio::test]
    async fn test_file_hash_calculation() {
        let extractor = create_test_extractor();
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        tokio::fs::write(&file_path, b"test content").await.unwrap();

        let hash1 = extractor.calculate_file_hash(&file_path).await.unwrap();
        let hash2 = extractor.calculate_file_hash(&file_path).await.unwrap();

        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let extractor = create_test_extractor();

        extractor.update_stats(true, false, 100).await;
        extractor.update_stats(false, false, 50).await;
        extractor.update_stats(true, true, 10).await;

        let stats = extractor.get_stats().await;

        assert_eq!(stats.get("total_extractions"), Some(&3));
        assert_eq!(stats.get("successful_extractions"), Some(&2));
        assert_eq!(stats.get("failed_extractions"), Some(&1));
        assert_eq!(stats.get("cache_hits"), Some(&1));
        assert_eq!(stats.get("total_processing_time_ms"), Some(&160));
    }

    #[test]
    fn test_reference_extraction() {
        let extractor = create_test_extractor();
        let text = r"
References

[1] Smith, J. (2023). Test Paper. Journal of Testing.
[2] Doe, J. et al. (2022). Another Paper. DOI: 10.1234/test
[3] Johnson, M. (2021). Third Paper. Conference Proceedings.
        ";

        let refs = extractor.extract_references(text);
        assert_eq!(refs.len(), 3);
        assert!(refs[1].doi.is_some());
        assert_eq!(refs[1].year, Some("2022".to_string()));
    }

    #[test]
    fn test_extraction_status_serialization() {
        let status = ExtractionStatus::Success;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"success\"");

        let status = ExtractionStatus::Failed;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"failed\"");
    }
}
