use crate::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Output mode for verification results
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutputMode {
    /// Full output with all details (default)
    #[default]
    Full,
    /// Notes mode: Returns verification notes and discrepancies only
    Notes,
    /// Corrected mode: Returns only the corrected/authoritative metadata
    Corrected,
}

/// Input for metadata verification
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerifyMetadataInput {
    /// DOI to verify (e.g., "10.1038/nature12373")
    #[schemars(description = "Digital Object Identifier for CrossRef lookup")]
    pub doi: Option<String>,

    /// PubMed ID to verify
    #[schemars(description = "PubMed ID for PubMed/NCBI lookup")]
    pub pmid: Option<String>,

    /// Paper title for fuzzy matching
    #[schemars(description = "Paper title for fuzzy matching across sources")]
    pub title: Option<String>,

    /// Author names for validation
    #[schemars(description = "Author names to validate against external sources")]
    pub authors: Option<Vec<String>>,

    /// Publication year for validation
    #[schemars(description = "Publication year for cross-validation")]
    pub year: Option<i32>,

    /// Semantic Scholar Paper ID
    #[schemars(description = "Semantic Scholar paper ID for lookup")]
    pub s2_paper_id: Option<String>,

    /// OpenAlex Work ID
    #[schemars(description = "OpenAlex work ID for lookup")]
    pub openalex_id: Option<String>,

    /// Sources to query (default: all available based on provided IDs)
    #[schemars(description = "Specific sources to query: crossref, pubmed, semantic_scholar, openalex")]
    pub sources: Option<Vec<String>>,

    /// Output mode: full, notes, or corrected (default: full)
    #[serde(default)]
    #[schemars(description = "Output mode: full (all details), notes (discrepancies only), corrected (authoritative metadata only)")]
    pub output_mode: VerificationOutputMode,
}

/// Verified metadata from external source
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerifiedMetadata {
    /// Paper title
    pub title: Option<String>,
    /// Author names
    pub authors: Vec<String>,
    /// Publication year
    pub year: Option<i32>,
    /// Journal name
    pub journal: Option<String>,
    /// Volume
    pub volume: Option<String>,
    /// Issue
    pub issue: Option<String>,
    /// Pages
    pub pages: Option<String>,
    /// DOI
    pub doi: Option<String>,
    /// PubMed ID
    pub pmid: Option<String>,
    /// Abstract
    pub abstract_text: Option<String>,
}

/// Result from a single verification source
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceVerificationResult {
    /// Source name (crossref, pubmed, semantic_scholar, openalex)
    pub source: String,
    /// Whether the lookup was successful
    pub success: bool,
    /// Verified metadata (if found)
    pub metadata: Option<VerifiedMetadata>,
    /// Confidence score for this source (0.0-1.0)
    pub confidence: f64,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
}

/// Aggregated verification result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerificationResult {
    /// Overall verification status
    pub status: VerificationStatus,
    /// Results from individual sources
    pub source_results: Vec<SourceVerificationResult>,
    /// Merged/best metadata from all sources
    pub merged_metadata: Option<VerifiedMetadata>,
    /// Overall confidence score (0.0-1.0)
    pub overall_confidence: f64,
    /// Discrepancies found between sources
    pub discrepancies: Vec<MetadataDiscrepancy>,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
}

/// Notes output mode result - focuses on discrepancies and verification notes
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerificationNotesResult {
    /// Overall verification status
    pub status: VerificationStatus,
    /// Discrepancies found between sources
    pub discrepancies: Vec<MetadataDiscrepancy>,
    /// Verification notes (human-readable summary)
    pub notes: Vec<VerificationNote>,
    /// Sources that were queried
    pub sources_queried: Vec<String>,
    /// Sources that returned data
    pub sources_responded: Vec<String>,
    /// Overall confidence score (0.0-1.0)
    pub overall_confidence: f64,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
}

/// A single verification note
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VerificationNote {
    /// Note severity level
    pub level: NoteLevel,
    /// Field this note applies to (if any)
    pub field: Option<String>,
    /// Human-readable message
    pub message: String,
}

/// Note severity level
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NoteLevel {
    /// Informational note
    Info,
    /// Warning - possible issue detected
    Warning,
    /// Error - verification failed
    Error,
}

/// Corrected output mode result - returns only the authoritative metadata
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CorrectedMetadataResult {
    /// Corrected/authoritative metadata
    pub metadata: VerifiedMetadata,
    /// Confidence score for the corrected metadata (0.0-1.0)
    pub confidence: f64,
    /// Authority source (primary source for the metadata)
    pub authority_source: String,
    /// Number of sources that agreed
    pub sources_agreed: usize,
    /// Fields that were corrected from input
    pub corrected_fields: Vec<String>,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
}

/// Combined verification output that can hold any output mode result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "output_type")]
pub enum VerificationOutput {
    /// Full verification result
    #[serde(rename = "full")]
    Full(VerificationResult),
    /// Notes-only result
    #[serde(rename = "notes")]
    Notes(VerificationNotesResult),
    /// Corrected metadata result
    #[serde(rename = "corrected")]
    Corrected(CorrectedMetadataResult),
}

/// Verification status
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    /// All sources agree
    Verified,
    /// Some sources found discrepancies
    PartialMatch,
    /// Could not verify (no sources returned data)
    Unverified,
    /// Verification failed (errors)
    Failed,
}

/// Discrepancy between sources
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataDiscrepancy {
    /// Field with discrepancy
    pub field: String,
    /// Values from different sources
    pub values: Vec<SourceValue>,
}

/// Value from a specific source
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceValue {
    /// Source name
    pub source: String,
    /// Value from this source
    pub value: String,
}

/// Input for batch metadata verification
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BatchVerifyMetadataInput {
    /// List of records to verify (1-N records)
    #[schemars(description = "List of metadata records to verify in batch")]
    pub records: Vec<VerifyMetadataInput>,

    /// Maximum concurrent verifications (default: 5)
    #[schemars(description = "Maximum number of concurrent verification requests")]
    pub max_concurrency: Option<usize>,
}

/// Result for a single record in batch verification
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BatchRecordResult {
    /// Index of the record in the input batch (0-based)
    pub index: usize,
    /// Input identifier (DOI, PMID, or title) for reference
    pub identifier: String,
    /// Verification result for this record
    pub result: VerificationResult,
}

/// Aggregated batch verification result
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BatchVerificationResult {
    /// Total number of records processed
    pub total_records: usize,
    /// Number of successfully verified records
    pub verified_count: usize,
    /// Number of partially matched records
    pub partial_match_count: usize,
    /// Number of unverified records
    pub unverified_count: usize,
    /// Number of failed records
    pub failed_count: usize,
    /// Individual results for each record
    pub results: Vec<BatchRecordResult>,
    /// Total processing time in milliseconds
    pub total_time_ms: u64,
}

/// Metadata verification tool
#[derive(Clone)]
pub struct VerifyMetadataTool {
    http_client: reqwest::Client,
    crossref_timeout: Duration,
    pubmed_timeout: Duration,
    semantic_scholar_timeout: Duration,
    openalex_timeout: Duration,
}

impl Default for VerifyMetadataTool {
    fn default() -> Self {
        Self::new()
    }
}

impl VerifyMetadataTool {
    /// Create a new verification tool
    #[must_use]
    pub fn new() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("research_hub_mcp/1.0.0 (https://github.com/research-hub-mcp)")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http_client,
            crossref_timeout: Duration::from_secs(10),
            pubmed_timeout: Duration::from_secs(10),
            semantic_scholar_timeout: Duration::from_secs(10),
            openalex_timeout: Duration::from_secs(10),
        }
    }

    /// Verify metadata against external sources
    pub async fn verify(&self, input: VerifyMetadataInput) -> Result<VerificationResult> {
        let start_time = std::time::Instant::now();
        info!("Starting metadata verification");

        let mut source_results = Vec::new();

        // Determine which sources to query
        let sources = input.sources.clone().unwrap_or_else(|| {
            let mut auto_sources = Vec::new();
            if input.doi.is_some() {
                auto_sources.push("crossref".to_string());
            }
            if input.pmid.is_some() {
                auto_sources.push("pubmed".to_string());
            }
            if input.s2_paper_id.is_some() || input.doi.is_some() || input.title.is_some() {
                auto_sources.push("semantic_scholar".to_string());
            }
            if input.openalex_id.is_some() || input.doi.is_some() {
                auto_sources.push("openalex".to_string());
            }
            if auto_sources.is_empty() && input.title.is_some() {
                // Fallback to title search on all sources
                auto_sources = vec![
                    "crossref".to_string(),
                    "semantic_scholar".to_string(),
                    "openalex".to_string(),
                ];
            }
            auto_sources
        });

        // Query each source
        for source in &sources {
            let result = match source.as_str() {
                "crossref" => self.verify_crossref(&input).await,
                "pubmed" => self.verify_pubmed(&input).await,
                "semantic_scholar" => self.verify_semantic_scholar(&input).await,
                "openalex" => self.verify_openalex(&input).await,
                _ => {
                    warn!("Unknown source: {}", source);
                    continue;
                }
            };
            source_results.push(result);
        }

        // Merge results and detect discrepancies
        let (merged_metadata, discrepancies) = self.merge_results(&source_results);

        // Calculate overall confidence
        let overall_confidence = self.calculate_overall_confidence(&source_results);

        // Determine status
        let status = if source_results.iter().all(|r| !r.success) {
            VerificationStatus::Failed
        } else if source_results.iter().any(|r| r.success) {
            if discrepancies.is_empty() {
                VerificationStatus::Verified
            } else {
                VerificationStatus::PartialMatch
            }
        } else {
            VerificationStatus::Unverified
        };

        let total_time = start_time.elapsed();

        info!(
            "Metadata verification completed: {:?} in {}ms",
            status,
            total_time.as_millis()
        );

        Ok(VerificationResult {
            status,
            source_results,
            merged_metadata,
            overall_confidence,
            discrepancies,
            total_time_ms: total_time.as_millis() as u64,
        })
    }

    /// Verify metadata and return result based on output mode
    pub async fn verify_with_output(
        &self,
        input: VerifyMetadataInput,
    ) -> Result<VerificationOutput> {
        let output_mode = input.output_mode.clone();
        let full_result = self.verify(input.clone()).await?;

        match output_mode {
            VerificationOutputMode::Full => Ok(VerificationOutput::Full(full_result)),
            VerificationOutputMode::Notes => {
                let notes_result = self.convert_to_notes(&full_result, &input);
                Ok(VerificationOutput::Notes(notes_result))
            }
            VerificationOutputMode::Corrected => {
                let corrected_result = self.convert_to_corrected(&full_result, &input);
                match corrected_result {
                    Some(result) => Ok(VerificationOutput::Corrected(result)),
                    None => {
                        // If no corrected metadata available, return as notes
                        let notes_result = self.convert_to_notes(&full_result, &input);
                        Ok(VerificationOutput::Notes(notes_result))
                    }
                }
            }
        }
    }

    /// Convert full result to notes output format
    fn convert_to_notes(
        &self,
        result: &VerificationResult,
        input: &VerifyMetadataInput,
    ) -> VerificationNotesResult {
        let mut notes = Vec::new();

        // Add status note
        let status_note = match result.status {
            VerificationStatus::Verified => VerificationNote {
                level: NoteLevel::Info,
                field: None,
                message: "All sources agree on the metadata.".to_string(),
            },
            VerificationStatus::PartialMatch => VerificationNote {
                level: NoteLevel::Warning,
                field: None,
                message: format!(
                    "Sources partially agree. {} discrepancies found.",
                    result.discrepancies.len()
                ),
            },
            VerificationStatus::Unverified => VerificationNote {
                level: NoteLevel::Warning,
                field: None,
                message: "Could not verify metadata. No sources returned data.".to_string(),
            },
            VerificationStatus::Failed => VerificationNote {
                level: NoteLevel::Error,
                field: None,
                message: "Verification failed. All source queries failed.".to_string(),
            },
        };
        notes.push(status_note);

        // Add notes for discrepancies
        for discrepancy in &result.discrepancies {
            let values_str: Vec<String> = discrepancy
                .values
                .iter()
                .map(|v| format!("{}: '{}'", v.source, v.value))
                .collect();

            notes.push(VerificationNote {
                level: NoteLevel::Warning,
                field: Some(discrepancy.field.clone()),
                message: format!(
                    "Discrepancy in '{}': {}",
                    discrepancy.field,
                    values_str.join(", ")
                ),
            });
        }

        // Add notes for input validation
        if input.doi.is_none() && input.pmid.is_none() && input.title.is_none() {
            notes.push(VerificationNote {
                level: NoteLevel::Error,
                field: None,
                message: "No identifier provided (DOI, PMID, or title required).".to_string(),
            });
        }

        // Add confidence note
        if result.overall_confidence < 0.5 {
            notes.push(VerificationNote {
                level: NoteLevel::Warning,
                field: None,
                message: format!(
                    "Low confidence score: {:.1}%. Consider verifying with additional sources.",
                    result.overall_confidence * 100.0
                ),
            });
        }

        let sources_queried: Vec<String> = result
            .source_results
            .iter()
            .map(|r| r.source.clone())
            .collect();

        let sources_responded: Vec<String> = result
            .source_results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.source.clone())
            .collect();

        VerificationNotesResult {
            status: result.status.clone(),
            discrepancies: result.discrepancies.clone(),
            notes,
            sources_queried,
            sources_responded,
            overall_confidence: result.overall_confidence,
            total_time_ms: result.total_time_ms,
        }
    }

    /// Convert full result to corrected output format
    fn convert_to_corrected(
        &self,
        result: &VerificationResult,
        input: &VerifyMetadataInput,
    ) -> Option<CorrectedMetadataResult> {
        let metadata = result.merged_metadata.clone()?;

        // Determine authority source (highest confidence source that succeeded)
        let authority_source = result
            .source_results
            .iter()
            .filter(|r| r.success)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| r.source.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Count sources that agreed
        let sources_agreed = result.source_results.iter().filter(|r| r.success).count();

        // Determine which fields were corrected from input
        let mut corrected_fields = Vec::new();

        if let Some(input_title) = &input.title {
            if let Some(verified_title) = &metadata.title {
                if input_title != verified_title {
                    corrected_fields.push("title".to_string());
                }
            }
        }

        if let Some(input_year) = input.year {
            if let Some(verified_year) = metadata.year {
                if input_year != verified_year {
                    corrected_fields.push("year".to_string());
                }
            }
        }

        if let Some(input_authors) = &input.authors {
            if !input_authors.is_empty() && !metadata.authors.is_empty() {
                // Simple check: if counts differ, consider corrected
                if input_authors.len() != metadata.authors.len() {
                    corrected_fields.push("authors".to_string());
                }
            }
        }

        Some(CorrectedMetadataResult {
            metadata,
            confidence: result.overall_confidence,
            authority_source,
            sources_agreed,
            corrected_fields,
            total_time_ms: result.total_time_ms,
        })
    }

    /// Verify multiple metadata records in batch
    pub async fn verify_batch(
        &self,
        input: BatchVerifyMetadataInput,
    ) -> Result<BatchVerificationResult> {
        let start_time = std::time::Instant::now();
        let total_records = input.records.len();
        let max_concurrency = input.max_concurrency.unwrap_or(5).min(20); // Cap at 20

        info!(
            "Starting batch metadata verification: {} records, max concurrency: {}",
            total_records, max_concurrency
        );

        if total_records == 0 {
            return Ok(BatchVerificationResult {
                total_records: 0,
                verified_count: 0,
                partial_match_count: 0,
                unverified_count: 0,
                failed_count: 0,
                results: Vec::new(),
                total_time_ms: 0,
            });
        }

        // Process records with semaphore for concurrency control
        let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrency));
        let tool = std::sync::Arc::new(self.clone());

        let mut handles = Vec::with_capacity(total_records);

        for (index, record) in input.records.into_iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await.map_err(|e| {
                crate::Error::Provider(format!("Failed to acquire semaphore: {}", e))
            })?;
            let tool = tool.clone();
            let identifier = Self::get_record_identifier(&record);

            let handle = tokio::spawn(async move {
                let result = tool.verify(record).await;
                drop(permit); // Release semaphore
                (index, identifier, result)
            });

            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::with_capacity(total_records);
        let mut verified_count = 0;
        let mut partial_match_count = 0;
        let mut unverified_count = 0;
        let mut failed_count = 0;

        for handle in handles {
            match handle.await {
                Ok((index, identifier, Ok(verification_result))) => {
                    match verification_result.status {
                        VerificationStatus::Verified => verified_count += 1,
                        VerificationStatus::PartialMatch => partial_match_count += 1,
                        VerificationStatus::Unverified => unverified_count += 1,
                        VerificationStatus::Failed => failed_count += 1,
                    }
                    results.push(BatchRecordResult {
                        index,
                        identifier,
                        result: verification_result,
                    });
                }
                Ok((index, identifier, Err(e))) => {
                    failed_count += 1;
                    warn!("Verification failed for record {}: {}", index, e);
                    results.push(BatchRecordResult {
                        index,
                        identifier,
                        result: VerificationResult {
                            status: VerificationStatus::Failed,
                            source_results: Vec::new(),
                            merged_metadata: None,
                            overall_confidence: 0.0,
                            discrepancies: Vec::new(),
                            total_time_ms: 0,
                        },
                    });
                }
                Err(e) => {
                    failed_count += 1;
                    warn!("Task join error: {}", e);
                    results.push(BatchRecordResult {
                        index: results.len(),
                        identifier: "unknown".to_string(),
                        result: VerificationResult {
                            status: VerificationStatus::Failed,
                            source_results: Vec::new(),
                            merged_metadata: None,
                            overall_confidence: 0.0,
                            discrepancies: Vec::new(),
                            total_time_ms: 0,
                        },
                    });
                }
            }
        }

        // Sort results by index
        results.sort_by_key(|r| r.index);

        let total_time = start_time.elapsed();

        info!(
            "Batch verification completed: {} total, {} verified, {} partial, {} unverified, {} failed in {}ms",
            total_records, verified_count, partial_match_count, unverified_count, failed_count, total_time.as_millis()
        );

        Ok(BatchVerificationResult {
            total_records,
            verified_count,
            partial_match_count,
            unverified_count,
            failed_count,
            results,
            total_time_ms: total_time.as_millis() as u64,
        })
    }

    /// Get a human-readable identifier for a record
    fn get_record_identifier(record: &VerifyMetadataInput) -> String {
        if let Some(doi) = &record.doi {
            format!("DOI:{}", doi)
        } else if let Some(pmid) = &record.pmid {
            format!("PMID:{}", pmid)
        } else if let Some(title) = &record.title {
            if title.len() > 50 {
                format!("Title:{}...", &title[..50])
            } else {
                format!("Title:{}", title)
            }
        } else if let Some(s2_id) = &record.s2_paper_id {
            format!("S2:{}", s2_id)
        } else if let Some(oa_id) = &record.openalex_id {
            format!("OpenAlex:{}", oa_id)
        } else {
            "unknown".to_string()
        }
    }

    /// Verify against CrossRef API
    async fn verify_crossref(&self, input: &VerifyMetadataInput) -> SourceVerificationResult {
        let start_time = std::time::Instant::now();

        // Need DOI or title for CrossRef
        let url = if let Some(doi) = &input.doi {
            format!("https://api.crossref.org/works/{}", doi)
        } else if let Some(title) = &input.title {
            format!(
                "https://api.crossref.org/works?query.title={}&rows=1",
                urlencoding::encode(title)
            )
        } else {
            return SourceVerificationResult {
                source: "crossref".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("No DOI or title provided for CrossRef lookup".to_string()),
                response_time_ms: 0,
            };
        };

        debug!("CrossRef query URL: {}", url);

        let result = tokio::time::timeout(self.crossref_timeout, self.http_client.get(&url).send())
            .await;

        let response_time = start_time.elapsed();

        match result {
            Ok(Ok(response)) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let metadata = self.parse_crossref_response(&json, input.doi.is_some());
                        let confidence = if metadata.is_some() { 0.95 } else { 0.0 };
                        SourceVerificationResult {
                            source: "crossref".to_string(),
                            success: metadata.is_some(),
                            metadata,
                            confidence,
                            error: None,
                            response_time_ms: response_time.as_millis() as u64,
                        }
                    }
                    Err(e) => SourceVerificationResult {
                        source: "crossref".to_string(),
                        success: false,
                        metadata: None,
                        confidence: 0.0,
                        error: Some(format!("Failed to parse CrossRef response: {}", e)),
                        response_time_ms: response_time.as_millis() as u64,
                    },
                }
            }
            Ok(Ok(response)) => SourceVerificationResult {
                source: "crossref".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("CrossRef returned status: {}", response.status())),
                response_time_ms: response_time.as_millis() as u64,
            },
            Ok(Err(e)) => SourceVerificationResult {
                source: "crossref".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("CrossRef request failed: {}", e)),
                response_time_ms: response_time.as_millis() as u64,
            },
            Err(_) => SourceVerificationResult {
                source: "crossref".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("CrossRef request timed out".to_string()),
                response_time_ms: response_time.as_millis() as u64,
            },
        }
    }

    /// Parse CrossRef API response
    fn parse_crossref_response(
        &self,
        json: &serde_json::Value,
        is_doi_lookup: bool,
    ) -> Option<VerifiedMetadata> {
        let message = if is_doi_lookup {
            json.get("message")?
        } else {
            json.get("message")?.get("items")?.get(0)?
        };

        let title = message
            .get("title")
            .and_then(|t| t.as_array())
            .and_then(|arr| arr.first())
            .and_then(|t| t.as_str())
            .map(String::from);

        let authors: Vec<String> = message
            .get("author")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|author| {
                        let given = author.get("given").and_then(|g| g.as_str()).unwrap_or("");
                        let family = author.get("family").and_then(|f| f.as_str()).unwrap_or("");
                        if family.is_empty() && given.is_empty() {
                            None
                        } else {
                            Some(format!("{} {}", given, family).trim().to_string())
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let year = message
            .get("published-print")
            .or_else(|| message.get("published-online"))
            .and_then(|p| p.get("date-parts"))
            .and_then(|dp| dp.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.as_array())
            .and_then(|arr| arr.first())
            .and_then(|y| y.as_i64())
            .map(|y| y as i32);

        let journal = message
            .get("container-title")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|j| j.as_str())
            .map(String::from);

        let doi = message
            .get("DOI")
            .and_then(|d| d.as_str())
            .map(String::from);

        let volume = message
            .get("volume")
            .and_then(|v| v.as_str())
            .map(String::from);

        let issue = message
            .get("issue")
            .and_then(|i| i.as_str())
            .map(String::from);

        let pages = message
            .get("page")
            .and_then(|p| p.as_str())
            .map(String::from);

        Some(VerifiedMetadata {
            title,
            authors,
            year,
            journal,
            volume,
            issue,
            pages,
            doi,
            pmid: None,
            abstract_text: None,
        })
    }

    /// Verify against PubMed API
    async fn verify_pubmed(&self, input: &VerifyMetadataInput) -> SourceVerificationResult {
        let start_time = std::time::Instant::now();

        let pmid = match &input.pmid {
            Some(id) => id.clone(),
            None => {
                return SourceVerificationResult {
                    source: "pubmed".to_string(),
                    success: false,
                    metadata: None,
                    confidence: 0.0,
                    error: Some("No PMID provided for PubMed lookup".to_string()),
                    response_time_ms: 0,
                };
            }
        };

        let url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}&retmode=json",
            pmid
        );

        debug!("PubMed query URL: {}", url);

        let result = tokio::time::timeout(self.pubmed_timeout, self.http_client.get(&url).send())
            .await;

        let response_time = start_time.elapsed();

        match result {
            Ok(Ok(response)) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let metadata = self.parse_pubmed_response(&json, &pmid);
                        let confidence = if metadata.is_some() { 0.95 } else { 0.0 };
                        SourceVerificationResult {
                            source: "pubmed".to_string(),
                            success: metadata.is_some(),
                            metadata,
                            confidence,
                            error: None,
                            response_time_ms: response_time.as_millis() as u64,
                        }
                    }
                    Err(e) => SourceVerificationResult {
                        source: "pubmed".to_string(),
                        success: false,
                        metadata: None,
                        confidence: 0.0,
                        error: Some(format!("Failed to parse PubMed response: {}", e)),
                        response_time_ms: response_time.as_millis() as u64,
                    },
                }
            }
            Ok(Ok(response)) => SourceVerificationResult {
                source: "pubmed".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("PubMed returned status: {}", response.status())),
                response_time_ms: response_time.as_millis() as u64,
            },
            Ok(Err(e)) => SourceVerificationResult {
                source: "pubmed".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("PubMed request failed: {}", e)),
                response_time_ms: response_time.as_millis() as u64,
            },
            Err(_) => SourceVerificationResult {
                source: "pubmed".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("PubMed request timed out".to_string()),
                response_time_ms: response_time.as_millis() as u64,
            },
        }
    }

    /// Parse PubMed API response
    fn parse_pubmed_response(
        &self,
        json: &serde_json::Value,
        pmid: &str,
    ) -> Option<VerifiedMetadata> {
        let result = json.get("result")?.get(pmid)?;

        let title = result.get("title").and_then(|t| t.as_str()).map(String::from);

        let authors: Vec<String> = result
            .get("authors")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|author| {
                        author.get("name").and_then(|n| n.as_str()).map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let year = result
            .get("pubdate")
            .and_then(|d| d.as_str())
            .and_then(|s| s.split_whitespace().last())
            .and_then(|y| y.parse().ok());

        let journal = result
            .get("source")
            .and_then(|s| s.as_str())
            .map(String::from);

        let volume = result
            .get("volume")
            .and_then(|v| v.as_str())
            .map(String::from);

        let issue = result
            .get("issue")
            .and_then(|i| i.as_str())
            .map(String::from);

        let pages = result
            .get("pages")
            .and_then(|p| p.as_str())
            .map(String::from);

        let doi = result
            .get("elocationid")
            .and_then(|e| e.as_str())
            .filter(|s| s.starts_with("doi:"))
            .map(|s| s.trim_start_matches("doi: ").to_string());

        Some(VerifiedMetadata {
            title,
            authors,
            year,
            journal,
            volume,
            issue,
            pages,
            doi,
            pmid: Some(pmid.to_string()),
            abstract_text: None,
        })
    }

    /// Verify against Semantic Scholar API
    async fn verify_semantic_scholar(
        &self,
        input: &VerifyMetadataInput,
    ) -> SourceVerificationResult {
        let start_time = std::time::Instant::now();

        let url = if let Some(s2_id) = &input.s2_paper_id {
            format!(
                "https://api.semanticscholar.org/graph/v1/paper/{}?fields=title,authors,year,venue,externalIds,abstract",
                s2_id
            )
        } else if let Some(doi) = &input.doi {
            format!(
                "https://api.semanticscholar.org/graph/v1/paper/DOI:{}?fields=title,authors,year,venue,externalIds,abstract",
                doi
            )
        } else if let Some(title) = &input.title {
            format!(
                "https://api.semanticscholar.org/graph/v1/paper/search?query={}&fields=title,authors,year,venue,externalIds,abstract&limit=1",
                urlencoding::encode(title)
            )
        } else {
            return SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("No identifier provided for Semantic Scholar lookup".to_string()),
                response_time_ms: 0,
            };
        };

        debug!("Semantic Scholar query URL: {}", url);

        let result = tokio::time::timeout(
            self.semantic_scholar_timeout,
            self.http_client.get(&url).send(),
        )
        .await;

        let response_time = start_time.elapsed();

        match result {
            Ok(Ok(response)) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let metadata =
                            self.parse_semantic_scholar_response(&json, input.title.is_some());
                        let confidence = if metadata.is_some() { 0.90 } else { 0.0 };
                        SourceVerificationResult {
                            source: "semantic_scholar".to_string(),
                            success: metadata.is_some(),
                            metadata,
                            confidence,
                            error: None,
                            response_time_ms: response_time.as_millis() as u64,
                        }
                    }
                    Err(e) => SourceVerificationResult {
                        source: "semantic_scholar".to_string(),
                        success: false,
                        metadata: None,
                        confidence: 0.0,
                        error: Some(format!("Failed to parse Semantic Scholar response: {}", e)),
                        response_time_ms: response_time.as_millis() as u64,
                    },
                }
            }
            Ok(Ok(response)) => SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!(
                    "Semantic Scholar returned status: {}",
                    response.status()
                )),
                response_time_ms: response_time.as_millis() as u64,
            },
            Ok(Err(e)) => SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("Semantic Scholar request failed: {}", e)),
                response_time_ms: response_time.as_millis() as u64,
            },
            Err(_) => SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("Semantic Scholar request timed out".to_string()),
                response_time_ms: response_time.as_millis() as u64,
            },
        }
    }

    /// Parse Semantic Scholar API response
    fn parse_semantic_scholar_response(
        &self,
        json: &serde_json::Value,
        is_search: bool,
    ) -> Option<VerifiedMetadata> {
        let paper = if is_search {
            json.get("data")?.as_array()?.first()?
        } else {
            json
        };

        let title = paper.get("title").and_then(|t| t.as_str()).map(String::from);

        let authors: Vec<String> = paper
            .get("authors")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|author| {
                        author.get("name").and_then(|n| n.as_str()).map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let year = paper
            .get("year")
            .and_then(|y| y.as_i64())
            .map(|y| y as i32);

        let journal = paper
            .get("venue")
            .and_then(|v| v.as_str())
            .map(String::from);

        let doi = paper
            .get("externalIds")
            .and_then(|ids| ids.get("DOI"))
            .and_then(|d| d.as_str())
            .map(String::from);

        let pmid = paper
            .get("externalIds")
            .and_then(|ids| ids.get("PubMed"))
            .and_then(|p| p.as_str())
            .map(String::from);

        let abstract_text = paper
            .get("abstract")
            .and_then(|a| a.as_str())
            .map(String::from);

        Some(VerifiedMetadata {
            title,
            authors,
            year,
            journal,
            volume: None,
            issue: None,
            pages: None,
            doi,
            pmid,
            abstract_text,
        })
    }

    /// Verify against OpenAlex API
    async fn verify_openalex(&self, input: &VerifyMetadataInput) -> SourceVerificationResult {
        let start_time = std::time::Instant::now();

        let url = if let Some(openalex_id) = &input.openalex_id {
            format!("https://api.openalex.org/works/{}", openalex_id)
        } else if let Some(doi) = &input.doi {
            format!("https://api.openalex.org/works/doi:{}", doi)
        } else if let Some(title) = &input.title {
            format!(
                "https://api.openalex.org/works?search={}&per_page=1",
                urlencoding::encode(title)
            )
        } else {
            return SourceVerificationResult {
                source: "openalex".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("No identifier provided for OpenAlex lookup".to_string()),
                response_time_ms: 0,
            };
        };

        debug!("OpenAlex query URL: {}", url);

        let result =
            tokio::time::timeout(self.openalex_timeout, self.http_client.get(&url).send()).await;

        let response_time = start_time.elapsed();

        match result {
            Ok(Ok(response)) if response.status().is_success() => {
                match response.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let metadata =
                            self.parse_openalex_response(&json, input.title.is_some());
                        let confidence = if metadata.is_some() { 0.90 } else { 0.0 };
                        SourceVerificationResult {
                            source: "openalex".to_string(),
                            success: metadata.is_some(),
                            metadata,
                            confidence,
                            error: None,
                            response_time_ms: response_time.as_millis() as u64,
                        }
                    }
                    Err(e) => SourceVerificationResult {
                        source: "openalex".to_string(),
                        success: false,
                        metadata: None,
                        confidence: 0.0,
                        error: Some(format!("Failed to parse OpenAlex response: {}", e)),
                        response_time_ms: response_time.as_millis() as u64,
                    },
                }
            }
            Ok(Ok(response)) => SourceVerificationResult {
                source: "openalex".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("OpenAlex returned status: {}", response.status())),
                response_time_ms: response_time.as_millis() as u64,
            },
            Ok(Err(e)) => SourceVerificationResult {
                source: "openalex".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some(format!("OpenAlex request failed: {}", e)),
                response_time_ms: response_time.as_millis() as u64,
            },
            Err(_) => SourceVerificationResult {
                source: "openalex".to_string(),
                success: false,
                metadata: None,
                confidence: 0.0,
                error: Some("OpenAlex request timed out".to_string()),
                response_time_ms: response_time.as_millis() as u64,
            },
        }
    }

    /// Parse OpenAlex API response
    fn parse_openalex_response(
        &self,
        json: &serde_json::Value,
        is_search: bool,
    ) -> Option<VerifiedMetadata> {
        let work = if is_search {
            json.get("results")?.as_array()?.first()?
        } else {
            json
        };

        let title = work.get("title").and_then(|t| t.as_str()).map(String::from);

        let authors: Vec<String> = work
            .get("authorships")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|authorship| {
                        authorship
                            .get("author")
                            .and_then(|a| a.get("display_name"))
                            .and_then(|n| n.as_str())
                            .map(String::from)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let year = work
            .get("publication_year")
            .and_then(|y| y.as_i64())
            .map(|y| y as i32);

        let journal = work
            .get("primary_location")
            .and_then(|loc| loc.get("source"))
            .and_then(|s| s.get("display_name"))
            .and_then(|n| n.as_str())
            .map(String::from);

        let doi = work
            .get("doi")
            .and_then(|d| d.as_str())
            .map(|s| s.trim_start_matches("https://doi.org/").to_string());

        let pmid = work
            .get("ids")
            .and_then(|ids| ids.get("pmid"))
            .and_then(|p| p.as_str())
            .map(|s| s.trim_start_matches("https://pubmed.ncbi.nlm.nih.gov/").to_string());

        let volume = work
            .get("biblio")
            .and_then(|b| b.get("volume"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let issue = work
            .get("biblio")
            .and_then(|b| b.get("issue"))
            .and_then(|i| i.as_str())
            .map(String::from);

        let pages = work.get("biblio").and_then(|b| {
            let first = b.get("first_page").and_then(|f| f.as_str());
            let last = b.get("last_page").and_then(|l| l.as_str());
            match (first, last) {
                (Some(f), Some(l)) => Some(format!("{}-{}", f, l)),
                (Some(f), None) => Some(f.to_string()),
                _ => None,
            }
        });

        Some(VerifiedMetadata {
            title,
            authors,
            year,
            journal,
            volume,
            issue,
            pages,
            doi,
            pmid,
            abstract_text: None,
        })
    }

    /// Merge results from multiple sources
    fn merge_results(
        &self,
        results: &[SourceVerificationResult],
    ) -> (Option<VerifiedMetadata>, Vec<MetadataDiscrepancy>) {
        let successful: Vec<_> = results
            .iter()
            .filter(|r| r.success && r.metadata.is_some())
            .collect();

        if successful.is_empty() {
            return (None, Vec::new());
        }

        // Start with highest confidence source
        let best = successful
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        let mut merged = best.metadata.clone().unwrap();
        let mut discrepancies = Vec::new();

        // Check for discrepancies in key fields
        let titles: Vec<_> = successful
            .iter()
            .filter_map(|r| r.metadata.as_ref()?.title.as_ref())
            .collect();

        if titles.len() > 1 {
            let unique: std::collections::HashSet<_> = titles.iter().collect();
            if unique.len() > 1 {
                discrepancies.push(MetadataDiscrepancy {
                    field: "title".to_string(),
                    values: successful
                        .iter()
                        .filter_map(|r| {
                            Some(SourceValue {
                                source: r.source.clone(),
                                value: r.metadata.as_ref()?.title.clone()?,
                            })
                        })
                        .collect(),
                });
            }
        }

        // Check year discrepancies
        let years: Vec<_> = successful
            .iter()
            .filter_map(|r| r.metadata.as_ref()?.year)
            .collect();

        if years.len() > 1 {
            let unique: std::collections::HashSet<_> = years.iter().collect();
            if unique.len() > 1 {
                discrepancies.push(MetadataDiscrepancy {
                    field: "year".to_string(),
                    values: successful
                        .iter()
                        .filter_map(|r| {
                            Some(SourceValue {
                                source: r.source.clone(),
                                value: r.metadata.as_ref()?.year?.to_string(),
                            })
                        })
                        .collect(),
                });
            }
        }

        // Fill in missing fields from other sources
        for result in &successful {
            if let Some(meta) = &result.metadata {
                if merged.title.is_none() {
                    merged.title = meta.title.clone();
                }
                if merged.authors.is_empty() {
                    merged.authors = meta.authors.clone();
                }
                if merged.year.is_none() {
                    merged.year = meta.year;
                }
                if merged.journal.is_none() {
                    merged.journal = meta.journal.clone();
                }
                if merged.doi.is_none() {
                    merged.doi = meta.doi.clone();
                }
                if merged.pmid.is_none() {
                    merged.pmid = meta.pmid.clone();
                }
                if merged.volume.is_none() {
                    merged.volume = meta.volume.clone();
                }
                if merged.issue.is_none() {
                    merged.issue = meta.issue.clone();
                }
                if merged.pages.is_none() {
                    merged.pages = meta.pages.clone();
                }
                if merged.abstract_text.is_none() {
                    merged.abstract_text = meta.abstract_text.clone();
                }
            }
        }

        (Some(merged), discrepancies)
    }

    /// Calculate overall confidence score
    fn calculate_overall_confidence(&self, results: &[SourceVerificationResult]) -> f64 {
        let successful: Vec<_> = results.iter().filter(|r| r.success).collect();

        if successful.is_empty() {
            return 0.0;
        }

        let avg_confidence: f64 =
            successful.iter().map(|r| r.confidence).sum::<f64>() / successful.len() as f64;

        // Boost confidence if multiple sources agree
        let source_boost = match successful.len() {
            1 => 0.0,
            2 => 0.05,
            3 => 0.08,
            _ => 0.10,
        };

        (avg_confidence + source_boost).min(1.0)
    }
}

impl std::fmt::Debug for VerifyMetadataTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifyMetadataTool")
            .field("crossref_timeout", &self.crossref_timeout)
            .field("pubmed_timeout", &self.pubmed_timeout)
            .field("semantic_scholar_timeout", &self.semantic_scholar_timeout)
            .field("openalex_timeout", &self.openalex_timeout)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_metadata_input_creation() {
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: Some("Test Paper".to_string()),
            authors: Some(vec!["John Doe".to_string()]),
            year: Some(2023),
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };

        assert_eq!(input.doi, Some("10.1038/nature12373".to_string()));
        assert!(input.pmid.is_none());
    }

    #[test]
    fn test_verify_metadata_input_all_fields() {
        let input = VerifyMetadataInput {
            doi: Some("10.1234/test".to_string()),
            pmid: Some("12345678".to_string()),
            title: Some("Complete Test".to_string()),
            authors: Some(vec!["Alice".to_string(), "Bob".to_string()]),
            year: Some(2024),
            s2_paper_id: Some("abc123".to_string()),
            openalex_id: Some("W12345".to_string()),
            sources: Some(vec!["crossref".to_string(), "pubmed".to_string()]),
            output_mode: VerificationOutputMode::default(),
        };

        assert!(input.doi.is_some());
        assert!(input.pmid.is_some());
        assert!(input.s2_paper_id.is_some());
        assert!(input.openalex_id.is_some());
        assert_eq!(input.sources.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_verification_status_serialization() {
        let status = VerificationStatus::Verified;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"verified\"");

        let status = VerificationStatus::PartialMatch;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"partial_match\"");

        let status = VerificationStatus::Unverified;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"unverified\"");

        let status = VerificationStatus::Failed;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"failed\"");
    }

    #[test]
    fn test_tool_creation() {
        let tool = VerifyMetadataTool::new();
        assert_eq!(tool.crossref_timeout, Duration::from_secs(10));
        assert_eq!(tool.pubmed_timeout, Duration::from_secs(10));
        assert_eq!(tool.semantic_scholar_timeout, Duration::from_secs(10));
        assert_eq!(tool.openalex_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_tool_default() {
        let tool = VerifyMetadataTool::default();
        assert_eq!(tool.crossref_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_confidence_calculation() {
        let tool = VerifyMetadataTool::new();

        // Single source
        let results = vec![SourceVerificationResult {
            source: "crossref".to_string(),
            success: true,
            metadata: Some(VerifiedMetadata {
                title: Some("Test".to_string()),
                authors: vec![],
                year: Some(2023),
                journal: None,
                volume: None,
                issue: None,
                pages: None,
                doi: None,
                pmid: None,
                abstract_text: None,
            }),
            confidence: 0.95,
            error: None,
            response_time_ms: 100,
        }];

        let confidence = tool.calculate_overall_confidence(&results);
        assert!((confidence - 0.95).abs() < 0.01);

        // Two sources
        let results = vec![
            SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.95,
                error: None,
                response_time_ms: 100,
            },
            SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 150,
            },
        ];

        let confidence = tool.calculate_overall_confidence(&results);
        // Average is 0.925, plus 0.05 boost = 0.975
        assert!(confidence > 0.95);
    }

    #[test]
    fn test_confidence_calculation_no_success() {
        let tool = VerifyMetadataTool::new();

        let results = vec![SourceVerificationResult {
            source: "crossref".to_string(),
            success: false,
            metadata: None,
            confidence: 0.0,
            error: Some("Failed".to_string()),
            response_time_ms: 100,
        }];

        let confidence = tool.calculate_overall_confidence(&results);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_confidence_calculation_multiple_sources_boost() {
        let tool = VerifyMetadataTool::new();

        // Three sources - should get 0.08 boost
        let results = vec![
            SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 100,
            },
            SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 150,
            },
            SourceVerificationResult {
                source: "openalex".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 120,
            },
        ];

        let confidence = tool.calculate_overall_confidence(&results);
        // Average is 0.90, plus 0.08 boost = 0.98
        assert!((confidence - 0.98).abs() < 0.01);
    }

    #[test]
    fn test_merge_results_empty() {
        let tool = VerifyMetadataTool::new();
        let results: Vec<SourceVerificationResult> = vec![];

        let (merged, discrepancies) = tool.merge_results(&results);
        assert!(merged.is_none());
        assert!(discrepancies.is_empty());
    }

    #[test]
    fn test_merge_results_single_source() {
        let tool = VerifyMetadataTool::new();

        let results = vec![SourceVerificationResult {
            source: "crossref".to_string(),
            success: true,
            metadata: Some(VerifiedMetadata {
                title: Some("Test Paper".to_string()),
                authors: vec!["John Doe".to_string()],
                year: Some(2023),
                journal: Some("Test Journal".to_string()),
                volume: Some("1".to_string()),
                issue: Some("2".to_string()),
                pages: Some("10-20".to_string()),
                doi: Some("10.1234/test".to_string()),
                pmid: None,
                abstract_text: None,
            }),
            confidence: 0.95,
            error: None,
            response_time_ms: 100,
        }];

        let (merged, discrepancies) = tool.merge_results(&results);
        assert!(merged.is_some());
        assert!(discrepancies.is_empty());

        let meta = merged.unwrap();
        assert_eq!(meta.title, Some("Test Paper".to_string()));
        assert_eq!(meta.year, Some(2023));
    }

    #[test]
    fn test_merge_results_with_discrepancy() {
        let tool = VerifyMetadataTool::new();

        let results = vec![
            SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Paper Title A".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.95,
                error: None,
                response_time_ms: 100,
            },
            SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Paper Title B".to_string()),
                    authors: vec![],
                    year: Some(2022), // Different year
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 150,
            },
        ];

        let (merged, discrepancies) = tool.merge_results(&results);
        assert!(merged.is_some());
        assert!(!discrepancies.is_empty());

        // Should have discrepancies for both title and year
        let fields: Vec<&str> = discrepancies.iter().map(|d| d.field.as_str()).collect();
        assert!(fields.contains(&"title"));
        assert!(fields.contains(&"year"));
    }

    #[test]
    fn test_merge_results_fills_missing_fields() {
        let tool = VerifyMetadataTool::new();

        let results = vec![
            SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: Some("10.1234/test".to_string()),
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.95,
                error: None,
                response_time_ms: 100,
            },
            SourceVerificationResult {
                source: "semantic_scholar".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec!["Author Name".to_string()],
                    year: Some(2023),
                    journal: Some("Test Journal".to_string()),
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: Some("12345".to_string()),
                    abstract_text: Some("Abstract text".to_string()),
                }),
                confidence: 0.90,
                error: None,
                response_time_ms: 150,
            },
        ];

        let (merged, _) = tool.merge_results(&results);
        let meta = merged.unwrap();

        // Should have DOI from crossref
        assert_eq!(meta.doi, Some("10.1234/test".to_string()));
        // Should have journal, pmid, abstract from semantic scholar
        assert_eq!(meta.journal, Some("Test Journal".to_string()));
        assert_eq!(meta.pmid, Some("12345".to_string()));
        assert!(meta.abstract_text.is_some());
    }

    #[test]
    fn test_verified_metadata_serialization() {
        let meta = VerifiedMetadata {
            title: Some("Test".to_string()),
            authors: vec!["Author".to_string()],
            year: Some(2023),
            journal: Some("Journal".to_string()),
            volume: Some("1".to_string()),
            issue: Some("2".to_string()),
            pages: Some("1-10".to_string()),
            doi: Some("10.1234/test".to_string()),
            pmid: Some("12345".to_string()),
            abstract_text: Some("Abstract".to_string()),
        };

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"title\":\"Test\""));
        assert!(json.contains("\"year\":2023"));

        let parsed: VerifiedMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.title, meta.title);
        assert_eq!(parsed.year, meta.year);
    }

    #[test]
    fn test_source_verification_result_serialization() {
        let result = SourceVerificationResult {
            source: "crossref".to_string(),
            success: true,
            metadata: None,
            confidence: 0.95,
            error: None,
            response_time_ms: 100,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"source\":\"crossref\""));
        assert!(json.contains("\"success\":true"));
        assert!(json.contains("\"confidence\":0.95"));
    }

    #[test]
    fn test_metadata_discrepancy() {
        let discrepancy = MetadataDiscrepancy {
            field: "year".to_string(),
            values: vec![
                SourceValue {
                    source: "crossref".to_string(),
                    value: "2023".to_string(),
                },
                SourceValue {
                    source: "pubmed".to_string(),
                    value: "2022".to_string(),
                },
            ],
        };

        assert_eq!(discrepancy.field, "year");
        assert_eq!(discrepancy.values.len(), 2);
    }

    #[test]
    fn test_verification_result_structure() {
        let result = VerificationResult {
            status: VerificationStatus::Verified,
            source_results: vec![],
            merged_metadata: None,
            overall_confidence: 0.95,
            discrepancies: vec![],
            total_time_ms: 100,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"status\":\"verified\""));
        assert!(json.contains("\"overall_confidence\":0.95"));
    }

    #[test]
    fn test_parse_crossref_response_doi_lookup() {
        let tool = VerifyMetadataTool::new();

        let json = serde_json::json!({
            "message": {
                "title": ["Test Paper Title"],
                "author": [
                    {"given": "John", "family": "Doe"},
                    {"given": "Jane", "family": "Smith"}
                ],
                "published-print": {
                    "date-parts": [[2023, 1, 15]]
                },
                "container-title": ["Nature"],
                "DOI": "10.1038/nature12373",
                "volume": "500",
                "issue": "7463",
                "page": "541-545"
            }
        });

        let result = tool.parse_crossref_response(&json, true);
        assert!(result.is_some());

        let meta = result.unwrap();
        assert_eq!(meta.title, Some("Test Paper Title".to_string()));
        assert_eq!(meta.authors.len(), 2);
        assert_eq!(meta.year, Some(2023));
        assert_eq!(meta.journal, Some("Nature".to_string()));
        assert_eq!(meta.doi, Some("10.1038/nature12373".to_string()));
    }

    #[test]
    fn test_parse_crossref_response_search() {
        let tool = VerifyMetadataTool::new();

        let json = serde_json::json!({
            "message": {
                "items": [
                    {
                        "title": ["Search Result"],
                        "author": [],
                        "DOI": "10.1234/test"
                    }
                ]
            }
        });

        let result = tool.parse_crossref_response(&json, false);
        assert!(result.is_some());
        assert_eq!(result.unwrap().title, Some("Search Result".to_string()));
    }

    #[test]
    fn test_parse_pubmed_response() {
        let tool = VerifyMetadataTool::new();

        let json = serde_json::json!({
            "result": {
                "12345678": {
                    "title": "PubMed Paper",
                    "authors": [
                        {"name": "Smith J"},
                        {"name": "Doe J"}
                    ],
                    "pubdate": "2023 Jan",
                    "source": "Nature Medicine",
                    "volume": "29",
                    "issue": "1",
                    "pages": "100-110"
                }
            }
        });

        let result = tool.parse_pubmed_response(&json, "12345678");
        assert!(result.is_some());

        let meta = result.unwrap();
        assert_eq!(meta.title, Some("PubMed Paper".to_string()));
        assert_eq!(meta.authors.len(), 2);
        assert_eq!(meta.journal, Some("Nature Medicine".to_string()));
    }

    #[test]
    fn test_parse_semantic_scholar_response_direct() {
        let tool = VerifyMetadataTool::new();

        let json = serde_json::json!({
            "title": "S2 Paper",
            "authors": [
                {"name": "Alice"},
                {"name": "Bob"}
            ],
            "year": 2023,
            "venue": "ICML",
            "externalIds": {
                "DOI": "10.1234/s2test",
                "PubMed": "87654321"
            },
            "abstract": "This is an abstract."
        });

        let result = tool.parse_semantic_scholar_response(&json, false);
        assert!(result.is_some());

        let meta = result.unwrap();
        assert_eq!(meta.title, Some("S2 Paper".to_string()));
        assert_eq!(meta.year, Some(2023));
        assert_eq!(meta.doi, Some("10.1234/s2test".to_string()));
        assert_eq!(meta.pmid, Some("87654321".to_string()));
    }

    #[test]
    fn test_parse_openalex_response() {
        let tool = VerifyMetadataTool::new();

        let json = serde_json::json!({
            "title": "OpenAlex Paper",
            "authorships": [
                {"author": {"display_name": "Charlie"}},
                {"author": {"display_name": "Diana"}}
            ],
            "publication_year": 2024,
            "primary_location": {
                "source": {"display_name": "Science"}
            },
            "doi": "https://doi.org/10.1234/oatest",
            "biblio": {
                "volume": "385",
                "issue": "6707",
                "first_page": "100",
                "last_page": "110"
            }
        });

        let result = tool.parse_openalex_response(&json, false);
        assert!(result.is_some());

        let meta = result.unwrap();
        assert_eq!(meta.title, Some("OpenAlex Paper".to_string()));
        assert_eq!(meta.authors.len(), 2);
        assert_eq!(meta.year, Some(2024));
        assert_eq!(meta.journal, Some("Science".to_string()));
        assert_eq!(meta.doi, Some("10.1234/oatest".to_string()));
        assert_eq!(meta.pages, Some("100-110".to_string()));
    }

    // Batch verification tests
    #[test]
    fn test_batch_verify_input_serialization() {
        let input = BatchVerifyMetadataInput {
            records: vec![
                VerifyMetadataInput {
                    doi: Some("10.1038/nature12373".to_string()),
                    pmid: None,
                    title: None,
                    authors: None,
                    year: None,
                    s2_paper_id: None,
                    openalex_id: None,
                    sources: None,
                    output_mode: VerificationOutputMode::default(),
                },
                VerifyMetadataInput {
                    doi: None,
                    pmid: Some("12345678".to_string()),
                    title: None,
                    authors: None,
                    year: None,
                    s2_paper_id: None,
                    openalex_id: None,
                    sources: None,
                    output_mode: VerificationOutputMode::default(),
                },
            ],
            max_concurrency: Some(3),
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: BatchVerifyMetadataInput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.records.len(), 2);
        assert_eq!(deserialized.max_concurrency, Some(3));
        assert_eq!(
            deserialized.records[0].doi,
            Some("10.1038/nature12373".to_string())
        );
        assert_eq!(
            deserialized.records[1].pmid,
            Some("12345678".to_string())
        );
    }

    #[test]
    fn test_batch_verification_result_serialization() {
        let result = BatchVerificationResult {
            total_records: 2,
            verified_count: 1,
            partial_match_count: 0,
            unverified_count: 0,
            failed_count: 1,
            results: vec![
                BatchRecordResult {
                    index: 0,
                    identifier: "DOI:10.1038/nature12373".to_string(),
                    result: VerificationResult {
                        status: VerificationStatus::Verified,
                        source_results: vec![],
                        merged_metadata: None,
                        overall_confidence: 0.95,
                        discrepancies: vec![],
                        total_time_ms: 100,
                    },
                },
                BatchRecordResult {
                    index: 1,
                    identifier: "PMID:12345678".to_string(),
                    result: VerificationResult {
                        status: VerificationStatus::Failed,
                        source_results: vec![],
                        merged_metadata: None,
                        overall_confidence: 0.0,
                        discrepancies: vec![],
                        total_time_ms: 50,
                    },
                },
            ],
            total_time_ms: 150,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BatchVerificationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.total_records, 2);
        assert_eq!(deserialized.verified_count, 1);
        assert_eq!(deserialized.failed_count, 1);
        assert_eq!(deserialized.results.len(), 2);
    }

    #[test]
    fn test_get_record_identifier_doi() {
        let record = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: Some("12345".to_string()), // DOI takes precedence
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };
        let identifier = VerifyMetadataTool::get_record_identifier(&record);
        assert_eq!(identifier, "DOI:10.1038/nature12373");
    }

    #[test]
    fn test_get_record_identifier_pmid() {
        let record = VerifyMetadataInput {
            doi: None,
            pmid: Some("12345678".to_string()),
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };
        let identifier = VerifyMetadataTool::get_record_identifier(&record);
        assert_eq!(identifier, "PMID:12345678");
    }

    #[test]
    fn test_get_record_identifier_title_short() {
        let record = VerifyMetadataInput {
            doi: None,
            pmid: None,
            title: Some("Short Title".to_string()),
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };
        let identifier = VerifyMetadataTool::get_record_identifier(&record);
        assert_eq!(identifier, "Title:Short Title");
    }

    #[test]
    fn test_get_record_identifier_title_long() {
        let record = VerifyMetadataInput {
            doi: None,
            pmid: None,
            title: Some("This is a very long paper title that exceeds fifty characters and should be truncated".to_string()),
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };
        let identifier = VerifyMetadataTool::get_record_identifier(&record);
        assert!(identifier.starts_with("Title:This is a very long paper title that exceeds"));
        assert!(identifier.ends_with("..."));
    }

    #[test]
    fn test_get_record_identifier_unknown() {
        let record = VerifyMetadataInput {
            doi: None,
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::default(),
        };
        let identifier = VerifyMetadataTool::get_record_identifier(&record);
        assert_eq!(identifier, "unknown");
    }

    #[tokio::test]
    async fn test_batch_verify_empty_input() {
        let tool = VerifyMetadataTool::new();
        let input = BatchVerifyMetadataInput {
            records: vec![],
            max_concurrency: None,
        };

        let result = tool.verify_batch(input).await.unwrap();
        assert_eq!(result.total_records, 0);
        assert_eq!(result.verified_count, 0);
        assert_eq!(result.results.len(), 0);
    }

    #[test]
    fn test_batch_record_result_fields() {
        let record_result = BatchRecordResult {
            index: 5,
            identifier: "DOI:10.1234/test".to_string(),
            result: VerificationResult {
                status: VerificationStatus::PartialMatch,
                source_results: vec![],
                merged_metadata: Some(VerifiedMetadata {
                    title: Some("Test Paper".to_string()),
                    authors: vec!["Author One".to_string()],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: Some("10.1234/test".to_string()),
                    pmid: None,
                    abstract_text: None,
                }),
                overall_confidence: 0.75,
                discrepancies: vec![
                    MetadataDiscrepancy {
                        field: "year".to_string(),
                        values: vec![
                            SourceValue {
                                source: "crossref".to_string(),
                                value: "2023".to_string(),
                            },
                            SourceValue {
                                source: "pubmed".to_string(),
                                value: "2022".to_string(),
                            },
                        ],
                    },
                ],
                total_time_ms: 200,
            },
        };

        assert_eq!(record_result.index, 5);
        assert_eq!(record_result.identifier, "DOI:10.1234/test");
        assert!(matches!(
            record_result.result.status,
            VerificationStatus::PartialMatch
        ));
        assert_eq!(record_result.result.discrepancies.len(), 1);
    }

    // Output mode tests (M2.2.11-12)
    #[test]
    fn test_output_mode_serialization() {
        assert_eq!(
            serde_json::to_string(&VerificationOutputMode::Full).unwrap(),
            "\"full\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationOutputMode::Notes).unwrap(),
            "\"notes\""
        );
        assert_eq!(
            serde_json::to_string(&VerificationOutputMode::Corrected).unwrap(),
            "\"corrected\""
        );
    }

    #[test]
    fn test_output_mode_deserialization() {
        let full: VerificationOutputMode = serde_json::from_str("\"full\"").unwrap();
        assert!(matches!(full, VerificationOutputMode::Full));

        let notes: VerificationOutputMode = serde_json::from_str("\"notes\"").unwrap();
        assert!(matches!(notes, VerificationOutputMode::Notes));

        let corrected: VerificationOutputMode = serde_json::from_str("\"corrected\"").unwrap();
        assert!(matches!(corrected, VerificationOutputMode::Corrected));
    }

    #[test]
    fn test_output_mode_default() {
        let mode = VerificationOutputMode::default();
        assert!(matches!(mode, VerificationOutputMode::Full));
    }

    #[test]
    fn test_verification_notes_result_structure() {
        let result = VerificationNotesResult {
            status: VerificationStatus::PartialMatch,
            discrepancies: vec![MetadataDiscrepancy {
                field: "year".to_string(),
                values: vec![
                    SourceValue {
                        source: "crossref".to_string(),
                        value: "2023".to_string(),
                    },
                    SourceValue {
                        source: "pubmed".to_string(),
                        value: "2022".to_string(),
                    },
                ],
            }],
            notes: vec![
                VerificationNote {
                    level: NoteLevel::Warning,
                    field: None,
                    message: "Sources partially agree.".to_string(),
                },
                VerificationNote {
                    level: NoteLevel::Warning,
                    field: Some("year".to_string()),
                    message: "Discrepancy in 'year'.".to_string(),
                },
            ],
            sources_queried: vec!["crossref".to_string(), "pubmed".to_string()],
            sources_responded: vec!["crossref".to_string(), "pubmed".to_string()],
            overall_confidence: 0.85,
            total_time_ms: 150,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"status\":\"partial_match\""));
        assert!(json.contains("\"discrepancies\""));
        assert!(json.contains("\"notes\""));
        assert!(json.contains("\"sources_queried\""));
    }

    #[test]
    fn test_note_level_serialization() {
        assert_eq!(
            serde_json::to_string(&NoteLevel::Info).unwrap(),
            "\"info\""
        );
        assert_eq!(
            serde_json::to_string(&NoteLevel::Warning).unwrap(),
            "\"warning\""
        );
        assert_eq!(
            serde_json::to_string(&NoteLevel::Error).unwrap(),
            "\"error\""
        );
    }

    #[test]
    fn test_corrected_metadata_result_structure() {
        let result = CorrectedMetadataResult {
            metadata: VerifiedMetadata {
                title: Some("Corrected Title".to_string()),
                authors: vec!["Author One".to_string(), "Author Two".to_string()],
                year: Some(2023),
                journal: Some("Nature".to_string()),
                volume: Some("600".to_string()),
                issue: Some("7890".to_string()),
                pages: Some("100-110".to_string()),
                doi: Some("10.1038/nature12373".to_string()),
                pmid: Some("12345678".to_string()),
                abstract_text: Some("Abstract text".to_string()),
            },
            confidence: 0.95,
            authority_source: "crossref".to_string(),
            sources_agreed: 3,
            corrected_fields: vec!["year".to_string()],
            total_time_ms: 200,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"metadata\""));
        assert!(json.contains("\"confidence\":0.95"));
        assert!(json.contains("\"authority_source\":\"crossref\""));
        assert!(json.contains("\"sources_agreed\":3"));
        assert!(json.contains("\"corrected_fields\":[\"year\"]"));
    }

    #[test]
    fn test_verification_output_full_variant() {
        let output = VerificationOutput::Full(VerificationResult {
            status: VerificationStatus::Verified,
            source_results: vec![],
            merged_metadata: None,
            overall_confidence: 0.95,
            discrepancies: vec![],
            total_time_ms: 100,
        });

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"output_type\":\"full\""));
        assert!(json.contains("\"status\":\"verified\""));
    }

    #[test]
    fn test_verification_output_notes_variant() {
        let output = VerificationOutput::Notes(VerificationNotesResult {
            status: VerificationStatus::PartialMatch,
            discrepancies: vec![],
            notes: vec![VerificationNote {
                level: NoteLevel::Info,
                field: None,
                message: "Test note".to_string(),
            }],
            sources_queried: vec!["crossref".to_string()],
            sources_responded: vec!["crossref".to_string()],
            overall_confidence: 0.90,
            total_time_ms: 50,
        });

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"output_type\":\"notes\""));
        assert!(json.contains("\"notes\""));
    }

    #[test]
    fn test_verification_output_corrected_variant() {
        let output = VerificationOutput::Corrected(CorrectedMetadataResult {
            metadata: VerifiedMetadata {
                title: Some("Test".to_string()),
                authors: vec![],
                year: Some(2023),
                journal: None,
                volume: None,
                issue: None,
                pages: None,
                doi: None,
                pmid: None,
                abstract_text: None,
            },
            confidence: 0.95,
            authority_source: "crossref".to_string(),
            sources_agreed: 2,
            corrected_fields: vec![],
            total_time_ms: 100,
        });

        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"output_type\":\"corrected\""));
        assert!(json.contains("\"authority_source\":\"crossref\""));
    }

    #[test]
    fn test_convert_to_notes_verified() {
        let tool = VerifyMetadataTool::new();
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Notes,
        };

        let result = VerificationResult {
            status: VerificationStatus::Verified,
            source_results: vec![SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("Test".to_string()),
                    authors: vec![],
                    year: Some(2023),
                    journal: None,
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: None,
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.95,
                error: None,
                response_time_ms: 100,
            }],
            merged_metadata: None,
            overall_confidence: 0.95,
            discrepancies: vec![],
            total_time_ms: 100,
        };

        let notes_result = tool.convert_to_notes(&result, &input);
        assert!(matches!(notes_result.status, VerificationStatus::Verified));
        assert!(!notes_result.notes.is_empty());
        // Should have at least the status note
        assert!(notes_result
            .notes
            .iter()
            .any(|n| matches!(n.level, NoteLevel::Info)));
    }

    #[test]
    fn test_convert_to_notes_with_discrepancies() {
        let tool = VerifyMetadataTool::new();
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Notes,
        };

        let result = VerificationResult {
            status: VerificationStatus::PartialMatch,
            source_results: vec![],
            merged_metadata: None,
            overall_confidence: 0.75,
            discrepancies: vec![MetadataDiscrepancy {
                field: "year".to_string(),
                values: vec![
                    SourceValue {
                        source: "crossref".to_string(),
                        value: "2023".to_string(),
                    },
                    SourceValue {
                        source: "pubmed".to_string(),
                        value: "2022".to_string(),
                    },
                ],
            }],
            total_time_ms: 150,
        };

        let notes_result = tool.convert_to_notes(&result, &input);
        assert_eq!(notes_result.discrepancies.len(), 1);
        // Should have notes for status and discrepancies
        assert!(notes_result.notes.len() >= 2);
    }

    #[test]
    fn test_convert_to_notes_low_confidence() {
        let tool = VerifyMetadataTool::new();
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Notes,
        };

        let result = VerificationResult {
            status: VerificationStatus::PartialMatch,
            source_results: vec![],
            merged_metadata: None,
            overall_confidence: 0.3, // Low confidence
            discrepancies: vec![],
            total_time_ms: 100,
        };

        let notes_result = tool.convert_to_notes(&result, &input);
        // Should have a low confidence warning note
        assert!(notes_result.notes.iter().any(|n| {
            matches!(n.level, NoteLevel::Warning) && n.message.contains("Low confidence")
        }));
    }

    #[test]
    fn test_convert_to_corrected() {
        let tool = VerifyMetadataTool::new();
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: Some("Old Title".to_string()), // Will be corrected
            authors: None,
            year: Some(2022), // Will be corrected to 2023
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Corrected,
        };

        let result = VerificationResult {
            status: VerificationStatus::Verified,
            source_results: vec![SourceVerificationResult {
                source: "crossref".to_string(),
                success: true,
                metadata: Some(VerifiedMetadata {
                    title: Some("New Title".to_string()),
                    authors: vec!["Author".to_string()],
                    year: Some(2023),
                    journal: Some("Nature".to_string()),
                    volume: None,
                    issue: None,
                    pages: None,
                    doi: Some("10.1038/nature12373".to_string()),
                    pmid: None,
                    abstract_text: None,
                }),
                confidence: 0.95,
                error: None,
                response_time_ms: 100,
            }],
            merged_metadata: Some(VerifiedMetadata {
                title: Some("New Title".to_string()),
                authors: vec!["Author".to_string()],
                year: Some(2023),
                journal: Some("Nature".to_string()),
                volume: None,
                issue: None,
                pages: None,
                doi: Some("10.1038/nature12373".to_string()),
                pmid: None,
                abstract_text: None,
            }),
            overall_confidence: 0.95,
            discrepancies: vec![],
            total_time_ms: 100,
        };

        let corrected = tool.convert_to_corrected(&result, &input);
        assert!(corrected.is_some());

        let corrected = corrected.unwrap();
        assert_eq!(corrected.authority_source, "crossref");
        assert_eq!(corrected.sources_agreed, 1);
        assert!(corrected.corrected_fields.contains(&"title".to_string()));
        assert!(corrected.corrected_fields.contains(&"year".to_string()));
    }

    #[test]
    fn test_convert_to_corrected_no_metadata() {
        let tool = VerifyMetadataTool::new();
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Corrected,
        };

        let result = VerificationResult {
            status: VerificationStatus::Failed,
            source_results: vec![],
            merged_metadata: None, // No metadata available
            overall_confidence: 0.0,
            discrepancies: vec![],
            total_time_ms: 50,
        };

        let corrected = tool.convert_to_corrected(&result, &input);
        assert!(corrected.is_none()); // Should return None when no metadata
    }

    #[test]
    fn test_input_with_output_mode_serialization() {
        let input = VerifyMetadataInput {
            doi: Some("10.1038/nature12373".to_string()),
            pmid: None,
            title: None,
            authors: None,
            year: None,
            s2_paper_id: None,
            openalex_id: None,
            sources: None,
            output_mode: VerificationOutputMode::Notes,
        };

        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"output_mode\":\"notes\""));

        let deserialized: VerifyMetadataInput = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            deserialized.output_mode,
            VerificationOutputMode::Notes
        ));
    }

    #[test]
    fn test_input_with_default_output_mode() {
        let json = r#"{"doi": "10.1038/nature12373"}"#;
        let input: VerifyMetadataInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input.output_mode, VerificationOutputMode::Full));
    }
}
