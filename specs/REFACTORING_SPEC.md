# Research Hub MCP Refactoring Specification

## Overview

This document outlines a comprehensive refactoring plan to address four major architectural issues in the research_hub_mcp project. The refactoring aims to improve flexibility, reduce coupling, and enhance the user experience.

**Version:** 1.0
**Date:** 2025-12-22
**Status:** Proposed

---

## Table of Contents

1. [Search Functionality Refactoring](#1-search-functionality-refactoring)
2. [Metadata Tools Refactoring](#2-metadata-tools-refactoring)
3. [Download Command Refactoring](#3-download-command-refactoring)
4. [Text2Table Components Refactoring](#4-text2table-components-refactoring)
5. [Implementation Priority](#5-implementation-priority)
6. [Migration Strategy](#6-migration-strategy)

---

## 1. Search Functionality Refactoring

### 1.1 Current Issues

The current search system attempts to unify all 15 providers under a common interface, which causes:

1. **Capability degradation** - Providers with rich query syntax (e.g., PubMed's advanced search operators) are reduced to basic keyword search
2. **Forced multi-provider execution** - Users cannot focus on a single authoritative source
3. **Lost provider-specific features** - arXiv categories, PubMed MeSH terms, CrossRef filters are inaccessible
4. **Complex routing logic** - The meta-search orchestration adds latency and complexity

### 1.2 Proposed Architecture

#### 1.2.1 New Tool: `search_source`

Replace the unified `search` tool with a source-specific search tool:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchSourceInput {
    /// The source to search (required, single source only)
    #[schemars(description = "Source name: arxiv, pubmed, crossref, semantic_scholar, etc.")]
    pub source: String,

    /// Raw query string passed directly to the source
    #[schemars(description = "Query string in source-native format")]
    pub query: String,

    /// Maximum results to return
    #[schemars(description = "Max results (1-100)")]
    pub limit: Option<u32>,

    /// Offset for pagination
    pub offset: Option<u32>,
}
```

#### 1.2.2 Source-Specific Query Formats

Each source exposes its native query capabilities:

| Source | Query Format | Examples |
|--------|--------------|----------|
| **pubmed** | PubMed Advanced Search syntax | `cancer[Title] AND therapy[MeSH]` |
| **arxiv** | arXiv API query syntax | `cat:cs.AI AND ti:transformer` |
| **crossref** | CrossRef query filters | `query.title:attention+mechanism` |
| **semantic_scholar** | S2 query + fields | `machine learning year:2024` |
| **openalex** | OpenAlex filter syntax | `title.search:neural networks` |
| **google_scholar** | Free text (limited) | `"deep learning" author:Hinton` |
| **biorxiv/medrxiv** | bioRxiv search syntax | `COVID-19 preprint` |
| **core** | CORE API query | `title:quantum computing` |
| **unpaywall** | DOI or title query | `10.1038/nature12373` |
| **openreview** | OpenReview search | `venue:ICLR.cc/2024` |
| **mdpi** | MDPI search | `sustainability water` |
| **ssrn** | SSRN keywords | `financial markets` |
| **researchgate** | RG search | `author:Smith cognitive` |
| **sci_hub** | DOI only | `10.1126/science.xxx` |

#### 1.2.3 New Tool: `list_sources`

Provide discovery of available sources and their capabilities:

```rust
#[tool]
async fn list_sources() -> Result<Value> {
    // Returns list of sources with:
    // - name, description
    // - supported_query_format (with examples)
    // - rate_limits
    // - health_status
}
```

#### 1.2.4 Provider Trait Changes

Modify `SourceProvider` trait to expose native query capability:

```rust
#[async_trait]
pub trait SourceProvider: Send + Sync {
    fn name(&self) -> &str;
    fn priority(&self) -> u8;

    /// Returns human-readable query format documentation
    fn query_format_help(&self) -> &str;

    /// Returns example queries for this source
    fn query_examples(&self) -> Vec<(&str, &str)>; // (query, description)

    /// Execute search with native query string
    async fn search_native(&self, query: &str, limit: u32, offset: u32)
        -> ProviderResult<Vec<PaperMetadata>>;

    // Keep existing methods for compatibility
    async fn search(&self, query: &SearchQuery, ctx: &SearchContext)
        -> ProviderResult<Vec<PaperMetadata>>;

    async fn get_by_doi(&self, doi: &str) -> ProviderResult<Option<PaperMetadata>>;
}
```

### 1.3 Files to Modify

| File | Changes |
|------|---------|
| `src/tools/search.rs` | Replace with `search_source.rs`, new input schema |
| `src/client/providers/traits.rs` | Add `query_format_help()`, `query_examples()`, `search_native()` |
| `src/client/providers/*.rs` | Implement native query for each provider |
| `src/client/meta_search.rs` | Simplify - remove multi-provider orchestration |
| `src/server/handler.rs` | Update tool registration |

### 1.4 Backward Compatibility

Optionally keep a `search_multi` tool for users who want to search multiple sources:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchMultiInput {
    /// List of (source, query) pairs
    pub searches: Vec<SourceQuery>,
    /// Whether to deduplicate results by DOI
    pub deduplicate: Option<bool>,
}
```

This requires multiple sequential calls to `search_source`, making the tradeoff explicit.

---

## 2. Metadata Tools Refactoring

### 2.1 Current Issues

The current `extract_metadata` tool mixes two concerns:
1. PDF metadata extraction (local file processing)
2. External metadata validation (network calls to CrossRef)

Additionally, there's no tool for validating/reconciling metadata from multiple sources.

### 2.2 Proposed Architecture

#### 2.2.1 Tool 1: `pdf_metadata` (Rename existing)

Focus purely on extracting metadata from local PDF files:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PdfMetadataInput {
    /// PDF file path(s) - single file or list
    pub files: OneOrMany<PathBuf>,

    /// Extract references from PDF
    pub extract_references: Option<bool>,

    /// Use cache for repeated extractions
    pub use_cache: Option<bool>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct PdfMetadataOutput {
    pub file_path: PathBuf,
    pub title: Option<String>,
    pub authors: Vec<Author>,
    pub doi: Option<String>,
    pub publication_date: Option<String>,
    pub journal: Option<String>,
    pub abstract_text: Option<String>,
    pub keywords: Vec<String>,
    pub references: Vec<Reference>,
    pub confidence_score: f64,
    pub extraction_method: String, // "pdf_text", "pdf_info", "grobid"
}
```

**Removed from this tool:**
- `validate_external` parameter
- CrossRef API integration

#### 2.2.2 Tool 2: `verify_metadata` (New)

Verify and reconcile metadata from multiple sources:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VerifyMetadataInput {
    /// Metadata records to verify (1 to N)
    pub records: Vec<MetadataRecord>,

    /// Output mode: "notes" for analysis, "corrected" for fixed values
    #[schemars(description = "Output mode: 'notes' | 'corrected'")]
    pub output_mode: Option<String>,

    /// Sources to query for verification
    #[schemars(description = "Sources: crossref, pubmed, semantic_scholar, openalex")]
    pub verification_sources: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MetadataRecord {
    /// Record identifier (for correlation in output)
    pub id: Option<String>,

    /// Known/suspected DOI
    pub doi: Option<String>,

    /// Known/suspected PubMed ID
    pub pmid: Option<String>,

    /// Known/suspected title
    pub title: Option<String>,

    /// Known/suspected authors
    pub authors: Option<Vec<String>>,

    /// Known/suspected year
    pub year: Option<u32>,

    /// Any other identifiers
    pub other_ids: Option<HashMap<String, String>>,
}
```

#### 2.2.3 Verification Logic

```rust
async fn verify_record(record: &MetadataRecord) -> VerificationResult {
    // 1. Query each available identifier
    let mut sources: HashMap<String, PaperMetadata> = HashMap::new();

    if let Some(doi) = &record.doi {
        if let Ok(meta) = crossref.get_by_doi(doi).await {
            sources.insert("crossref_doi", meta);
        }
    }

    if let Some(pmid) = &record.pmid {
        if let Ok(meta) = pubmed.get_by_pmid(pmid).await {
            sources.insert("pubmed_pmid", meta);
        }
    }

    // 2. Compare results across sources
    let comparison = compare_metadata_sources(&sources, record);

    // 3. Determine most authoritative source
    let authority = determine_authority(&comparison);

    // 4. Return verification result
    VerificationResult {
        record_id: record.id.clone(),
        sources_queried: sources.keys().collect(),
        matches: comparison.matches,
        conflicts: comparison.conflicts,
        suggested_corrections: authority.corrections,
        confidence: authority.confidence,
        notes: comparison.analysis_notes,
    }
}
```

#### 2.2.4 Output Modes

**Mode: "notes"** (default)
```json
{
  "record_id": "paper_1",
  "verification_status": "conflict_detected",
  "sources_queried": ["crossref", "pubmed", "semantic_scholar"],
  "conflicts": [
    {
      "field": "doi",
      "input_value": "10.1000/wrong",
      "source_values": {
        "pubmed": "10.1000/correct",
        "semantic_scholar": "10.1000/correct"
      },
      "recommendation": "Use DOI from PubMed/S2 (2/3 sources agree)"
    }
  ],
  "matches": ["title", "authors", "year"],
  "notes": "DOI appears incorrect. Title matches across all sources."
}
```

**Mode: "corrected"**
```json
{
  "record_id": "paper_1",
  "corrected": {
    "doi": "10.1000/correct",
    "title": "Original Paper Title",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024
  },
  "corrections_made": [
    {"field": "doi", "from": "10.1000/wrong", "to": "10.1000/correct"}
  ],
  "confidence": 0.95
}
```

### 2.3 Files to Modify

| File | Changes |
|------|---------|
| `src/tools/metadata.rs` | Rename to `pdf_metadata.rs`, remove external validation |
| `src/tools/verify_metadata.rs` | New file - verification logic |
| `src/server/handler.rs` | Register both tools |

---

## 3. Download Command Refactoring

### 3.1 Current Architecture Analysis

#### 3.1.1 Current Implementation Split (Rust vs Python)

The download system is split across two languages:

**Rust Layer** (`src/tools/download.rs`, `src/python_embed.rs`):
- `DownloadTool` struct - orchestrates download flow
- `download_paper()` - main entry point
- `download_paper_impl()` - primary download logic
- `resolve_download_source()` - metadata resolution via `MetaSearchClient`
- `execute_download_by_cdp()` - calls Python CDP download
- `attempt_plugin_fallback()` - calls Python plugin system
- `determine_file_path()` - path resolution logic
- File validation, hash calculation, post-processing

**Python Layer** (`python/rust_research_py/plugins/`):
- **10 Publisher-Specific Plugins**:
  | Plugin | DOI Prefix | Access Model |
  |--------|-----------|--------------|
  | `nature_pdf_downloader.py` | `10.1038` | Subscription + OA |
  | `wiley_pdf_downloader.py` | `10.1002`, `10.1111`, etc. (26 prefixes) | Subscription |
  | `springer_pdf_downloader.py` | `10.1007`, `10.1186`, etc. | Subscription + OA |
  | `biorxiv_pdf_downloader.py` | `10.1101` | Open Access |
  | `oxford_pdf_downloader.py` | `10.1093` | Subscription |
  | `mdpi_pdf_downloader.py` | `10.3390` | Open Access |
  | `frontiers_pdf_downloader.py` | `10.3389` | Open Access |
  | `plos_pdf_downloader.py` | `10.1371` | Open Access |
  | `pnas_pdf_downloader.py` | `10.1073` | Subscription + OA |
  | `hindawi_pdf_downloader.py` | `10.1155` | Open Access |

- **Core Infrastructure**:
  - `common.py` - `BasePlugin` abstract class, `DownloadResult` dataclass
  - `utils.py` - Plugin registry, DOI detection, `execute_download_by_cdp()`
  - `plugin_runner.py` - CLI entry point for plugin invocation

- **Download Strategies per Plugin**:
  1. Direct HTTP (fastest, for OA content)
  2. Browser Download Handler (Playwright `expect_download()`)
  3. CDP Fetch Interception (for JS-heavy/protected sites)

#### 3.1.2 Current Flow

```
download_paper(input)
    │
    ├─► [DOI provided?]
    │       │
    │       ├─► YES: attempt_plugin_fallback()
    │       │         │
    │       │         ├─► PyO3 → run_plugin_download()
    │       │         │         │
    │       │         │         ├─► detect_publisher_patterns(doi)
    │       │         │         ├─► Initialize matching plugin
    │       │         │         └─► plugin.download() [3 strategies]
    │       │         │
    │       │         ├─► SUCCESS → return result
    │       │         └─► FAIL → download_with_primary_fallback()
    │       │
    │       └─► NO: download_paper_impl() directly
    │
    └─► download_paper_impl()
            │
            ├─► resolve_download_source() [MetaSearchClient]
            ├─► determine_file_path()
            ├─► execute_download_by_cdp() [PyO3 → Python CDP]
            └─► apply_post_processing() [pdf2text if Markdown]
```

#### 3.1.3 Current Issues (Detailed)

1. **Sequential Fallback** - Plugin tried first, then CDP, never in parallel
2. **No Early Termination** - Can't cancel plugin download if CDP succeeds first
3. **Metadata Coupling** - `resolve_download_source()` always runs full search
4. **FFI Overhead** - Each Python call blocks Rust async runtime via `spawn_blocking`
5. **Plugin Detection Inefficiency** - Iterates all plugins to find DOI match
6. **No Unified Interface** - Rust HTTP and Python plugins have different contracts

### 3.2 Proposed Architecture

#### 3.2.1 Separation of Concerns

Split download into three phases:

```
Phase 1: Metadata Resolution (if needed)
    └─> Returns: doi, title, pdf_urls[]

Phase 2: Parallel Download Race
    └─> Multiple download methods compete
    └─> First successful download wins
    └─> Others cancelled

Phase 3: Post-Processing (if requested)
    └─> pdf2text, verification, etc.
```

#### 3.2.2 New Download Input Schema

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DownloadInput {
    // Identifier (at least one required)
    pub doi: Option<String>,
    pub url: Option<String>,
    pub pmid: Option<String>,

    // Download configuration
    pub directory: Option<PathBuf>,
    pub filename: Option<String>,
    pub category: Option<String>,

    /// Download methods to try (all tried in parallel)
    #[schemars(description = "Methods: direct_http, cdp, unpaywall, arxiv, pubmed, sci_hub")]
    pub methods: Option<Vec<String>>,

    /// Cancel other methods when one succeeds
    pub race_mode: Option<bool>, // default: true

    // Post-processing
    pub output_format: Option<OutputFormat>,
    pub verify_integrity: Option<bool>,

    // Behavior
    pub overwrite: Option<bool>,
    pub headless: Option<bool>,
}
```

#### 3.2.3 Race-Based Download Implementation

```rust
pub struct DownloadRace {
    methods: Vec<Box<dyn DownloadMethod>>,
    cancel_token: CancellationToken,
}

impl DownloadRace {
    pub async fn execute(&self, target: &DownloadTarget) -> Result<DownloadResult> {
        let cancel = self.cancel_token.clone();

        // Start all methods concurrently
        let futures: Vec<_> = self.methods.iter()
            .map(|method| {
                let cancel = cancel.clone();
                let target = target.clone();
                async move {
                    tokio::select! {
                        result = method.download(&target) => {
                            if result.is_ok() {
                                cancel.cancel(); // Signal others to stop
                            }
                            (method.name(), result)
                        }
                        _ = cancel.cancelled() => {
                            (method.name(), Err(Error::Cancelled))
                        }
                    }
                }
            })
            .collect();

        // Wait for first success
        let results = futures::future::join_all(futures).await;

        // Return first successful result
        for (method_name, result) in results {
            if let Ok(download) = result {
                return Ok(download.with_method(method_name));
            }
        }

        // All failed - aggregate errors
        Err(Error::AllMethodsFailed(results))
    }
}
```

#### 3.2.4 Download Method Trait

```rust
#[async_trait]
pub trait DownloadMethod: Send + Sync {
    fn name(&self) -> &str;

    /// Check if this method can handle the target
    fn can_handle(&self, target: &DownloadTarget) -> bool;

    /// Attempt to download
    async fn download(&self, target: &DownloadTarget) -> Result<TempDownload>;

    /// Estimated priority (higher = preferred if multiple succeed)
    fn priority(&self) -> u8;
}
```

#### 3.2.5 Download Method Categories

**Category A: Rust-Native Methods** (async, cancellable)
| Method | Priority | Handles | Description |
|--------|----------|---------|-------------|
| `direct_http` | 80 | URLs | Simple reqwest HTTP GET |
| `arxiv_direct` | 95 | arXiv IDs | Direct `arxiv.org/pdf/{id}.pdf` |
| `pmc_direct` | 90 | PMC IDs | Direct PMC PDF link |
| `unpaywall_api` | 85 | DOIs | Unpaywall API → OA URL |

**Category B: Python Plugin Methods** (via PyO3, blocking task)
| Method | Priority | DOI Prefix | Publisher |
|--------|----------|------------|-----------|
| `plugin_nature` | 88 | `10.1038` | Nature |
| `plugin_wiley` | 82 | `10.1002`, `10.1111`... | Wiley |
| `plugin_springer` | 85 | `10.1007`, `10.1186`... | Springer |
| `plugin_biorxiv` | 90 | `10.1101` | bioRxiv/medRxiv |
| `plugin_oxford` | 80 | `10.1093` | Oxford Academic |
| `plugin_mdpi` | 87 | `10.3390` | MDPI |
| `plugin_frontiers` | 87 | `10.3389` | Frontiers |
| `plugin_plos` | 87 | `10.1371` | PLOS |
| `plugin_pnas` | 83 | `10.1073` | PNAS |
| `plugin_hindawi` | 86 | `10.1155` | Hindawi |

**Category C: Universal Fallback Methods**
| Method | Priority | Handles | Description |
|--------|----------|---------|-------------|
| `cdp_universal` | 70 | Any URL | Playwright CDP |
| `sci_hub` | 10 | DOIs | Last resort |

#### 3.2.6 Unified Download Method Interface

To enable race mode across Rust and Python methods, we need a unified interface:

**Rust Trait Definition:**
```rust
#[async_trait]
pub trait DownloadMethod: Send + Sync {
    /// Unique method identifier
    fn name(&self) -> &str;

    /// DOI prefixes this method handles (empty = URL-based only)
    fn supported_doi_prefixes(&self) -> &[&str];

    /// Check if method can handle the target
    fn can_handle(&self, target: &DownloadTarget) -> bool;

    /// Priority for selection when multiple methods succeed
    fn priority(&self) -> u8;

    /// Execute download with cancellation support
    async fn download(
        &self,
        target: &DownloadTarget,
        output_path: &Path,
        cancel: CancellationToken,
    ) -> Result<TempDownload>;

    /// Whether this method requires Python runtime
    fn requires_python(&self) -> bool { false }
}
```

**Python Plugin Wrapper:**
```rust
pub struct PythonPluginMethod {
    plugin_name: String,
    doi_prefixes: Vec<String>,
    priority: u8,
}

#[async_trait]
impl DownloadMethod for PythonPluginMethod {
    fn name(&self) -> &str { &self.plugin_name }

    fn supported_doi_prefixes(&self) -> &[&str] {
        self.doi_prefixes.iter().map(|s| s.as_str()).collect()
    }

    fn requires_python(&self) -> bool { true }

    async fn download(
        &self,
        target: &DownloadTarget,
        output_path: &Path,
        cancel: CancellationToken,
    ) -> Result<TempDownload> {
        let plugin_name = self.plugin_name.clone();
        let doi = target.doi.clone().ok_or(Error::NoDoi)?;
        let output_dir = output_path.parent().unwrap().to_path_buf();

        // Run in blocking task with cancellation check
        tokio::select! {
            result = tokio::task::spawn_blocking(move || {
                crate::python_embed::run_specific_plugin(
                    &plugin_name,
                    &doi,
                    &output_dir,
                )
            }) => {
                result.map_err(|e| Error::TaskPanic(e))?
            }
            _ = cancel.cancelled() => {
                Err(Error::Cancelled)
            }
        }
    }
}
```

#### 3.2.7 Race Coordinator Implementation

```rust
pub struct DownloadRaceCoordinator {
    rust_methods: Vec<Arc<dyn DownloadMethod>>,
    python_methods: Vec<Arc<PythonPluginMethod>>,
    config: DownloadConfig,
}

impl DownloadRaceCoordinator {
    /// Execute download with race mode
    pub async fn download_race(
        &self,
        target: &DownloadTarget,
        methods: Option<Vec<String>>,
    ) -> Result<DownloadResult> {
        let cancel = CancellationToken::new();

        // Filter applicable methods
        let applicable = self.filter_applicable_methods(target, methods);

        // Separate into rust-native (truly async) and python (blocking)
        let (rust_methods, python_methods): (Vec<_>, Vec<_>) =
            applicable.into_iter().partition(|m| !m.requires_python());

        // Start rust methods as true async tasks
        let rust_futures: Vec<_> = rust_methods.iter().map(|method| {
            let method = method.clone();
            let target = target.clone();
            let cancel = cancel.clone();
            async move {
                let result = method.download(&target, &temp_path, cancel).await;
                (method.name().to_string(), method.priority(), result)
            }
        }).collect();

        // Start python methods with limited concurrency (avoid GIL contention)
        let python_semaphore = Arc::new(Semaphore::new(2)); // Max 2 concurrent Python calls
        let python_futures: Vec<_> = python_methods.iter().map(|method| {
            let method = method.clone();
            let target = target.clone();
            let cancel = cancel.clone();
            let sem = python_semaphore.clone();
            async move {
                let _permit = sem.acquire().await;
                let result = method.download(&target, &temp_path, cancel).await;
                (method.name().to_string(), method.priority(), result)
            }
        }).collect();

        // Combine all futures
        let all_futures = rust_futures.into_iter().chain(python_futures);

        // Race: wait for first success
        let results = Self::race_first_success(all_futures, cancel.clone()).await;

        // Return best result (highest priority among successes)
        Self::select_best_result(results)
    }

    async fn race_first_success<F, T>(
        futures: impl IntoIterator<Item = F>,
        cancel: CancellationToken,
    ) -> Vec<(String, u8, Result<T>)>
    where
        F: Future<Output = (String, u8, Result<T>)> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut handles = Vec::new();

        for fut in futures {
            let tx = tx.clone();
            let cancel = cancel.clone();
            let handle = tokio::spawn(async move {
                let result = fut.await;
                let is_success = result.2.is_ok();
                let _ = tx.send(result);
                if is_success {
                    cancel.cancel(); // Signal others to stop
                }
            });
            handles.push(handle);
        }
        drop(tx); // Close sender when all spawned

        // Collect results
        let mut results = Vec::new();
        while let Some(result) = rx.recv().await {
            results.push(result);
        }

        results
    }
}
```

#### 3.2.8 Python-Side Changes

Update `utils.py` to support specific plugin invocation:

```python
# python/rust_research_py/plugins/utils.py

async def run_specific_plugin(
    plugin_name: str,
    doi: str,
    output_dir: Path,
    filename: Optional[str] = None,
    headless: bool = True,
    cancel_event: Optional[asyncio.Event] = None,
) -> DownloadResult:
    """Run a specific plugin by name (for race mode)."""
    if plugin_name not in PLUGIN_REGISTRY:
        return DownloadResult(
            success=False,
            error=f"Unknown plugin: {plugin_name}"
        )

    plugin_class = PLUGIN_REGISTRY[plugin_name]
    plugin = plugin_class(headless=headless)

    try:
        async with plugin:
            # Build URL from DOI
            url = plugin.build_download_url(doi)
            if not url:
                return DownloadResult(
                    success=False,
                    error=f"Cannot build URL for DOI: {doi}"
                )

            # Download with cancellation support
            return await plugin.download(
                url=url,
                output_dir=output_dir,
                filename=filename,
                doi=doi,
                cancel_event=cancel_event,
            )
    except asyncio.CancelledError:
        return DownloadResult(success=False, error="Cancelled")
```

Add cancellation support to `BasePlugin`:

```python
# python/rust_research_py/plugins/common.py

class BasePlugin(ABC):
    async def download(
        self,
        url: str,
        output_dir: Path,
        filename: Optional[str] = None,
        wait_time: float = 2.0,
        doi: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> DownloadResult:
        """Download with cancellation support."""
        # Check cancellation before each major step
        if cancel_event and cancel_event.is_set():
            return DownloadResult(success=False, error="Cancelled")

        # ... existing download logic with periodic cancellation checks
```

### 3.3 Metadata Resolution Refactoring

Move metadata resolution to separate async step:

```rust
pub struct MetadataResolver {
    providers: Vec<Arc<dyn SourceProvider>>,
}

impl MetadataResolver {
    /// Resolve metadata and PDF URLs from identifiers
    pub async fn resolve(&self, input: &DownloadInput) -> Result<DownloadTarget> {
        let mut target = DownloadTarget::default();

        // DOI lookup
        if let Some(doi) = &input.doi {
            let results = self.parallel_doi_lookup(doi).await;
            target.merge_results(results);
        }

        // PMID lookup
        if let Some(pmid) = &input.pmid {
            let meta = self.pubmed_lookup(pmid).await?;
            target.merge_metadata(meta);
        }

        // Direct URL
        if let Some(url) = &input.url {
            target.add_url(url.clone(), "user_provided");
        }

        Ok(target)
    }
}
```

### 3.4 Files to Modify

#### Rust Files

| File | Changes |
|------|---------|
| `src/tools/download.rs` | Refactor to use `DownloadRaceCoordinator` |
| `src/client/download_methods/mod.rs` | **NEW**: Module declaration |
| `src/client/download_methods/traits.rs` | **NEW**: `DownloadMethod` trait |
| `src/client/download_methods/direct_http.rs` | **NEW**: Rust HTTP method |
| `src/client/download_methods/arxiv.rs` | **NEW**: arXiv direct method |
| `src/client/download_methods/pmc.rs` | **NEW**: PMC direct method |
| `src/client/download_methods/unpaywall.rs` | **NEW**: Unpaywall API method |
| `src/client/download_methods/python_plugin.rs` | **NEW**: PyO3 plugin wrapper |
| `src/client/download_methods/cdp_universal.rs` | **NEW**: Universal CDP fallback |
| `src/client/download_methods/coordinator.rs` | **NEW**: Race coordinator |
| `src/python_embed.rs` | Add `run_specific_plugin()` function |

#### Python Files

| File | Changes |
|------|---------|
| `python/rust_research_py/plugins/common.py` | Add `cancel_event` to `BasePlugin.download()` |
| `python/rust_research_py/plugins/utils.py` | Add `run_specific_plugin()` function |
| `python/rust_research_py/plugins/downloader/*.py` | Add cancellation checks in download loops |

### 3.5 Implementation Phases

**Phase 3a: Foundation (2-3 days)**
1. Create `DownloadMethod` trait
2. Implement `DirectHttpMethod` (pure Rust)
3. Implement `ArxivDirectMethod` (pure Rust)
4. Unit tests for Rust methods

**Phase 3b: Python Integration (2-3 days)**
1. Add cancellation support to Python plugins
2. Create `PythonPluginMethod` wrapper
3. Add `run_specific_plugin()` to `python_embed.rs`
4. Integration tests

**Phase 3c: Race Coordinator (2-3 days)**
1. Implement `DownloadRaceCoordinator`
2. Update `DownloadTool` to use coordinator
3. Add `methods` and `race_mode` parameters
4. End-to-end tests

### 3.6 Considerations

#### GIL Contention
- Limit Python plugin concurrency to 2 via semaphore
- Rust methods run fully async without blocking

#### Cancellation Propagation
- `CancellationToken` for Rust → `tokio::select!`
- Python cancellation via `asyncio.Event` (polled periodically)
- Browser operations may not be immediately cancellable

#### Temp File Management
- Each method writes to unique temp file
- Only winner's file is moved to final location
- Cleanup on cancellation/failure

#### Error Aggregation
- Collect all method errors if all fail
- Report which methods were tried and why they failed

---

## 4. Text2Table Components Refactoring

### 4.1 Current Issues

1. **t2t-server** - Only wraps vLLM startup, overly complex
2. **t2t-cli** - Separate `run` and `run-batch` commands with redundant parameters
3. **Parameter duplication** - Same options defined twice across commands
4. **Input/output format rigidity** - JSONL-only batch, explicit format flags

### 4.2 Proposed Architecture

#### 4.2.1 Simplified t2t-server

Focus purely on vLLM service management:

```python
# python/rust_research_py/text2table/server.py

@click.command()
@click.option('--model', default='Qwen/Qwen3-30B-A3B-Instruct-2507')
@click.option('--port', default=8000, type=int)
@click.option('--host', default='0.0.0.0')
@click.option('--tensor-parallel-size', default=1, type=int)
@click.option('--gpu-memory-utilization', default=0.9, type=float)
@click.option('--max-model-len', default=None, type=int)
@click.option('--trust-remote-code', is_flag=True)
def main(model, port, host, tensor_parallel_size, gpu_memory_utilization,
         max_model_len, trust_remote_code):
    """Start vLLM server for text2table."""
    start_vllm_server(
        model=model,
        host=host,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )
```

Remove: Complex fallback logic, environment variable handling in CLI

#### 4.2.2 Unified t2t-cli

Single command that auto-adapts to input:

```python
# python/rust_research_py/text2table/cli.py

@click.command()
@click.argument('input', type=click.Path(exists=True), required=False)
@click.option('--text', '-t', help='Direct text input (instead of file)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--label', '-l', multiple=True, help='Column labels to extract')
@click.option('--labels-file', type=click.Path(exists=True))
# Core options
@click.option('--server-url', envvar='TEXT2TABLE_VLLM_URL', required=True)
@click.option('--gliner-url', envvar='TEXT2TABLE_GLINER_URL')
@click.option('--enable-thinking', is_flag=True)
# Processing options
@click.option('--threshold', default=0.5, type=float)
@click.option('--max-new-tokens', default=4096, type=int)
@click.option('--request-timeout', default=600, type=int)
# Batch options (only apply when input is tabular)
@click.option('--text-column', default='text', help='Column containing text')
@click.option('--id-column', help='Column for record ID')
@click.option('--concurrency', default=4, type=int)
def text2table(input, text, output, label, labels_file, server_url, ...):
    """Extract structured tables from text.

    INPUT can be:
    - Omitted (use --text for direct input)
    - A .txt file (single text)
    - A .tsv/.csv file (batch mode)
    - A .jsonl file (batch mode)

    OUTPUT format determined by extension:
    - .tsv/.csv: Tab/comma-separated output
    - .jsonl: JSON Lines output
    - Omitted: TSV to stdout
    """
    # Detect input mode
    if text:
        mode = 'single'
        texts = [text]
    elif input:
        ext = Path(input).suffix.lower()
        if ext == '.txt':
            mode = 'single'
            texts = [Path(input).read_text()]
        elif ext in ('.tsv', '.csv'):
            mode = 'batch'
            texts = load_tabular(input, text_column)
        elif ext == '.jsonl':
            mode = 'batch'
            texts = load_jsonl(input, text_column)
        else:
            raise click.BadParameter(f"Unsupported input format: {ext}")
    else:
        raise click.UsageError("Either INPUT file or --text required")

    # Detect output format
    if output:
        out_ext = Path(output).suffix.lower()
        out_format = 'jsonl' if out_ext == '.jsonl' else 'tsv'
    else:
        out_format = 'tsv'  # stdout default

    # Process
    if mode == 'single':
        result = run_single(texts[0], labels, ...)
        output_result(result, output, out_format)
    else:
        results = run_batch(texts, labels, concurrency, ...)
        output_results(results, output, out_format)
```

#### 4.2.3 Shared Parameter Handling

Create a dataclass for shared parameters:

```python
@dataclass
class Text2TableConfig:
    """Shared configuration for text2table operations."""
    server_url: str
    gliner_url: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    threshold: float = 0.5
    max_new_tokens: int = 4096
    request_timeout: int = 600
    enable_thinking: bool = False
    enable_row_validation: bool = False
    row_validation_mode: str = "substring"
    gliner_model: str = "Ihor/gliner-biomed-large-v1.0"
    gliner_soft_threshold: float = 0.3

    @classmethod
    def from_env(cls) -> 'Text2TableConfig':
        """Load config from environment variables."""
        return cls(
            server_url=os.environ.get('TEXT2TABLE_VLLM_URL', ''),
            gliner_url=os.environ.get('TEXT2TABLE_GLINER_URL'),
            # ... etc
        )

    @classmethod
    def from_click_context(cls, ctx: click.Context) -> 'Text2TableConfig':
        """Load config from Click context."""
        params = ctx.params
        return cls(
            server_url=params['server_url'],
            # ... etc
        )
```

#### 4.2.4 Input/Output Format Detection

```python
def detect_input_format(path: Path) -> tuple[str, Callable]:
    """Detect input format and return loader function."""
    ext = path.suffix.lower()
    if ext == '.txt':
        return 'single', lambda p: [p.read_text()]
    elif ext == '.tsv':
        return 'batch', lambda p: load_tsv(p)
    elif ext == '.csv':
        return 'batch', lambda p: load_csv(p)
    elif ext == '.jsonl':
        return 'batch', lambda p: load_jsonl(p)
    else:
        raise ValueError(f"Unsupported format: {ext}")

def detect_output_format(path: Optional[Path]) -> str:
    """Detect output format from path extension."""
    if path is None:
        return 'tsv'  # stdout default
    ext = path.suffix.lower()
    if ext == '.jsonl':
        return 'jsonl'
    elif ext == '.csv':
        return 'csv'
    else:
        return 'tsv'  # default for .tsv and unknown
```

### 4.3 Parameter Consolidation

Remove duplicated parameters and use defaults:

| Parameter | New Default | Notes |
|-----------|-------------|-------|
| `--max-new-tokens` | 4096 | Unified across all modes |
| `--request-timeout` | 600 | Unified (was 120 for single, 600 for batch) |
| `--concurrency` | 4 | Only applies to batch mode |
| `--gliner-soft-threshold` | 0.3 | Remove duplicate definition |

### 4.4 Entry Points Update

Update `pyproject.toml`:

```toml
[project.scripts]
# Remove: text2table-server = "rust_research_py.text2table.cli:text2table_server"
t2t-server = "rust_research_py.text2table.server:main"
t2t = "rust_research_py.text2table.cli:text2table"
```

### 4.5 Files to Modify

| File | Changes |
|------|---------|
| `python/rust_research_py/text2table/cli.py` | Merge run/run-batch, add format detection |
| `python/rust_research_py/text2table/server.py` | Simplify to bare vLLM starter |
| `python/rust_research_py/text2table/config.py` | New - shared config dataclass |
| `python/pyproject.toml` | Update entry points |
| `src/tools/text2table.rs` | Update to match new Python API |
| `src/python_embed.rs` | Simplify Python calls |

---

## 5. Implementation Priority

### Phase 1: High Impact, Low Risk
1. **t2t-cli consolidation** - Pure Python, no Rust changes
2. **t2t-server simplification** - Pure Python
3. **pdf_metadata rename** - Minimal Rust changes

### Phase 2: Medium Complexity
4. **verify_metadata** new tool - New Rust code, uses existing providers
5. **search_source** refactoring - Provider changes needed

### Phase 3: Higher Complexity
6. **download race-mode** - Significant architecture change
7. **download method abstraction** - New trait system

### Estimated Effort

| Component | Effort | Risk |
|-----------|--------|------|
| t2t-cli/server | 2-3 days | Low |
| pdf_metadata rename | 1 day | Low |
| verify_metadata | 3-4 days | Medium |
| search_source | 4-5 days | Medium |
| download refactor | 5-7 days | High |

---

## 6. Migration Strategy

### 6.1 Backward Compatibility

1. **Deprecated tools** - Keep old tool names as aliases for 2 versions
2. **Warning messages** - Log deprecation warnings when old tools used
3. **Documentation** - Update examples in CLAUDE.md and README

### 6.2 Testing Strategy

1. **Unit tests** - Each new tool/method
2. **Integration tests** - End-to-end workflows
3. **Regression tests** - Ensure old functionality preserved
4. **Performance tests** - Especially for download race mode

### 6.3 Rollout Plan

1. Implement in feature branch
2. Add comprehensive tests
3. Document changes in CHANGELOG.md
4. Tag as minor version bump (breaking changes = major bump)
5. Update CLAUDE.md with new tool documentation

---

## Appendix A: New File Structure

```
src/
├── tools/
│   ├── search_source.rs      # NEW: Single-source search
│   ├── list_sources.rs       # NEW: Source discovery
│   ├── pdf_metadata.rs       # RENAMED from metadata.rs
│   ├── verify_metadata.rs    # NEW: Metadata verification
│   ├── download.rs           # REFACTORED: Race-based download
│   └── ...
├── client/
│   ├── download_methods/     # NEW: Download method implementations
│   │   ├── mod.rs
│   │   ├── traits.rs
│   │   ├── direct_http.rs
│   │   ├── cdp.rs
│   │   ├── arxiv.rs
│   │   └── unpaywall.rs
│   ├── providers/            # MODIFIED: Add native query support
│   └── ...
python/
└── rust_research_py/
    └── text2table/
        ├── cli.py            # SIMPLIFIED: Single command
        ├── server.py         # SIMPLIFIED: vLLM only
        ├── config.py         # NEW: Shared config
        └── ...
```

## Appendix B: Schema Examples

### B.1 search_source Example

```json
{
  "source": "pubmed",
  "query": "COVID-19[Title] AND vaccine[MeSH] AND 2024[Year]",
  "limit": 20
}
```

### B.2 verify_metadata Example

```json
{
  "records": [
    {
      "id": "paper_1",
      "doi": "10.1000/suspected",
      "pmid": "12345678",
      "title": "A Paper About Something"
    }
  ],
  "output_mode": "corrected",
  "verification_sources": ["crossref", "pubmed"]
}
```

### B.3 download with race mode Example

```json
{
  "doi": "10.1038/nature12373",
  "methods": ["unpaywall", "direct_http", "cdp"],
  "race_mode": true,
  "directory": "~/papers",
  "output_format": "pdf"
}
```
