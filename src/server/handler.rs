use crate::client::providers::{
    ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, GoogleScholarProvider,
    MdpiProvider, MedrxivProvider, OpenAlexProvider, OpenReviewProvider, PubMedCentralProvider,
    PubMedProvider, ResearchGateProvider, SciHubProvider, SemanticScholarProvider, SsrnProvider,
    UnpaywallProvider,
};
use crate::tools::{
    bibliography::BibliographyInput,
    code_search::CodeSearchInput,
    download::{
        BatchDownloadInput as ActualBatchDownloadInput, DownloadInput as ActualDownloadInput,
        DownloadOutputFormat,
    },
    list_sources::{ListSourcesInput, ListSourcesTool},
    pdf_metadata::MetadataInput as ActualMetadataInput,
    search_source::{SearchSourceInput, SearchSourceTool},
    verify_metadata::{VerifyMetadataInput, VerifyMetadataTool},
};
use crate::{
    BibliographyTool, CodeSearchTool, Config, DownloadTool, MetaSearchClient, MetadataExtractor,
    Result,
};
use chrono::Utc;
use rmcp::{
    model::{
        CallToolRequestParam, CallToolResult, Content, Implementation, InitializeRequestParam,
        InitializeResult, ListToolsResult, PaginatedRequestParam, ProtocolVersion,
        ServerCapabilities, ServerInfo, Tool,
    },
    service::{RequestContext, RoleServer},
    ErrorData, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    future::Future,
    sync::Arc,
    time::SystemTime,
};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

// Tool input structures
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DownloadInput {
    /// DOI or URL of the paper to download
    pub identifier: String,
    /// Optional output directory
    pub output_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MetadataInput {
    /// Path to the PDF file or DOI
    pub input: String,
}

/// Cache entry for paper categories from recent searches
#[derive(Debug, Clone)]
struct CategoryCacheEntry {
    category: Option<String>,
    #[allow(dead_code)]
    timestamp: SystemTime,
}

/// Main MCP server handler implementing rmcp
#[derive(Debug)]
pub struct ResearchServerHandler {
    #[allow(dead_code)]
    config: Arc<Config>,
    download_tool: Arc<DownloadTool>,
    metadata_extractor: Arc<MetadataExtractor>,
    code_search_tool: Arc<CodeSearchTool>,
    bibliography_tool: Arc<BibliographyTool>,
    verify_metadata_tool: Arc<VerifyMetadataTool>,
    search_source_tool: Arc<SearchSourceTool>,
    list_sources_tool: Arc<ListSourcesTool>,
    /// Cache of DOI -> Category mappings from recent searches
    category_cache: Arc<RwLock<HashMap<String, CategoryCacheEntry>>>,
}

impl ResearchServerHandler {
    pub fn new(config: Arc<Config>) -> Result<Self> {
        info!("Initializing Research MCP server handler");

        // Initialize MetaSearch client with config
        let meta_config = crate::client::MetaSearchConfig::from_config(&config);
        let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);

        // Initialize download tool
        let download_tool = DownloadTool::new(client, config.clone())?;

        // Initialize metadata extractor
        let metadata_extractor = MetadataExtractor::new(config.clone())?;

        // Initialize code search tool
        let code_search_tool = CodeSearchTool::new(config.clone())?;

        // Initialize bibliography tool
        let bibliography_tool = BibliographyTool::new(config.clone())?;

        // Initialize verify metadata tool
        let verify_metadata_tool = VerifyMetadataTool::new();

        // Initialize search source tool with providers
        let mut search_source_tool = SearchSourceTool::new();

        // Initialize list sources tool with providers
        let mut list_sources_tool = ListSourcesTool::new();

        // Register all providers with both tools
        // Note: Each provider needs to be created twice since Box<dyn SourceProvider>
        // cannot be cloned (trait objects don't support Clone)
        macro_rules! register_providers {
            ($tool:expr, $($provider:expr),+ $(,)?) => {
                $(
                    if let Ok(p) = $provider {
                        $tool.register_provider(Box::new(p));
                    }
                )+
            };
        }

        register_providers!(
            search_source_tool,
            ArxivProvider::new(),
            CrossRefProvider::new(None),
            SemanticScholarProvider::new(None),
            PubMedCentralProvider::new(None),
            PubMedProvider::new(None),
            OpenAlexProvider::new(),
            BiorxivProvider::new(),
            MedrxivProvider::new(),
            OpenReviewProvider::new(),
            CoreProvider::new(None),
            MdpiProvider::new(),
            SsrnProvider::new(),
            UnpaywallProvider::new_with_default_email(),
            ResearchGateProvider::new(),
            GoogleScholarProvider::new(std::env::var("GOOGLE_SCHOLAR_API_KEY").ok()),
            SciHubProvider::new()
        );

        register_providers!(
            list_sources_tool,
            ArxivProvider::new(),
            CrossRefProvider::new(None),
            SemanticScholarProvider::new(None),
            PubMedCentralProvider::new(None),
            PubMedProvider::new(None),
            OpenAlexProvider::new(),
            BiorxivProvider::new(),
            MedrxivProvider::new(),
            OpenReviewProvider::new(),
            CoreProvider::new(None),
            MdpiProvider::new(),
            SsrnProvider::new(),
            UnpaywallProvider::new_with_default_email(),
            ResearchGateProvider::new(),
            GoogleScholarProvider::new(std::env::var("GOOGLE_SCHOLAR_API_KEY").ok()),
            SciHubProvider::new()
        );

        Ok(Self {
            config,
            download_tool: Arc::new(download_tool),
            metadata_extractor: Arc::new(metadata_extractor),
            code_search_tool: Arc::new(code_search_tool),
            bibliography_tool: Arc::new(bibliography_tool),
            verify_metadata_tool: Arc::new(verify_metadata_tool),
            search_source_tool: Arc::new(search_source_tool),
            list_sources_tool: Arc::new(list_sources_tool),
            category_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Health check for the server
    #[instrument(skip(self))]
    pub async fn ping(&self) -> Result<()> {
        debug!("Ping received - server is healthy");
        Ok(())
    }

    /// Get cached category for a DOI
    async fn get_cached_category(&self, doi: &str) -> Option<String> {
        let cache = self.category_cache.read().await;
        if let Some(entry) = cache.get(doi) {
            debug!(
                "Found cached category '{}' for DOI '{}'",
                entry.category.as_deref().unwrap_or("None"),
                doi
            );
            entry.category.clone()
        } else {
            debug!("No cached category found for DOI '{}'", doi);
            None
        }
    }
}

impl ServerHandler for ResearchServerHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(format!("🔬 Research Hub MCP Server v{} - Enhanced academic paper search and retrieval.\n\nProvides tools to:\n• 🔍 Search across 12+ academic sources (arXiv, CrossRef, PubMed, etc.)\n• 📥 Download papers with intelligent fallback protection\n• 📝 Convert PDFs to Markdown via pdf2text on request\n• 📊 Extract metadata from PDFs\n• 🔍 Search code patterns in downloaded papers (NEW)\n• 📚 Generate citations in multiple formats (NEW)\n\nDesigned for personal academic research and Claude Code workflows.", env!("CARGO_PKG_VERSION"))),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }

    #[instrument(skip(self, request, context))]
    fn initialize(
        &self,
        request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> impl Future<Output = std::result::Result<InitializeResult, ErrorData>> + Send + '_ {
        info!("MCP server initializing");

        async move {
            // Set peer info if not already set
            if context.peer.peer_info().is_none() {
                context.peer.set_peer_info(request);
            }

            Ok(InitializeResult {
                protocol_version: ProtocolVersion::default(),
                capabilities: ServerCapabilities::builder().enable_tools().build(),
                server_info: Implementation {
                    name: "knowledge_accumulator_mcp".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
                instructions: Some("A MCP server for accumulating and organizing academic knowledge. Provides tools to search, download, and categorize academic papers.".into()),
            })
        }
    }

    #[instrument(skip(self, _request, _context))]
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = std::result::Result<ListToolsResult, ErrorData>> + Send + '_ {
        info!("Listing available tools");

        async move {
            let tools = vec![
                Tool {
                    name: "debug_test".into(), 
                    description: Some("Simple test tool for debugging - just echoes back what it receives".into()),
                    input_schema: Arc::new(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Test message to echo back"
                            }
                        },
                        "required": ["message"]
                    }).as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "download_paper".into(), 
                    description: Some("Download a paper PDF by DOI with plugin fallback. Optionally render Markdown via pdf2text.".into()),
                    input_schema: Arc::new(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "doi": {
                                "type": "string",
                                "description": "DOI of the paper to download (e.g., '10.1038/nature12373')"
                            },
                            "filename": {
                                "type": "string", 
                                "description": "Optional custom filename for the downloaded PDF"
                            },
                            "format": {
                                "type": "string",
                                "description": "Output format: 'pdf' (default) or 'markdown' (runs pdf2text after download)",
                                "enum": ["pdf", "markdown"],
                                "default": "pdf"
                            }
                        },
                        "required": ["doi"]
                    }).as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "download_papers_batch".into(),
                    description: Some("Download multiple papers concurrently (MAX 100 papers per batch, 1-20 concurrent). For >100 papers, split into multiple batches. 5-10x faster than individual downloads.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(ActualBatchDownloadInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "search_code".into(),
                    description: Some("Search for code patterns within downloaded research papers using regex".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(CodeSearchInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "generate_bibliography".into(),
                    description: Some("Generate citations from DOIs with parallel fetching (processes unlimited DOIs with 30 concurrent fetches). Supports BibTeX, APA, MLA, Chicago, IEEE, Harvard formats.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(BibliographyInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "verify_metadata".into(),
                    description: Some("Verify and enrich paper metadata by cross-referencing multiple academic sources (CrossRef, PubMed, Semantic Scholar, OpenAlex). Identifies discrepancies and provides confidence scores.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(VerifyMetadataInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "search_source".into(),
                    description: Some("Search a specific academic source using its native query syntax. Use help=true to get query format documentation for any source.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(SearchSourceInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                Tool {
                    name: "list_sources".into(),
                    description: Some("List all available academic sources with their capabilities, query syntax, and examples. Filter by full_text or metadata_only.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(ListSourcesInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
                // New: pdf_metadata (replaces extract_metadata)
                Tool {
                    name: "pdf_metadata".into(),
                    description: Some("Extract metadata from PDF files. Single file or batch processing (12 concurrent for batch_files array). Returns title, authors, DOI, abstract, etc.".into()),
                    input_schema: Arc::new(serde_json::to_value(schemars::schema_for!(ActualMetadataInput)).unwrap().as_object().unwrap().clone()),
                    output_schema: None,
                    annotations: None,
                },
            ];

            Ok(ListToolsResult {
                tools,
                next_cursor: None,
            })
        }
    }

    #[instrument(skip(self, request, _context))]
    fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = std::result::Result<CallToolResult, ErrorData>> + Send + '_ {
        info!("Tool called: {}", request.name);

        let download_tool = Arc::clone(&self.download_tool);
        let metadata_extractor = Arc::clone(&self.metadata_extractor);
        let code_search_tool = Arc::clone(&self.code_search_tool);
        let bibliography_tool = Arc::clone(&self.bibliography_tool);
        let verify_metadata_tool = Arc::clone(&self.verify_metadata_tool);
        let search_source_tool = Arc::clone(&self.search_source_tool);
        let list_sources_tool = Arc::clone(&self.list_sources_tool);

        async move {
            match request.name.as_ref() {
                "debug_test" => {
                    info!("Debug tool called with arguments: {:?}", request.arguments);
                    let message = request
                        .arguments
                        .and_then(|args| {
                            args.get("message")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                        })
                        .unwrap_or_else(|| "No message provided".to_string());

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(format!("Debug echo: {message}"))]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                "download_paper" => {
                    // Simple parsing for simplified schema
                    let args = request.arguments.unwrap_or_default();
                    let doi = args.get("doi").and_then(|v| v.as_str()).ok_or_else(|| {
                        ErrorData::invalid_params(
                            "Missing required 'doi' parameter".to_string(),
                            None,
                        )
                    })?;
                    let filename = args
                        .get("filename")
                        .and_then(|v| v.as_str())
                        .map(ToString::to_string);
                    let output_format = args
                        .get("format")
                        .and_then(|v| v.as_str())
                        .map(|v| match v.to_lowercase().as_str() {
                            "markdown" => DownloadOutputFormat::Markdown,
                            _ => DownloadOutputFormat::Pdf,
                        })
                        .unwrap_or(DownloadOutputFormat::Pdf);

                    // Look up category from recent search results
                    let category = self.get_cached_category(doi).await;

                    let input = ActualDownloadInput {
                        doi: Some(doi.to_string()),
                        url: None,
                        filename,
                        directory: None,
                        category,
                        overwrite: false,
                        verify_integrity: true,
                        output_format,
                        headless: true,
                        enable_local_grobid: false,
                    };

                    debug!("Attempting download with input: {:?}", input);
                    match download_tool.download_paper(input).await {
                        Ok(result) => {
                            debug!("Download result received: {:?}", result.status);
                            debug!(
                                "File size: {:?}, file path: {:?}",
                                result.file_size, result.file_path
                            );

                            // Validate that the file actually has content
                            let file_size = result.file_size.unwrap_or(0);
                            if file_size == 0 {
                                debug!("Download succeeded but file size is 0 - cleaning up");
                                // Clean up zero-byte file if it exists
                                if let Some(file_path) = &result.file_path {
                                    if file_path.exists() {
                                        debug!("Removing zero-byte file: {:?}", file_path);
                                        let _ = std::fs::remove_file(file_path);
                                    }
                                }
                                Ok(CallToolResult {
                                    content: Some(vec![Content::text(format!("⚠️ Download failed - no content received\n\nDOI: {doi}\n\n🔍 Debug Info:\n• Download ID: {}\n• Duration: {:.2}s\n• Status: {:?}\n• File created but empty\n\nThe paper was found but no downloadable content is available. This could be because:\n• The paper is too new or recently published\n• It's behind a paywall not covered by available sources\n• The DOI might be incorrect\n• Network issues during download\n\nTry checking the publisher's website or your institutional access.",
                                        result.download_id, result.duration_seconds, result.status))]),
                                    structured_content: None,
                                    is_error: Some(true),
                                })
                            } else {
                                debug!("Download successful - file size: {} bytes", file_size);
                                let duration_info = if result.duration_seconds > 0.0 {
                                    format!(
                                        "\n⏱️ Time: {:.1}s\n🚀 Speed: {:.1} KB/s",
                                        result.duration_seconds,
                                        result.average_speed as f64 / 1024.0
                                    )
                                } else {
                                    String::new()
                                };

                                let hash_info = result
                                    .sha256_hash
                                    .map(|h| format!("\n🔐 SHA256: {}...", &h[..16]))
                                    .unwrap_or_default();

                                let mut success_message = format!(
                                    "✅ Download successful!\n\n📄 File: {}\n📦 Size: {} KB{}{}",
                                    result
                                        .file_path
                                        .as_ref()
                                        .map_or("Unknown".to_string(), |p| p.display().to_string()),
                                    file_size / 1024,
                                    duration_info,
                                    hash_info
                                );

                                if let Some(plugin) = result.used_plugin.as_deref() {
                                    success_message
                                        .push_str(&format!("\n🔌 Plugin fallback: {plugin}"));
                                }

                                if let Some(markdown_path) = &result.markdown_path {
                                    success_message.push_str(&format!(
                                        "\n📝 Markdown: {}",
                                        markdown_path.display()
                                    ));
                                }

                                if let Some(warning) = &result.post_process_error {
                                    success_message.push_str(&format!(
                                        "\n⚠️ Post-processing warning: {}",
                                        warning
                                    ));
                                }

                                Ok(CallToolResult {
                                    content: Some(vec![Content::text(success_message)]),
                                    structured_content: None,
                                    is_error: Some(false),
                                })
                            }
                        }
                        Err(e) => {
                            debug!("Download failed with error: {}", e);
                            debug!("Error type: {:?}", std::any::type_name_of_val(&e));

                            // Generate timestamp for debugging
                            let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S UTC");

                            // Return a helpful error message with debug information
                            let error_msg = match e.to_string().as_str() {
                                msg if msg.contains("No PDF available")
                                    || msg.contains("not found in any provider") =>
                                {
                                    format!("⚠️ Paper not available for download\n\n\
                                            DOI: {doi}\n\n\
                                            🔍 Debug Info:\n\
                                            • Time: {}\n\
                                            • Error: {}\n\
                                            • All available sources checked\n\n\
                                            This paper is not currently available through any source. This could be because:\n\
                                            • The paper is too new (published recently)\n\
                                            • It's from a publisher not covered by available sources\n\
                                            • The DOI might be incorrect or malformed\n\
                                            • Temporary service unavailability\n\n\
                                            💡 Alternatives:\n\
                                            • Try searching for the paper on Google Scholar\n\
                                            • Check if your institution has access\n\
                                            • Try arXiv or other preprint servers\n\
                                            • Contact the authors directly\n\
                                            • Verify the DOI is correct", timestamp, msg)
                                }
                                msg if msg.contains("Network")
                                    || msg.contains("timeout")
                                    || msg.contains("Connection") =>
                                {
                                    format!(
                                        "⚠️ Network error while downloading\n\n\
                                            DOI: {doi}\n\n\
                                            🔍 Debug Info:\n\
                                            • Time: {}\n\
                                            • Error: {}\n\
                                            • Network connectivity issue detected\n\n\
                                            Please check your internet connection and try again.\n\
                                            If the problem persists, the source servers may be temporarily unavailable.",
                                        timestamp, msg
                                    )
                                }
                                msg if msg.contains("Permission")
                                    || msg.contains("Claude Desktop") =>
                                {
                                    format!(
                                        "⚠️ File system permission error\n\n\
                                            DOI: {doi}\n\n\
                                            🔍 Debug Info:\n\
                                            • Time: {}\n\
                                            • Error: {}\n\n\
                                            This appears to be a permission issue with accessing the download directory.\n\
                                            Please check the error message for specific instructions to resolve.",
                                        timestamp, msg
                                    )
                                }
                                _ => {
                                    format!(
                                        "⚠️ Download failed\n\n\
                                            DOI: {doi}\n\n\
                                            🔍 Debug Info:\n\
                                            • Time: {}\n\
                                            • Error Type: {}\n\
                                            • Error: {}\n\n\
                                            Please try again or use a different DOI. If this error persists,\n\
                                            it may indicate an issue with the paper source or network connectivity.",
                                        timestamp, std::any::type_name_of_val(&e), e
                                    )
                                }
                            };
                            Ok(CallToolResult {
                                content: Some(vec![Content::text(error_msg)]),
                                structured_content: None,
                                is_error: Some(true),
                            })
                        }
                    }
                }
                "download_papers_batch" => {
                    let input: ActualBatchDownloadInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(
                            format!("Invalid batch download input: {e}"),
                            None,
                        )
                    })?;

                    debug!("Starting batch download with {} papers", input.papers.len());
                    match download_tool.download_papers_batch(input).await {
                        Ok(result) => {
                            debug!(
                                "Batch download completed: {}/{} successful",
                                result.summary.successful, result.summary.total_requested
                            );

                            let success_rate = if result.summary.total_requested > 0 {
                                (result.summary.successful as f64
                                    / result.summary.total_requested as f64)
                                    * 100.0
                            } else {
                                0.0
                            };

                            let mut content = format!(
                                "✅ Batch Download Complete!\n\n\
                                📊 Summary:\n\
                                • Total requested: {}\n\
                                • Successful: {} ({:.1}%)\n\
                                • Failed: {}\n\
                                • Skipped: {}\n\
                                • Total time: {:.1}s\n\
                                • Total data: {:.1} MB\n\
                                • Average speed: {:.1} KB/s\n",
                                result.summary.total_requested,
                                result.summary.successful,
                                success_rate,
                                result.summary.failed,
                                result.summary.skipped,
                                result.total_duration_seconds,
                                result.summary.total_bytes as f64 / 1_048_576.0, // Convert to MB
                                result.summary.average_speed as f64 / 1024.0     // Convert to KB/s
                            );

                            // Add details about successful downloads
                            let successful_downloads: Vec<_> = result
                                .results
                                .iter()
                                .filter(|r| r.result.is_some())
                                .collect();

                            if !successful_downloads.is_empty() {
                                content.push_str("\n📁 Downloaded Papers:\n");
                                for item in successful_downloads.iter().take(10) {
                                    // Limit to first 10 for readability
                                    if let Some(ref download_result) = item.result {
                                        if let Some(ref file_path) = download_result.file_path {
                                            let file_name = file_path
                                                .file_name()
                                                .and_then(|name| name.to_str())
                                                .unwrap_or("unknown");
                                            let size_mb = download_result.file_size.unwrap_or(0)
                                                as f64
                                                / 1_048_576.0;
                                            content.push_str(&format!(
                                                "• {} ({:.1} MB)\n",
                                                file_name, size_mb
                                            ));
                                        }
                                    }
                                }
                                if successful_downloads.len() > 10 {
                                    content.push_str(&format!(
                                        "• ... and {} more files\n",
                                        successful_downloads.len() - 10
                                    ));
                                }
                            }

                            // Add error details if there were failures
                            if result.summary.failed > 0 && !result.summary.failed_items.is_empty()
                            {
                                content.push_str("\n❌ Failed Downloads:\n");
                                for failed_item in result.summary.failed_items.iter().take(5) {
                                    // Limit to first 5
                                    content.push_str(&format!("• {}\n", failed_item));
                                }
                                if result.summary.failed_items.len() > 5 {
                                    content.push_str(&format!(
                                        "• ... and {} more failures\n",
                                        result.summary.failed_items.len() - 5
                                    ));
                                }
                            }

                            Ok(CallToolResult {
                                content: Some(vec![Content::text(content)]),
                                structured_content: None,
                                is_error: Some(result.summary.failed > result.summary.successful),
                            })
                        }
                        Err(e) => {
                            debug!("Batch download failed: {}", e);
                            let error_msg = format!(
                                "⚠️ Batch download failed\n\n\
                                Error: {}\n\n\
                                This could be due to:\n\
                                • Invalid input parameters\n\
                                • Network connectivity issues\n\
                                • Resource constraints\n\
                                • Provider limitations\n\n\
                                Please check your input and try again.",
                                e
                            );

                            Ok(CallToolResult {
                                content: Some(vec![Content::text(error_msg)]),
                                structured_content: None,
                                is_error: Some(true),
                            })
                        }
                    }
                }
                "pdf_metadata" => {
                    // New canonical name for extract_metadata
                    let input: ActualMetadataInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid metadata input: {e}"), None)
                    })?;

                    let result = metadata_extractor
                        .extract_metadata(input)
                        .await
                        .map_err(|e| {
                            ErrorData::internal_error(
                                format!("Metadata extraction failed: {e}"),
                                None,
                            )
                        })?;

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(
                            serde_json::to_string_pretty(&result).map_err(|e| {
                                ErrorData::internal_error(
                                    format!("Serialization failed: {e}"),
                                    None,
                                )
                            })?,
                        )]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                "search_code" => {
                    let input: CodeSearchInput = serde_json::from_value(serde_json::Value::Object(
                        request.arguments.unwrap_or_default(),
                    ))
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid code search input: {e}"), None)
                    })?;

                    let results = code_search_tool.search(input).await.map_err(|e| {
                        ErrorData::internal_error(format!("Code search failed: {e}"), None)
                    })?;

                    if results.is_empty() {
                        Ok(CallToolResult {
                            content: Some(vec![Content::text(
                                "🔍 No code patterns found matching your search criteria."
                                    .to_string(),
                            )]),
                            structured_content: None,
                            is_error: Some(false),
                        })
                    } else {
                        let formatted_results = results
                            .iter()
                            .map(|result| {
                                let matches_text = result
                                    .matches
                                    .iter()
                                    .take(5) // Limit to first 5 matches per file
                                    .map(|m| {
                                        let context_before = if m.context_before.is_empty() {
                                            String::new()
                                        } else {
                                            format!("  {}\n", m.context_before.join("\n  "))
                                        };

                                        let context_after = if m.context_after.is_empty() {
                                            String::new()
                                        } else {
                                            format!("\n  {}", m.context_after.join("\n  "))
                                        };

                                        let lang_info = m
                                            .language
                                            .as_ref()
                                            .map(|l| format!(" [{l}]"))
                                            .unwrap_or_default();

                                        format!(
                                            "{}► Line {}{}: {}{}",
                                            context_before,
                                            m.line_number,
                                            lang_info,
                                            m.line,
                                            context_after
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n\n");

                                let title_info = result
                                    .paper_title
                                    .as_ref()
                                    .map(|t| format!("📄 Paper: {t}\n"))
                                    .unwrap_or_default();

                                format!(
                                    "📁 File: {}\n{}🎯 {} matches found:\n\n{}",
                                    result.file_path,
                                    title_info,
                                    result.total_matches,
                                    matches_text
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(&format!("\n\n{}\n\n", "─".repeat(60)));

                        Ok(CallToolResult {
                            content: Some(vec![Content::text(format!(
                                "🔍 Found {} files with matching code patterns:\n\n{}",
                                results.len(),
                                formatted_results
                            ))]),
                            structured_content: None,
                            is_error: Some(false),
                        })
                    }
                }
                "generate_bibliography" => {
                    let input: BibliographyInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid bibliography input: {e}"), None)
                    })?;

                    let result = bibliography_tool.generate(input).await.map_err(|e| {
                        ErrorData::internal_error(
                            format!("Bibliography generation failed: {e}"),
                            None,
                        )
                    })?;

                    let mut output = format!(
                        "📚 Generated {} citations in {:?} format:\n\n",
                        result.citations.len(),
                        result.format
                    );

                    output.push_str(&result.bibliography);

                    if !result.errors.is_empty() {
                        output.push_str("\n\n⚠️ Errors encountered:\n");
                        for error in &result.errors {
                            output
                                .push_str(&format!("• {}: {}\n", error.identifier, error.message));
                        }
                    }

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(output)]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                "verify_metadata" => {
                    let input: VerifyMetadataInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid verify_metadata input: {e}"), None)
                    })?;

                    let result = verify_metadata_tool.verify(input).await.map_err(|e| {
                        ErrorData::internal_error(
                            format!("Metadata verification failed: {e}"),
                            None,
                        )
                    })?;

                    let mut output = format!(
                        "🔍 Metadata Verification Result\n\nStatus: {:?}\nConfidence: {:.1}%\nTime: {}ms\n",
                        result.status,
                        result.overall_confidence * 100.0,
                        result.total_time_ms
                    );

                    if !result.source_results.is_empty() {
                        output.push_str("\n📚 Source Results:\n");
                        for src in &result.source_results {
                            output.push_str(&format!(
                                "• {}: {} (confidence: {:.1}%, {}ms)\n",
                                src.source,
                                if src.success { "✓" } else { "✗" },
                                src.confidence * 100.0,
                                src.response_time_ms
                            ));
                        }
                    }

                    if let Some(meta) = &result.merged_metadata {
                        output.push_str("\n📄 Merged Metadata:\n");
                        if let Some(title) = &meta.title {
                            output.push_str(&format!("Title: {}\n", title));
                        }
                        if !meta.authors.is_empty() {
                            output.push_str(&format!("Authors: {}\n", meta.authors.join(", ")));
                        }
                        if let Some(year) = meta.year {
                            output.push_str(&format!("Year: {}\n", year));
                        }
                        if let Some(journal) = &meta.journal {
                            output.push_str(&format!("Journal: {}\n", journal));
                        }
                        if let Some(doi) = &meta.doi {
                            output.push_str(&format!("DOI: {}\n", doi));
                        }
                    }

                    if !result.discrepancies.is_empty() {
                        output.push_str("\n⚠️ Discrepancies Found:\n");
                        for disc in &result.discrepancies {
                            output.push_str(&format!("• {}: ", disc.field));
                            for val in &disc.values {
                                output.push_str(&format!("{}: \"{}\" ", val.source, val.value));
                            }
                            output.push('\n');
                        }
                    }

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(output)]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                "search_source" => {
                    let input: SearchSourceInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid search_source input: {e}"), None)
                    })?;

                    let result = search_source_tool.search(input).await.map_err(|e| {
                        ErrorData::internal_error(
                            format!("Source search failed: {e}"),
                            None,
                        )
                    })?;

                    // Check if this is a help request
                    if let Some(info) = &result.source_info {
                        let mut output = format!(
                            "📖 {} Query Help\n\n{}\n\n🔍 Query Format:\n{}\n",
                            info.name,
                            info.description,
                            info.query_format_help
                        );

                        if !info.query_examples.is_empty() {
                            output.push_str("\n📝 Examples:\n");
                            for example in &info.query_examples {
                                output.push_str(&format!("• {} - {}\n", example.query, example.description));
                            }
                        }

                        if !info.supported_search_types.is_empty() {
                            output.push_str(&format!(
                                "\n🏷️ Supported Types: {}\n",
                                info.supported_search_types.join(", ")
                            ));
                        }

                        if let Some(syntax) = &info.native_query_syntax {
                            output.push_str(&format!("\n📚 Native Syntax:\n{}\n", syntax));
                        }

                        return Ok(CallToolResult {
                            content: Some(vec![Content::text(output)]),
                            structured_content: None,
                            is_error: Some(false),
                        });
                    }

                    // Regular search result
                    let mut output = format!(
                        "🔎 Search Results from {}\nQuery: \"{}\"\nFound: {} papers",
                        result.source,
                        result.query,
                        result.papers.len()
                    );

                    if let Some(total) = result.total_available {
                        output.push_str(&format!(" (of {} total)", total));
                    }
                    output.push_str(&format!("\nTime: {}ms\n\n", result.search_time_ms));

                    for (i, paper) in result.papers.iter().enumerate() {
                        let title = paper.title.as_deref().unwrap_or("Untitled");
                        let authors = if paper.authors.is_empty() {
                            "Unknown".to_string()
                        } else {
                            paper.authors.join(", ")
                        };
                        let year = paper.year.map(|y| y.to_string()).unwrap_or_else(|| "N/A".to_string());
                        let doi = if paper.doi.is_empty() { "N/A" } else { &paper.doi };

                        output.push_str(&format!(
                            "{}. {}\n   Authors: {}\n   Year: {}\n   DOI: {}\n\n",
                            i + 1, title, authors, year, doi
                        ));
                    }

                    if result.has_more {
                        output.push_str("📄 More results available. Use offset parameter to paginate.\n");
                    }

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(output)]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                "list_sources" => {
                    let input: ListSourcesInput = serde_json::from_value(
                        serde_json::Value::Object(request.arguments.unwrap_or_default()),
                    )
                    .map_err(|e| {
                        ErrorData::invalid_params(format!("Invalid list_sources input: {e}"), None)
                    })?;

                    let result = list_sources_tool.list(input).await.map_err(|e| {
                        ErrorData::internal_error(
                            format!("Failed to list sources: {e}"),
                            None,
                        )
                    })?;

                    let mut output = format!(
                        "📚 Available Academic Sources\n\nTotal: {} sources\n",
                        result.total
                    );

                    if result.total == 0 {
                        output.push_str("\nNo sources available. Check server configuration.\n");
                    } else {
                        output.push('\n');
                        for source in &result.sources {
                            let full_text_badge = if source.supports_full_text {
                                "📄 Full-text"
                            } else {
                                "📋 Metadata-only"
                            };

                            let health_badge = match source.healthy {
                                Some(true) => " ✅ Healthy",
                                Some(false) => " ❌ Unhealthy",
                                None => "",
                            };

                            output.push_str(&format!(
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
                                🔍 {} (priority: {}){}\n\
                                {}\n\
                                {}\n\n\
                                📝 Query Format:\n{}\n",
                                source.name,
                                source.priority,
                                health_badge,
                                full_text_badge,
                                source.description,
                                source.query_format_help
                            ));

                            if let Some(examples) = &source.query_examples {
                                output.push_str("\n💡 Examples:\n");
                                for example in examples.iter().take(3) {
                                    output.push_str(&format!(
                                        "  • {} - {}\n",
                                        example.query, example.description
                                    ));
                                }
                            }

                            if let Some(syntax_url) = &source.native_query_syntax {
                                output.push_str(&format!("\n🔗 Syntax Docs: {}\n", syntax_url));
                            }

                            output.push('\n');
                        }
                    }

                    output.push_str(&format!(
                        "\n🏷️ Available Filters: {}\n",
                        result.available_filters.join(", ")
                    ));

                    Ok(CallToolResult {
                        content: Some(vec![Content::text(output)]),
                        structured_content: None,
                        is_error: Some(false),
                    })
                }
                _ => Err(ErrorData::invalid_request(
                    format!("Unknown tool: {}", request.name),
                    None,
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_handler() -> ResearchServerHandler {
        let config = Config::default();
        ResearchServerHandler::new(Arc::new(config)).unwrap()
    }

    #[tokio::test]
    async fn test_handler_creation() {
        let handler = create_test_handler();
        assert!(handler.config.research_source.endpoints.len() > 0);
    }

    #[tokio::test]
    async fn test_ping() {
        let handler = create_test_handler();
        let result = handler.ping().await;
        assert!(result.is_ok());
    }
}
