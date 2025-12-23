#![allow(clippy::uninlined_format_args)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::ignored_unit_patterns)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use rust_research_mcp::{
    client::{
        providers::{
            ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, GoogleScholarProvider,
            MdpiProvider, MedrxivProvider, OpenAlexProvider, OpenReviewProvider,
            PubMedCentralProvider, PubMedProvider, ResearchGateProvider, SciHubProvider,
            SemanticScholarProvider, SsrnProvider, UnpaywallProvider,
        },
        MetaSearchClient, MetaSearchConfig,
    },
    tools::{
        download::{DownloadInput, DownloadOutputFormat, DownloadTool},
        pdf_metadata::{MetadataExtractor, MetadataInput},
        search_source::{SearchSourceInput, SearchSourceTool},
    },
    Config, ConfigOverrides,
};
use tracing::{error, info, Level};

#[derive(Parser)]
#[command(name = "rust-research")]
#[command(about = "Terminal CLI for search/download workflows")]
#[command(version)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Override download directory
    #[arg(long)]
    download_dir: Option<PathBuf>,

    /// Override log level (trace, debug, info, warn, error)
    #[arg(long)]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Search papers from a specific academic source (use 'list-sources' to see available sources)
    Search {
        /// Query text (use native query syntax for the source)
        query: String,
        /// Source to search (e.g., arxiv, pubmed, `pubmed_central`, `semantic_scholar`)
        #[arg(short, long)]
        source: String,
        /// Max results to return
        #[arg(short, long, default_value_t = 10)]
        limit: u32,
        /// Output file path (format detected from extension: .tsv, .json; default: TSV to stdout)
        #[arg(short, long)]
        output_file: Option<PathBuf>,
    },
    /// List available academic sources
    ListSources,
    /// Download a paper by DOI or direct URL
    Download {
        /// DOI of the paper
        #[arg(long)]
        doi: Option<String>,
        /// Direct PDF URL
        #[arg(long)]
        url: Option<String>,
        /// Custom filename
        #[arg(long)]
        filename: Option<String>,
        /// Target directory
        #[arg(long)]
        directory: Option<PathBuf>,
        /// Category folder
        #[arg(long)]
        category: Option<String>,
        /// Overwrite existing files
        #[arg(long)]
        overwrite: bool,
        /// Disable SHA256 verification
        #[arg(long)]
        no_verify: bool,
        /// Convert to Markdown after download
        #[arg(long)]
        markdown: bool,
        /// Disable headless browser mode (show browser window)
        #[arg(long)]
        disable_headless: bool,
        /// Use local GROBID server instead of public endpoint
        #[arg(long)]
        enable_local_grobid: bool,
    },
    /// Install Python dependencies
    Install,
    /// Extract metadata from a PDF file (local extraction only)
    Pdf2metadata {
        /// Path to a PDF file
        input: String,
    },
    /// Verify and reconcile metadata from external sources (CrossRef, PubMed, etc.)
    /// Input: TSV/JSON/JSONL file with metadata records (columns: id, doi, pmid, title, authors, year)
    /// Output: Verified/corrected metadata in same format
    VerifyMetadata {
        /// Input file path (TSV/JSON/JSONL), or use stdin if omitted
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Output file path, or stdout if omitted
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output mode: full, notes, or corrected
        #[arg(long, default_value = "corrected")]
        output_mode: String,
        /// Output format: tsv or json (auto-detected from output file extension if not specified)
        #[arg(long)]
        format: Option<String>,
        /// Sources to query: crossref, pubmed, semantic_scholar, openalex (comma-separated)
        #[arg(long)]
        sources: Option<String>,
        /// Maximum concurrent verifications
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
    /// Convert PDF to structured JSON and Markdown using GROBID
    Pdf2text {
        /// Path to a single PDF file
        #[arg(long)]
        pdf_file: Option<PathBuf>,
        /// Directory containing PDF files
        #[arg(long)]
        pdf_dir: Option<PathBuf>,
        /// Output directory for extracted content (required)
        #[arg(long)]
        output_dir: PathBuf,
        /// GROBID server URL
        #[arg(long)]
        grobid_url: Option<String>,
        /// Don't auto-start a local GROBID server
        #[arg(long)]
        no_auto_start: bool,
        /// Don't extract figures
        #[arg(long)]
        no_figures: bool,
        /// Don't extract tables
        #[arg(long)]
        no_tables: bool,
        /// Copy source PDF into output bundle
        #[arg(long)]
        copy_pdf: bool,
        /// Overwrite existing outputs
        #[arg(long)]
        overwrite: bool,
        /// Don't generate Markdown
        #[arg(long)]
        no_markdown: bool,
    },
    /// Start the text2table vLLM server
    T2TServer {
        /// Model name
        #[arg(long, default_value = "Qwen/Qwen3-30B-A3B-Instruct-2507")]
        model: String,

        /// Server host
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Server port
        #[arg(long, default_value_t = 8000)]
        port: u16,

        /// Number of GPUs for tensor parallelism
        #[arg(long, default_value_t = 1)]
        tensor_parallel_size: usize,

        /// GPU memory utilization (0.0 - 1.0)
        #[arg(long, default_value_t = 0.9)]
        gpu_memory_utilization: f32,

        /// Maximum model context length
        #[arg(long)]
        max_model_len: Option<usize>,

        /// Trust remote code (required for some models)
        #[arg(long, default_value_t = true)]
        trust_remote_code: bool,

        /// Model weights cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Run text2table extraction pipeline (batch processing for 1 to N records)
    T2TCli {
        /// Input file (.tsv, .csv, .jsonl) - batch processing
        #[arg(long, required = true)]
        input_file: PathBuf,

        /// Output file (format detected from extension: .tsv, .csv, .jsonl)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Labels to extract (repeat for multiple)
        #[arg(short, long)]
        label: Vec<String>,

        /// Path to file containing labels (one per line)
        #[arg(long)]
        labels_file: Option<PathBuf>,

        /// Column containing text (if not specified, all columns are used as key:value pairs)
        #[arg(long)]
        text_column: Option<String>,

        /// Column for record ID
        #[arg(long)]
        id_column: Option<String>,

        /// Concurrency limit
        #[arg(long, default_value_t = 4)]
        concurrency: usize,

        /// Custom user prompt
        #[arg(long)]
        prompt: Option<String>,

        /// `GLiNER` threshold
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,

        /// GLiNER model name
        #[arg(long, default_value = "Ihor/gliner-biomed-large-v1.0")]
        gliner_model: String,

        /// GLiNER soft threshold
        #[arg(long)]
        gliner_soft_threshold: Option<f64>,

        /// vLLM Model name
        #[arg(long)]
        model: Option<String>,

        /// Enable thinking mode
        #[arg(long)]
        enable_thinking: bool,

        /// vLLM Server URL
        #[arg(long, env = "TEXT2TABLE_VLLM_URL")]
        server_url: Option<String>,

        /// GLiNER Service URL
        #[arg(long, env = "TEXT2TABLE_GLINER_URL")]
        gliner_url: Option<String>,

        /// Disable GLiNER usage
        #[arg(long)]
        disable_gliner: bool,

        /// Enable row validation
        #[arg(long)]
        enable_row_validation: bool,

        /// Row validation mode
        #[arg(long, default_value = "substring")]
        row_validation_mode: String,

        /// API Key
        #[arg(long, env = "TEXT2TABLE_API_KEY")]
        api_key: Option<String>,

        /// GLiNER API Key
        #[arg(long, env = "TEXT2TABLE_GLINER_API_KEY")]
        gliner_api_key: Option<String>,
    },
}


/// Create a search_source tool with all providers registered
fn create_search_source_tool() -> SearchSourceTool {
    let mut tool = SearchSourceTool::new();

    // Register all providers
    macro_rules! register_provider {
        ($provider:expr) => {
            if let Ok(p) = $provider {
                tool.register_provider(Box::new(p));
            }
        };
    }

    register_provider!(ArxivProvider::new());
    register_provider!(CrossRefProvider::new(None));
    register_provider!(SemanticScholarProvider::new(None));
    register_provider!(PubMedCentralProvider::new(None));
    register_provider!(PubMedProvider::new(None));
    register_provider!(OpenAlexProvider::new());
    register_provider!(BiorxivProvider::new());
    register_provider!(MedrxivProvider::new());
    register_provider!(OpenReviewProvider::new());
    register_provider!(CoreProvider::new(None));
    register_provider!(MdpiProvider::new());
    register_provider!(SsrnProvider::new());
    register_provider!(UnpaywallProvider::new_with_default_email());
    register_provider!(ResearchGateProvider::new());
    register_provider!(GoogleScholarProvider::new(std::env::var("GOOGLE_SCHOLAR_API_KEY").ok()));
    register_provider!(SciHubProvider::new());

    tool
}

/// Metadata record for batch verification
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct MetadataRecord {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    doi: Option<String>,
    #[serde(default)]
    pmid: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    authors: Option<Vec<String>>,
    #[serde(default)]
    year: Option<i32>,
}

/// Parse metadata records from TSV/JSON/JSONL content
fn parse_metadata_records(content: &str, format: &str) -> anyhow::Result<Vec<MetadataRecord>> {
    match format.to_lowercase().as_str() {
        "json" => {
            // Try parsing as JSON array first
            if let Ok(records) = serde_json::from_str::<Vec<MetadataRecord>>(content) {
                return Ok(records);
            }
            // Try as single object
            let record: MetadataRecord = serde_json::from_str(content)?;
            Ok(vec![record])
        }
        "jsonl" => {
            let mut records = Vec::new();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let record: MetadataRecord = serde_json::from_str(line)?;
                records.push(record);
            }
            Ok(records)
        }
        _ => {
            // Parse TSV (default): id, doi, pmid, title, authors, year
            let mut records = Vec::new();
            let mut lines = content.lines();

            // Check for header
            let first_line = lines.next().unwrap_or("");
            let has_header = first_line.to_lowercase().contains("doi")
                || first_line.to_lowercase().contains("title")
                || first_line.to_lowercase().contains("pmid");

            let parse_line = |line: &str, idx: usize| -> Option<MetadataRecord> {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.is_empty() || parts.iter().all(|p| p.trim().is_empty()) {
                    return None;
                }

                let get_field = |i: usize| -> Option<String> {
                    parts.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()).map(ToString::to_string)
                };

                Some(MetadataRecord {
                    id: get_field(0).or_else(|| Some(format!("record_{}", idx))),
                    doi: get_field(1),
                    pmid: get_field(2),
                    title: get_field(3),
                    authors: get_field(4).map(|a| a.split(';').map(|s| s.trim().to_string()).collect()),
                    year: get_field(5).and_then(|y| y.parse().ok()),
                })
            };

            // Parse first line if not header
            if !has_header {
                if let Some(record) = parse_line(first_line, 0) {
                    records.push(record);
                }
            }

            // Parse remaining lines
            for (idx, line) in lines.enumerate() {
                if let Some(record) = parse_line(line, idx + 1) {
                    records.push(record);
                }
            }

            Ok(records)
        }
    }
}

/// Format verification results for output
fn format_verification_results(
    results: &[(Option<String>, rust_research_mcp::Result<rust_research_mcp::tools::verify_metadata::VerificationResult>)],
    format: &str,
) -> anyhow::Result<String> {
    use std::fmt::Write;
    match format.to_lowercase().as_str() {
        "json" => {
            let output: Vec<serde_json::Value> = results
                .iter()
                .map(|(id, result)| {
                    let mut obj = serde_json::json!({
                        "id": id,
                    });
                    match result {
                        Ok(r) => {
                            obj["status"] = serde_json::json!("success");
                            obj["result"] = serde_json::to_value(r).unwrap_or_default();
                        }
                        Err(e) => {
                            obj["status"] = serde_json::json!("error");
                            obj["error"] = serde_json::json!(e.to_string());
                        }
                    }
                    obj
                })
                .collect();
            Ok(serde_json::to_string_pretty(&output)?)
        }
        "jsonl" => {
            let lines: Vec<String> = results
                .iter()
                .map(|(id, result)| {
                    let mut obj = serde_json::json!({ "id": id });
                    match result {
                        Ok(r) => {
                            obj["status"] = serde_json::json!("success");
                            obj["result"] = serde_json::to_value(r).unwrap_or_default();
                        }
                        Err(e) => {
                            obj["status"] = serde_json::json!("error");
                            obj["error"] = serde_json::json!(e.to_string());
                        }
                    }
                    serde_json::to_string(&obj).unwrap_or_default()
                })
                .collect();
            Ok(lines.join("\n"))
        }
        _ => {
            // TSV output (default): id, status, doi, title, authors, year, journal, confidence
            let mut output = String::new();
            output.push_str("id\tstatus\tdoi\ttitle\tauthors\tyear\tjournal\tconfidence\n");

            for (id, result) in results {
                let id_str = id.as_deref().unwrap_or("");
                match result {
                    Ok(r) => {
                        let meta = r.merged_metadata.as_ref();
                        let doi = meta.and_then(|m| m.doi.as_deref()).unwrap_or("");
                        let title = meta.and_then(|m| m.title.as_deref()).unwrap_or("");
                        let authors = meta
                            .map(|m| m.authors.join("; "))
                            .unwrap_or_default();
                        let year = meta
                            .and_then(|m| m.year)
                            .map(|y| y.to_string())
                            .unwrap_or_default();
                        let journal = meta
                            .and_then(|m| m.journal.as_deref())
                            .unwrap_or("");
                        let confidence = format!("{:.2}", r.overall_confidence);
                        writeln!(
                            output,
                            "{}\tsuccess\t{}\t{}\t{}\t{}\t{}\t{}",
                            id_str, doi, title, authors, year, journal, confidence
                        )?;
                    }
                    Err(e) => {
                        writeln!(output, "{}\terror\t\t\t\t\t\t{}", id_str, e)?;
                    }
                }
            }
            Ok(output)
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    init_tracing(&cli);

    let overrides = ConfigOverrides {
        server_port: None,
        server_host: None,
        log_level: cli.log_level.clone(),
        profile: None,
        download_directory: cli.download_dir.clone(),
    };

    let config = if let Some(path) = cli.config.as_ref() {
        Config::load_with_overrides(Some(path), &overrides)?
    } else {
        Config::load_with_overrides(None, &overrides)?
    };

    let config = Arc::new(config);
    let meta_config = MetaSearchConfig::from_config(&config);
    let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);

    match cli.command {
        Commands::Search {
            query,
            source,
            limit,
            output_file,
        } => {
            let search_tool = create_search_source_tool();
            let input = SearchSourceInput {
                source: source.clone(),
                query: query.clone(),
                limit,
                offset: 0,
                search_type: None,
                help: false,
            };

            match search_tool.search(input).await {
                Ok(result) => {
                    let count = result.total_available.unwrap_or(result.papers.len() as u32);
                    info!("Found {} results from {} for '{}'", count, source, query);

                    // Determine output format from file extension
                    let output_format = output_file
                        .as_ref()
                        .and_then(|p| p.extension())
                        .and_then(|e| e.to_str())
                        .map(str::to_lowercase)
                        .unwrap_or_else(|| "tsv".to_string());

                    // Format output
                    let output_content = if output_format == "json" {
                        // JSON output
                        serde_json::to_string_pretty(&result.papers)
                            .unwrap_or_else(|_| "[]".to_string())
                    } else {
                        // TSV output (default)
                        let mut lines = vec!["doi\tpmid\ttitle\tauthors\tyear\tjournal\tkeywords\tabstract".to_string()];
                        for paper in &result.papers {
                            let doi = &paper.doi;
                            let pmid = paper.pmid.clone().unwrap_or_default();
                            let title = paper.title.clone().unwrap_or_default();
                            let authors = paper.authors.join("; ");
                            let year = paper.year.map(|y| y.to_string()).unwrap_or_default();
                            let journal = paper.journal.clone().unwrap_or_default();
                            let keywords = paper.keywords.join("; ");
                            let abstract_text = paper.abstract_text.clone().unwrap_or_default()
                                .replace(['\t', '\n'], " ");
                            lines.push(format!(
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                doi, pmid, title, authors, year, journal, keywords, abstract_text
                            ));
                        }
                        lines.join("\n")
                    };

                    // Write to file or stdout
                    if let Some(path) = output_file {
                        if let Some(parent) = path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&path, &output_content)?;
                        info!("Results saved to {}", path.display());
                    } else {
                        // Print to stdout
                        println!("{}", output_content);
                    }
                }
                Err(e) => {
                    error!("Search failed: {}", e);
                }
            }
        }
        Commands::ListSources => {
            let tool = create_search_source_tool();
            info!("Available academic sources:");
            for provider_name in tool.available_sources() {
                if let Some(info) = tool.get_source_info(&provider_name) {
                    info!("  {} - {}", info.name, info.description);
                }
            }
        }
        Commands::Install => {
            info!("Installing Python dependencies...");
            match rust_research_mcp::python_embed::install_python_package() {
                Ok(_) => {
                    info!("Python dependencies installed successfully.");
                }
                Err(e) => {
                    error!("Failed to install Python dependencies: {}", e);
                    return Err(anyhow::anyhow!(
                        "Failed to install Python dependencies: {}",
                        e
                    ));
                }
            }
        }
        Commands::Download {
            doi,
            url,
            filename,
            directory,
            category,
            overwrite,
            no_verify,
            markdown,
            disable_headless,
            enable_local_grobid,
        } => {
            let download_tool = DownloadTool::new(client, config.clone())?;
            let input = DownloadInput {
                doi,
                url,
                filename,
                directory: directory.map(|p| p.to_string_lossy().to_string()),
                category,
                overwrite,
                verify_integrity: !no_verify,
                output_format: if markdown {
                    DownloadOutputFormat::Markdown
                } else {
                    DownloadOutputFormat::Pdf
                },
                headless: !disable_headless,
                enable_local_grobid,
            };

            let result = download_tool.download_paper(input).await?;
            if let Some(err) = result.error {
                error!("Download failed: {}", err);
                return Ok(());
            }

            let file_path = result
                .file_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unknown".into());
            info!("Download succeeded: {}", file_path);
            if let Some(size) = result.file_size {
                info!("Size: {:.2} MB", size as f64 / 1_048_576.0);
            }
            if let Some(plugin) = result.used_plugin.as_deref() {
                info!("Plugin fallback used: {}", plugin);
            }
            if let Some(md) = result.markdown_path.as_ref() {
                info!("Markdown generated: {}", md.display());
            }
            if let Some(warning) = result.post_process_error.as_ref() {
                info!("Post-processing warning: {}", warning);
            }
        }
        Commands::Pdf2metadata { input } => {
            let extractor = MetadataExtractor::new(config)?;
            let meta_input = MetadataInput {
                file_path: input.clone(),
                use_cache: false,
                extract_references: true,
                batch_files: None,
            };
            let result = extractor.extract_metadata(meta_input).await?;
            info!("{}", serde_json::to_string_pretty(&result)?);
        }
        Commands::VerifyMetadata {
            input,
            output,
            output_mode,
            format,
            sources,
            concurrency,
        } => {
            use futures::stream::{self, StreamExt};
            use rust_research_mcp::tools::verify_metadata::{
                VerifyMetadataInput, VerifyMetadataTool, VerificationOutputMode,
            };
            use std::io::BufRead;
            use std::sync::Arc;

            // Parse output mode
            let mode = match output_mode.to_lowercase().as_str() {
                "notes" => VerificationOutputMode::Notes,
                "corrected" => VerificationOutputMode::Corrected,
                _ => VerificationOutputMode::Full,
            };

            // Parse sources
            let source_list: Option<Vec<String>> = sources
                .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());

            // Detect input format and read records
            let (records, input_format) = if let Some(ref path) = input {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("tsv");
                let content = std::fs::read_to_string(path)?;
                let recs = parse_metadata_records(&content, ext)?;
                (recs, ext.to_string())
            } else {
                // Read from stdin
                let stdin = std::io::stdin();
                let content: String = stdin.lock().lines().collect::<Result<Vec<_>, _>>()?.join("\n");
                let recs = parse_metadata_records(&content, "tsv")?;
                (recs, "tsv".to_string())
            };

            if records.is_empty() {
                return Err(anyhow::anyhow!("No valid metadata records found in input"));
            }

            info!("Processing {} metadata records with concurrency {}", records.len(), concurrency);

            // Determine output format
            let out_format = format.unwrap_or_else(|| {
                output.as_ref()
                    .and_then(|p| p.extension())
                    .and_then(|e| e.to_str())
                    .map(ToString::to_string)
                    .unwrap_or_else(|| input_format.clone())
            });

            // Create tool and process records concurrently
            let tool = Arc::new(VerifyMetadataTool::new());

            let results: Vec<_> = stream::iter(records)
                .map(|record| {
                    let tool = tool.clone();
                    let mode = mode.clone();
                    let sources = source_list.clone();
                    async move {
                        let input = VerifyMetadataInput {
                            doi: record.doi,
                            pmid: record.pmid,
                            title: record.title,
                            authors: record.authors,
                            year: record.year,
                            s2_paper_id: None,
                            openalex_id: None,
                            sources,
                            output_mode: mode,
                        };
                        let result = tool.verify(input).await;
                        (record.id, result)
                    }
                })
                .buffer_unordered(concurrency)
                .collect()
                .await;

            // Format output
            let output_content = format_verification_results(&results, &out_format)?;

            // Write output
            if let Some(ref path) = output {
                std::fs::write(path, &output_content)?;
                info!("Results written to: {}", path.display());
            } else {
                print!("{}", output_content);
            }
        }
        Commands::Pdf2text {
            pdf_file,
            pdf_dir,
            output_dir,
            grobid_url,
            no_auto_start,
            no_figures,
            no_tables,
            copy_pdf,
            overwrite,
            no_markdown,
        } => {
            use rust_research_mcp::tools::pdf2text::{Pdf2TextInput, Pdf2TextTool};
            use tokio::signal;

            let pdf2text_tool = Pdf2TextTool::new(config.clone())?;
            let input = Pdf2TextInput {
                pdf_file: pdf_file.map(|p| p.to_string_lossy().to_string()),
                pdf_dir: pdf_dir.map(|p| p.to_string_lossy().to_string()),
                output_dir: output_dir.to_string_lossy().to_string(),
                grobid_url,
                no_auto_start,
                no_figures,
                no_tables,
                copy_pdf,
                overwrite,
                no_markdown,
            };

            // Use tokio::select! to handle Ctrl+C gracefully
            tokio::select! {
                result = pdf2text_tool.convert(input) => {
                    match result {
                        Ok(result) => {
                            if result.success {
                                info!("Conversion succeeded!");
                                info!("Files processed: {}", result.files_processed);
                                if let Some(json) = result.json_path {
                                    info!("JSON output: {}", json);
                                }
                                if let Some(md) = result.markdown_path {
                                    info!("Markdown output: {}", md);
                                }
                            } else {
                                error!("Conversion failed!");
                                if let Some(err) = result.error {
                                    error!("Error: {}", err);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Conversion error: {}", e);
                            return Err(e.into());
                        }
                    }
                }
                _ = signal::ctrl_c() => {
                    info!("Received Ctrl+C, shutting down...");
                    return Ok(());
                }
            }
        }
        Commands::T2TServer {
            model,
            host,
            port,
            tensor_parallel_size,
            gpu_memory_utilization,
            max_model_len,
            trust_remote_code,
            cache_dir,
        } => {
            use rust_research_mcp::python_embed::run_text2table_server;

            run_text2table_server(
                &model,
                &host,
                port,
                tensor_parallel_size,
                gpu_memory_utilization,
                max_model_len,
                trust_remote_code,
                cache_dir.as_deref(),
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        }
        Commands::T2TCli {
            input_file,
            output,
            label,
            labels_file,
            text_column,
            id_column,
            concurrency,
            prompt,
            threshold,
            gliner_model,
            gliner_soft_threshold,
            model,
            enable_thinking,
            server_url,
            gliner_url,
            disable_gliner,
            enable_row_validation,
            row_validation_mode,
            api_key,
            gliner_api_key,
        } => {
            use rust_research_mcp::python_embed::run_text2table_cli;

            // Validate labels
            if label.is_empty() && labels_file.is_none() {
                return Err(anyhow::anyhow!("At least one label is required via --label or --labels-file"));
            }

            let server_url = server_url.ok_or_else(|| {
                anyhow::anyhow!("Server URL is required. Set TEXT2TABLE_VLLM_URL or use --server-url")
            })?;

            info!("Running text2table pipeline on {:?}...", input_file);

            tokio::select! {
                result = tokio::task::spawn_blocking(move || {
                    run_text2table_cli(
                        &input_file,
                        output.as_deref(),
                        &label,
                        labels_file.as_deref(),
                        text_column.as_deref(),
                        id_column.as_deref(),
                        concurrency,
                        prompt.as_deref(),
                        threshold,
                        &gliner_model,
                        gliner_soft_threshold,
                        model.as_deref(),
                        enable_thinking,
                        &server_url,
                        gliner_url.as_deref(),
                        disable_gliner,
                        enable_row_validation,
                        &row_validation_mode,
                        api_key.as_deref(),
                        gliner_api_key.as_deref(),
                    )
                }) => {
                    let inner_result = result.map_err(|e| anyhow::anyhow!("Task panicked: {e}"))?;
                    inner_result.map_err(|e| anyhow::anyhow!("{e}"))?;
                    info!("Text2table processing completed successfully");
                }
                _ = wait_for_shutdown_signal() => {
                    info!("Received shutdown signal, terminating text2table...");
                    std::process::exit(130);
                }
            }
        }
    }

    Ok(())
}

fn init_tracing(cli: &Cli) {
    let level = if let Some(raw) = cli.log_level.as_deref() {
        match raw.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        }
    } else if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(level)
        .with_writer(std::io::stderr)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
    info!("CLI logging initialized at {:?} level", level);
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(sigterm) => sigterm,
            Err(err) => {
                error!("Failed to register SIGTERM handler: {}", err);
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
