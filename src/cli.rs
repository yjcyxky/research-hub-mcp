#![allow(clippy::uninlined_format_args)]

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use rust_research_mcp::{
    client::{
        providers::{
            ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, GoogleScholarProvider,
            MdpiProvider, MedrxivProvider, OpenAlexProvider, OpenReviewProvider,
            PubMedCentralProvider, ResearchGateProvider, SciHubProvider, SemanticScholarProvider,
            SsrnProvider, UnpaywallProvider,
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
use serde_json;
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
enum Commands {
    /// Search papers from a specific academic source (use 'list-sources' to see available sources)
    Search {
        /// Query text (use native query syntax for the source)
        query: String,
        /// Source to search (e.g., arxiv, pubmed_central, semantic_scholar)
        #[arg(short, long)]
        source: String,
        /// Max results to return
        #[arg(short, long, default_value_t = 10)]
        limit: u32,
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
    /// Extract metadata from a PDF file
    Metadata {
        /// Path to a PDF file
        input: String,
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
    /// Run text2table extraction pipeline
    T2TCli {
        #[command(subcommand)]
        command: T2TSubcommands,
    },
}

#[derive(Subcommand)]
enum T2TSubcommands {
    /// Process a single text input
    Run {
        /// Raw text to process
        #[arg(long)]
        text: Option<String>,

        /// Path to text file
        #[arg(long)]
        text_file: Option<PathBuf>,

        /// Labels to extract (repeat for multiple)
        #[arg(short, long)]
        label: Vec<String>,

        /// Path to file containing labels (one per line)
        #[arg(long)]
        labels_file: Option<PathBuf>,

        /// Custom user prompt
        #[arg(long)]
        prompt: Option<String>,

        /// GLiNER threshold
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
    /// Batch process from TSV file
    Batch {
        /// Input TSV file
        #[arg(long)]
        input_file: PathBuf,

        /// Output file (defaults to output.jsonl or output.tsv based on --output-format)
        #[arg(long)]
        output_file: Option<PathBuf>,

        /// Output format (jsonl or tsv)
        #[arg(long, value_enum, default_value_t = BatchOutputFormat::Jsonl)]
        output_format: BatchOutputFormat,

        /// Concurrency limit
        #[arg(long, default_value_t = 4)]
        concurrency: usize,

        // --- Copy of Run args (flattening via struct would be better but keeping it simple here) ---
        /// Labels to extract (repeat for multiple)
        #[arg(short, long)]
        label: Vec<String>,

        /// Path to file containing labels (one per line)
        #[arg(long)]
        labels_file: Option<PathBuf>,

        /// Custom user prompt
        #[arg(long)]
        prompt: Option<String>,

        /// GLiNER threshold
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

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum BatchOutputFormat {
    Jsonl,
    Tsv,
}

impl BatchOutputFormat {
    fn as_str(&self) -> &'static str {
        match self {
            BatchOutputFormat::Jsonl => "jsonl",
            BatchOutputFormat::Tsv => "tsv",
        }
    }
}

/// Create a search_source tool with all providers registered
fn create_search_source_tool() -> anyhow::Result<SearchSourceTool> {
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

    Ok(tool)
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
        } => {
            let search_tool = create_search_source_tool()?;
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
                    for (idx, paper) in result.papers.iter().enumerate() {
                        let title = paper.title.clone().unwrap_or_else(|| "No title".into());
                        let doi = paper.doi.clone();
                        info!(
                            "{}. {} [DOI: {}]",
                            idx + 1,
                            title,
                            if doi.is_empty() { "N/A".into() } else { doi }
                        );
                    }
                }
                Err(e) => {
                    error!("Search failed: {}", e);
                }
            }
        }
        Commands::ListSources => {
            let tool = create_search_source_tool()?;
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
        Commands::Metadata { input } => {
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
        Commands::T2TCli { command } => {
            use rust_research_mcp::tools::text2table::{
                Text2TableBatchInput, Text2TableInput, Text2TableTool,
            };

            let tool = Text2TableTool::new(config.clone())?;

            match command {
                T2TSubcommands::Run {
                    text,
                    text_file,
                    label,
                    labels_file,
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
                    let input = Text2TableInput {
                        text,
                        text_file: text_file.map(|p| p.to_string_lossy().to_string()),
                        labels: label,
                        labels_file: labels_file.map(|p| p.to_string_lossy().to_string()),
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
                    };

                    tokio::select! {
                        result = tool.generate(input) => {
                            let result = result?;
                            if result.success {
                                info!("Generation succeeded!");
                                if let Some(table) = result.table {
                                    println!("{}", table);
                                }
                                if let Some(thinking) = result.thinking {
                                    info!("Thinking: {}", thinking);
                                }
                            } else {
                                error!("Generation failed: {:?}", result.error);
                            }
                        }
                        _ = wait_for_shutdown_signal() => {
                            info!("Received shutdown signal, terminating text2table run...");
                            std::process::exit(130);
                        }
                    }
                }
                T2TSubcommands::Batch {
                    input_file,
                    output_file,
                    output_format,
                    concurrency,
                    label,
                    labels_file,
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
                    let config = Text2TableInput {
                        text: None, // Will be filled per row
                        text_file: None,
                        labels: label,
                        labels_file: labels_file.map(|p| p.to_string_lossy().to_string()),
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
                    };

                    let input = Text2TableBatchInput {
                        input_file: input_file.to_string_lossy().to_string(),
                        output_file: output_file.map(|p| p.to_string_lossy().to_string()),
                        output_format: output_format.as_str().to_string(),
                        concurrency,
                        config,
                    };

                    tokio::select! {
                        result = tool.process_batch(input) => {
                            result?;
                        }
                        _ = wait_for_shutdown_signal() => {
                            info!("Received shutdown signal, terminating text2table batch...");
                            std::process::exit(130);
                        }
                    }
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
