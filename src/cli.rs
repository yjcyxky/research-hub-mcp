#![allow(clippy::uninlined_format_args)]

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use rust_research_mcp::{
    client::{MetaSearchClient, MetaSearchConfig},
    tools::{
        download::{DownloadInput, DownloadOutputFormat, DownloadTool},
        metadata::{MetadataExtractor, MetadataInput},
        search::{SearchInput, SearchTool, SearchType as ToolSearchType},
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
    /// Search papers by DOI, title, or author
    Search {
        /// Query text (DOI, title, author, etc.)
        query: String,
        /// Max results to return
        #[arg(short, long, default_value_t = 10)]
        limit: u32,
        /// Offset for pagination
        #[arg(long, default_value_t = 0)]
        offset: u32,
        /// Search mode
        #[arg(long, value_enum, default_value_t = SearchMode::Auto)]
        mode: SearchMode,
        /// Comma-separated list of primary providers to use (e.g., pubmed_central,google_scholar,biorxiv)
        #[arg(long, value_delimiter = ',')]
        sources: Option<Vec<String>>,
        /// Comma-separated list of metadata-only providers (e.g., crossref)
        #[arg(long, value_delimiter = ',')]
        metadata_sources: Option<Vec<String>>,
    },
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
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum SearchMode {
    Auto,
    Doi,
    Title,
    Author,
    AuthorYear,
    TitleAbstract,
}

impl From<SearchMode> for ToolSearchType {
    fn from(value: SearchMode) -> Self {
        match value {
            SearchMode::Auto => Self::Auto,
            SearchMode::Doi => Self::Doi,
            SearchMode::Title => Self::Title,
            SearchMode::Author => Self::Author,
            SearchMode::AuthorYear => Self::AuthorYear,
            SearchMode::TitleAbstract => Self::TitleAbstract,
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
            limit,
            offset,
            mode,
            sources,
            metadata_sources,
        } => {
            let search_tool = SearchTool::new(config.clone())?;
            let input = SearchInput {
                query: query.clone(),
                search_type: mode.into(),
                limit,
                offset,
                sources,
                metadata_sources,
            };
            let results = search_tool.search_papers(input).await?;

            info!(
                "Found {} of {} results for '{}'",
                results.returned_count, results.total_count, query
            );
            for (idx, paper) in results.papers.iter().enumerate() {
                let title = paper
                    .metadata
                    .title
                    .clone()
                    .unwrap_or_else(|| "No title".into());
                let doi = paper.metadata.doi.clone();
                let source = paper.source.clone();
                info!(
                    "{}. {} [DOI: {}] (source: {})",
                    idx + 1,
                    title,
                    if doi.is_empty() { "N/A".into() } else { doi },
                    source
                );
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
                validate_external: false,
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

            let result = pdf2text_tool.convert(input).await?;
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
