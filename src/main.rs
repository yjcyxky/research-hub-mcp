#![allow(clippy::field_reassign_with_default)]

use anyhow::Result;
use clap::{Parser, Subcommand};
use rust_research_mcp::{Config, ConfigOverrides, DaemonConfig, DaemonService, PidFile, Server};
use std::sync::Arc;
use tracing::{debug, error, info};

#[derive(Parser)]
#[command(name = "rust-research-mcp")]
#[command(about = "A MCP server for academic research and knowledge accumulation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,

    /// Run as daemon
    #[arg(short, long)]
    daemon: bool,

    /// PID file path (for daemon mode)
    #[arg(long)]
    pid_file: Option<std::path::PathBuf>,

    /// Health check port
    #[arg(long, default_value = "8090")]
    health_port: u16,

    /// Override server port
    #[arg(long)]
    port: Option<u16>,

    /// Override server host
    #[arg(long)]
    host: Option<String>,

    /// Override log level (trace, debug, info, warn, error)
    #[arg(long)]
    log_level: Option<String>,

    /// Set environment profile (development, production)
    #[arg(long)]
    profile: Option<String>,

    /// Override download directory path
    #[arg(long)]
    download_dir: Option<std::path::PathBuf>,

    /// Generate JSON schema for configuration
    #[arg(long)]
    generate_schema: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Install Python dependencies
    Install,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(if cli.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        })
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting rust-research-mcp server");

    // Handle subcommands
    if let Some(Commands::Install) = cli.command {
        info!("Installing Python dependencies...");
        match rust_research_mcp::python_embed::install_python_package() {
            Ok(_) => {
                info!("Python dependencies installed successfully.");
                return Ok(());
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

    // Handle schema generation request
    if cli.generate_schema {
        let schema = Config::generate_schema();
        println!("{}", serde_json::to_string_pretty(&schema)?);
        return Ok(());
    }

    // Build configuration overrides from CLI arguments
    let overrides = ConfigOverrides {
        server_port: cli.port,
        server_host: cli.host,
        log_level: cli.log_level.or_else(|| {
            if cli.verbose {
                Some("debug".to_string())
            } else {
                None
            }
        }),
        profile: cli.profile,
        download_directory: cli.download_dir,
    };

    // Load configuration with proper precedence
    let config = if let Some(config_path) = cli.config.as_ref() {
        info!("Loading config from: {}", config_path.display());
        Config::load_with_overrides(Some(config_path), &overrides)?
    } else {
        Config::load_with_overrides(None, &overrides)?
    };

    // Log configuration (safely)
    let safe_config = config.safe_for_logging();
    info!(
        "Loaded configuration: profile={}, schema_version={}",
        safe_config.profile, safe_config.schema_version
    );
    debug!("Configuration details: {:#?}", safe_config);

    // Check if running in daemon mode
    if cli.daemon {
        info!("Starting in daemon mode");

        // Configure daemon
        let mut daemon_config = DaemonConfig::default();
        daemon_config.daemon = true;
        daemon_config.health_port = cli.health_port;
        daemon_config.pid_file = cli.pid_file.or_else(|| Some(PidFile::standard_path()));

        // Create and start daemon service
        let config = Arc::new(config);
        let mut daemon = DaemonService::new(config, daemon_config)?;

        match daemon.start().await {
            Ok(()) => {
                info!("Daemon service stopped");
                Ok(())
            }
            Err(e) => {
                error!("Daemon service error: {}", e);
                Err(e.into())
            }
        }
    } else {
        // Run in foreground mode
        let server = Server::new(config);

        match server.run().await {
            Ok(()) => {
                info!("Server shutdown completed successfully");
                Ok(())
            }
            Err(e) => {
                error!("Server error: {}", e);
                Err(e.into())
            }
        }
    }
}
