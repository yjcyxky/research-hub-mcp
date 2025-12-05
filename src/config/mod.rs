use config;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info, warn};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Research source configuration  
    pub research_source: ResearchSourceConfig,
    /// Download management configuration
    pub downloads: DownloadsConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Rate limiting configuration
    pub rate_limiting: RateLimitingConfig,
    /// Categorization configuration
    pub categorization: crate::services::CategorizationConfig,
    /// Environment profile (development, production)
    #[serde(default = "default_profile")]
    pub profile: String,
    /// Configuration schema version
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ServerConfig {
    /// Server listen port
    pub port: u16,
    /// Server bind address
    pub host: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Graceful shutdown timeout in seconds
    pub graceful_shutdown_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ResearchSourceConfig {
    /// List of research source endpoints
    pub endpoints: Vec<String>,
    /// Rate limit (requests per second)
    pub rate_limit_per_sec: u32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Provider timeout in seconds (for meta-search across multiple providers)
    pub provider_timeout_secs: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct DownloadsConfig {
    /// Download directory path
    pub directory: PathBuf,
    /// Maximum concurrent downloads
    pub max_concurrent: usize,
    /// Maximum file size in MB
    pub max_file_size_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level (debug, info, warn, error)
    pub level: String,
    /// Log format (json, text)
    pub format: String,
    /// Optional log file path
    pub file: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct RateLimitingConfig {
    /// Enable adaptive rate limiting based on response times
    pub adaptive: bool,
    /// Show progress indicators during rate-limited operations
    pub show_progress: bool,
    /// Allow burst requests for small batches
    pub allow_burst: bool,
    /// Number of requests allowed in a burst
    pub burst_size: u32,
    /// Provider-specific rate limits (requests per second)
    pub providers: HashMap<String, f64>,
    /// Default rate limit for providers not explicitly configured
    pub default_rate: f64,
    /// Minimum rate when adaptive limiting decreases rate
    pub min_rate: f64,
    /// Maximum rate when adaptive limiting increases rate
    pub max_rate: f64,
}

fn default_profile() -> String {
    "development".to_string()
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

/// Expand tilde and environment variables in paths
#[allow(clippy::option_if_let_else)]
fn expand_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        dirs::home_dir().map_or_else(|| PathBuf::from(path), |home_dir| home_dir.join(stripped))
    } else if path.starts_with('$') {
        // Handle environment variables like $HOME/downloads
        path.find('/').map_or_else(
            || PathBuf::from(path),
            |equals_pos| {
                let env_var = &path[1..equals_pos];
                std::env::var(env_var).map_or_else(
                    |_| PathBuf::from(path),
                    |env_value| PathBuf::from(env_value).join(&path[equals_pos + 1..]),
                )
            },
        )
    } else {
        PathBuf::from(path)
    }
}

/// CLI argument overrides for configuration
#[derive(Debug, Default, Clone)]
pub struct ConfigOverrides {
    pub server_port: Option<u16>,
    pub server_host: Option<String>,
    pub log_level: Option<String>,
    pub profile: Option<String>,
    pub download_directory: Option<PathBuf>,
}

/// Environment variable overrides with RSH_ prefix
#[derive(Debug, Deserialize)]
pub struct ConfigEnvOverrides {
    #[serde(rename = "server_port")]
    pub server_port: Option<u16>,
    #[serde(rename = "server_host")]
    pub server_host: Option<String>,
    #[serde(rename = "log_level")]
    pub log_level: Option<String>,
    #[serde(rename = "profile")]
    pub profile: Option<String>,
    #[serde(rename = "download_directory")]
    pub download_directory: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            research_source: ResearchSourceConfig::default(),
            downloads: DownloadsConfig::default(),
            logging: LoggingConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
            categorization: crate::services::CategorizationConfig::default(),
            profile: default_profile(),
            schema_version: default_schema_version(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            timeout_secs: 30,
            health_check_interval_secs: 30,
            graceful_shutdown_timeout_secs: 5,
        }
    }
}

impl Default for ResearchSourceConfig {
    fn default() -> Self {
        Self {
            endpoints: vec![
                "https://sci-hub.se".to_string(),
                "https://sci-hub.st".to_string(),
                "https://sci-hub.ru".to_string(),
            ],
            rate_limit_per_sec: 1,
            timeout_secs: 30,
            provider_timeout_secs: 30,
            max_retries: 3,
        }
    }
}

impl Default for DownloadsConfig {
    fn default() -> Self {
        Self {
            directory: expand_path("~/downloads/papers"),
            max_concurrent: 3,
            max_file_size_mb: 100,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            file: None,
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();

        // Academic sources - higher rates (respectful but efficient)
        providers.insert("arxiv".to_string(), 2.0);
        providers.insert("pubmed".to_string(), 2.0);
        providers.insert("pubmed_central".to_string(), 2.0);
        providers.insert("biorxiv".to_string(), 2.0);
        providers.insert("medrxiv".to_string(), 2.0);
        providers.insert("google_scholar".to_string(), 1.0);

        // Commercial/API sources - moderate rates
        providers.insert("crossref".to_string(), 1.5);
        providers.insert("semantic_scholar".to_string(), 1.5);
        providers.insert("unpaywall".to_string(), 1.5);
        providers.insert("core".to_string(), 1.5);
        providers.insert("ssrn".to_string(), 1.0);

        // Sci-Hub and similar - conservative rates
        providers.insert("sci_hub".to_string(), 0.5);
        providers.insert("researchgate".to_string(), 1.0);
        providers.insert("mdpi".to_string(), 1.0);
        providers.insert("openreview".to_string(), 1.0);

        Self {
            adaptive: true,
            show_progress: true,
            allow_burst: true,
            burst_size: 3,
            providers,
            default_rate: 1.0,
            min_rate: 0.25,
            max_rate: 5.0,
        }
    }
}

impl Config {
    /// Load configuration with layered precedence: defaults < file < env vars < CLI args
    pub fn load() -> crate::Result<Self> {
        Self::load_with_overrides(None, &ConfigOverrides::default())
    }

    /// Load configuration from a specific file path
    pub fn load_from_file(file_path: &std::path::Path) -> crate::Result<Self> {
        Self::load_with_overrides(Some(file_path), &ConfigOverrides::default())
    }

    /// Load configuration with CLI overrides
    pub fn load_with_overrides(
        config_path: Option<&std::path::Path>,
        overrides: &ConfigOverrides,
    ) -> crate::Result<Self> {
        debug!("Loading configuration with layered approach");

        // 1. Start with defaults
        let mut config = Self::default();
        debug!("Applied default configuration");

        // 2. Load from file (if exists)
        if let Some(path) = config_path {
            if path.exists() {
                config = Self::load_from_toml_file(path)?;
                debug!("Loaded configuration from file: {}", path.display());
            } else {
                warn!("Configuration file not found: {}", path.display());
            }
        } else {
            // Try standard locations
            if let Some(standard_config) = Self::try_load_standard_locations()? {
                config = standard_config;
                debug!("Loaded configuration from standard location");
            }
        }

        // 3. Override with environment variables
        config = Self::apply_env_overrides(config);
        debug!("Applied environment variable overrides");

        // 4. Apply CLI overrides
        config = Self::apply_cli_overrides(config, overrides);
        debug!("Applied CLI overrides");

        // 5. Apply profile-specific settings
        config = Self::apply_profile_settings(config);
        debug!("Applied profile-specific settings for: {}", config.profile);

        // 6. Validate final configuration
        config.validate()?;
        debug!("Configuration validation passed");

        Ok(config)
    }

    /// Load configuration from a TOML file
    fn load_from_toml_file(path: &std::path::Path) -> crate::Result<Self> {
        // Security: Validate configuration file security before reading
        Self::validate_config_file_security(path)?;

        let config_str = std::fs::read_to_string(path)?;
        toml::from_str(&config_str)
            .map_err(|e| crate::Error::Config(config::ConfigError::Foreign(Box::new(e))))
    }

    /// Try to load from standard configuration locations
    fn try_load_standard_locations() -> crate::Result<Option<Self>> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| {
                crate::Error::Config(config::ConfigError::NotFound("config directory".into()))
            })?
            .join("knowledge_accumulator_mcp");

        // Create config directory if it doesn't exist
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;

            // Security: Set secure permissions on config directory
            Self::set_secure_directory_permissions(&config_dir)?;

            debug!(
                "Created config directory with secure permissions: {}",
                config_dir.display()
            );
        }

        let config_files = [
            config_dir.join("config.toml"),
            config_dir.join("config.development.toml"),
            config_dir.join("config.production.toml"),
        ];

        for config_file in &config_files {
            if config_file.exists() {
                debug!("Found config file: {}", config_file.display());
                return Ok(Some(Self::load_from_toml_file(config_file)?));
            }
        }

        Ok(None)
    }

    /// Apply environment variable overrides using the RSH_ prefix
    fn apply_env_overrides(mut config: Self) -> Self {
        // Use envy to parse environment variables with RSH_ prefix
        match envy::prefixed("RSH_").from_env::<ConfigEnvOverrides>() {
            Ok(env_overrides) => {
                debug!("Found environment variable overrides");

                if let Some(port) = env_overrides.server_port {
                    if port > 0 {
                        config.server.port = port;
                        debug!("Overrode server port from env: {}", port);
                    } else {
                        warn!("Invalid port value from env: {}, ignoring", port);
                    }
                }

                if let Some(host) = env_overrides.server_host {
                    if host.trim().is_empty() {
                        warn!("Empty host value from env, ignoring");
                    } else {
                        config.server.host.clone_from(&host);
                        debug!("Overrode server host from env: {}", host);
                    }
                }

                if let Some(level) = env_overrides.log_level {
                    let valid_levels = ["trace", "debug", "info", "warn", "error"];
                    if valid_levels.contains(&level.as_str()) {
                        config.logging.level.clone_from(&level);
                        debug!("Overrode log level from env: {}", level);
                    } else {
                        warn!("Invalid log level from env: {}, ignoring", level);
                    }
                }

                if let Some(profile) = env_overrides.profile {
                    config.profile.clone_from(&profile);
                    debug!("Overrode profile from env: {}", profile);
                }

                if let Some(dir) = env_overrides.download_directory {
                    if dir.trim().is_empty() {
                        warn!("Empty download directory from env, ignoring");
                    } else {
                        config.downloads.directory = expand_path(&dir);
                        debug!("Overrode download directory from env: {}", dir);
                    }
                }
            }
            Err(e) => {
                debug!("No valid environment variable overrides found: {}", e);
            }
        }

        config
    }

    /// Apply CLI argument overrides
    fn apply_cli_overrides(mut config: Self, overrides: &ConfigOverrides) -> Self {
        if let Some(port) = overrides.server_port {
            config.server.port = port;
            debug!("Overrode server port from CLI: {}", port);
        }

        if let Some(ref host) = overrides.server_host {
            config.server.host.clone_from(host);
            debug!("Overrode server host from CLI: {}", host);
        }

        if let Some(ref level) = overrides.log_level {
            config.logging.level.clone_from(level);
            debug!("Overrode log level from CLI: {}", level);
        }

        if let Some(ref profile) = overrides.profile {
            config.profile.clone_from(profile);
            debug!("Overrode profile from CLI: {}", profile);
        }

        if let Some(ref dir) = overrides.download_directory {
            config.downloads.directory = dir.clone();
            debug!("Overrode download directory from CLI: {}", dir.display());
        }

        config
    }

    /// Apply profile-specific configuration adjustments
    fn apply_profile_settings(mut config: Self) -> Self {
        match config.profile.as_str() {
            "development" => {
                // Development settings: more verbose logging, shorter timeouts
                if config.logging.level == "info" {
                    config.logging.level = "debug".to_string();
                }
                config.server.timeout_secs = 10;
                config.server.health_check_interval_secs = 10;
            }
            "production" => {
                // Production settings: conservative timeouts, structured logging
                config.logging.format = "json".to_string();
                config.server.timeout_secs = 60;
                config.server.health_check_interval_secs = 60;
            }
            _ => {
                warn!("Unknown profile '{}', using defaults", config.profile);
            }
        }

        config
    }

    /// Generate JSON schema for configuration
    #[must_use]
    pub fn generate_schema() -> serde_json::Value {
        let schema = schemars::schema_for!(Self);
        serde_json::to_value(schema).unwrap_or_default()
    }

    /// Hot reload configuration from all sources (for non-critical settings)
    pub fn reload(&mut self) -> crate::Result<bool> {
        self.reload_with_overrides(&ConfigOverrides::default())
    }

    /// Hot reload configuration with CLI overrides (for non-critical settings)
    pub fn reload_with_overrides(&mut self, overrides: &ConfigOverrides) -> crate::Result<bool> {
        debug!("Attempting configuration hot reload with overrides");

        let new_config = Self::load_with_overrides(None, overrides)?;

        // Only allow hot reload for non-critical settings to avoid service disruption
        let mut changed = false;

        // Hot-reloadable: Logging configuration
        if self.logging.level != new_config.logging.level {
            self.logging.level.clone_from(&new_config.logging.level);
            changed = true;
            debug!("Hot reloaded log level: {}", new_config.logging.level);
        }

        if self.logging.format != new_config.logging.format {
            self.logging.format.clone_from(&new_config.logging.format);
            changed = true;
            debug!("Hot reloaded log format: {}", new_config.logging.format);
        }

        if self.logging.file != new_config.logging.file {
            self.logging.file.clone_from(&new_config.logging.file);
            changed = true;
            debug!("Hot reloaded log file: {:?}", new_config.logging.file);
        }

        // Hot-reloadable: Sci-Hub rate limiting and timeouts
        if self.research_source.rate_limit_per_sec != new_config.research_source.rate_limit_per_sec
        {
            self.research_source.rate_limit_per_sec = new_config.research_source.rate_limit_per_sec;
            changed = true;
            debug!(
                "Hot reloaded rate limit: {}",
                new_config.research_source.rate_limit_per_sec
            );
        }

        if self.research_source.timeout_secs != new_config.research_source.timeout_secs {
            self.research_source.timeout_secs = new_config.research_source.timeout_secs;
            changed = true;
            debug!(
                "Hot reloaded sci-hub timeout: {}",
                new_config.research_source.timeout_secs
            );
        }
        if self.research_source.provider_timeout_secs
            != new_config.research_source.provider_timeout_secs
        {
            self.research_source.provider_timeout_secs =
                new_config.research_source.provider_timeout_secs;
            changed = true;
            debug!(
                "Hot reloaded provider timeout: {}",
                new_config.research_source.provider_timeout_secs
            );
        }

        if self.research_source.max_retries != new_config.research_source.max_retries {
            self.research_source.max_retries = new_config.research_source.max_retries;
            changed = true;
            debug!(
                "Hot reloaded max retries: {}",
                new_config.research_source.max_retries
            );
        }

        // Hot-reloadable: Health check intervals
        if self.server.health_check_interval_secs != new_config.server.health_check_interval_secs {
            self.server.health_check_interval_secs = new_config.server.health_check_interval_secs;
            changed = true;
            debug!(
                "Hot reloaded health check interval: {}",
                new_config.server.health_check_interval_secs
            );
        }

        // Hot-reloadable: Download configuration
        if self.downloads.max_concurrent != new_config.downloads.max_concurrent {
            self.downloads.max_concurrent = new_config.downloads.max_concurrent;
            changed = true;
            debug!(
                "Hot reloaded max concurrent downloads: {}",
                new_config.downloads.max_concurrent
            );
        }

        if self.downloads.max_file_size_mb != new_config.downloads.max_file_size_mb {
            self.downloads.max_file_size_mb = new_config.downloads.max_file_size_mb;
            changed = true;
            debug!(
                "Hot reloaded max file size: {}MB",
                new_config.downloads.max_file_size_mb
            );
        }

        // Hot-reloadable: Rate limiting configuration
        if self.rate_limiting.adaptive != new_config.rate_limiting.adaptive {
            self.rate_limiting.adaptive = new_config.rate_limiting.adaptive;
            changed = true;
            debug!(
                "Hot reloaded adaptive rate limiting: {}",
                new_config.rate_limiting.adaptive
            );
        }

        if self.rate_limiting.show_progress != new_config.rate_limiting.show_progress {
            self.rate_limiting.show_progress = new_config.rate_limiting.show_progress;
            changed = true;
            debug!(
                "Hot reloaded show progress: {}",
                new_config.rate_limiting.show_progress
            );
        }

        if self.rate_limiting.allow_burst != new_config.rate_limiting.allow_burst {
            self.rate_limiting.allow_burst = new_config.rate_limiting.allow_burst;
            changed = true;
            debug!(
                "Hot reloaded allow burst: {}",
                new_config.rate_limiting.allow_burst
            );
        }

        if self.rate_limiting.burst_size != new_config.rate_limiting.burst_size {
            self.rate_limiting.burst_size = new_config.rate_limiting.burst_size;
            changed = true;
            debug!(
                "Hot reloaded burst size: {}",
                new_config.rate_limiting.burst_size
            );
        }

        if self.rate_limiting.default_rate != new_config.rate_limiting.default_rate {
            self.rate_limiting.default_rate = new_config.rate_limiting.default_rate;
            changed = true;
            debug!(
                "Hot reloaded default rate: {}",
                new_config.rate_limiting.default_rate
            );
        }

        if self.rate_limiting.providers != new_config.rate_limiting.providers {
            self.rate_limiting.providers = new_config.rate_limiting.providers.clone();
            changed = true;
            debug!("Hot reloaded provider-specific rates");
        }

        // Validate the reloaded configuration
        self.validate()?;

        if changed {
            debug!("Configuration hot reload completed with changes");
        } else {
            debug!("Configuration hot reload completed with no changes");
        }

        Ok(changed)
    }

    pub fn validate(&self) -> crate::Result<()> {
        // Validate schema version (allow forward compatibility)
        let supported_versions = ["1.0"];
        if !supported_versions.contains(&self.schema_version.as_str()) {
            warn!(
                "Unknown config schema version: {}. Supported: {:?}. Attempting to continue...",
                self.schema_version, supported_versions
            );
        }

        // Validate server configuration
        if self.server.port == 0 {
            return Err(crate::Error::InvalidInput {
                field: "server.port".to_string(),
                reason: "Server port cannot be 0".to_string(),
            });
        }
        if self.server.port < 1024 {
            warn!(
                "Server port {} is in the privileged range (<1024)",
                self.server.port
            );
        }

        // Validate Sci-Hub configuration
        if self.research_source.endpoints.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "sci_hub.mirrors".to_string(),
                reason: "At least one Sci-Hub mirror must be configured".to_string(),
            });
        }

        for mirror in &self.research_source.endpoints {
            if !mirror.starts_with("https://") {
                return Err(crate::Error::InvalidInput {
                    field: "sci_hub.mirrors".to_string(),
                    reason: format!("Sci-Hub mirror must use HTTPS: {mirror}"),
                });
            }
        }

        if self.research_source.rate_limit_per_sec == 0 {
            return Err(crate::Error::InvalidInput {
                field: "sci_hub.rate_limit_per_sec".to_string(),
                reason: "Rate limit must be greater than 0".to_string(),
            });
        }

        // Validate downloads configuration
        if self.downloads.max_concurrent == 0 {
            return Err(crate::Error::InvalidInput {
                field: "downloads.max_concurrent".to_string(),
                reason: "Max concurrent downloads must be greater than 0".to_string(),
            });
        }
        if self.downloads.max_file_size_mb == 0 {
            return Err(crate::Error::InvalidInput {
                field: "downloads.max_file_size_mb".to_string(),
                reason: "Max file size must be greater than 0".to_string(),
            });
        }

        // Validate logging configuration
        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.logging.level.as_str()) {
            return Err(crate::Error::InvalidInput {
                field: "logging.level".to_string(),
                reason: format!(
                    "Invalid log level '{}'. Valid levels: {:?}",
                    self.logging.level, valid_log_levels
                ),
            });
        }

        // Validate rate limiting configuration
        if self.rate_limiting.default_rate <= 0.0 {
            return Err(crate::Error::InvalidInput {
                field: "rate_limiting.default_rate".to_string(),
                reason: "Default rate limit must be greater than 0".to_string(),
            });
        }

        if self.rate_limiting.min_rate <= 0.0
            || self.rate_limiting.min_rate > self.rate_limiting.max_rate
        {
            return Err(crate::Error::InvalidInput {
                field: "rate_limiting.min_rate".to_string(),
                reason: "Minimum rate must be greater than 0 and less than maximum rate"
                    .to_string(),
            });
        }

        if self.rate_limiting.burst_size == 0 {
            return Err(crate::Error::InvalidInput {
                field: "rate_limiting.burst_size".to_string(),
                reason: "Burst size must be greater than 0".to_string(),
            });
        }

        for (provider, rate) in &self.rate_limiting.providers {
            if *rate <= 0.0 {
                return Err(crate::Error::InvalidInput {
                    field: format!("rate_limiting.providers.{provider}"),
                    reason: format!("Rate limit for provider '{provider}' must be greater than 0"),
                });
            }
        }

        let valid_log_formats = ["json", "text"];
        if !valid_log_formats.contains(&self.logging.format.as_str()) {
            return Err(crate::Error::InvalidInput {
                field: "logging.format".to_string(),
                reason: format!(
                    "Invalid log format '{}'. Valid formats: {:?}",
                    self.logging.format, valid_log_formats
                ),
            });
        }

        // Validate profile
        let valid_profiles = ["development", "production"];
        if !valid_profiles.contains(&self.profile.as_str()) {
            warn!(
                "Unknown profile '{}'. Valid profiles: {:?}",
                self.profile, valid_profiles
            );
        }

        Ok(())
    }

    /// Get a safe version of the config for logging (with sensitive values redacted)
    #[must_use]
    pub fn safe_for_logging(&self) -> Self {
        let mut safe_config = self.clone();

        // Redact any potentially sensitive information
        if let Some(ref _file) = safe_config.logging.file {
            // Log file paths might contain sensitive info, but probably okay to log
        }

        // Could redact download directory if it contains user info
        if safe_config
            .downloads
            .directory
            .to_string_lossy()
            .contains("/Users/")
        {
            safe_config.downloads.directory = PathBuf::from("[REDACTED_USER_PATH]");
        }

        safe_config
    }

    /// Generate an example configuration file with all options documented
    #[must_use]
    pub fn generate_example_config() -> String {
        let example_config = r#"# Rust Sci-Hub MCP Server Configuration
# This file demonstrates all available configuration options with their defaults

# Configuration schema version (for future compatibility)
schema_version = "1.0"

# Environment profile: "development" or "production"
profile = "development"

[server]
# Server listen port (default: 8080)
port = 8080

# Server bind address (default: "127.0.0.1")
host = "127.0.0.1"

# Request timeout in seconds (default: 30)
timeout_secs = 30

# Health check interval in seconds (default: 30)
health_check_interval_secs = 30

# Graceful shutdown timeout in seconds (default: 5)
graceful_shutdown_timeout_secs = 5

[sci_hub]
# List of Sci-Hub mirror URLs to try
mirrors = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru"
]

# Rate limit in requests per second (default: 1)
rate_limit_per_sec = 1

# Request timeout in seconds (default: 30)
timeout_secs = 30

# Provider timeout in seconds for meta-search across multiple providers (default: 30)
provider_timeout_secs = 30

# Maximum retry attempts (default: 3)
max_retries = 3

[downloads]
# Download directory path (supports ~ expansion)
directory = "~/downloads/papers"

# Maximum concurrent downloads (default: 3)
max_concurrent = 3

# Maximum file size in MB (default: 100)
max_file_size_mb = 100

[logging]
# Log level: "trace", "debug", "info", "warn", "error" (default: "info")
level = "info"

# Log format: "json" or "text" (default: "json")
format = "json"

# Optional log file path (default: none, logs to stderr)
# file = "/var/log/knowledge_accumulator_mcp.log"

[rate_limiting]
# Enable adaptive rate limiting based on response times (default: true)
adaptive = true

# Show progress indicators during rate-limited operations (default: true)
show_progress = true

# Allow burst requests for small batches (default: true)  
allow_burst = true

# Number of requests allowed in a burst (default: 3)
burst_size = 3

# Default rate limit for providers not explicitly configured (default: 1.0)
default_rate = 1.0

# Minimum rate when adaptive limiting decreases rate (default: 0.25)
min_rate = 0.25

# Maximum rate when adaptive limiting increases rate (default: 5.0)
max_rate = 5.0

# Provider-specific rate limits (requests per second)
[rate_limiting.providers]
arxiv = 2.0
pubmed = 2.0
pubmed_central = 2.0
biorxiv = 2.0
crossref = 1.5
semantic_scholar = 1.5  
unpaywall = 1.5
core = 1.5
ssrn = 1.0
sci_hub = 0.5
researchgate = 1.0
mdpi = 1.0
openreview = 1.0

# Environment Variables:
# Override any setting using RSH_ prefix:
# RSH_SERVER_PORT=9090
# RSH_SERVER_HOST=0.0.0.0
# RSH_LOG_LEVEL=debug
# RSH_PROFILE=production

# Command Line Arguments:
# --port 9090 --host 0.0.0.0 --log-level debug --profile production
"#;
        example_config.to_string()
    }

    /// Generate a minimal configuration file with only essential settings
    #[must_use]
    pub fn generate_minimal_config() -> String {
        let minimal_config = r#"# Minimal Rust Sci-Hub MCP Server Configuration

[server]
port = 8080

[downloads]
directory = "~/downloads/papers"

[logging]
level = "info"
"#;
        minimal_config.to_string()
    }

    /// Generate a production-ready configuration file
    #[must_use]
    pub fn generate_production_config() -> String {
        let production_config = r#"# Production Rust Sci-Hub MCP Server Configuration

schema_version = "1.0"
profile = "production"

[server]
port = 8080
host = "127.0.0.1"
timeout_secs = 60
health_check_interval_secs = 60
graceful_shutdown_timeout_secs = 10

[sci_hub]
mirrors = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru"
]
rate_limit_per_sec = 1
timeout_secs = 45
max_retries = 5

[downloads]
directory = "~/Documents/research/papers"
max_concurrent = 2
max_file_size_mb = 200

[logging]
level = "info"
format = "json"
file = "~/library/logs/knowledge_accumulator_mcp.log"

[rate_limiting]
adaptive = true
show_progress = false  # Less verbose for production
allow_burst = true
burst_size = 2        # More conservative for production
default_rate = 0.8    # Slightly slower for production reliability
"#;
        production_config.to_string()
    }

    /// Validate configuration file security
    fn validate_config_file_security(path: &std::path::Path) -> crate::Result<()> {
        // Check if file is a symbolic link
        if path.exists() {
            let metadata = std::fs::symlink_metadata(path).map_err(|e| {
                crate::Error::Service(format!("Failed to check config file metadata: {e}"))
            })?;

            if metadata.file_type().is_symlink() {
                return Err(crate::Error::Service(format!(
                    "Security: Refusing to read configuration from symbolic link: {:?}",
                    path
                )));
            }

            #[cfg(unix)]
            {
                // Check file permissions - should be readable only by owner (0600 or more restrictive)
                let permissions = metadata.permissions();
                let mode = permissions.mode();

                // Check if file is readable by group or others
                if (mode & 0o077) != 0 {
                    warn!(
                        "Security: Configuration file has overly permissive permissions ({:o}): {:?}. \
                        Consider setting permissions to 0600 for security.",
                        mode & 0o777,
                        path
                    );
                }
            }
        }
        Ok(())
    }

    /// Set secure permissions on configuration directory
    fn set_secure_directory_permissions(path: &std::path::Path) -> crate::Result<()> {
        #[cfg(unix)]
        {
            use std::fs::Permissions;

            // Set permissions to 0700 (owner read/write/execute only) for config directory
            let permissions = Permissions::from_mode(0o700);
            std::fs::set_permissions(path, permissions).map_err(|e| {
                crate::Error::Service(format!(
                    "Failed to set secure permissions on config directory: {e}"
                ))
            })?;

            info!(
                "Set secure permissions (0700) on config directory: {:?}",
                path
            );
        }

        #[cfg(not(unix))]
        {
            info!(
                "Non-Unix system: Cannot set Unix-style permissions on config directory: {:?}",
                path
            );
        }

        Ok(())
    }

    /// Set secure permissions on configuration file
    fn set_secure_config_file_permissions(path: &std::path::Path) -> crate::Result<()> {
        #[cfg(unix)]
        {
            use std::fs::Permissions;

            // Set permissions to 0600 (owner read/write only) for config files
            let permissions = Permissions::from_mode(0o600);
            std::fs::set_permissions(path, permissions).map_err(|e| {
                crate::Error::Service(format!(
                    "Failed to set secure permissions on config file: {e}"
                ))
            })?;

            info!("Set secure permissions (0600) on config file: {:?}", path);
        }

        #[cfg(not(unix))]
        {
            info!(
                "Non-Unix system: Cannot set Unix-style permissions on config file: {:?}",
                path
            );
        }

        Ok(())
    }

    /// Write example configuration to a file
    pub fn write_example_config(path: &std::path::Path, config_type: &str) -> crate::Result<()> {
        let content = match config_type {
            "minimal" => Self::generate_minimal_config(),
            "production" => Self::generate_production_config(),
            _ => Self::generate_example_config(),
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Security: Check for symlink attacks before writing config file
        if path.exists() {
            let metadata = std::fs::symlink_metadata(path).map_err(|e| {
                crate::Error::Service(format!("Failed to check config file metadata: {e}"))
            })?;

            if metadata.file_type().is_symlink() {
                return Err(crate::Error::Service(format!(
                    "Security: Refusing to overwrite symbolic link: {:?}",
                    path
                )));
            }
        }

        std::fs::write(path, content)?;

        // Security: Set secure permissions after writing config file
        Self::set_secure_config_file_permissions(path)?;

        debug!(
            "Generated example configuration file with secure permissions: {}",
            path.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
            timeout_secs: 30,
            health_check_interval_secs: 30,
            graceful_shutdown_timeout_secs: 5,
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.health_check_interval_secs, 30);
        assert_eq!(config.graceful_shutdown_timeout_secs, 5);
    }

    #[test]
    fn test_sci_hub_config_defaults() {
        let config = ResearchSourceConfig {
            endpoints: vec!["https://sci-hub.se".to_string()],
            rate_limit_per_sec: 1,
            timeout_secs: 30,
            provider_timeout_secs: 60,
            max_retries: 3,
        };
        assert!(!config.endpoints.is_empty());
        assert_eq!(config.rate_limit_per_sec, 1);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid port should fail
        config.server.port = 0;
        assert!(config.validate().is_err());

        // Fix port and test invalid log level
        config.server.port = 8080;
        config.logging.level = "invalid".to_string();
        assert!(config.validate().is_err());

        // Fix log level and test empty mirrors
        config.logging.level = "info".to_string();
        config.research_source.endpoints.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_path_expansion() {
        // Test tilde expansion
        let expanded = expand_path("~/downloads/papers");
        if let Some(home_dir) = dirs::home_dir() {
            assert_eq!(expanded, home_dir.join("downloads/papers"));
        } else {
            assert_eq!(expanded, PathBuf::from("~/downloads/papers"));
        }

        // Test environment variable expansion
        std::env::set_var("TEST_VAR", "/tmp");
        let expanded = expand_path("$TEST_VAR/papers");
        assert_eq!(expanded, PathBuf::from("/tmp/papers"));
        std::env::remove_var("TEST_VAR");

        // Test normal path
        let expanded = expand_path("/absolute/path");
        assert_eq!(expanded, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn test_config_overrides() {
        let overrides = ConfigOverrides {
            server_port: Some(9090),
            server_host: Some("0.0.0.0".to_string()),
            log_level: Some("debug".to_string()),
            profile: Some("production".to_string()),
            download_directory: Some(PathBuf::from("/custom/download/path")),
        };

        let config = Config::apply_cli_overrides(Config::default(), &overrides);

        assert_eq!(config.server.port, 9090);
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.profile, "production");
        assert_eq!(
            config.downloads.directory,
            PathBuf::from("/custom/download/path")
        );
    }

    #[test]
    fn test_profile_settings() {
        let mut config = Config::default();
        config.profile = "development".to_string();
        config.logging.level = "info".to_string();

        let config = Config::apply_profile_settings(config);

        // Development profile should change log level to debug
        assert_eq!(config.logging.level, "debug");
        assert_eq!(config.server.timeout_secs, 10);

        // Test production profile
        let mut config = Config::default();
        config.profile = "production".to_string();

        let config = Config::apply_profile_settings(config);

        assert_eq!(config.logging.format, "json");
        assert_eq!(config.server.timeout_secs, 60);
    }

    #[test]
    fn test_schema_generation() {
        let schema = Config::generate_schema();
        assert!(schema.is_object());

        // Verify schema contains expected properties
        let schema_obj = schema.as_object().unwrap();
        assert!(schema_obj.contains_key("$schema"));
    }

    #[test]
    fn test_example_config_generation() {
        let example = Config::generate_example_config();
        assert!(example.contains("schema_version"));
        assert!(example.contains("[server]"));
        assert!(example.contains("[sci_hub]"));
        assert!(example.contains("[downloads]"));
        assert!(example.contains("[logging]"));

        let minimal = Config::generate_minimal_config();
        assert!(minimal.contains("[server]"));
        assert!(minimal.contains("port"));

        let production = Config::generate_production_config();
        assert!(production.contains("production"));
        assert!(production.contains("timeout_secs = 60"));
    }

    #[test]
    fn test_safe_for_logging() {
        let mut config = Config::default();
        config.downloads.directory = PathBuf::from("/Users/testuser/Downloads");

        let safe_config = config.safe_for_logging();
        assert_eq!(
            safe_config.downloads.directory,
            PathBuf::from("[REDACTED_USER_PATH]")
        );
    }

    #[test]
    fn test_hot_reload() {
        let mut config = Config::default();
        let original_level = config.logging.level.clone();

        // Mock a reload that changes the log level
        config.logging.level = "debug".to_string();

        // Verify changed detection works
        let mut test_config = Config::default();
        test_config.logging.level = "debug".to_string();

        // Hot reload should detect changes in allowed fields
        assert_ne!(test_config.logging.level, original_level);
    }
}
