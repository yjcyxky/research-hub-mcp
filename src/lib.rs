//! Rust Research MCP - Academic Paper Search and Metadata Extraction
//!
//! This crate provides a Model Context Protocol (MCP) server for searching and downloading
//! academic papers from multiple sources including `arXiv`, `Semantic Scholar`, `CrossRef`, and more.

#![allow(clippy::cognitive_complexity)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::single_match_else)]
#![allow(clippy::match_single_binding)]
#![allow(clippy::unnecessary_debug_formatting)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::unused_self)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::unnecessary_first_then_check)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::if_not_else)]
#![allow(clippy::unnecessary_semicolon)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::float_cmp)]
#![allow(clippy::format_push_string)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::field_reassign_with_default)]

// pub mod adapters;
pub mod client;
pub mod config;
// pub mod di;
pub mod error;
// pub mod ports;
// pub mod repositories;
pub mod resilience;
pub mod server;
pub mod service;
pub mod services;
pub mod tools;
pub mod python_embed;

// pub use adapters::{
//     MetaSearchAdapter, MultiProviderAdapter, PaperDownloadAdapter, PdfMetadataAdapter,
// };
pub use client::{Doi, MetaSearchClient, MetaSearchConfig, PaperMetadata};
pub use config::{Config, ConfigOverrides};
// pub use di::{ServiceContainer, ServiceScope};
pub use error::{Error, Result};
// pub use ports::{DownloadServicePort, MetadataServicePort, ProviderServicePort, SearchServicePort};
// pub use repositories::{
//     CacheRepository, ConfigRepository, InMemoryCacheRepository, InMemoryConfigRepository,
//     InMemoryPaperRepository, PaperQuery, PaperRepository, Repository, RepositoryError,
//     RepositoryResult, RepositoryStats,
// };
pub use resilience::health::HealthCheckManager;
pub use resilience::{CircuitBreaker, RetryConfig, RetryPolicy, TimeoutConfig, TimeoutExt};
pub use server::Server;
pub use service::{DaemonConfig, DaemonService, HealthCheck, PidFile, SignalHandler};
pub use tools::{
    BibliographyTool, CategorizeTool, CodeSearchTool, DownloadTool, ListSourcesTool,
    MetadataExtractor, Pdf2TextTool, SearchSourceTool, VerifyMetadataTool,
};
