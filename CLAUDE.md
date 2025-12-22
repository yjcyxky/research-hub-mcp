# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic research MCP server providing multi-provider paper search, download, and metadata extraction. Built with Rust using the rmcp framework. Personal research use only.

## Commands

```bash
cargo nextest run                    # Run tests (parallel)
cargo nextest run TEST_NAME          # Run specific test
cargo nextest run --no-capture       # Tests with output visible
cargo clippy -- -D warnings          # Must pass before commit
cargo fmt                            # Format code
cargo build --release                # Production build
cargo run -- serve                   # Start MCP server
cargo tarpaulin --out Html           # Coverage report
cargo audit                          # Security check
```

## Architecture

The codebase uses a modular architecture with clear separation of concerns:

```
src/
├── main.rs              # CLI entry with clap for arg parsing
├── cli.rs               # Secondary CLI binary (rust-research)
├── lib.rs               # Public API exports
├── server/              # MCP server (rmcp framework)
│   ├── handler.rs       # MCP request handling
│   └── transport.rs     # Transport layer
├── tools/               # MCP tool implementations
│   ├── search.rs        # Multi-provider paper search
│   ├── search_source.rs # Source-specific search with native query syntax
│   ├── download.rs      # Paper download with fallback
│   ├── pdf_metadata.rs  # PDF metadata extraction
│   ├── verify_metadata.rs # Cross-source metadata verification
│   ├── pdf2text.rs      # PDF text extraction
│   ├── text2table.rs    # Text-to-table extraction
│   ├── bibliography.rs  # Citation generation (BibTeX, APA, etc.)
│   ├── categorize.rs    # Paper categorization
│   └── code_search.rs   # Regex code pattern search
├── client/              # External API integration
│   ├── meta_search.rs   # Meta-search orchestration
│   ├── mirror.rs        # Mirror management
│   ├── rate_limiter.rs  # Rate limiting
│   ├── circuit_breaker_service.rs
│   └── providers/       # 15 academic source implementations
│       ├── arxiv.rs, crossref.rs, semantic_scholar.rs
│       ├── pubmed_central.rs, openalex.rs, openreview.rs
│       ├── biorxiv.rs, medrxiv.rs, core.rs, mdpi.rs
│       ├── unpaywall.rs, ssrn.rs, researchgate.rs
│       ├── google_scholar.rs, sci_hub.rs
│       └── traits.rs    # SourceProvider trait
├── resilience/          # Fault tolerance
│   ├── circuit_breaker.rs
│   ├── retry.rs
│   ├── timeout.rs
│   └── health.rs
├── service/             # Daemon management
│   ├── daemon.rs, pid.rs, signals.rs, health.rs
├── config/              # Configuration (TOML, env vars)
├── python_embed.rs      # PyO3 Python embedding for PDF processing
└── error.rs             # thiserror-based error types
```

**Key patterns:**
- Tools implement MCP interface via `#[tool]` macro from rmcp
- Providers implement `SourceProvider` trait (`client/providers/traits.rs`)
- Meta-search orchestrates multiple providers with intelligent routing
- Circuit breakers + rate limiting for external API resilience

## MCP Tool Pattern

```rust
use rmcp::prelude::*;
use schemars::JsonSchema;

#[derive(Debug, Deserialize, JsonSchema)]
struct InputSchema {
    #[schemars(description = "Clear description")]
    field: String,
}

#[tool]
async fn tool_name(input: InputSchema) -> Result<Value> {
    // Validate input
    // Call service with timeout
    // Return JSON response
}
```

## Provider Pattern

All providers implement `SourceProvider` trait in `src/client/providers/traits.rs`:

```rust
#[async_trait]
pub trait SourceProvider: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn priority(&self) -> u8;
    fn supported_search_types(&self) -> Vec<SearchType>;
    fn supports_full_text(&self) -> bool;
    fn base_delay(&self) -> Duration;

    // Query format documentation (added 2025-12-22)
    fn query_format_help(&self) -> &'static str;
    fn query_examples(&self) -> Vec<(&'static str, &'static str)>;
    fn native_query_syntax(&self) -> Option<&'static str>;

    async fn search(&self, query: &SearchQuery, ctx: &SearchContext) -> ProviderResult<Vec<PaperMetadata>>;
    async fn get_by_doi(&self, doi: &str, ctx: &SearchContext) -> Result<Option<PaperMetadata>, ProviderError>;
    async fn health_check(&self, ctx: &SearchContext) -> Result<bool, ProviderError>;
}
```

## MCP Tools Reference

### search_source
Search a specific academic source with native query syntax support.

```json
{
  "source": "arxiv",           // Required: arxiv, pubmed, crossref, semantic_scholar, etc.
  "query": "au:Hinton ti:deep learning",  // Source-specific query
  "limit": 10,                 // Optional, default: 10
  "offset": 0                  // Optional, default: 0
}
```

### verify_metadata
Cross-reference metadata against multiple authoritative sources.

```json
{
  "doi": "10.1038/nature12373",  // DOI to verify
  "sources": ["crossref", "pubmed", "semantic_scholar", "openalex"]  // Optional
}
```

Returns confidence scores and identifies discrepancies across sources.

### pdf_metadata
Extract metadata from local PDF files.

```json
{
  "file_path": "/path/to/paper.pdf"
}
```

## Code Style

- Use `thiserror` for errors - no raw strings
- Use `tracing` for logging - never println
- Use `?` operator - avoid unwrap/expect
- Keep functions under 50 lines
- Test naming: `test_<function>_<case>_<expected>`
- Always use timeout for I/O operations

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{MockServer, Mock, ResponseTemplate};

    #[tokio::test]
    async fn test_function_success() {
        let mock_server = MockServer::start().await;
        Mock::given(matchers::method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        let result = function_under_test().await;
        assert!(result.is_ok());
    }
}
```

For long-running tests:
```rust
#[tokio::test(flavor = "multi_thread")]
async fn long_test() { /* ... */ }
```

## Configuration

Order of precedence:
1. CLI arguments (clap)
2. Environment variables (`RUST_RESEARCH_MCP_*`)
3. config.toml
4. Defaults

## Python Embedding

Uses PyO3 for PDF processing (`src/python_embed.rs`). Python package in `python/` with PDF extraction capabilities.

### Python CLI Commands (t2t)

```bash
# Text-to-table extraction
t2t text2table input.txt -o output.json        # Single file
t2t text2table batch.csv -o results.jsonl      # Batch processing (CSV/TSV/JSONL input)

# Server mode
t2t-server --model Qwen/Qwen2.5-7B --port 8000

# Options
--max-new-tokens 4096      # Max tokens for generation (default: 4096)
--request-timeout 600      # Request timeout in seconds (default: 600)
--gliner-threshold 0.5     # Entity extraction threshold
```

Input format auto-detection:
- `.txt` - Single text file
- `.csv` - Batch CSV (text_column header)
- `.tsv` - Batch TSV (text_column header)
- `.jsonl` - Batch JSONL (text field)

## TDD Workflow

1. Write failing test first
2. Run `cargo nextest run TEST_NAME` to verify failure
3. Write minimal code to pass
4. Verify test passes
5. Refactor if needed
6. Run full suite + clippy before commit

## Git Workflow

```bash
git checkout -b feature/task-name
cargo fmt && cargo clippy -- -D warnings
cargo nextest run
git add -A && git commit -m "feat: description"
```

## Deployment

Two binaries: `rust-research-mcp` (MCP server) and `rust-research` (CLI tool).

Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "rust-research-mcp": {
      "command": "/path/to/rust-research-mcp",
      "args": ["--download-dir", "~/downloads/research_papers"],
      "env": {"RUST_LOG": "info"}
    }
  }
}
```

Daemon mode: `rust-research-mcp --daemon --pid-file /var/run/rust-research-mcp.pid --health-port 8090`

## Migration Guide: search_papers to search_source

The `search_papers` tool has been replaced with `search_source` for more precise control over academic source queries.

### Key Changes

| Old (search_papers) | New (search_source) |
|---------------------|---------------------|
| Multi-provider parallel search | Single-source targeted search |
| Automatic source selection | Explicit source parameter required |
| `mode` parameter (auto/doi/title) | `search_type` parameter (optional) |
| Aggregated results | Source-specific results |

### Usage Examples

**Old approach:**
```json
{
  "tool": "search_papers",
  "arguments": {
    "query": "machine learning",
    "limit": 10,
    "mode": "title"
  }
}
```

**New approach:**
```json
{
  "tool": "search_source",
  "arguments": {
    "source": "semantic_scholar",
    "query": "machine learning",
    "limit": 10
  }
}
```

### Native Query Syntax

Each source supports its own native query syntax for optimal results:

- **ArXiv**: `ti:neural networks` (title), `au:Smith` (author), `abs:deep learning` (abstract)
- **PubMed Central**: `cancer[tiab]` (title/abstract), `2023[pdat]` (publication date)
- **Semantic Scholar**: Direct keyword search, use `fieldsOfStudy:` for field filtering
- **CrossRef**: Standard query text, DOI resolution

Use `list_sources` tool to see all available sources with query format help:
```json
{
  "tool": "list_sources",
  "arguments": {"include_examples": true}
}
```

### CLI Migration

**Old:**
```bash
rust-research search "machine learning" --mode title --limit 10
```

**New:**
```bash
rust-research search "machine learning" --source semantic_scholar --limit 10
rust-research list-sources  # See available sources
```
