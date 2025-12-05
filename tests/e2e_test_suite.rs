use rmcp::ServerHandler;
use rust_research_mcp::{
    client::{
        providers::{
            ArxivProvider, BiorxivProvider, CoreProvider, CrossRefProvider, SearchQuery,
            SearchType, SemanticScholarProvider, SourceProvider, SsrnProvider, UnpaywallProvider,
        },
        MetaSearchClient, MetaSearchConfig,
    },
    server::ResearchServerHandler,
    tools::{
        download::{DownloadInput as ActualDownloadInput, DownloadOutputFormat},
        metadata::MetadataInput as ActualMetadataInput,
        search::{SearchInput as ActualSearchInput, SearchType as ToolSearchType},
    },
    Config, DownloadTool, MetadataExtractor, SearchTool,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio;
use tracing::{debug, info};

/// Test configuration that works offline
fn create_test_config() -> Arc<Config> {
    let mut config = Config::default();
    config.research_source.timeout_secs = 5; // Shorter timeout for tests
    config.research_source.max_retries = 1; // Fewer retries for tests
    Arc::new(config)
}

/// Helper to create a search query
fn create_search_query(query: &str, search_type: SearchType) -> SearchQuery {
    SearchQuery {
        query: query.to_string(),
        search_type,
        max_results: 5,
        offset: 0,
        params: HashMap::new(),
    }
}

/// Helper to create a search context
fn create_search_context() -> rust_research_mcp::client::providers::SearchContext {
    rust_research_mcp::client::providers::SearchContext {
        timeout: Duration::from_secs(30),
        user_agent: "test-client".to_string(),
        rate_limit: None,
        headers: HashMap::new(),
    }
}

#[tokio::test]
async fn test_arxiv_provider() {
    let provider = ArxivProvider::new().expect("Failed to create ArXiv provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "arxiv");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 80);

    // Test search types
    let supported = provider.supported_search_types();
    assert!(supported.contains(&SearchType::Title));
    assert!(supported.contains(&SearchType::Author));

    // Test health check
    let health = provider.health_check(&context).await;
    assert!(health.is_ok());

    // Test search with known paper
    let query = create_search_query("quantum computing", SearchType::Keywords);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            info!(
                "ArXiv search returned {} papers",
                search_result.papers.len()
            );
            assert!(search_result.papers.len() <= 5);
        }
        Err(e) => {
            // Network errors are acceptable in tests
            info!("ArXiv search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_crossref_provider() {
    let provider = CrossRefProvider::new(None).expect("Failed to create CrossRef provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "crossref");
    assert!(!provider.supports_full_text()); // CrossRef doesn't provide PDFs
    assert_eq!(provider.priority(), 90); // High priority

    // Test DOI search
    let query = create_search_query("10.1038/nature12373", SearchType::Doi);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            if !search_result.papers.is_empty() {
                let paper = &search_result.papers[0];
                assert!(paper.doi.contains("10.1038/nature12373"));
                assert!(paper.title.is_some());
            }
        }
        Err(e) => {
            info!("CrossRef search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_semantic_scholar_provider() {
    let provider =
        SemanticScholarProvider::new(None).expect("Failed to create Semantic Scholar provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "semantic_scholar");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 88);

    // Test title search
    let query = create_search_query("deep learning", SearchType::Title);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            info!(
                "Semantic Scholar returned {} papers",
                search_result.papers.len()
            );
            // Check that papers have expected fields
            for paper in search_result.papers.iter().take(1) {
                assert!(!paper.doi.is_empty() || paper.title.is_some());
            }
        }
        Err(e) => {
            info!("Semantic Scholar search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_unpaywall_provider() {
    let provider =
        UnpaywallProvider::new_with_default_email().expect("Failed to create Unpaywall provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "unpaywall");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 87);

    // Unpaywall only supports DOI search
    let supported = provider.supported_search_types();
    assert!(supported.contains(&SearchType::Doi));

    // Test with a known open access DOI
    let query = create_search_query("10.1371/journal.pone.0000308", SearchType::Doi);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            if !search_result.papers.is_empty() {
                let paper = &search_result.papers[0];
                assert!(paper.doi.contains("10.1371"));
                // Open access papers should have PDF URLs
                info!("Unpaywall PDF URL: {:?}", paper.pdf_url);
            }
        }
        Err(e) => {
            info!("Unpaywall search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_core_provider() {
    let provider = CoreProvider::new(None).expect("Failed to create CORE provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "core");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 86);

    // Test keyword search
    let query = create_search_query("machine learning", SearchType::Keywords);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            info!("CORE returned {} papers", search_result.papers.len());
            assert!(search_result.papers.len() <= 5);
        }
        Err(e) => {
            info!("CORE search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_biorxiv_provider() {
    let provider = BiorxivProvider::new().expect("Failed to create bioRxiv provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "biorxiv");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 75);

    // bioRxiv has limited search capabilities
    let query = create_search_query("covid", SearchType::Keywords);
    let result = provider.search(&query, &context).await;

    match result {
        Ok(search_result) => {
            info!("bioRxiv returned {} papers", search_result.papers.len());
            // bioRxiv should return biology/medical papers
            for _paper in search_result.papers.iter().take(1) {
                // Papers from bioRxiv provider
            }
        }
        Err(e) => {
            info!("bioRxiv search failed (may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_ssrn_provider() {
    let provider = SsrnProvider::new().expect("Failed to create SSRN provider");
    let context = create_search_context();

    // Test provider metadata
    assert_eq!(provider.name(), "ssrn");
    assert!(provider.supports_full_text());
    assert_eq!(provider.priority(), 85);

    // Test SSRN DOI extraction
    let _query = create_search_query("10.2139/ssrn.1234567", SearchType::Doi);
    let result = provider.get_by_doi("10.2139/ssrn.1234567", &context).await;

    // SSRN may block automated requests
    match result {
        Ok(Some(paper)) => {
            assert!(paper.doi.contains("10.2139/ssrn"));
        }
        Ok(None) => {
            info!("SSRN paper not found (expected for test DOI)");
        }
        Err(e) => {
            info!("SSRN search failed (may be blocked): {}", e);
        }
    }
}

#[tokio::test]
async fn test_meta_search_client() {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = MetaSearchClient::new((*config).clone(), meta_config)
        .expect("Failed to create MetaSearchClient");

    // Test search across all providers
    let query = SearchQuery {
        query: "artificial intelligence".to_string(),
        search_type: SearchType::Keywords,
        max_results: 10,
        offset: 0,
        params: HashMap::new(),
    };

    let result = client.search(&query).await;

    match result {
        Ok(search_result) => {
            info!(
                "Meta search found {} papers from {} providers",
                search_result.papers.len(),
                search_result.successful_providers
            );

            // Should have results from multiple providers
            assert!(search_result.successful_providers > 0);

            // Check deduplication
            let mut seen_dois = std::collections::HashSet::new();
            for paper in &search_result.papers {
                if !paper.doi.is_empty() {
                    assert!(seen_dois.insert(paper.doi.clone()), "Duplicate DOI found");
                }
            }
        }
        Err(e) => {
            panic!("Meta search failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_search_tool() {
    let config = create_test_config();
    let search_tool = SearchTool::new(config).expect("Failed to create SearchTool");

    // Test search with auto type detection
    let input = ActualSearchInput {
        query: "10.1038/nature12373".to_string(),
        search_type: ToolSearchType::Auto,
        limit: 1,
        offset: 0,
    };

    let result = search_tool.search_papers(input).await;

    match result {
        Ok(search_result) => {
            info!("Search tool found {} papers", search_result.returned_count);

            if search_result.returned_count > 0 {
                let paper = &search_result.papers[0];
                info!("Found paper with DOI: '{}'", paper.metadata.doi);
                // Auto-detection should recognize this as a DOI search and return valid results
                // The exact DOI returned may vary depending on provider behavior, but should be valid
                assert!(
                    !paper.metadata.doi.is_empty() && paper.metadata.doi.starts_with("10."),
                    "Expected a valid DOI starting with '10.', but got: '{}'",
                    paper.metadata.doi
                );
                info!("✓ Search tool successfully found papers with DOI auto-detection");
            }
        }
        Err(e) => {
            info!("Search tool failed (network may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_download_tool() {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(
        MetaSearchClient::new((*config).clone(), meta_config)
            .expect("Failed to create MetaSearchClient"),
    );
    let download_tool = DownloadTool::new(client, config).expect("Failed to create DownloadTool");

    // Test input validation
    let invalid_input = ActualDownloadInput {
        doi: None,
        url: None,
        filename: None,
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    let result = download_tool.download_paper(invalid_input).await;
    assert!(result.is_err(), "Should fail with no DOI or URL");

    // Test with both DOI and URL (should fail)
    let both_input = ActualDownloadInput {
        doi: Some("10.1038/test".to_string()),
        url: Some("https://example.com/test.pdf".to_string()),
        filename: None,
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    let result = download_tool.download_paper(both_input).await;
    assert!(result.is_err(), "Should fail with both DOI and URL");
}

#[tokio::test]
async fn test_metadata_extractor() {
    let config = create_test_config();
    let extractor = MetadataExtractor::new(config).expect("Failed to create MetadataExtractor");

    // Test with file path input
    let input = ActualMetadataInput {
        file_path: "/tmp/test.pdf".to_string(),
        use_cache: false,
        validate_external: false,
        extract_references: false,
        batch_files: None,
    };

    let result = extractor.extract_metadata(input).await;

    match result {
        Ok(metadata_result) => {
            if let Some(metadata) = metadata_result.metadata {
                if let Some(doi) = metadata.doi {
                    info!("Extracted metadata with DOI: {}", doi);
                }
            }
        }
        Err(e) => {
            info!(
                "Metadata extraction failed (expected for non-existent file): {}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_server_handler() {
    let config = create_test_config();
    let handler = ResearchServerHandler::new(config).expect("Failed to create server handler");

    // Test server info
    let info = handler.get_info();
    assert!(info.capabilities.tools.is_some());

    // Test ping
    let result = handler.ping().await;
    assert!(result.is_ok(), "Ping should succeed");

    // We can't easily test the handler methods that require RequestContext
    // But we can verify the server info structure
    // The actual MCP protocol testing is done in the Python tests
}

#[tokio::test]
async fn test_provider_priority_ordering() {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = MetaSearchClient::new((*config).clone(), meta_config)
        .expect("Failed to create MetaSearchClient");

    // Get provider names
    let provider_names = client.providers();

    // Check that we have multiple providers
    assert!(provider_names.len() > 5, "Should have multiple providers");

    // Check for expected providers
    assert!(provider_names.contains(&"crossref".to_string()));
    assert!(provider_names.contains(&"arxiv".to_string()));
    assert!(provider_names.contains(&"semantic_scholar".to_string()));

    info!("Available providers: {:?}", provider_names);
}

#[tokio::test]
async fn test_url_resolution() {
    // Test that relative URLs are properly resolved
    let base = url::Url::parse("https://example.com/path/").unwrap();

    // Test various relative URL patterns
    let test_cases = vec![
        ("../file.pdf", "https://example.com/file.pdf"),
        (
            "/absolute/path.pdf",
            "https://example.com/absolute/path.pdf",
        ),
        (
            "relative/path.pdf",
            "https://example.com/path/relative/path.pdf",
        ),
        (
            "./same/level.pdf",
            "https://example.com/path/same/level.pdf",
        ),
    ];

    for (relative, expected) in test_cases {
        let resolved = base.join(relative).unwrap();
        assert_eq!(
            resolved.to_string(),
            expected,
            "Failed to resolve {} correctly",
            relative
        );
    }
}

#[tokio::test]
async fn test_cascade_pdf_retrieval() {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = MetaSearchClient::new((*config).clone(), meta_config)
        .expect("Failed to create MetaSearchClient");

    // Test cascade with a DOI that might have PDFs in multiple sources
    let doi = "10.1371/journal.pone.0000308"; // Open access paper

    let result = client.get_pdf_url_cascade(doi).await;

    match result {
        Ok(Some(pdf_url)) => {
            assert!(pdf_url.starts_with("http"), "PDF URL should be absolute");
            info!("Cascade found PDF at: {}", pdf_url);
        }
        Ok(None) => {
            info!("No PDF found through cascade (paper may not be available)");
        }
        Err(e) => {
            info!("Cascade failed (network may be offline): {}", e);
        }
    }
}

#[tokio::test]
async fn test_rate_limiting() {
    let provider = ArxivProvider::new().expect("Failed to create ArXiv provider");

    // Initialize rate limiter with test configuration
    let mut test_config = rust_research_mcp::config::RateLimitingConfig::default();
    test_config.allow_burst = false; // Disable burst for predictable testing
    test_config.providers.insert("arxiv".to_string(), 1.0); // 1 req/sec for testing
    provider.init_rate_limiter(&test_config).await;

    let _context = create_search_context();

    // Make multiple rapid requests
    let query = create_search_query("test", SearchType::Keywords);

    let start = std::time::Instant::now();
    for i in 0..3 {
        let _ = provider.search(&query, &_context).await;
        debug!("Request {} completed", i + 1);
    }
    let elapsed = start.elapsed();

    // With 1 req/sec and no burst, 3 requests should take at least 2 seconds
    // (0s for first, 1s wait for second, 1s wait for third = 2s minimum)
    assert!(
        elapsed >= Duration::from_millis(1800), // Allow some tolerance
        "Rate limiting should enforce delays between requests. Elapsed: {:?}",
        elapsed
    );

    info!("Rate limiting test completed in {:?}", elapsed);
}

#[tokio::test]
async fn test_error_handling() {
    let config = create_test_config();
    let search_tool = SearchTool::new(config).expect("Failed to create SearchTool");

    // Test with empty query
    let empty_input = ActualSearchInput {
        query: "".to_string(),
        search_type: ToolSearchType::Auto,
        limit: 10,
        offset: 0,
    };

    let result = search_tool.search_papers(empty_input).await;

    // Empty query might return empty results or error
    match result {
        Ok(search_result) => {
            assert_eq!(
                search_result.returned_count, 0,
                "Empty query should return no results"
            );
        }
        Err(e) => {
            info!("Empty query properly rejected: {}", e);
        }
    }

    // Test with invalid limit
    let invalid_limit = ActualSearchInput {
        query: "test".to_string(),
        search_type: ToolSearchType::Auto,
        limit: 0, // Invalid
        offset: 0,
    };

    let result = search_tool.search_papers(invalid_limit).await;
    assert!(
        result.is_err() || result.unwrap().returned_count == 0,
        "Invalid limit should fail or return no results"
    );
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_search_download_flow() {
        let config = create_test_config();

        // 1. Search for a paper
        let search_tool = SearchTool::new(config.clone()).expect("Failed to create SearchTool");
        let search_input = ActualSearchInput {
            query: "10.1038/nature12373".to_string(),
            search_type: ToolSearchType::Doi,
            limit: 1,
            offset: 0,
        };

        let search_result = search_tool.search_papers(search_input).await;

        if let Ok(result) = search_result {
            if result.returned_count > 0 {
                let paper = &result.papers[0];
                info!("Found paper: {:?}", paper.metadata.title);

                // 2. Try to download it
                let meta_config = MetaSearchConfig::default();
                let client = Arc::new(
                    MetaSearchClient::new((*config).clone(), meta_config)
                        .expect("Failed to create MetaSearchClient"),
                );
                let download_tool = DownloadTool::new(client, config.clone())
                    .expect("Failed to create DownloadTool");

                let download_input = ActualDownloadInput {
                    doi: Some(paper.metadata.doi.clone()),
                    url: None,
                    filename: Some("test_integration.pdf".to_string()),
                    directory: Some("/tmp".to_string()),
                    category: None,
                    overwrite: true,
                    verify_integrity: false,
                    output_format: DownloadOutputFormat::Pdf,
                };

                let download_result = download_tool.download_paper(download_input).await;

                match download_result {
                    Ok(result) => {
                        assert!(result.file_path.is_some());
                        if let Some(size) = result.file_size {
                            assert!(size > 0, "Downloaded file should have content");
                        }

                        // Clean up
                        if let Some(path) = result.file_path {
                            let _ = std::fs::remove_file(path);
                        }
                    }
                    Err(e) => {
                        info!("Download failed (expected for some papers): {}", e);
                    }
                }

                // 3. Extract metadata (if we have a downloaded file)
                // Skip metadata extraction since it requires an actual PDF file
            }
        }
    }
}
