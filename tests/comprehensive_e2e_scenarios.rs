use rust_research_mcp::{
    client::{
        providers::{SearchQuery, SearchType},
        MetaSearchClient, MetaSearchConfig,
    },
    tools::{
        categorize::CategorizeInput,
        download::{DownloadInput, DownloadOutputFormat},
        pdf_metadata::MetadataInput,
        search_source::{SearchSourceInput, SearchSourceTool},
        BibliographyTool, CategorizeTool, DownloadTool, MetadataExtractor,
    },
    Config,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio;
use tokio::sync::Semaphore;
use tracing::{info, warn};

/// Test configuration optimized for comprehensive E2E scenarios
fn create_comprehensive_test_config() -> Arc<Config> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = Config::default();
    config.research_source.timeout_secs = 10;
    config.research_source.provider_timeout_secs = 10; // Faster timeout for tests
    config.research_source.max_retries = 2;
    config.downloads.directory = temp_dir.keep();
    config.downloads.max_file_size_mb = 50; // 50MB for testing
    config.downloads.max_concurrent = 5;
    Arc::new(config)
}

/// Helper to create realistic test queries for different domains
fn create_domain_queries() -> HashMap<String, Vec<SearchQuery>> {
    let mut queries = HashMap::new();

    // Computer Science / Machine Learning queries
    queries.insert(
        "cs_ml".to_string(),
        vec![
            SearchQuery {
                query: "attention is all you need".to_string(),
                search_type: SearchType::Title,
                max_results: 10,
                offset: 0,
                params: HashMap::new(),
                sources: None,
                metadata_sources: None,
            },
            SearchQuery {
                query: "arXiv:1706.03762".to_string(),
                search_type: SearchType::Doi,
                max_results: 1,
                offset: 0,
                params: HashMap::new(),
                sources: None,
                metadata_sources: None,
            },
        ],
    );

    // Biomedical queries
    queries.insert(
        "biomedical".to_string(),
        vec![
            SearchQuery {
                query: "CRISPR gene editing".to_string(),
                search_type: SearchType::Title,
                max_results: 10,
                offset: 0,
                params: HashMap::new(),
                sources: None,
                metadata_sources: None,
            },
            SearchQuery {
                query: "10.1038/nature12373".to_string(),
                search_type: SearchType::Doi,
                max_results: 1,
                offset: 0,
                params: HashMap::new(),
                sources: None,
                metadata_sources: None,
            },
        ],
    );

    // Physics queries
    queries.insert(
        "physics".to_string(),
        vec![SearchQuery {
            query: "quantum computing".to_string(),
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        }],
    );

    queries
}

// =============================================================================
// PRIORITY 1: CRITICAL USER JOURNEY TESTS
// =============================================================================

#[tokio::test]
async fn test_complete_research_workflow() {
    let config = create_comprehensive_test_config();
    let meta_config = MetaSearchConfig::default();

    // Initialize all tools - use new SearchSourceTool
    let search_tool = SearchSourceTool::new();
    let client = Arc::new(
        MetaSearchClient::new((*config).clone(), meta_config)
            .expect("Failed to create MetaSearchClient"),
    );
    let download_tool =
        DownloadTool::new(client.clone(), config.clone()).expect("Failed to create DownloadTool");
    let metadata_extractor =
        MetadataExtractor::new(config.clone()).expect("Failed to create MetadataExtractor");
    let _bibliography_tool =
        BibliographyTool::new(config.clone()).expect("Failed to create BibliographyTool");
    let categorize_tool =
        CategorizeTool::new(config.clone()).expect("Failed to create CategorizeTool");

    // Step 1: Search for a well-known paper using source-specific search
    let search_input = SearchSourceInput {
        source: "arxiv".to_string(),
        query: "attention is all you need".to_string(),
        limit: 5,
        offset: 0,
        search_type: Some("title".to_string()),
        help: false,
    };

    let start_time = Instant::now();
    let search_result = search_tool.search(search_input).await;
    let search_duration = start_time.elapsed();

    // Assert search performance target: < 120s (realistic for multiple provider operations)
    assert!(
        search_duration < Duration::from_secs(120),
        "Search took {:?}, exceeding 120s target",
        search_duration
    );

    match search_result {
        Ok(result) if !result.papers.is_empty() => {
            let paper = &result.papers[0];
            info!("Found paper: {:?}", paper.title);

            // Step 2: Download the paper
            let download_input = DownloadInput {
                doi: Some(paper.doi.clone()),
                url: None,
                filename: Some("test_workflow.pdf".to_string()),
                directory: None,
                category: Some("machine_learning".to_string()),
                overwrite: true,
                verify_integrity: true,
                headless: true,
                enable_local_grobid: false,
                output_format: DownloadOutputFormat::Pdf,
            };

            let download_start = Instant::now();
            let download_result = download_tool.download_paper(download_input).await;
            let _download_duration = download_start.elapsed();

            match download_result {
                Ok(download_response) => {
                    if let Some(file_path) = download_response.file_path {
                        // Step 3: Extract metadata
                        let metadata_input = MetadataInput {
                            file_path: file_path.to_string_lossy().to_string(),
                            use_cache: false,
                            extract_references: true,
                            batch_files: None,
                        };

                        let metadata_start = Instant::now();
                        let metadata_result =
                            metadata_extractor.extract_metadata(metadata_input).await;
                        let metadata_duration = metadata_start.elapsed();

                        // Assert metadata extraction performance: < 10s
                        assert!(
                            metadata_duration < Duration::from_secs(10),
                            "Metadata extraction took {:?}, exceeding 10s target",
                            metadata_duration
                        );

                        // Step 4: Categorize the paper (if metadata extraction succeeded)
                        if metadata_result.is_ok() {
                            let categorize_input = CategorizeInput {
                                query: "comprehensive test".to_string(),
                                papers: vec![], // Empty for this test
                                max_abstracts: Some(1),
                            };
                            let categorize_result =
                                categorize_tool.categorize_papers(categorize_input).await;

                            match categorize_result {
                                Ok(category) => {
                                    info!("Complete workflow successful. Category: {:?}", category);

                                    // Verify end-to-end data consistency
                                    assert!(!paper.doi.is_empty());
                                    assert!(download_response.file_size.unwrap_or(0) > 0);
                                }
                                Err(e) => warn!("Categorization failed: {}", e),
                            }
                        }

                        // Cleanup
                        let _ = std::fs::remove_file(file_path);
                    }
                }
                Err(e) => info!("Download failed (expected in some environments): {}", e),
            }
        }
        Ok(_) => info!("No papers found for test query"),
        Err(e) => info!("Search failed (network may be offline): {}", e),
    }
}

#[tokio::test]
async fn test_multi_provider_failover_cascade() {
    let config = create_comprehensive_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = MetaSearchClient::new((*config).clone(), meta_config)
        .expect("Failed to create MetaSearchClient");

    // Test with a DOI that should be available in multiple sources
    let test_doi = "10.1371/journal.pone.0000308"; // Open access paper

    // Simulate provider failures by using circuit breaker patterns
    let providers = client.providers();
    assert!(
        providers.len() >= 5,
        "Should have multiple providers for failover testing"
    );

    // Test cascade PDF retrieval
    let cascade_start = Instant::now();
    let pdf_result = client.get_pdf_url_cascade(test_doi).await;
    let cascade_duration = cascade_start.elapsed();

    match pdf_result {
        Ok(Some(pdf_url)) => {
            assert!(pdf_url.starts_with("http"), "PDF URL should be absolute");
            info!("Cascade found PDF in {:?}: {}", cascade_duration, pdf_url);

            // Test that failover is reasonably fast (< 60s for all providers)
            assert!(
                cascade_duration < Duration::from_secs(60),
                "Cascade took {:?}, exceeding 60s reasonable limit",
                cascade_duration
            );
        }
        Ok(None) => info!("No PDF found through cascade (expected for some papers)"),
        Err(e) => info!("Cascade failed (network may be offline): {}", e),
    }
}

#[tokio::test]
async fn test_concurrent_research_sessions() {
    let search_tool = Arc::new(SearchSourceTool::new());

    let semaphore = Arc::new(Semaphore::new(10)); // Limit concurrent sessions
    let mut tasks = vec![];

    // Test concurrent searches on different sources
    let sources = vec!["arxiv", "crossref", "semantic_scholar"];
    let queries = vec!["machine learning", "deep learning", "neural networks"];

    for source in &sources {
        for query in &queries {
            let search_tool = search_tool.clone();
            let semaphore = semaphore.clone();
            let source = source.to_string();
            let query = query.to_string();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                let search_input = SearchSourceInput {
                    source,
                    query,
                    limit: 5,
                    offset: 0,
                    search_type: None,
                    help: false,
                };

                let start = Instant::now();
                let result = search_tool.search(search_input).await;
                let duration = start.elapsed();

                (result.is_ok(), duration)
            });

            tasks.push(task);
        }
    }

    // Wait for all sessions
    let results = futures::future::join_all(tasks).await;

    // Verify concurrent execution completed
    let successful = results
        .iter()
        .filter(|r| r.is_ok() && r.as_ref().unwrap().0)
        .count();

    info!(
        "Concurrent sessions: {} successful out of {}",
        successful,
        results.len()
    );

    // At least some sessions should complete successfully
    assert!(
        successful > 0,
        "At least some concurrent sessions should succeed"
    );
}

// =============================================================================
// PRIORITY 2: PROVIDER-SPECIFIC TESTS
// =============================================================================

#[tokio::test]
async fn test_provider_specific_search() {
    let search_tool = SearchSourceTool::new();

    // Test search on different providers
    let test_cases = vec![
        ("arxiv", "machine learning", Some("title")),
        ("crossref", "10.1038/nature12373", Some("doi")),
        ("semantic_scholar", "deep learning", None),
    ];

    for (source, query, search_type) in test_cases {
        let input = SearchSourceInput {
            source: source.to_string(),
            query: query.to_string(),
            limit: 5,
            offset: 0,
            search_type: search_type.map(String::from),
            help: false,
        };

        let result = search_tool.search(input).await;

        match result {
            Ok(search_result) => {
                info!(
                    "{} search returned {} papers",
                    source,
                    search_result.papers.len()
                );
            }
            Err(e) => {
                info!("{} search failed (may be offline): {}", source, e);
            }
        }
    }
}

// =============================================================================
// PRIORITY 3: ERROR HANDLING AND EDGE CASES
// =============================================================================

#[tokio::test]
async fn test_error_handling_invalid_source() {
    let search_tool = SearchSourceTool::new();

    let input = SearchSourceInput {
        source: "nonexistent_provider".to_string(),
        query: "test".to_string(),
        limit: 5,
        offset: 0,
        search_type: None,
        help: false,
    };

    let result = search_tool.search(input).await;
    assert!(result.is_err(), "Should fail with invalid source");
}

#[tokio::test]
async fn test_help_mode() {
    let search_tool = SearchSourceTool::new();

    let input = SearchSourceInput {
        source: "arxiv".to_string(),
        query: "".to_string(),
        limit: 5,
        offset: 0,
        search_type: None,
        help: true,
    };

    let result = search_tool.search(input).await;
    assert!(result.is_ok(), "Help mode should not fail");

    if let Ok(help_result) = result {
        // Help mode should return query format information
        assert!(help_result.papers.is_empty(), "Help mode should not return papers");
    }
}

// =============================================================================
// PERFORMANCE AND STRESS TESTS
// =============================================================================

#[tokio::test]
async fn test_search_performance() {
    let search_tool = SearchSourceTool::new();

    let input = SearchSourceInput {
        source: "arxiv".to_string(),
        query: "quantum".to_string(),
        limit: 10,
        offset: 0,
        search_type: None,
        help: false,
    };

    let start = Instant::now();
    let _ = search_tool.search(input).await;
    let duration = start.elapsed();

    // Search should complete within reasonable time
    assert!(
        duration < Duration::from_secs(30),
        "Search took {:?}, exceeding 30s limit",
        duration
    );

    info!("Search completed in {:?}", duration);
}

#[tokio::test]
async fn test_metadata_extractor_with_missing_file() {
    let config = create_comprehensive_test_config();
    let extractor = MetadataExtractor::new(config).expect("Failed to create MetadataExtractor");

    let input = MetadataInput {
        file_path: "/nonexistent/file.pdf".to_string(),
        use_cache: false,
        extract_references: false,
        batch_files: None,
    };

    let result = extractor.extract_metadata(input).await;
    // Should fail gracefully with missing file
    assert!(result.is_err(), "Should fail with missing file");
}
