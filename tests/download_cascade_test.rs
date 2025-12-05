use rust_research_mcp::client::meta_search::{MetaSearchClient, MetaSearchConfig};
use rust_research_mcp::client::providers::{SearchQuery, SearchType};
use rust_research_mcp::tools::download::{DownloadInput, DownloadOutputFormat, DownloadTool};
use rust_research_mcp::{Config, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

/// Create a test configuration
fn create_test_config() -> Config {
    let mut config = Config::default();
    // Use a temporary directory for downloads in tests
    config.downloads.directory = TempDir::new().unwrap().keep();
    config
}

/// Test the download cascade behavior with mock data
#[tokio::test]
async fn test_download_cascade_with_provider_failures() -> Result<()> {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new(config.clone(), meta_config)?);
    let download_tool = DownloadTool::new(client.clone(), Arc::new(config))?;

    // Test with a DOI that should trigger cascade behavior
    let download_input = DownloadInput {
        doi: Some("10.invalid/test_cascade_should_fail".to_string()),
        url: None,
        filename: Some("test_cascade.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    // This should attempt the cascade and eventually fail gracefully
    let result = download_tool.download_paper(download_input).await;

    // We expect this to fail because it's a test DOI, but it should fail gracefully
    // with a proper error message indicating no providers found the paper
    assert!(result.is_err());

    let error_message = result.unwrap_err().to_string();
    println!("Error message: {}", error_message);
    assert!(
        error_message.contains("DOI")
            || error_message.contains("not found")
            || error_message.contains("Invalid")
            || error_message.contains("Either DOI or URL must be provided")
            || error_message.contains("provider")
            || error_message.contains("failed")
    );

    Ok(())
}

/// Test URL resolution error handling
#[tokio::test]
async fn test_url_validation_error_handling() -> Result<()> {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new(config.clone(), meta_config)?);
    let download_tool = DownloadTool::new(client.clone(), Arc::new(config))?;

    // Test with an invalid URL
    let download_input = DownloadInput {
        doi: None,
        url: Some("not-a-valid-url".to_string()),
        filename: Some("test.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    let result = download_tool.download_paper(download_input).await;
    assert!(result.is_err());

    let error_message = result.unwrap_err().to_string();
    assert!(error_message.contains("Invalid URL") || error_message.contains("url"));

    Ok(())
}

/// Test input validation
#[tokio::test]
async fn test_input_validation() -> Result<()> {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new(config.clone(), meta_config)?);
    let download_tool = DownloadTool::new(client.clone(), Arc::new(config))?;

    // Test with no DOI or URL
    let download_input = DownloadInput {
        doi: None,
        url: None,
        filename: Some("test.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
    };

    let result = download_tool.download_paper(download_input).await;
    assert!(result.is_err());

    let error_message = result.unwrap_err().to_string();
    assert!(error_message.contains("Either DOI or URL must be provided"));

    Ok(())
}

/// Test provider health check and error handling
#[tokio::test]
async fn test_provider_health_and_error_handling() -> Result<()> {
    let config = create_test_config();
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new(config, meta_config)?);

    // Test provider health checks
    let health_results = client.health_check().await;

    // Should have multiple providers
    assert!(!health_results.is_empty());

    // Should include our main providers
    assert!(health_results.contains_key("arxiv"));
    assert!(health_results.contains_key("crossref"));
    assert!(health_results.contains_key("sci_hub"));

    Ok(())
}

/// Test search query creation and validation
#[tokio::test]
async fn test_search_query_validation() -> Result<()> {
    // Test various search query types
    let valid_queries = vec![
        SearchQuery {
            query: "10.1038/nature12373".to_string(),
            search_type: SearchType::Doi,
            max_results: 1,
            offset: 0,
            params: HashMap::new(),
        },
        SearchQuery {
            query: "quantum computing".to_string(),
            search_type: SearchType::Keywords,
            max_results: 10,
            offset: 0,
            params: HashMap::new(),
        },
        SearchQuery {
            query: "John Smith".to_string(),
            search_type: SearchType::Author,
            max_results: 5,
            offset: 0,
            params: HashMap::new(),
        },
    ];

    for query in valid_queries {
        // These queries should be structurally valid
        assert!(!query.query.is_empty());
        assert!(query.max_results > 0);
        // offset is u32, so always >= 0
    }

    Ok(())
}

/// Test error categorization for different provider failures
#[tokio::test]
async fn test_provider_error_categorization() -> Result<()> {
    use rust_research_mcp::error::{Error, ErrorCategory};

    // Test different error types and their categorization
    let errors = vec![
        (
            Error::SciHub {
                code: 403,
                message: "Forbidden".to_string(),
            },
            ErrorCategory::Transient,
        ),
        (
            Error::SciHub {
                code: 404,
                message: "Not Found".to_string(),
            },
            ErrorCategory::Permanent,
        ),
        (
            Error::SciHub {
                code: 429,
                message: "Rate Limited".to_string(),
            },
            ErrorCategory::RateLimited,
        ),
        (
            Error::SciHub {
                code: 500,
                message: "Server Error".to_string(),
            },
            ErrorCategory::Transient,
        ),
    ];

    for (error, expected_category) in errors {
        assert_eq!(error.category(), expected_category);

        // Test retry logic
        let should_retry = matches!(
            expected_category,
            ErrorCategory::Transient | ErrorCategory::RateLimited
        );
        assert_eq!(error.is_retryable(), should_retry);
    }

    Ok(())
}
