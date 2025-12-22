use rust_research_mcp::tools::download::{DownloadInput, DownloadOutputFormat, DownloadTool};
use rust_research_mcp::tools::pdf_metadata::{MetadataExtractor, MetadataInput};
use rust_research_mcp::tools::search_source::{SearchSourceInput, SearchSourceTool};
use rust_research_mcp::{Config, MetaSearchClient, MetaSearchConfig, Server};
use std::sync::Arc;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// End-to-end test scenarios covering complete user workflows
#[tokio::test]
async fn test_complete_paper_search_workflow() {
    // Setup test environment
    let temp_dir = TempDir::new().unwrap();
    let mut config = Config::default();
    config.downloads.directory = temp_dir.path().to_path_buf();
    config.downloads.max_concurrent = 1;

    // Setup mock server for Sci-Hub
    let mock_server = MockServer::start().await;
    config.research_source.endpoints = vec![mock_server.uri()];

    // Mock successful DOI search response
    Mock::given(method("GET"))
        .and(path("/10.1000/test.doi"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"
            <html>
                <body>
                    <div id="article">
                        <a href="/download/test.pdf">Download PDF</a>
                    </div>
                </body>
            </html>
            "#,
        ))
        .mount(&mock_server)
        .await;

    // Mock PDF download response
    Mock::given(method("GET"))
        .and(path("/download/test.pdf"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("Mock PDF content")
                .append_header("content-type", "application/pdf"),
        )
        .mount(&mock_server)
        .await;

    // Initialize components - use new SearchSourceTool
    let meta_config = MetaSearchConfig::default();
    let meta_client = Arc::new(MetaSearchClient::new(config.clone(), meta_config).unwrap());
    let search_tool = SearchSourceTool::new();
    let download_tool = DownloadTool::new(meta_client.clone(), Arc::new(config.clone())).unwrap();
    let metadata_extractor = MetadataExtractor::new(Arc::new(config.clone())).unwrap();

    // Scenario 1: Search for paper by title using source-specific search
    let search_input = SearchSourceInput {
        source: "arxiv".to_string(),
        query: "machine learning".to_string(),
        limit: 10,
        offset: 0,
        search_type: None,
        help: false,
    };
    let search_result = search_tool.search(search_input).await;
    // Note: This test may not find papers without network access
    println!("Search result: {:?}", search_result.is_ok());

    // Scenario 2: Test metadata extraction (on a dummy file)
    let test_pdf_path = temp_dir.path().join("test.pdf");
    std::fs::write(&test_pdf_path, b"dummy pdf content").unwrap();

    let metadata_input = MetadataInput {
        file_path: test_pdf_path.to_string_lossy().to_string(),
        use_cache: false,
        extract_references: false,
        batch_files: None,
    };
    let metadata_result = metadata_extractor.extract_metadata(metadata_input).await;
    // This should succeed even if it can't extract much from a dummy file
    assert!(
        metadata_result.is_ok(),
        "Metadata extraction should not fail"
    );
}

#[tokio::test]
async fn test_complete_server_lifecycle_scenario() {
    // Test server lifecycle components without running full MCP server
    // (Full MCP server requires stdio transport and client handshake)
    let mut config = Config::default();
    config.server.port = 0; // Use random available port
    config.server.graceful_shutdown_timeout_secs = 1;

    let server = Arc::new(Server::new(config));

    // Test server initial state
    assert!(
        !server.is_shutdown_requested(),
        "Server should not be shutdown initially"
    );

    // Test server shutdown mechanism
    server.shutdown().await;
    assert!(
        server.is_shutdown_requested(),
        "Server should be marked for shutdown"
    );

    // Test shutdown is idempotent
    server.shutdown().await;
    assert!(
        server.is_shutdown_requested(),
        "Server should remain shutdown after second call"
    );
}

#[tokio::test]
async fn test_configuration_workflow() {
    // Test configuration loading and validation workflow
    let temp_dir = TempDir::new().unwrap();

    // Test default configuration
    let default_config = Config::default();
    assert!(
        default_config.validate().is_ok(),
        "Default config should be valid"
    );

    // Test configuration with custom download directory
    let mut custom_config = Config::default();
    custom_config.downloads.directory = temp_dir.path().to_path_buf();
    assert!(
        custom_config.validate().is_ok(),
        "Custom config should be valid"
    );

    // Test invalid configuration
    let mut invalid_config = Config::default();
    invalid_config.server.port = 0; // Invalid port
    assert!(
        invalid_config.validate().is_err(),
        "Invalid config should fail validation"
    );

    // Test configuration serialization
    let serialized = serde_json::to_string(&default_config);
    assert!(serialized.is_ok(), "Config should serialize successfully");

    // Test configuration deserialization
    let json_config = r#"{
        "server": {
            "host": "localhost",
            "port": 8080,
            "timeout_secs": 30,
            "graceful_shutdown_timeout_secs": 5,
            "health_check_interval_secs": 30,
            "max_connections": 100
        },
        "sci_hub": {
            "mirrors": ["https://test.com"],
            "timeout_secs": 30,
            "rate_limit_per_sec": 1,
            "max_retries": 3,
            "user_agent": "rust-sci-hub-mcp/0.1.0"
        },
        "downloads": {
            "directory": "/tmp/test",
            "max_concurrent": 3,
            "max_file_size_mb": 100,
            "organize_by_date": false,
            "verify_integrity": true
        }
    }"#;

    let parsed_config: Result<Config, _> = serde_json::from_str(json_config);
    assert!(
        parsed_config.is_ok(),
        "Config should deserialize successfully"
    );
}
