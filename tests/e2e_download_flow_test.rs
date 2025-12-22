use rust_research_mcp::client::meta_search::{MetaSearchClient, MetaSearchConfig};
use rust_research_mcp::tools::download::{DownloadInput, DownloadOutputFormat, DownloadTool};
use rust_research_mcp::tools::pdf_metadata::{MetadataExtractor, MetadataInput};
use rust_research_mcp::{Config, Result};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::fs;

/// Create a test configuration with temporary directories
fn create_test_config() -> Config {
    let mut config = Config::default();
    let temp_dir = TempDir::new().unwrap();
    config.downloads.directory = temp_dir.keep();
    config
}

/// End-to-End test for the complete download and metadata extraction flow
#[tokio::test]
async fn test_e2e_download_and_metadata_flow() -> Result<()> {
    let config = Arc::new(create_test_config());
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);
    let download_tool = DownloadTool::new(client.clone(), config.clone())?;
    let metadata_extractor = MetadataExtractor::new(config.clone())?;

    // Test with a well-known arXiv paper that should be accessible
    let download_input = DownloadInput {
        doi: Some("arXiv:1706.03762".to_string()), // "Attention Is All You Need" - famous transformer paper
        url: None,
        filename: Some("transformer_paper.pdf".to_string()),
        directory: None,
        category: Some("machine_learning".to_string()),
        overwrite: true,
        verify_integrity: false, // Skip integrity check for speed
        output_format: DownloadOutputFormat::Pdf,
        headless: true,
        enable_local_grobid: false,
    };

    // Attempt the download (this might fail in CI environments without internet)
    let download_result = download_tool.download_paper(download_input).await;

    match download_result {
        Ok(result) => {
            // If download succeeded, test metadata extraction
            if let Some(file_path) = result.file_path {
                println!("Successfully downloaded paper to: {:?}", file_path);

                // Test metadata extraction on the downloaded file
                let metadata_input = MetadataInput {
                    file_path: file_path.to_string_lossy().to_string(),
                    use_cache: false,
                    extract_references: false,
                    batch_files: None,
                };

                let metadata_result = metadata_extractor.extract_metadata(metadata_input).await?;

                // Verify the metadata extraction worked
                match metadata_result.status {
                    rust_research_mcp::tools::pdf_metadata::ExtractionStatus::Success
                    | rust_research_mcp::tools::pdf_metadata::ExtractionStatus::Partial => {
                        println!("Metadata extraction successful");
                        if let Some(metadata) = metadata_result.metadata {
                            println!("Extracted title: {:?}", metadata.title);
                            println!("Authors: {:?}", metadata.authors.len());
                        }
                    }
                    _ => {
                        println!("Metadata extraction failed: {:?}", metadata_result.error);
                    }
                }
            }
        }
        Err(e) => {
            // Download failed - this is expected in CI environments or if the paper isn't available
            println!("Download failed (expected in test environment): {}", e);

            // Verify the error message is informative
            let error_str = e.to_string();
            assert!(
                error_str.contains("not found")
                    || error_str.contains("providers checked")
                    || error_str.contains("PDF")
                    || error_str.contains("DOI")
                    || error_str.contains("network")
                    || error_str.contains("timeout"),
                "Error message should be informative: {}",
                error_str
            );
        }
    }

    Ok(())
}

/// Test error handling for invalid files in metadata extraction
#[tokio::test]
async fn test_metadata_extraction_error_handling() -> Result<()> {
    let config = Arc::new(create_test_config());
    let metadata_extractor = MetadataExtractor::new(config)?;

    // Test with non-existent file
    let metadata_input = MetadataInput {
        file_path: "/nonexistent/file.pdf".to_string(),
        use_cache: false,
        extract_references: false,
        batch_files: None,
    };

    let result = metadata_extractor.extract_metadata(metadata_input).await?;
    assert!(matches!(
        result.status,
        rust_research_mcp::tools::pdf_metadata::ExtractionStatus::Failed
    ));
    assert!(result.error.is_some());
    assert!(result.error.as_ref().unwrap().contains("not found"));

    // Test with invalid PDF file
    let temp_dir = TempDir::new()?;
    let fake_pdf_path = temp_dir.path().join("fake.pdf");
    fs::write(&fake_pdf_path, b"This is not a PDF file").await?;

    let metadata_input = MetadataInput {
        file_path: fake_pdf_path.to_string_lossy().to_string(),
        use_cache: false,
        extract_references: false,
        batch_files: None,
    };

    let result = metadata_extractor.extract_metadata(metadata_input).await?;
    assert!(matches!(
        result.status,
        rust_research_mcp::tools::pdf_metadata::ExtractionStatus::Failed
    ));
    assert!(result.error.is_some());

    let error_msg = result.error.as_ref().unwrap();
    assert!(error_msg.contains("Invalid file header") || error_msg.contains("PDF"));

    Ok(())
}

/// Test URL validation and error messages in download tool
#[tokio::test]
async fn test_url_resolution_errors() -> Result<()> {
    let config = Arc::new(create_test_config());
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);
    let download_tool = DownloadTool::new(client, config)?;

    // Test relative URL that would cause the original error
    let download_input = DownloadInput {
        doi: None,
        url: Some("relative/path/without/base".to_string()), // This should be rejected during validation
        filename: Some("test.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
        headless: true,
        enable_local_grobid: false,
    };

    let result = download_tool.download_paper(download_input).await;
    assert!(result.is_err());

    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("Invalid URL")
            || error_message.contains("relative URL")
            || error_message.contains("url"),
        "Error should mention URL issue: {}",
        error_message
    );

    Ok(())
}

/// Test that provider failures result in informative error messages
#[tokio::test]
async fn test_provider_failure_messages() -> Result<()> {
    let config = Arc::new(create_test_config());
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);
    let download_tool = DownloadTool::new(client, config)?;

    // Test with a DOI that definitely doesn't exist
    let download_input = DownloadInput {
        doi: Some("10.9999/definitely.not.a.real.doi.12345".to_string()),
        url: None,
        filename: Some("nonexistent.pdf".to_string()),
        directory: None,
        category: None,
        overwrite: false,
        verify_integrity: false,
        output_format: DownloadOutputFormat::Pdf,
        headless: true,
        enable_local_grobid: false,
    };

    let result = download_tool.download_paper(download_input).await;
    assert!(result.is_err());

    let error_message = result.unwrap_err().to_string();

    // Verify the error message contains helpful information about the failure
    assert!(
        error_message.contains("not found")
            || error_message.contains("providers checked")
            || error_message.contains("DOI")
            || error_message.contains("provider(s)")
            || error_message.contains("no downloadable PDF available")
            || error_message.contains("Download request failed")
            || error_message.contains("Service error"),
        "Error should be informative about provider search: {}",
        error_message
    );

    Ok(())
}

/// Test concurrent downloads (basic stress test)
#[tokio::test]
async fn test_concurrent_downloads() -> Result<()> {
    let config = Arc::new(create_test_config());
    let meta_config = MetaSearchConfig::default();
    let client = Arc::new(MetaSearchClient::new((*config).clone(), meta_config)?);
    let download_tool = DownloadTool::new(client, config)?;

    // Create multiple download requests
    let mut handles = Vec::new();

    for i in 0..3 {
        let tool_clone = download_tool.clone();
        let handle = tokio::spawn(async move {
            let download_input = DownloadInput {
                doi: Some(format!("10.invalid/test_should_fail.{}", i)),
                url: None,
                filename: Some(format!("concurrent_{}.pdf", i)),
                directory: None,
                category: None,
                overwrite: false,
                verify_integrity: false,
                output_format: DownloadOutputFormat::Pdf,
                headless: true,
                enable_local_grobid: false,
            };

            tool_clone.download_paper(download_input).await
        });
        handles.push(handle);
    }

    // Wait for all downloads to complete (they should all fail gracefully)
    let results = futures::future::join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        let download_result = result.expect("Task should complete");

        // All should fail (test DOIs), but gracefully
        assert!(
            download_result.is_err(),
            "Download {} should fail with test DOI",
            i
        );

        // Error messages should be reasonable
        let error_msg = download_result.unwrap_err().to_string();
        assert!(
            !error_msg.is_empty(),
            "Error message should not be empty for download {}",
            i
        );
    }

    Ok(())
}
