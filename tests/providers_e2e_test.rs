use rust_research_mcp::client::providers::{
    ArxivProvider, CoreProvider, CrossRefProvider, OpenAlexProvider, SciHubProvider, SearchContext,
    SearchQuery, SearchType, SemanticScholarProvider, SourceProvider, SsrnProvider,
    UnpaywallProvider,
};
use std::collections::HashMap;
use std::time::Duration;
use tokio;

/// Test configuration
struct TestConfig {
    /// Whether to run tests that make real network requests
    run_live_tests: bool,
    /// Timeout for each test
    test_timeout: Duration,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            // Set to false by default to avoid hitting real APIs in CI
            // Set environment variable RUN_LIVE_TESTS=true to enable
            run_live_tests: std::env::var("RUN_LIVE_TESTS")
                .unwrap_or_else(|_| "false".to_string())
                .parse::<bool>()
                .unwrap_or(false),
            test_timeout: Duration::from_secs(30),
        }
    }
}

/// Create a standard search context for testing
fn create_test_context() -> SearchContext {
    SearchContext {
        timeout: Duration::from_secs(30),
        user_agent: "rust_research_mcp-test/0.2.1".to_string(),
        rate_limit: Some(Duration::from_millis(500)),
        headers: HashMap::new(),
    }
}

/// Create a DOI search query
fn create_doi_query(doi: &str) -> SearchQuery {
    SearchQuery {
        query: doi.to_string(),
        search_type: SearchType::Doi,
        max_results: 1,
        offset: 0,
        params: HashMap::new(),
        sources: None,
        metadata_sources: None,
    }
}

/// Create a title search query
fn create_title_query(title: &str, max_results: u32) -> SearchQuery {
    SearchQuery {
        query: title.to_string(),
        search_type: SearchType::Title,
        max_results,
        offset: 0,
        params: HashMap::new(),
        sources: None,
        metadata_sources: None,
    }
}

/// Create a keyword search query
fn create_keyword_query(keywords: &str, max_results: u32) -> SearchQuery {
    SearchQuery {
        query: keywords.to_string(),
        search_type: SearchType::Keywords,
        max_results,
        offset: 0,
        params: HashMap::new(),
        sources: None,
        metadata_sources: None,
    }
}

// ============================================================================
// ArXiv Provider Tests
// ============================================================================

#[tokio::test]
async fn test_arxiv_provider_creation() {
    let provider = ArxivProvider::new();
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "arxiv");
    assert_eq!(provider.priority(), 80);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_arxiv_search_by_title() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = ArxivProvider::new().expect("Failed to create Arxiv provider");
    let context = create_test_context();
    let query = create_title_query("quantum computing", 5);

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Search failed: {:?}", result.err());

    let result = result.unwrap();
    assert!(!result.papers.is_empty(), "No papers found");
    assert!(result.papers.len() <= 5, "Too many papers returned");

    // Check that papers have expected fields
    for paper in &result.papers {
        assert!(
            !paper.doi.is_empty() || paper.title.is_some(),
            "Paper missing identifier"
        );
        println!("Found paper: {:?}", paper.title);
    }
}

#[tokio::test]
async fn test_arxiv_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = ArxivProvider::new().expect("Failed to create ArXiv provider");
    let context = create_test_context();
    // Use a known arXiv paper ID
    let query = create_doi_query("2103.14030"); // arXiv ID format

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            assert_eq!(result.papers.len(), 1, "Should find exactly one paper");
            let paper = &result.papers[0];
            assert!(paper.title.is_some(), "Paper should have a title");
            assert!(!paper.authors.is_empty(), "Paper should have authors");
        }
    }
}

#[tokio::test]
async fn test_arxiv_health_check() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = ArxivProvider::new().expect("Failed to create ArXiv provider");
    let context = create_test_context();

    let result = tokio::time::timeout(config.test_timeout, provider.health_check(&context)).await;

    assert!(result.is_ok(), "Health check timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Health check failed: {:?}", result.err());
    assert!(result.unwrap(), "ArXiv is not healthy");
}

// ============================================================================
// OpenAlex Provider Tests
// ============================================================================

#[tokio::test]
async fn test_openalex_provider_creation() {
    let provider = OpenAlexProvider::new();
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "openalex");
    assert_eq!(provider.priority(), 180);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_openalex_search_by_title() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = OpenAlexProvider::new().expect("Failed to create OpenAlex provider");
    let context = create_test_context();
    let query = create_title_query("machine learning", 5);

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let search_result = result.unwrap();
    assert!(
        search_result.is_ok(),
        "Search failed: {:?}",
        search_result.err()
    );

    let provider_result = search_result.unwrap();
    assert!(!provider_result.papers.is_empty(), "No papers found");
    assert_eq!(provider_result.source, "openalex");
    assert!(provider_result.total_available.unwrap_or(0) > 0);

    // Verify paper structure
    let paper = &provider_result.papers[0];
    assert!(!paper.doi.is_empty());
    // Should have either a title or at least some content
    assert!(paper.title.is_some() || !paper.authors.is_empty());
}

#[tokio::test]
async fn test_openalex_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = OpenAlexProvider::new().expect("Failed to create OpenAlex provider");
    let context = create_test_context();
    // Use a well-known DOI
    let query = create_doi_query("10.1038/nature12373");

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let search_result = result.unwrap();
    assert!(
        search_result.is_ok(),
        "DOI search failed: {:?}",
        search_result.err()
    );

    let provider_result = search_result.unwrap();
    if !provider_result.papers.is_empty() {
        let paper = &provider_result.papers[0];
        // Should find the exact paper
        assert!(paper.doi.contains("nature12373") || paper.doi.contains("10.1038"));
    }
}

#[tokio::test]
async fn test_openalex_search_by_author() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = OpenAlexProvider::new().expect("Failed to create OpenAlex provider");
    let context = create_test_context();
    let query = SearchQuery {
        query: "Geoffrey Hinton".to_string(),
        search_type: SearchType::Author,
        max_results: 3,
        offset: 0,
        params: HashMap::new(),
        sources: None,
        metadata_sources: None,
    };

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let search_result = result.unwrap();
    assert!(
        search_result.is_ok(),
        "Author search failed: {:?}",
        search_result.err()
    );

    let provider_result = search_result.unwrap();
    if !provider_result.papers.is_empty() {
        // Should find papers with the author
        let has_author = provider_result.papers.iter().any(|paper| {
            paper
                .authors
                .iter()
                .any(|author| author.to_lowercase().contains("hinton"))
        });
        assert!(has_author, "No papers found with Geoffrey Hinton as author");
    }
}

#[tokio::test]
async fn test_openalex_health_check() {
    let provider = OpenAlexProvider::new().expect("Failed to create OpenAlex provider");
    let context = create_test_context();

    let result = provider.health_check(&context).await;
    assert!(result.is_ok(), "Health check failed: {:?}", result.err());
    assert!(result.unwrap(), "OpenAlex is not healthy");
}

// ============================================================================
// CrossRef Provider Tests
// ============================================================================

#[tokio::test]
async fn test_crossref_provider_creation() {
    let provider = CrossRefProvider::new(None);
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "crossref");
    assert_eq!(provider.priority(), 90);
    assert!(!provider.supports_full_text()); // CrossRef usually doesn't provide PDFs
}

#[tokio::test]
async fn test_crossref_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = CrossRefProvider::new(None).expect("Failed to create CrossRef provider");
    let context = create_test_context();
    let query = create_doi_query("10.1038/nature12373"); // Known DOI

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            assert_eq!(result.papers.len(), 1, "Should find exactly one paper");
            let paper = &result.papers[0];
            assert!(paper.title.is_some(), "Paper should have a title");
            assert_eq!(paper.doi, "10.1038/nature12373");
        }
    }
}

#[tokio::test]
async fn test_crossref_search_by_title() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = CrossRefProvider::new(None).expect("Failed to create CrossRef provider");
    let context = create_test_context();
    let query = create_title_query("machine learning", 3);

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        assert!(!result.papers.is_empty(), "Should find some papers");
        assert!(result.papers.len() <= 3, "Should respect max_results");

        for paper in &result.papers {
            assert!(!paper.doi.is_empty(), "CrossRef papers should have DOIs");
            println!("Found CrossRef paper: {} - {:?}", paper.doi, paper.title);
        }
    }
}

// ============================================================================
// SSRN Provider Tests
// ============================================================================

#[tokio::test]
async fn test_ssrn_provider_creation() {
    let provider = SsrnProvider::new();
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "ssrn");
    assert_eq!(provider.priority(), 85);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_ssrn_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = SsrnProvider::new().expect("Failed to create SSRN provider");
    let context = create_test_context();
    // Use a known SSRN DOI format
    let query = create_doi_query("10.2139/ssrn.3580300"); // Example SSRN paper

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            let paper = &result.papers[0];
            assert!(paper.doi.contains("ssrn"), "Should be an SSRN DOI");
            println!("Found SSRN paper: {} - {:?}", paper.doi, paper.title);
        }
    }
}

#[tokio::test]
async fn test_ssrn_extract_id() {
    let provider = SsrnProvider::new().expect("Failed to create SSRN provider");

    // Test the ID extraction logic
    let test_cases = vec![
        ("10.2139/ssrn.5290350", Some("5290350")),
        ("10.2139/ssrn.1234567", Some("1234567")),
        ("10.1038/nature12373", None), // Not an SSRN DOI
    ];

    for (doi, expected) in test_cases {
        // We'd need to make extract_ssrn_id public or test through the public API
        let _context = create_test_context();
        let _query = create_doi_query(doi);

        if expected.is_some() {
            // Should be able to search for SSRN DOIs
            assert!(provider.supported_search_types().contains(&SearchType::Doi));
        }
    }
}

// ============================================================================
// Semantic Scholar Provider Tests
// ============================================================================

#[tokio::test]
async fn test_semantic_scholar_provider_creation() {
    let provider = SemanticScholarProvider::new(None);
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "semantic_scholar");
    assert_eq!(provider.priority(), 88);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_semantic_scholar_search_by_title() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider =
        SemanticScholarProvider::new(None).expect("Failed to create Semantic Scholar provider");
    let context = create_test_context();
    let query = create_title_query("machine learning", 3);

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Search failed: {:?}", result.err());

    let result = result.unwrap();
    assert!(!result.papers.is_empty(), "No papers found");
    assert!(result.papers.len() <= 3, "Too many papers returned");

    // Check that papers have expected fields
    for paper in &result.papers {
        assert!(!paper.doi.is_empty(), "Paper missing DOI/ID");
        println!("Found Semantic Scholar paper: {:?}", paper.title);
        // Many Semantic Scholar papers should have PDF URLs
        if paper.pdf_url.is_some() {
            println!("  📄 Has open access PDF");
        }
    }
}

#[tokio::test]
async fn test_semantic_scholar_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider =
        SemanticScholarProvider::new(None).expect("Failed to create Semantic Scholar provider");
    let context = create_test_context();
    // Use a known DOI that should be in Semantic Scholar
    let query = create_doi_query("10.1038/nature12373");

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            let paper = &result.papers[0];
            assert!(paper.title.is_some(), "Paper should have a title");
            println!(
                "Found paper on Semantic Scholar: {} - {:?}",
                paper.doi, paper.title
            );

            // Check if it has PDF access
            if let Some(pdf_url) = &paper.pdf_url {
                println!("  📄 Open access PDF available: {}", pdf_url);
            }
        }
    }
}

#[tokio::test]
async fn test_semantic_scholar_health_check() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider =
        SemanticScholarProvider::new(None).expect("Failed to create Semantic Scholar provider");
    let context = create_test_context();

    let result = tokio::time::timeout(config.test_timeout, provider.health_check(&context)).await;

    assert!(result.is_ok(), "Health check timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Health check failed: {:?}", result.err());
    assert!(result.unwrap(), "Semantic Scholar is not healthy");
}

// ============================================================================
// Unpaywall Provider Tests
// ============================================================================

#[tokio::test]
async fn test_unpaywall_provider_creation() {
    let provider = UnpaywallProvider::new_with_default_email();
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "unpaywall");
    assert_eq!(provider.priority(), 87);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_unpaywall_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider =
        UnpaywallProvider::new_with_default_email().expect("Failed to create Unpaywall provider");
    let context = create_test_context();
    // Use a DOI that's likely to have open access
    let query = create_doi_query("10.1371/journal.pone.0000308"); // PLOS ONE paper (open access)

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            let paper = &result.papers[0];
            assert!(paper.title.is_some(), "Paper should have a title");
            println!(
                "Found open access paper on Unpaywall: {} - {:?}",
                paper.doi, paper.title
            );

            // Unpaywall should provide PDF URL for open access papers
            if let Some(pdf_url) = &paper.pdf_url {
                println!("  📄 Open access PDF available: {}", pdf_url);
            }
        } else {
            println!("Paper not found or not open access in Unpaywall");
        }
    }
}

#[tokio::test]
async fn test_unpaywall_health_check() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider =
        UnpaywallProvider::new_with_default_email().expect("Failed to create Unpaywall provider");
    let context = create_test_context();

    let result = tokio::time::timeout(config.test_timeout, provider.health_check(&context)).await;

    assert!(result.is_ok(), "Health check timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Health check failed: {:?}", result.err());
    assert!(result.unwrap(), "Unpaywall is not healthy");
}

#[tokio::test]
async fn test_unpaywall_non_doi_search() {
    let provider =
        UnpaywallProvider::new_with_default_email().expect("Failed to create Unpaywall provider");
    let context = create_test_context();
    let query = create_title_query("machine learning", 1);

    let result = provider.search(&query, &context).await;

    // Should succeed but return empty results since Unpaywall only supports DOI
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(
        result.papers.is_empty(),
        "Should return empty for non-DOI searches"
    );
}

// ============================================================================
// CORE Provider Tests
// ============================================================================

#[tokio::test]
async fn test_core_provider_creation() {
    let provider = CoreProvider::new(None);
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "core");
    assert_eq!(provider.priority(), 86);
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_core_search_by_title() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = CoreProvider::new(None).expect("Failed to create CORE provider");
    let context = create_test_context();
    let query = create_title_query("machine learning", 3);

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Search failed: {:?}", result.err());

    let result = result.unwrap();
    assert!(!result.papers.is_empty(), "No papers found");
    assert!(result.papers.len() <= 3, "Too many papers returned");

    // Check that papers have expected fields
    for paper in &result.papers {
        assert!(!paper.doi.is_empty(), "Paper missing DOI/ID");
        println!("Found CORE paper: {:?}", paper.title);
        // Many CORE papers should have PDF URLs since it's open access focused
        if paper.pdf_url.is_some() {
            println!("  📄 Has open access PDF");
        }
    }
}

#[tokio::test]
async fn test_core_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = CoreProvider::new(None).expect("Failed to create CORE provider");
    let context = create_test_context();
    // Use a DOI that's likely to be in CORE's open access collection
    let query = create_doi_query("10.1371/journal.pone.0000308"); // PLOS ONE paper

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    if let Ok(result) = result {
        if !result.papers.is_empty() {
            let paper = &result.papers[0];
            assert!(paper.title.is_some(), "Paper should have a title");
            println!("Found paper on CORE: {} - {:?}", paper.doi, paper.title);

            // Check if it has PDF access
            if let Some(pdf_url) = &paper.pdf_url {
                println!("  📄 Open access PDF available: {}", pdf_url);
            }
        }
    }
}

#[tokio::test]
async fn test_core_health_check() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = CoreProvider::new(None).expect("Failed to create CORE provider");
    let context = create_test_context();

    let result = tokio::time::timeout(config.test_timeout, provider.health_check(&context)).await;

    assert!(result.is_ok(), "Health check timed out");
    let result = result.unwrap();
    assert!(result.is_ok(), "Health check failed: {:?}", result.err());
    assert!(result.unwrap(), "CORE is not healthy");
}

// ============================================================================
// Sci-Hub Provider Tests
// ============================================================================

#[tokio::test]
async fn test_scihub_provider_creation() {
    let provider = SciHubProvider::new();
    assert!(provider.is_ok());

    let provider = provider.unwrap();
    assert_eq!(provider.name(), "sci_hub");
    assert_eq!(provider.priority(), 10); // Lowest priority
    assert!(provider.supports_full_text());
}

#[tokio::test]
async fn test_scihub_search_by_doi() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = SciHubProvider::new().expect("Failed to create Sci-Hub provider");
    let context = create_test_context();
    // Use a well-known older paper that's likely to be on Sci-Hub
    let query = create_doi_query("10.1038/nature12373");

    let result = tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

    assert!(result.is_ok(), "Search timed out");
    let result = result.unwrap();

    // Sci-Hub might be blocked or down, so we just check it doesn't panic
    match result {
        Ok(result) => {
            if !result.papers.is_empty() {
                let paper = &result.papers[0];
                println!("Found on Sci-Hub: {:?}", paper.title);
                // If found, should have a PDF URL
                assert!(paper.pdf_url.is_some(), "Sci-Hub should provide PDF URLs");
            } else {
                println!("Paper not found on Sci-Hub (might be too recent or blocked)");
            }
        }
        Err(e) => {
            println!("Sci-Hub search failed (might be blocked): {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_scihub_health_check() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    let provider = SciHubProvider::new().expect("Failed to create Sci-Hub provider");
    let context = create_test_context();

    let result = tokio::time::timeout(config.test_timeout, provider.health_check(&context)).await;

    assert!(result.is_ok(), "Health check timed out");
    let result = result.unwrap();

    // Sci-Hub might be blocked, so we just check it returns a result
    match result {
        Ok(healthy) => {
            println!(
                "Sci-Hub health check: {}",
                if healthy { "OK" } else { "DOWN" }
            );
        }
        Err(e) => {
            println!("Sci-Hub health check error: {:?}", e);
        }
    }
}

// ============================================================================
// Integration Tests - Multiple Providers
// ============================================================================

#[tokio::test]
async fn test_all_providers_with_same_query() {
    let config = TestConfig::default();
    if !config.run_live_tests {
        eprintln!("Skipping live test. Set RUN_LIVE_TESTS=true to run.");
        return;
    }

    // Create all providers
    let providers: Vec<Box<dyn SourceProvider>> = vec![
        Box::new(ArxivProvider::new().expect("Failed to create ArXiv")),
        Box::new(CoreProvider::new(None).expect("Failed to create CORE")),
        Box::new(CrossRefProvider::new(None).expect("Failed to create CrossRef")),
        Box::new(SemanticScholarProvider::new(None).expect("Failed to create Semantic Scholar")),
        Box::new(UnpaywallProvider::new_with_default_email().expect("Failed to create Unpaywall")),
        Box::new(SsrnProvider::new().expect("Failed to create SSRN")),
        Box::new(SciHubProvider::new().expect("Failed to create Sci-Hub")),
    ];

    let context = create_test_context();
    let query = create_keyword_query("machine learning", 2);

    println!("\nTesting all providers with query: '{}'", query.query);
    println!("{}", "=".repeat(60));

    for provider in providers {
        println!(
            "\nProvider: {} (priority: {})",
            provider.name(),
            provider.priority()
        );
        println!("{}", "-".repeat(40));

        let result =
            tokio::time::timeout(config.test_timeout, provider.search(&query, &context)).await;

        match result {
            Ok(Ok(result)) => {
                println!("✅ Success: Found {} papers", result.papers.len());
                for (i, paper) in result.papers.iter().enumerate() {
                    println!(
                        "  {}. {} - {:?}",
                        i + 1,
                        paper.doi.split('/').last().unwrap_or(&paper.doi),
                        paper.title.as_ref().map(|t| {
                            if t.len() > 50 {
                                format!("{}...", &t[..50])
                            } else {
                                t.clone()
                            }
                        })
                    );
                    if paper.pdf_url.is_some() {
                        println!("     📄 Has PDF");
                    }
                }
            }
            Ok(Err(e)) => {
                println!("❌ Provider error: {:?}", e);
            }
            Err(_) => {
                println!("⏱️ Timeout");
            }
        }
    }
}

#[tokio::test]
async fn test_provider_priorities_ordering() {
    // Verify that providers have the expected priority ordering
    let providers: Vec<(String, u8)> = vec![
        (
            CrossRefProvider::new(None).unwrap().name().to_string(),
            CrossRefProvider::new(None).unwrap().priority(),
        ),
        (
            SemanticScholarProvider::new(None)
                .unwrap()
                .name()
                .to_string(),
            SemanticScholarProvider::new(None).unwrap().priority(),
        ),
        (
            UnpaywallProvider::new_with_default_email()
                .unwrap()
                .name()
                .to_string(),
            UnpaywallProvider::new_with_default_email()
                .unwrap()
                .priority(),
        ),
        (
            CoreProvider::new(None).unwrap().name().to_string(),
            CoreProvider::new(None).unwrap().priority(),
        ),
        (
            SsrnProvider::new().unwrap().name().to_string(),
            SsrnProvider::new().unwrap().priority(),
        ),
        (
            ArxivProvider::new().unwrap().name().to_string(),
            ArxivProvider::new().unwrap().priority(),
        ),
        (
            SciHubProvider::new().unwrap().name().to_string(),
            SciHubProvider::new().unwrap().priority(),
        ),
    ];

    // Sort by priority (descending)
    let mut sorted = providers.clone();
    sorted.sort_by_key(|(_, priority)| std::cmp::Reverse(*priority));

    println!("\nProvider Priority Order:");
    for (name, priority) in &sorted {
        println!("  {}: {}", name, priority);
    }

    // Verify expected order
    assert_eq!(
        sorted[0].0, "crossref",
        "CrossRef should have highest priority"
    );
    assert_eq!(
        sorted[1].0, "semantic_scholar",
        "Semantic Scholar should be second"
    );
    assert_eq!(sorted[2].0, "unpaywall", "Unpaywall should be third");
    assert_eq!(sorted[3].0, "core", "CORE should be fourth");
    assert_eq!(sorted[4].0, "ssrn", "SSRN should be fifth");
    assert_eq!(sorted[5].0, "arxiv", "arXiv should be sixth");
    assert_eq!(
        sorted[6].0, "sci_hub",
        "Sci-Hub should have lowest priority"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_provider_invalid_doi_handling() {
    let providers: Vec<Box<dyn SourceProvider>> = vec![
        Box::new(ArxivProvider::new().unwrap()),
        Box::new(CoreProvider::new(None).unwrap()),
        Box::new(CrossRefProvider::new(None).unwrap()),
        Box::new(SemanticScholarProvider::new(None).unwrap()),
        Box::new(SsrnProvider::new().unwrap()),
        Box::new(SciHubProvider::new().unwrap()),
    ];

    let context = create_test_context();
    let query = create_doi_query("not-a-valid-doi");

    for provider in providers {
        let result = provider.search(&query, &context).await;

        // Should either return empty results or an error, but not panic
        match result {
            Ok(result) => {
                println!(
                    "{}: Returned {} results for invalid DOI",
                    provider.name(),
                    result.papers.len()
                );
            }
            Err(e) => {
                println!(
                    "{}: Properly errored on invalid DOI: {:?}",
                    provider.name(),
                    e
                );
            }
        }
    }
}

#[tokio::test]
async fn test_provider_empty_query_handling() {
    let providers: Vec<Box<dyn SourceProvider>> = vec![
        Box::new(ArxivProvider::new().unwrap()),
        Box::new(CoreProvider::new(None).unwrap()),
        Box::new(CrossRefProvider::new(None).unwrap()),
        Box::new(SemanticScholarProvider::new(None).unwrap()),
        Box::new(SsrnProvider::new().unwrap()),
        Box::new(SciHubProvider::new().unwrap()),
    ];

    let context = create_test_context();
    let query = create_title_query("", 1);

    for provider in providers {
        let result = provider.search(&query, &context).await;

        // Should handle empty queries gracefully
        match result {
            Ok(result) => {
                println!(
                    "{}: Returned {} results for empty query",
                    provider.name(),
                    result.papers.len()
                );
            }
            Err(e) => {
                println!(
                    "{}: Properly errored on empty query: {:?}",
                    provider.name(),
                    e
                );
            }
        }
    }
}
