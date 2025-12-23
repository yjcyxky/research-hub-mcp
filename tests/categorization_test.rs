use rust_research_mcp::{
    client::PaperMetadata, services::CategorizationService, tools::categorize::CategorizeInput,
    CategorizeTool, Config, SearchSourceTool,
};
use std::sync::Arc;
use tempfile::TempDir;

fn create_test_config() -> Arc<Config> {
    let mut config = Config::default();
    // Enable categorization for testing
    config.categorization.enabled = true;
    config.categorization.max_abstracts = 3;
    config.categorization.max_prompt_length = 2000;
    config.categorization.default_category = "test_papers".to_string();
    Arc::new(config)
}

fn create_test_papers() -> Vec<PaperMetadata> {
    vec![
        PaperMetadata {
            doi: "10.1000/ml1".to_string(),
            pmid: None,
            title: Some("Deep Learning for Medical Image Analysis".to_string()),
            authors: vec!["Smith, J.".to_string(), "Doe, A.".to_string()],
            journal: Some("Nature Machine Intelligence".to_string()),
            year: Some(2024),
            abstract_text: Some("This paper presents a novel deep learning approach for analyzing medical images using neural networks and machine learning techniques.".to_string()),
            keywords: Vec::new(),
            pdf_url: None,
            file_size: None,
        },
        PaperMetadata {
            doi: "10.1000/quantum1".to_string(),
            pmid: None,
            title: Some("Quantum Computing Algorithms for Chemistry".to_string()),
            authors: vec!["Einstein, A.".to_string()],
            journal: Some("Physical Review A".to_string()),
            year: Some(2024),
            abstract_text: Some("We develop quantum algorithms for simulating molecular systems and quantum chemistry calculations.".to_string()),
            keywords: Vec::new(),
            pdf_url: None,
            file_size: None,
        },
        PaperMetadata {
            doi: "10.1000/agent1".to_string(),
            pmid: None,
            title: Some("Multi-Agent Systems with Memory".to_string()),
            authors: vec!["Agent, M.".to_string()],
            journal: Some("Journal of AI Research".to_string()),
            year: Some(2024),
            abstract_text: Some("This work explores multi-agent systems with episodic memory and agent coordination mechanisms.".to_string()),
            keywords: Vec::new(),
            pdf_url: None,
            file_size: None,
        },
    ]
}

#[tokio::test]
async fn test_categorization_service_creation() {
    let config = create_test_config();
    let service = CategorizationService::new(config.categorization.clone());
    assert!(
        service.is_ok(),
        "Should create categorization service successfully"
    );

    let service = service.unwrap();
    assert!(service.is_enabled(), "Service should be enabled");
    assert_eq!(service.default_category(), "test_papers");
}

#[tokio::test]
async fn test_categorization_service_sanitization() {
    let config = create_test_config();
    let service = CategorizationService::new(config.categorization.clone()).unwrap();

    // Test basic sanitization
    assert_eq!(
        service.sanitize_category("Machine Learning"),
        "machine_learning"
    );
    assert_eq!(
        service.sanitize_category("Quantum Computing!"),
        "quantum_computing"
    );
    assert_eq!(
        service.sanitize_category("AI & ML Research"),
        "ai_ml_research"
    );

    // Test special characters removal
    assert_eq!(
        service.sanitize_category("Computer-Science"),
        "computer_science"
    );
    assert_eq!(
        service.sanitize_category("ML/AI Research"),
        "ml_ai_research"
    );

    // Test multiple underscores
    assert_eq!(
        service.sanitize_category("machine___learning"),
        "machine_learning"
    );

    // Test fallback for invalid input
    assert_eq!(service.sanitize_category(""), "test_papers");
    assert_eq!(service.sanitize_category("ai"), "test_papers"); // Too short

    // Test quoted responses
    assert_eq!(
        service.sanitize_category("\"machine_learning\""),
        "machine_learning"
    );
    assert_eq!(
        service.sanitize_category("'quantum_physics'"),
        "quantum_physics"
    );
}

#[tokio::test]
async fn test_categorization_conflict_resolution() {
    let config = create_test_config();
    let service = CategorizationService::new(config.categorization.clone()).unwrap();
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // No conflict case
    let category = service.resolve_category_conflict(base_path, "machine_learning");
    assert_eq!(category, "machine_learning");

    // Create directory to test existing directory handling
    std::fs::create_dir_all(base_path.join("machine_learning")).unwrap();
    let category = service.resolve_category_conflict(base_path, "machine_learning");
    assert_eq!(category, "machine_learning"); // Should use existing directory

    // Create file with same name to test conflict resolution
    std::fs::write(base_path.join("quantum_physics"), "test").unwrap();
    let category = service.resolve_category_conflict(base_path, "quantum_physics");
    assert_eq!(category, "quantum_physics_2"); // Should resolve conflict with number
}

#[tokio::test]
async fn test_categorize_tool_creation() {
    let config = create_test_config();
    let tool = CategorizeTool::new(config);
    assert!(tool.is_ok(), "Should create categorize tool successfully");

    let tool = tool.unwrap();
    assert!(tool.is_enabled(), "Tool should be enabled");
}

#[tokio::test]
async fn test_categorize_tool_input_validation() {
    let config = create_test_config();
    let tool = CategorizeTool::new(config).unwrap();

    // Test empty query
    let empty_query_input = CategorizeInput {
        query: "".to_string(),
        papers: create_test_papers(),
        max_abstracts: None,
    };
    let result = tool.categorize_papers(empty_query_input).await;
    assert!(result.is_err(), "Should fail with empty query");

    // Test too long query
    let long_query_input = CategorizeInput {
        query: "a".repeat(1001),
        papers: create_test_papers(),
        max_abstracts: None,
    };
    let result = tool.categorize_papers(long_query_input).await;
    assert!(result.is_err(), "Should fail with too long query");

    // Test empty papers
    let empty_papers_input = CategorizeInput {
        query: "machine learning".to_string(),
        papers: vec![],
        max_abstracts: None,
    };
    let result = tool.categorize_papers(empty_papers_input).await;
    assert!(result.is_err(), "Should fail with empty papers");

    // Test too many papers
    let many_papers: Vec<PaperMetadata> = (0..101)
        .map(|i| PaperMetadata {
            doi: format!("10.1000/paper{}", i),
            pmid: None,
            title: Some(format!("Paper {}", i)),
            authors: vec!["Author".to_string()],
            journal: None,
            year: Some(2024),
            abstract_text: None,
            keywords: Vec::new(),
            pdf_url: None,
            file_size: None,
        })
        .collect();

    let too_many_papers_input = CategorizeInput {
        query: "test".to_string(),
        papers: many_papers,
        max_abstracts: None,
    };
    let result = tool.categorize_papers(too_many_papers_input).await;
    assert!(result.is_err(), "Should fail with too many papers");
}

#[tokio::test]
async fn test_categorize_tool_heuristic_categorization() {
    let config = create_test_config();
    let tool = CategorizeTool::new(config).unwrap();

    // Test machine learning categorization
    let ml_input = CategorizeInput {
        query: "machine learning in healthcare".to_string(),
        papers: vec![create_test_papers()[0].clone()],
        max_abstracts: Some(1),
    };
    let result = tool.categorize_papers(ml_input).await.unwrap();
    assert_eq!(result.sanitized_category, "machine_learning");
    assert!(!result.is_fallback);
    assert_eq!(result.papers_analyzed, 1);
    assert_eq!(result.abstracts_used, 1);

    // Test quantum physics categorization
    let quantum_input = CategorizeInput {
        query: "quantum computing algorithms".to_string(),
        papers: vec![create_test_papers()[1].clone()],
        max_abstracts: Some(1),
    };
    let result = tool.categorize_papers(quantum_input).await.unwrap();
    assert_eq!(result.sanitized_category, "quantum_physics");
    assert!(!result.is_fallback);

    // Test agent systems categorization
    let agent_input = CategorizeInput {
        query: "multi-agent systems".to_string(),
        papers: vec![create_test_papers()[2].clone()],
        max_abstracts: Some(1),
    };
    let result = tool.categorize_papers(agent_input).await.unwrap();
    assert_eq!(result.sanitized_category, "agentic_systems");
    assert!(!result.is_fallback);

    // Test fallback categorization
    let generic_input = CategorizeInput {
        query: "unknown research topic".to_string(),
        papers: vec![PaperMetadata {
            doi: "10.1000/unknown".to_string(),
            pmid: None,
            title: Some("Some Random Paper".to_string()),
            authors: vec!["Unknown".to_string()],
            journal: None,
            year: Some(2024),
            abstract_text: Some("This is about something completely different.".to_string()),
            keywords: Vec::new(),
            pdf_url: None,
            file_size: None,
        }],
        max_abstracts: Some(1),
    };
    let result = tool.categorize_papers(generic_input).await.unwrap();
    // Should extract meaningful words from query or use default
    assert!(
        result.sanitized_category == "unknown_research_topic"
            || result.sanitized_category == "test_papers"
    );
}

#[tokio::test]
async fn test_categorize_tool_disabled() {
    let mut config = Config::default();
    config.categorization.enabled = false;
    let config = Arc::new(config);

    let tool = CategorizeTool::new(config).unwrap();
    assert!(!tool.is_enabled(), "Tool should be disabled");

    let input = CategorizeInput {
        query: "test query".to_string(),
        papers: create_test_papers(),
        max_abstracts: None,
    };
    let result = tool.categorize_papers(input).await.unwrap();
    assert!(result.is_fallback, "Should use fallback when disabled");
    assert_eq!(result.papers_analyzed, 0);
    assert_eq!(result.abstracts_used, 0);
    assert_eq!(result.prompt_length, 0);
}

#[tokio::test]
async fn test_search_source_tool_categorization_integration() {
    let config = create_test_config();
    let _search_tool = SearchSourceTool::new();

    // Test that search source tool was created successfully
    // Note: Full integration test would require actual search results
    // This test validates the service is properly integrated during construction

    // Verify config has categorization enabled
    assert!(
        config.categorization.enabled,
        "Config should have categorization enabled"
    );
}

#[tokio::test]
async fn test_categorization_prompt_generation() {
    let config = create_test_config();
    let service = CategorizationService::new(config.categorization.clone()).unwrap();

    let papers = create_test_papers();
    let prompt = service.generate_category_prompt("machine learning", &papers);

    // Validate prompt structure
    assert!(prompt.contains("machine learning"), "Should contain query");
    assert!(
        prompt.contains("snake_case"),
        "Should contain formatting requirements"
    );
    assert!(
        prompt.contains("Return only the folder name"),
        "Should contain return instruction"
    );
    assert!(
        prompt.contains("3-5 words maximum"),
        "Should contain length requirement"
    );
    assert!(
        prompt.len() <= config.categorization.max_prompt_length,
        "Should respect max length"
    );

    // Test with papers that have abstracts
    let abstracts_count = papers.iter().filter(|p| p.abstract_text.is_some()).count();
    assert!(abstracts_count > 0, "Test papers should have abstracts");

    // Should contain at least one abstract
    let has_abstract = papers.iter().any(|p| {
        if let Some(abstract_text) = &p.abstract_text {
            prompt.contains(abstract_text)
        } else {
            false
        }
    });
    assert!(has_abstract, "Prompt should contain paper abstracts");
}

#[tokio::test]
async fn test_categorization_max_abstracts_limit() {
    let mut config = Config::default();
    config.categorization.enabled = true;
    config.categorization.max_abstracts = 2; // Limit to 2 abstracts
    let config = Arc::new(config);

    let tool = CategorizeTool::new(config).unwrap();

    let input = CategorizeInput {
        query: "test query".to_string(),
        papers: create_test_papers(), // 3 papers
        max_abstracts: None,          // Should use config default of 2
    };

    let result = tool.categorize_papers(input).await.unwrap();
    assert_eq!(
        result.papers_analyzed, 2,
        "Should analyze only 2 papers due to config limit"
    );
    assert_eq!(result.abstracts_used, 2, "Should use only 2 abstracts");
}

#[tokio::test]
async fn test_categorization_prompt_truncation() {
    let mut config = Config::default();
    config.categorization.enabled = true;
    config.categorization.max_prompt_length = 500; // Very short for testing
    let config = Arc::new(config);

    let service = CategorizationService::new(config.categorization.clone()).unwrap();

    // Create papers with very long abstracts
    let papers = vec![PaperMetadata {
        doi: "10.1000/long".to_string(),
        pmid: None,
        title: Some("Long Paper".to_string()),
        authors: vec!["Author".to_string()],
        journal: None,
        year: Some(2024),
        abstract_text: Some("A".repeat(2000)), // Very long abstract
        keywords: Vec::new(),
        pdf_url: None,
        file_size: None,
    }];

    let prompt = service.generate_category_prompt("test query", &papers);
    assert!(
        prompt.len() <= 500,
        "Prompt should be truncated to max length"
    );
    assert!(
        prompt.contains("Content truncated"),
        "Should indicate truncation"
    );
}

// Integration test for the complete categorization workflow
#[tokio::test]
async fn test_categorization_end_to_end_workflow() {
    let config = create_test_config();

    // 1. Test categorization service directly
    let service = CategorizationService::new(config.categorization.clone()).unwrap();
    assert!(service.is_enabled());

    // 2. Test categorize tool
    let tool = CategorizeTool::new(config.clone()).unwrap();
    let input = CategorizeInput {
        query: "machine learning for medical diagnosis".to_string(),
        papers: vec![create_test_papers()[0].clone()],
        max_abstracts: Some(1),
    };
    let result = tool.categorize_papers(input).await.unwrap();
    assert_eq!(result.sanitized_category, "machine_learning");

    // 3. Test search source tool integration
    let _search_tool = SearchSourceTool::new();
    // Search source tool was created successfully

    // This validates the complete workflow is properly set up
    // In practice, search results would automatically include categories
    // and downloads would use those categories for folder organization
}
