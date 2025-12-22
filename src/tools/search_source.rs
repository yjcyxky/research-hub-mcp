use crate::client::providers::{ProviderError, SearchContext, SearchQuery, SearchType, SourceProvider};
use crate::client::PaperMetadata;
use crate::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Input for source-specific search
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchSourceInput {
    /// Target source (arxiv, pubmed, crossref, semantic_scholar, etc.)
    #[schemars(description = "Target source: arxiv, pubmed, crossref, semantic_scholar, openalex, biorxiv, medrxiv")]
    pub source: String,

    /// Search query using source-native syntax
    #[schemars(description = "Search query (use source-native syntax for best results)")]
    pub query: String,

    /// Maximum results to return (default: 10)
    #[serde(default = "default_limit")]
    #[schemars(description = "Maximum results (1-100, default: 10)")]
    pub limit: u32,

    /// Pagination offset (default: 0)
    #[serde(default)]
    #[schemars(description = "Pagination offset for subsequent pages")]
    pub offset: u32,

    /// Search type hint (auto, doi, title, author, keywords)
    #[serde(default)]
    #[schemars(description = "Search type: auto, doi, title, author, keywords, subject")]
    pub search_type: Option<String>,

    /// Return query format help instead of searching
    #[serde(default)]
    #[schemars(description = "Set to true to get query format help for this source")]
    pub help: bool,
}

fn default_limit() -> u32 {
    10
}

/// Source information with query help
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SourceInfo {
    /// Source name
    pub name: String,
    /// Source description
    pub description: String,
    /// Query format help
    pub query_format_help: String,
    /// Example queries
    pub query_examples: Vec<QueryExample>,
    /// Supported search types
    pub supported_search_types: Vec<String>,
    /// Native query syntax documentation (if available)
    pub native_query_syntax: Option<String>,
}

/// Query example
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryExample {
    /// Example query string
    pub query: String,
    /// Description of what this query finds
    pub description: String,
}

/// Search result from a specific source
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchSourceResult {
    /// Source that was searched
    pub source: String,
    /// Search query used
    pub query: String,
    /// Papers found
    pub papers: Vec<PaperMetadata>,
    /// Total results available (if known)
    pub total_available: Option<u32>,
    /// Whether more results are available
    pub has_more: bool,
    /// Search duration in milliseconds
    pub search_time_ms: u64,
    /// Source information (when help=true)
    pub source_info: Option<SourceInfo>,
}

/// Source-specific search tool
pub struct SearchSourceTool {
    providers: HashMap<String, Box<dyn SourceProvider>>,
}

impl SearchSourceTool {
    /// Create a new search source tool
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider
    pub fn register_provider(&mut self, provider: Box<dyn SourceProvider>) {
        let name = provider.name().to_string();
        self.providers.insert(name, provider);
    }

    /// Get available sources
    pub fn available_sources(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Get source info with query help
    pub fn get_source_info(&self, source_name: &str) -> Option<SourceInfo> {
        let provider = self.providers.get(source_name)?;

        Some(SourceInfo {
            name: provider.name().to_string(),
            description: provider.description().to_string(),
            query_format_help: provider.query_format_help().to_string(),
            query_examples: provider
                .query_examples()
                .into_iter()
                .map(|(query, desc)| QueryExample {
                    query: query.to_string(),
                    description: desc.to_string(),
                })
                .collect(),
            supported_search_types: provider
                .supported_search_types()
                .into_iter()
                .map(|t| format!("{:?}", t).to_lowercase())
                .collect(),
            native_query_syntax: provider.native_query_syntax().map(String::from),
        })
    }

    /// Search a specific source
    pub async fn search(&self, input: SearchSourceInput) -> Result<SearchSourceResult> {
        let source_name = input.source.to_lowercase();

        // Check if this is a help request
        if input.help {
            return self.handle_help_request(&source_name, &input.query);
        }

        // Find the provider
        let provider = self.providers.get(&source_name).ok_or_else(|| {
            let available: Vec<_> = self.providers.keys().collect();
            crate::Error::InvalidInput {
                field: "source".to_string(),
                reason: format!(
                    "Unknown source '{}'. Available: {:?}",
                    source_name, available
                ),
            }
        })?;

        info!("Searching {} for: {}", source_name, input.query);

        // Parse search type
        let search_type = match input.search_type.as_deref() {
            Some("doi") => SearchType::Doi,
            Some("title") => SearchType::Title,
            Some("author") => SearchType::Author,
            Some("keywords") => SearchType::Keywords,
            Some("subject") => SearchType::Subject,
            Some("title_abstract") => SearchType::TitleAbstract,
            _ => SearchType::Auto,
        };

        // Build search query
        let query = SearchQuery {
            query: input.query.clone(),
            search_type,
            max_results: input.limit.min(100),
            offset: input.offset,
            params: HashMap::new(),
            sources: None,
            metadata_sources: None,
        };

        // Build context
        let context = SearchContext {
            timeout: Duration::from_secs(30),
            user_agent: "research_hub_mcp/1.0.0".to_string(),
            rate_limit: None,
            headers: HashMap::new(),
        };

        let start = std::time::Instant::now();

        // Execute search
        let result = provider.search(&query, &context).await.map_err(|e| {
            match e {
                ProviderError::Network(msg) => crate::Error::Provider(format!(
                    "{} network error: {}",
                    source_name, msg
                )),
                ProviderError::RateLimit => crate::Error::RateLimitExceeded {
                    retry_after: std::time::Duration::from_secs(60),
                },
                ProviderError::Timeout => crate::Error::Timeout {
                    timeout: std::time::Duration::from_secs(30),
                },
                ProviderError::InvalidQuery(msg) => crate::Error::InvalidInput {
                    field: "query".to_string(),
                    reason: msg,
                },
                _ => crate::Error::Provider(e.to_string()),
            }
        })?;

        let search_time = start.elapsed();

        debug!(
            "Search completed: {} results in {}ms",
            result.papers.len(),
            search_time.as_millis()
        );

        Ok(SearchSourceResult {
            source: source_name,
            query: input.query,
            papers: result.papers,
            total_available: result.total_available,
            has_more: result.has_more,
            search_time_ms: search_time.as_millis() as u64,
            source_info: None,
        })
    }

    /// Handle help request
    fn handle_help_request(&self, source: &str, _query: &str) -> Result<SearchSourceResult> {
        if source.is_empty() || source == "all" {
            // Return info about all sources
            let all_sources: Vec<SourceInfo> = self
                .providers
                .keys()
                .filter_map(|name| self.get_source_info(name))
                .collect();

            // For "all" sources help, we return info in a special way
            warn!("Help requested for all sources. {} providers available.", all_sources.len());

            Ok(SearchSourceResult {
                source: "all".to_string(),
                query: String::new(),
                papers: vec![],
                total_available: Some(all_sources.len() as u32),
                has_more: false,
                search_time_ms: 0,
                source_info: if all_sources.len() == 1 {
                    all_sources.into_iter().next()
                } else {
                    // Return first source info as placeholder
                    // In practice, the full list would be in response metadata
                    all_sources.into_iter().next()
                },
            })
        } else if let Some(info) = self.get_source_info(source) {
            Ok(SearchSourceResult {
                source: source.to_string(),
                query: String::new(),
                papers: vec![],
                total_available: None,
                has_more: false,
                search_time_ms: 0,
                source_info: Some(info),
            })
        } else {
            Err(crate::Error::InvalidInput {
                field: "source".to_string(),
                reason: format!("Unknown source: {}", source),
            })
        }
    }
}

impl Default for SearchSourceTool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SearchSourceTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchSourceTool")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_source_input() {
        let input = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "machine learning".to_string(),
            limit: 10,
            offset: 0,
            search_type: None,
            help: false,
        };

        assert_eq!(input.source, "arxiv");
        assert_eq!(input.limit, 10);
    }

    #[test]
    fn test_search_source_input_all_fields() {
        let input = SearchSourceInput {
            source: "semantic_scholar".to_string(),
            query: "deep learning transformer".to_string(),
            limit: 50,
            offset: 20,
            search_type: Some("title".to_string()),
            help: true,
        };

        assert_eq!(input.source, "semantic_scholar");
        assert_eq!(input.query, "deep learning transformer");
        assert_eq!(input.limit, 50);
        assert_eq!(input.offset, 20);
        assert_eq!(input.search_type, Some("title".to_string()));
        assert!(input.help);
    }

    #[test]
    fn test_search_source_input_serialization() {
        let input = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "quantum computing".to_string(),
            limit: 10,
            offset: 0,
            search_type: Some("keywords".to_string()),
            help: false,
        };

        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"source\":\"arxiv\""));
        assert!(json.contains("\"query\":\"quantum computing\""));

        let parsed: SearchSourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, input.source);
        assert_eq!(parsed.query, input.query);
    }

    #[test]
    fn test_search_source_input_deserialization_defaults() {
        let json = r#"{"source": "pubmed", "query": "cancer"}"#;
        let input: SearchSourceInput = serde_json::from_str(json).unwrap();

        assert_eq!(input.source, "pubmed");
        assert_eq!(input.query, "cancer");
        assert_eq!(input.limit, 10); // default
        assert_eq!(input.offset, 0); // default
        assert!(input.search_type.is_none());
        assert!(!input.help); // default false
    }

    #[test]
    fn test_default_limit() {
        assert_eq!(default_limit(), 10);
    }

    #[test]
    fn test_tool_creation() {
        let tool = SearchSourceTool::new();
        assert!(tool.available_sources().is_empty());
    }

    #[test]
    fn test_tool_default() {
        let tool = SearchSourceTool::default();
        assert!(tool.available_sources().is_empty());
    }

    #[test]
    fn test_source_info_structure() {
        let info = SourceInfo {
            name: "arxiv".to_string(),
            description: "ArXiv preprint server".to_string(),
            query_format_help: "Use field prefixes".to_string(),
            query_examples: vec![
                QueryExample {
                    query: "ti:transformer".to_string(),
                    description: "Search by title".to_string(),
                },
            ],
            supported_search_types: vec!["title".to_string(), "author".to_string()],
            native_query_syntax: Some("ArXiv syntax docs".to_string()),
        };

        assert_eq!(info.name, "arxiv");
        assert_eq!(info.query_examples.len(), 1);
        assert!(info.native_query_syntax.is_some());
    }

    #[test]
    fn test_source_info_serialization() {
        let info = SourceInfo {
            name: "crossref".to_string(),
            description: "CrossRef metadata".to_string(),
            query_format_help: "Standard queries".to_string(),
            query_examples: vec![],
            supported_search_types: vec!["doi".to_string()],
            native_query_syntax: None,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"name\":\"crossref\""));
        assert!(json.contains("\"supported_search_types\":[\"doi\"]"));
    }

    #[test]
    fn test_query_example_structure() {
        let example = QueryExample {
            query: "au:Einstein".to_string(),
            description: "Search papers by Einstein".to_string(),
        };

        assert_eq!(example.query, "au:Einstein");
        assert_eq!(example.description, "Search papers by Einstein");

        let json = serde_json::to_string(&example).unwrap();
        assert!(json.contains("au:Einstein"));
    }

    #[test]
    fn test_search_source_result_structure() {
        let result = SearchSourceResult {
            source: "arxiv".to_string(),
            query: "machine learning".to_string(),
            papers: vec![],
            total_available: Some(100),
            has_more: true,
            search_time_ms: 250,
            source_info: None,
        };

        assert_eq!(result.source, "arxiv");
        assert_eq!(result.total_available, Some(100));
        assert!(result.has_more);
        assert_eq!(result.search_time_ms, 250);
    }

    #[test]
    fn test_search_source_result_serialization() {
        let result = SearchSourceResult {
            source: "pubmed".to_string(),
            query: "covid-19".to_string(),
            papers: vec![],
            total_available: None,
            has_more: false,
            search_time_ms: 100,
            source_info: Some(SourceInfo {
                name: "pubmed".to_string(),
                description: "PubMed database".to_string(),
                query_format_help: "Use MeSH terms".to_string(),
                query_examples: vec![],
                supported_search_types: vec![],
                native_query_syntax: None,
            }),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"source\":\"pubmed\""));
        assert!(json.contains("\"query\":\"covid-19\""));
        assert!(json.contains("\"has_more\":false"));
        assert!(json.contains("\"source_info\""));
    }

    #[test]
    fn test_search_source_result_with_source_info() {
        let result = SearchSourceResult {
            source: "semantic_scholar".to_string(),
            query: "".to_string(),
            papers: vec![],
            total_available: None,
            has_more: false,
            search_time_ms: 0,
            source_info: Some(SourceInfo {
                name: "semantic_scholar".to_string(),
                description: "AI-powered search".to_string(),
                query_format_help: "Natural language".to_string(),
                query_examples: vec![
                    QueryExample {
                        query: "attention is all you need".to_string(),
                        description: "Search by paper title".to_string(),
                    },
                ],
                supported_search_types: vec!["auto".to_string(), "title".to_string()],
                native_query_syntax: None,
            }),
        };

        assert!(result.source_info.is_some());
        let info = result.source_info.unwrap();
        assert_eq!(info.query_examples.len(), 1);
    }

    #[test]
    fn test_tool_debug_format() {
        let tool = SearchSourceTool::new();
        let debug_str = format!("{:?}", tool);
        assert!(debug_str.contains("SearchSourceTool"));
        assert!(debug_str.contains("providers"));
    }

    #[test]
    fn test_available_sources_empty() {
        let tool = SearchSourceTool::new();
        let sources = tool.available_sources();
        assert!(sources.is_empty());
    }

    #[test]
    fn test_get_source_info_not_found() {
        let tool = SearchSourceTool::new();
        let info = tool.get_source_info("nonexistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_search_type_parsing() {
        // Test that search types are correctly represented
        let types = vec!["doi", "title", "author", "keywords", "subject", "title_abstract", "auto"];

        for search_type in types {
            let input = SearchSourceInput {
                source: "test".to_string(),
                query: "test".to_string(),
                limit: 10,
                offset: 0,
                search_type: Some(search_type.to_string()),
                help: false,
            };
            assert_eq!(input.search_type.as_deref(), Some(search_type));
        }
    }

    #[test]
    fn test_limit_clamping_in_input() {
        // Test limit values at boundaries
        let input_min = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "test".to_string(),
            limit: 1,
            offset: 0,
            search_type: None,
            help: false,
        };
        assert_eq!(input_min.limit, 1);

        let input_max = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "test".to_string(),
            limit: 100,
            offset: 0,
            search_type: None,
            help: false,
        };
        assert_eq!(input_max.limit, 100);

        // Values over 100 should be accepted at input level (clamped at search level)
        let input_over = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "test".to_string(),
            limit: 200,
            offset: 0,
            search_type: None,
            help: false,
        };
        assert_eq!(input_over.limit, 200);
    }

    #[test]
    fn test_offset_values() {
        let input = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "test".to_string(),
            limit: 10,
            offset: 50,
            search_type: None,
            help: false,
        };
        assert_eq!(input.offset, 50);
    }

    #[test]
    fn test_help_mode_input() {
        let input = SearchSourceInput {
            source: "arxiv".to_string(),
            query: "".to_string(),
            limit: 10,
            offset: 0,
            search_type: None,
            help: true,
        };

        assert!(input.help);
        assert!(input.query.is_empty());
    }

    #[test]
    fn test_source_info_without_native_syntax() {
        let info = SourceInfo {
            name: "openalex".to_string(),
            description: "OpenAlex API".to_string(),
            query_format_help: "Filter syntax".to_string(),
            query_examples: vec![],
            supported_search_types: vec!["auto".to_string()],
            native_query_syntax: None,
        };

        assert!(info.native_query_syntax.is_none());
    }

    #[test]
    fn test_multiple_query_examples() {
        let info = SourceInfo {
            name: "arxiv".to_string(),
            description: "ArXiv".to_string(),
            query_format_help: "Help".to_string(),
            query_examples: vec![
                QueryExample {
                    query: "ti:neural".to_string(),
                    description: "Title search".to_string(),
                },
                QueryExample {
                    query: "au:Hinton".to_string(),
                    description: "Author search".to_string(),
                },
                QueryExample {
                    query: "cat:cs.AI".to_string(),
                    description: "Category search".to_string(),
                },
            ],
            supported_search_types: vec!["title".to_string(), "author".to_string()],
            native_query_syntax: Some("ArXiv query syntax".to_string()),
        };

        assert_eq!(info.query_examples.len(), 3);
        assert_eq!(info.query_examples[0].query, "ti:neural");
        assert_eq!(info.query_examples[1].query, "au:Hinton");
        assert_eq!(info.query_examples[2].query, "cat:cs.AI");
    }

    #[test]
    fn test_search_result_empty_papers() {
        let result = SearchSourceResult {
            source: "arxiv".to_string(),
            query: "nonexistent unique query xyz123".to_string(),
            papers: vec![],
            total_available: Some(0),
            has_more: false,
            search_time_ms: 50,
            source_info: None,
        };

        assert!(result.papers.is_empty());
        assert_eq!(result.total_available, Some(0));
        assert!(!result.has_more);
    }

    // M3.5.5: Test native query format for each provider
    mod native_query_format_tests {
        use super::*;
        use crate::client::providers::*;

        // Helper function to verify provider query format basics
        fn verify_provider_query_format<P: SourceProvider>(provider: &P) {
            // Every provider should have query format help
            let help = provider.query_format_help();
            assert!(!help.is_empty(), "{} should have query format help", provider.name());

            // Every provider should have at least one example
            let examples = provider.query_examples();
            assert!(
                !examples.is_empty(),
                "{} should have at least one query example",
                provider.name()
            );

            // Each example should have both query and description
            for (query, desc) in examples {
                assert!(!query.is_empty(), "{} query example should not be empty", provider.name());
                assert!(!desc.is_empty(), "{} description should not be empty", provider.name());
            }
        }

        #[test]
        fn test_arxiv_query_format() {
            let provider = ArxivProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("ti:") || help.contains("au:") || help.contains("prefix"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_crossref_query_format() {
            let provider = CrossRefProvider::new(None).unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("author") || help.contains("filter"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_semantic_scholar_query_format() {
            let provider = SemanticScholarProvider::new(None).unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("keyword") || help.contains("query") || help.contains("search"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_pubmed_central_query_format() {
            let provider = PubMedCentralProvider::new(None).unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("MeSH") || help.contains("field") || help.contains("["));
        }

        #[test]
        fn test_openalex_query_format() {
            let provider = OpenAlexProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("filter") || help.contains("search") || help.contains("API"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_biorxiv_query_format() {
            let provider = BiorxivProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("preprint") || help.contains("date") || help.contains("biology"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_medrxiv_query_format() {
            let provider = MedrxivProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("medical") || help.contains("health") || help.contains("preprint"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_openreview_query_format() {
            let provider = OpenReviewProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("venue") || help.contains("conference") || help.contains("paper"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_core_query_format() {
            let provider = CoreProvider::new(None).unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("Lucene") || help.contains("field") || help.contains("syntax"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_mdpi_query_format() {
            let provider = MdpiProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("journal") || help.contains("open access") || help.contains("MDPI"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_ssrn_query_format() {
            let provider = SsrnProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("preprint") || help.contains("research") || help.contains("SSRN"));
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_unpaywall_query_format() {
            let provider = UnpaywallProvider::new_with_default_email().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("DOI") || help.contains("open access") || help.contains("lookup"));
            // Unpaywall is DOI-only, so may not have native query syntax
        }

        #[test]
        fn test_researchgate_query_format() {
            let provider = ResearchGateProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("DOI") || help.contains("PDF") || help.contains("ResearchGate"));
            // ResearchGate may not have complex native query syntax
        }

        #[test]
        fn test_google_scholar_query_format() {
            let provider = GoogleScholarProvider::new(None).unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(
                help.contains("author:") || help.contains("intitle:") || help.contains("operator"),
                "Google Scholar help should contain search operators"
            );
            assert!(provider.native_query_syntax().is_some());
        }

        #[test]
        fn test_scihub_query_format() {
            let provider = SciHubProvider::new().unwrap();
            verify_provider_query_format(&provider);

            let help = provider.query_format_help();
            assert!(help.contains("DOI") || help.contains("PDF") || help.contains("download"));
            // Sci-Hub is DOI-based, may not have complex query syntax
        }

        // Test that all providers have consistent format
        #[test]
        fn test_all_providers_have_consistent_format() {
            let providers: Vec<(&str, Box<dyn SourceProvider>)> = vec![
                ("arxiv", Box::new(ArxivProvider::new().unwrap())),
                ("crossref", Box::new(CrossRefProvider::new(None).unwrap())),
                ("semantic_scholar", Box::new(SemanticScholarProvider::new(None).unwrap())),
                ("pubmed_central", Box::new(PubMedCentralProvider::new(None).unwrap())),
                ("openalex", Box::new(OpenAlexProvider::new().unwrap())),
                ("biorxiv", Box::new(BiorxivProvider::new().unwrap())),
                ("medrxiv", Box::new(MedrxivProvider::new().unwrap())),
                ("openreview", Box::new(OpenReviewProvider::new().unwrap())),
                ("core", Box::new(CoreProvider::new(None).unwrap())),
                ("mdpi", Box::new(MdpiProvider::new().unwrap())),
                ("ssrn", Box::new(SsrnProvider::new().unwrap())),
                ("unpaywall", Box::new(UnpaywallProvider::new_with_default_email().unwrap())),
                ("researchgate", Box::new(ResearchGateProvider::new().unwrap())),
                ("google_scholar", Box::new(GoogleScholarProvider::new(None).unwrap())),
                ("sci_hub", Box::new(SciHubProvider::new().unwrap())),
            ];

            for (name, provider) in providers {
                assert_eq!(
                    provider.name(),
                    name,
                    "Provider name should match expected"
                );

                let help = provider.query_format_help();
                assert!(
                    help.len() >= 20,
                    "{} query_format_help should be descriptive (got {} chars)",
                    name,
                    help.len()
                );

                let examples = provider.query_examples();
                assert!(
                    !examples.is_empty(),
                    "{} should have at least one example",
                    name
                );
            }
        }

        // Test query examples are valid and diverse
        #[test]
        fn test_query_examples_diversity() {
            let provider = ArxivProvider::new().unwrap();
            let examples = provider.query_examples();

            // ArXiv should have examples for different query types
            let has_title_example = examples.iter().any(|(q, _)| q.contains("ti:"));
            let has_author_example = examples.iter().any(|(q, _)| q.contains("au:"));

            assert!(
                has_title_example || has_author_example,
                "ArXiv should have examples demonstrating different query prefixes"
            );
        }

        // Test native query syntax URLs are valid
        #[test]
        fn test_native_query_syntax_urls() {
            let providers_with_docs: Vec<Box<dyn SourceProvider>> = vec![
                Box::new(ArxivProvider::new().unwrap()),
                Box::new(CrossRefProvider::new(None).unwrap()),
                Box::new(SemanticScholarProvider::new(None).unwrap()),
                Box::new(OpenAlexProvider::new().unwrap()),
            ];

            for provider in providers_with_docs {
                if let Some(syntax) = provider.native_query_syntax() {
                    // If it's a URL, it should start with http
                    if syntax.starts_with("http") {
                        assert!(
                            syntax.starts_with("https://") || syntax.starts_with("http://"),
                            "{} native_query_syntax URL should be valid: {}",
                            provider.name(),
                            syntax
                        );
                    }
                }
            }
        }
    }
}
