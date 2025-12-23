use crate::{Config, Result};
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument};

/// Input parameters for the code search tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CodeSearchInput {
    /// Pattern to search for (regex supported)
    pub pattern: String,

    /// Programming language filter (optional)
    pub language: Option<String>,

    /// Directory to search in (defaults to download directory)
    pub search_dir: Option<String>,

    /// Maximum number of results to return
    #[serde(default = "default_limit")]
    pub limit: u32,

    /// Include context lines around matches
    #[serde(default = "default_context")]
    pub context_lines: u32,
}

const fn default_limit() -> u32 {
    20
}

const fn default_context() -> u32 {
    3
}

/// Result of a code search operation
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CodeSearchResult {
    /// File path where code was found
    pub file_path: String,

    /// Paper title (if extractable)
    pub paper_title: Option<String>,

    /// Matched code snippets
    pub matches: Vec<CodeMatch>,

    /// Total number of matches found
    pub total_matches: usize,
}

/// Individual code match
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CodeMatch {
    /// Line number in the file
    pub line_number: usize,

    /// The matched line
    pub line: String,

    /// Context before the match
    pub context_before: Vec<String>,

    /// Context after the match
    pub context_after: Vec<String>,

    /// Detected programming language
    pub language: Option<String>,
}

/// Code search tool for finding patterns in research papers
#[derive(Debug, Clone)]
pub struct CodeSearchTool {
    config: Arc<Config>,
}

impl CodeSearchTool {
    /// Create a new code search tool
    pub const fn new(config: Arc<Config>) -> Result<Self> {
        Ok(Self { config })
    }

    /// Search for code patterns in downloaded papers
    #[instrument(skip(self))]
    pub async fn search(&self, input: CodeSearchInput) -> Result<Vec<CodeSearchResult>> {
        info!("Searching for code pattern: {}", input.pattern);

        // Compile regex pattern
        let regex = Regex::new(&input.pattern).map_err(|e| crate::Error::InvalidInput {
            field: "pattern".to_string(),
            reason: format!("Invalid regex pattern: {e}"),
        })?;

        // Determine search directory
        let search_dir = input.search_dir.clone().unwrap_or_else(|| {
            self.config
                .downloads
                .directory
                .to_string_lossy()
                .to_string()
        });

        let search_path = Path::new(&search_dir);
        if !search_path.exists() {
            return Err(crate::Error::InvalidInput {
                field: "search_dir".to_string(),
                reason: format!("Search directory not found: {search_dir}"),
            });
        }

        // Search for code in PDF text files (we'll need to extract text first)
        let mut results = Vec::new();
        let mut total_processed = 0;

        // Walk through directory
        for entry in fs::read_dir(search_path)? {
            let entry = entry?;
            let path = entry.path();

            // Process PDF files and text files
            if path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                if let Ok(text) = self.extract_text_from_pdf(&path) {
                    if let Some(result) = self.search_in_text(
                        &text,
                        path.to_str().unwrap_or("unknown"),
                        &regex,
                        &input,
                    ) {
                        results.push(result);
                        total_processed += 1;

                        if total_processed >= input.limit {
                            break;
                        }
                    }
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(text) = fs::read_to_string(&path) {
                    if let Some(result) = self.search_in_text(
                        &text,
                        path.to_str().unwrap_or("unknown"),
                        &regex,
                        &input,
                    ) {
                        results.push(result);
                        total_processed += 1;

                        if total_processed >= input.limit {
                            break;
                        }
                    }
                }
            }
        }

        info!("Found {} results with code patterns", results.len());
        Ok(results)
    }

    /// Extract text from PDF (simplified version - would need proper PDF library)
    fn extract_text_from_pdf(&self, path: &Path) -> Result<String> {
        debug!("Extracting text from PDF: {:?}", path);

        // For now, we'll use a simple approach
        // In production, we'd use a proper PDF extraction library
        if let Ok(bytes) = fs::read(path) {
            // Look for text patterns in PDF
            let text = String::from_utf8_lossy(&bytes);

            // Extract code blocks (simplified heuristic)
            let code_blocks = self.extract_code_blocks(&text);
            Ok(code_blocks.join("\n"))
        } else {
            Err(crate::Error::InvalidInput {
                field: "file_path".to_string(),
                reason: format!("Failed to read PDF: {path:?}"),
            })
        }
    }

    /// Extract code blocks from text
    fn extract_code_blocks(&self, text: &str) -> Vec<String> {
        let mut blocks = Vec::new();
        let mut in_code_block = false;
        let mut current_block = Vec::new();

        for line in text.lines() {
            // Simple heuristics for code detection
            if self.looks_like_code(line) {
                in_code_block = true;
                current_block.push(line.to_string());
            } else if in_code_block && line.trim().is_empty() {
                // Empty line might end code block
                if !current_block.is_empty() {
                    blocks.push(current_block.join("\n"));
                    current_block.clear();
                }
                in_code_block = false;
            } else if in_code_block {
                current_block.push(line.to_string());
            }
        }

        if !current_block.is_empty() {
            blocks.push(current_block.join("\n"));
        }

        blocks
    }

    /// Check if a line looks like code
    fn looks_like_code(&self, line: &str) -> bool {
        let trimmed = line.trim();

        // Common code patterns
        trimmed.starts_with("def ")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("public ")
            || trimmed.starts_with("private ")
            || trimmed.starts_with("const ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("var ")
            || trimmed.starts_with("if ")
            || trimmed.starts_with("for ")
            || trimmed.starts_with("while ")
            || trimmed.starts_with("return ")
            || trimmed.contains("();")
            || trimmed.contains("(){")
            || trimmed.contains(" = ")
            || trimmed.contains("->")
            || trimmed.contains("=>")
            || (trimmed.starts_with('{') && trimmed.ends_with('}'))
            || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    }

    /// Search for pattern in text
    fn search_in_text(
        &self,
        text: &str,
        file_path: &str,
        regex: &Regex,
        input: &CodeSearchInput,
    ) -> Option<CodeSearchResult> {
        let mut matches = Vec::new();
        let lines: Vec<&str> = text.lines().collect();

        for (idx, line) in lines.iter().enumerate() {
            if regex.is_match(line) {
                // Check language filter
                if let Some(ref lang) = input.language {
                    if self.detect_language(line).is_none_or(|l| l != *lang) {
                        continue;
                    }
                }

                // Get context
                let start = idx.saturating_sub(input.context_lines as usize);
                let end = (idx + input.context_lines as usize + 1).min(lines.len());

                let context_before: Vec<String> =
                    lines[start..idx].iter().map(|s| (*s).to_string()).collect();

                let context_after: Vec<String> = lines[(idx + 1)..end]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();

                matches.push(CodeMatch {
                    line_number: idx + 1,
                    line: (*line).to_string(),
                    context_before,
                    context_after,
                    language: self.detect_language(line),
                });

                if matches.len() >= input.limit as usize {
                    break;
                }
            }
        }

        if matches.is_empty() {
            None
        } else {
            Some(CodeSearchResult {
                file_path: file_path.to_string(),
                paper_title: self.extract_paper_title(text),
                total_matches: matches.len(),
                matches,
            })
        }
    }

    /// Detect programming language from code line
    fn detect_language(&self, line: &str) -> Option<String> {
        let trimmed = line.trim();

        if trimmed.starts_with("def ") || trimmed.starts_with("import ") {
            Some("python".to_string())
        } else if trimmed.starts_with("function ") || trimmed.contains("const ") {
            Some("javascript".to_string())
        } else if trimmed.starts_with("fn ") || trimmed.contains("let mut") {
            Some("rust".to_string())
        } else if trimmed.starts_with("public ") || trimmed.starts_with("private ") {
            Some("java".to_string())
        } else if trimmed.contains("#include") || trimmed.contains("std::") {
            Some("cpp".to_string())
        } else {
            None
        }
    }

    /// Extract paper title from text (simplified)
    fn extract_paper_title(&self, text: &str) -> Option<String> {
        // Look for common title patterns
        for line in text.lines().take(50) {
            let trimmed = line.trim();
            if trimmed.len() > 10 && trimmed.len() < 200 {
                // Simple heuristic: early lines with reasonable length
                if !trimmed.starts_with("Abstract")
                    && !trimmed.starts_with("Keywords")
                    && !trimmed.contains('@')
                    && !trimmed.contains("http")
                {
                    return Some(trimmed.to_string());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_detection() {
        let config = Arc::new(Config::default());
        let tool = CodeSearchTool::new(config).unwrap();

        assert!(tool.looks_like_code("def main():"));
        assert!(tool.looks_like_code("    return x + y"));
        assert!(tool.looks_like_code("const value = 42;"));
        assert!(!tool.looks_like_code("This is regular text"));
    }

    #[test]
    fn test_language_detection() {
        let config = Arc::new(Config::default());
        let tool = CodeSearchTool::new(config).unwrap();

        assert_eq!(
            tool.detect_language("def main():"),
            Some("python".to_string())
        );
        assert_eq!(
            tool.detect_language("fn main() {"),
            Some("rust".to_string())
        );
        assert_eq!(
            tool.detect_language("function test() {"),
            Some("javascript".to_string())
        );
    }
}
