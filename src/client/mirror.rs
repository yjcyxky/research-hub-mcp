use crate::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use url::Url;

/// Health status of a Sci-Hub mirror
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MirrorHealth {
    /// Mirror is healthy and responding
    Healthy,
    /// Mirror is responding slowly
    Degraded,
    /// Mirror is not responding
    Unhealthy,
    /// Mirror health is unknown (not yet tested)
    #[default]
    Unknown,
}

/// Represents a Sci-Hub mirror with health information
#[derive(Debug, Clone)]
pub struct Mirror {
    /// Mirror URL
    pub url: Url,
    /// Current health status
    pub health: MirrorHealth,
    /// Last health check timestamp
    pub last_check: Option<SystemTime>,
    /// Response time in milliseconds
    pub response_time: Option<Duration>,
    /// Number of consecutive failures
    pub failure_count: u32,
    /// Whether this mirror is currently enabled
    pub enabled: bool,
}

impl Mirror {
    /// Create a new mirror from a URL string
    pub fn new(url_str: &str) -> Result<Self> {
        let url = Url::parse(url_str).map_err(|e| crate::Error::InvalidInput {
            field: "mirror_url".to_string(),
            reason: format!("Invalid mirror URL '{url_str}': {e}"),
        })?;

        if url.scheme() != "https" {
            return Err(crate::Error::InvalidInput {
                field: "mirror_url".to_string(),
                reason: format!("Mirror URL must use HTTPS: {url_str}"),
            });
        }

        Ok(Self {
            url,
            health: MirrorHealth::Unknown,
            last_check: None,
            response_time: None,
            failure_count: 0,
            enabled: true,
        })
    }

    /// Check if this mirror should be used based on health and failure count
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.enabled
            && self.failure_count < 3
            && matches!(
                self.health,
                MirrorHealth::Healthy | MirrorHealth::Degraded | MirrorHealth::Unknown
            )
    }

    /// Mark this mirror as failed
    pub fn mark_failure(&mut self) {
        self.failure_count += 1;
        self.health = MirrorHealth::Unhealthy;
        self.last_check = Some(SystemTime::now());

        if self.failure_count >= 3 {
            warn!(
                "Mirror {} marked as unhealthy after {} failures",
                self.url, self.failure_count
            );
        }
    }

    /// Mark this mirror as successful
    pub fn mark_success(&mut self, response_time: Duration) {
        self.failure_count = 0;
        self.response_time = Some(response_time);
        self.last_check = Some(SystemTime::now());

        // Determine health based on response time
        self.health = if response_time < Duration::from_millis(2000) {
            MirrorHealth::Healthy
        } else if response_time < Duration::from_millis(5000) {
            MirrorHealth::Degraded
        } else {
            MirrorHealth::Unhealthy
        };

        debug!(
            "Mirror {} health check: {:?} ({}ms)",
            self.url,
            self.health,
            response_time.as_millis()
        );
    }

    /// Check if a health check is needed
    #[must_use]
    pub fn needs_health_check(&self) -> bool {
        self.last_check.map_or(true, |last_check| {
            let check_interval = match self.health {
                MirrorHealth::Healthy => Duration::from_secs(300), // 5 minutes
                MirrorHealth::Degraded => Duration::from_secs(120), // 2 minutes
                MirrorHealth::Unhealthy => Duration::from_secs(60), // 1 minute
                MirrorHealth::Unknown => Duration::from_secs(10),  // 10 seconds
            };

            SystemTime::now()
                .duration_since(last_check)
                .unwrap_or(Duration::ZERO)
                > check_interval
        })
    }
}

/// Manages multiple Sci-Hub mirrors with health checking and rotation
pub struct MirrorManager {
    mirrors: Arc<RwLock<Vec<Mirror>>>,
    client: Client,
    current_index: Arc<RwLock<usize>>,
}

impl MirrorManager {
    /// Create a new mirror manager with the given mirror URLs
    pub fn new(mirror_urls: Vec<String>, client: Client) -> Result<Self> {
        let mut mirrors = Vec::new();

        for url_str in mirror_urls {
            match Mirror::new(&url_str) {
                Ok(mirror) => mirrors.push(mirror),
                Err(e) => {
                    warn!("Skipping invalid mirror URL '{}': {}", url_str, e);
                }
            }
        }

        if mirrors.is_empty() {
            return Err(crate::Error::InvalidInput {
                field: "mirrors".to_string(),
                reason: "No valid Sci-Hub mirrors configured".to_string(),
            });
        }

        info!("Initialized mirror manager with {} mirrors", mirrors.len());

        Ok(Self {
            mirrors: Arc::new(RwLock::new(mirrors)),
            client,
            current_index: Arc::new(RwLock::new(0)),
        })
    }

    /// Get the next available mirror, rotating through healthy mirrors
    pub async fn get_next_mirror(&self) -> Option<Mirror> {
        let mirrors = self.mirrors.read().await;

        // First, try to find a healthy mirror starting from current index
        let start_index = *self.current_index.read().await;
        let usable_mirrors: Vec<(usize, &Mirror)> = mirrors
            .iter()
            .enumerate()
            .cycle()
            .skip(start_index)
            .take(mirrors.len())
            .filter(|(_, mirror)| mirror.is_usable())
            .collect();

        if let Some((index, mirror)) = usable_mirrors.first() {
            // Update current index for round-robin rotation
            *self.current_index.write().await = (*index + 1) % mirrors.len();

            return Some((*mirror).clone());
        }

        // If no healthy mirrors, try to use any enabled mirror
        mirrors.iter().find(|mirror| mirror.enabled).cloned()
    }

    /// Perform health checks on all mirrors that need checking
    pub async fn health_check_all(&self) {
        let mirrors_to_check = {
            let mirrors = self.mirrors.read().await;
            mirrors
                .iter()
                .enumerate()
                .filter(|(_, mirror)| mirror.needs_health_check())
                .map(|(i, mirror)| (i, mirror.clone()))
                .collect::<Vec<_>>()
        };

        if mirrors_to_check.is_empty() {
            debug!("No mirrors need health checking");
            return;
        }

        info!(
            "Performing health checks on {} mirrors",
            mirrors_to_check.len()
        );

        // Perform health checks concurrently
        let health_check_tasks = mirrors_to_check.into_iter().map(|(index, mirror)| {
            let client = self.client.clone();
            let mirrors = self.mirrors.clone();

            tokio::spawn(async move {
                let start_time = SystemTime::now();

                // Simple health check: HEAD request to the mirror root
                let health_check_result = client
                    .head(mirror.url.as_str())
                    .timeout(Duration::from_secs(10))
                    .send()
                    .await;

                let response_time = start_time.elapsed().unwrap_or(Duration::from_secs(10));

                // Update mirror health
                let mut mirrors_write = mirrors.write().await;
                if let Some(mirror_ref) = mirrors_write.get_mut(index) {
                    match health_check_result {
                        Ok(response) if response.status().is_success() => {
                            mirror_ref.mark_success(response_time);
                        }
                        Ok(response) => {
                            warn!(
                                "Mirror {} health check failed with status: {}",
                                mirror_ref.url,
                                response.status()
                            );
                            mirror_ref.mark_failure();
                        }
                        Err(e) => {
                            warn!("Mirror {} health check failed: {}", mirror_ref.url, e);
                            mirror_ref.mark_failure();
                        }
                    }
                }
            })
        });

        // Wait for all health checks to complete
        futures::future::join_all(health_check_tasks).await;

        debug!("Health checks completed");
    }

    /// Mark a mirror as failed (called when a request fails)
    pub async fn mark_mirror_failed(&self, mirror_url: &Url) {
        let mut mirrors = self.mirrors.write().await;

        if let Some(mirror) = mirrors.iter_mut().find(|m| &m.url == mirror_url) {
            mirror.mark_failure();
            error!("Marked mirror {} as failed", mirror_url);
        }
    }

    /// Mark a mirror as successful (called when a request succeeds)
    pub async fn mark_mirror_success(&self, mirror_url: &Url, response_time: Duration) {
        let mut mirrors = self.mirrors.write().await;

        if let Some(mirror) = mirrors.iter_mut().find(|m| &m.url == mirror_url) {
            mirror.mark_success(response_time);
            debug!("Marked mirror {} as successful", mirror_url);
        }
    }

    /// Get current status of all mirrors
    pub async fn get_mirror_status(&self) -> Vec<Mirror> {
        self.mirrors.read().await.clone()
    }

    /// Get count of healthy mirrors
    pub async fn healthy_mirror_count(&self) -> usize {
        let mirrors = self.mirrors.read().await;
        mirrors.iter().filter(|m| m.is_usable()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_creation() {
        let mirror = Mirror::new("https://sci-hub.se").unwrap();
        assert_eq!(mirror.url.as_str(), "https://sci-hub.se/");
        assert_eq!(mirror.health, MirrorHealth::Unknown);
        assert!(mirror.is_usable());
    }

    #[test]
    fn test_mirror_invalid_url() {
        assert!(Mirror::new("not-a-url").is_err());
        assert!(Mirror::new("http://insecure.com").is_err());
    }

    #[test]
    fn test_mirror_failure_tracking() {
        let mut mirror = Mirror::new("https://sci-hub.se").unwrap();

        // Should be usable initially
        assert!(mirror.is_usable());

        // Mark failures - each failure sets health to Unhealthy
        mirror.mark_failure();
        assert_eq!(mirror.failure_count, 1);
        assert!(!mirror.is_usable()); // No longer usable due to Unhealthy status

        mirror.mark_failure();
        mirror.mark_failure();
        assert_eq!(mirror.failure_count, 3);
        assert!(!mirror.is_usable()); // Still not usable
    }

    #[test]
    fn test_mirror_success_resets_failures() {
        let mut mirror = Mirror::new("https://sci-hub.se").unwrap();

        mirror.mark_failure();
        mirror.mark_failure();
        assert_eq!(mirror.failure_count, 2);

        mirror.mark_success(Duration::from_millis(500));
        assert_eq!(mirror.failure_count, 0);
        assert_eq!(mirror.health, MirrorHealth::Healthy);
    }

    #[tokio::test]
    async fn test_mirror_manager_creation() {
        let client = Client::new();
        let mirror_urls = vec![
            "https://sci-hub.se".to_string(),
            "https://sci-hub.st".to_string(),
        ];

        let manager = MirrorManager::new(mirror_urls, client).unwrap();
        assert_eq!(manager.healthy_mirror_count().await, 2); // All unknown initially, which are usable
    }
}
