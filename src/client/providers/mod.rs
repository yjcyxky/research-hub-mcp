pub mod arxiv;
pub mod biorxiv;
pub mod core;
pub mod crossref;
pub mod google_scholar;
pub mod mdpi;
pub mod medrxiv;
pub mod openalex;
pub mod openreview;
pub mod pubmed_central;
pub mod researchgate;
pub mod sci_hub;
pub mod semantic_scholar;
pub mod ssrn;
pub mod traits;
pub mod unpaywall;

pub use arxiv::ArxivProvider;
pub use biorxiv::BiorxivProvider;
pub use core::CoreProvider;
pub use crossref::CrossRefProvider;
pub use google_scholar::GoogleScholarProvider;
pub use mdpi::MdpiProvider;
pub use medrxiv::MedrxivProvider;
pub use openalex::OpenAlexProvider;
pub use openreview::OpenReviewProvider;
pub use pubmed_central::PubMedCentralProvider;
pub use researchgate::ResearchGateProvider;
pub use sci_hub::SciHubProvider;
pub use semantic_scholar::SemanticScholarProvider;
pub use ssrn::SsrnProvider;
pub use traits::{
    ProviderError, ProviderResult, SearchContext, SearchQuery, SearchType, SourceProvider,
};
pub use unpaywall::UnpaywallProvider;
