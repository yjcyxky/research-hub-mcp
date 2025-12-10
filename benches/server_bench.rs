use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_research_mcp::{server::ResearchServerHandler, Config, Server};
use std::sync::Arc;
use tokio::runtime::Runtime;

fn benchmark_server_creation(c: &mut Criterion) {
    c.bench_function("server_creation", |b| {
        b.iter(|| {
            let config = black_box(Config::default());
            let _server = black_box(Server::new(config));
        })
    });
}

fn benchmark_handler_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("handler_initialization", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = black_box(Config::default());
                let handler =
                    ResearchServerHandler::new(Arc::new(config)).unwrap();
                black_box(handler)
            })
        })
    });
}

fn benchmark_ping_response(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("ping_response", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = black_box(Config::default());
                let handler = ResearchServerHandler::new(Arc::new(config)).unwrap();
                black_box(handler.ping().await.unwrap())
            })
        })
    });
}

fn benchmark_config_validation(c: &mut Criterion) {
    c.bench_function("config_validation", |b| {
        b.iter(|| {
            let config = black_box(Config::default());
            black_box(config.validate().unwrap())
        })
    });
}

criterion_group!(
    benches,
    benchmark_server_creation,
    benchmark_handler_initialization,
    benchmark_ping_response,
    benchmark_config_validation
);
criterion_main!(benches);
