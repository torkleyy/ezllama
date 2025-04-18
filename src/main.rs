use anyhow::Result;
use ezllama::LlamaParams;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .or_else(|_| EnvFilter::try_new("info"))
                .unwrap(),
        )
        .event_format(
            tracing_subscriber::fmt::format::format()
                .compact()
                .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
                    "%H:%M:%S.%3f".to_owned(),
                )),
        )
        .init();

    let params = LlamaParams {
        model_path: PathBuf::from("/Users/thomas/models/llm/gemma-1.1-7b-it.Q4_K_M.gguf"),
        prompt: Some("Rust is the".to_string()),
        n_len: 32,
        ..Default::default()
    };

    ezllama::run_llama(params)
}
