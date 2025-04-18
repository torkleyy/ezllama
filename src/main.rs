use anyhow::Result;
use ezllama::{LlamaParams, Model, ModelParams};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::DEBUG.into())
                .from_env_lossy(),
        )
        .event_format(
            tracing_subscriber::fmt::format::format()
                .compact()
                .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
                    "%H:%M:%S.%3f".to_owned(),
                )),
        )
        .init();

    let model_params = ModelParams {
        model_path: PathBuf::from("/Users/thomas/models/llm/gemma-1.1-7b-it.Q4_K_M.gguf"),
        ..Default::default()
    };

    let mut model = Model::new(&model_params)?;

    // Generate text multiple times with the same model
    let output1 = model.generate("Rust is a", 32, None)?;
    println!("First generation: {}", output1);

    let output2 = model.generate("Python is a", 32, None)?;
    println!("Second generation: {}", output2);

    Ok(())
}
