use anyhow::Result;
use ezllama::{Model, ModelParams};
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

    // Example 2: Using the model wrapper for repeated use
    let model_params = ModelParams {
        model_path: PathBuf::from("arcee-agent.Q4_1.gguf"),
        ..Default::default()
    };

    let mut model = Model::new(&model_params)?;

    // Generate text multiple times with the same model
    let output1 = model.generate("Once upon a time", 128)?;
    println!("First generation: {}", output1);

    let output2 = model.generate(". End of story. Now let's talk about programming.", 64)?;
    println!("Second generation: {}", output2);

    Ok(())
}
