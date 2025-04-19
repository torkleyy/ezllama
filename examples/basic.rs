use ezllama::Result;
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

    // Initialize the model
    let model_params = ModelParams {
        model_path: PathBuf::from("arcee-agent.Q4_1.gguf"),
        ..Default::default()
    };

    let model = Model::new(&model_params)?;

    // Example 1: Simple text completion using TextSession
    println!("\n=== Example 1: Text Completion with TextSession ===\n");
    let mut text_session = model.create_text_session(&model_params)?;
    let output1 = text_session.prompt("Once upon a time", 128)?;
    println!("Prompt: Once upon a time");
    println!("Completion: {}\n", output1);

    // Example 2: Single message chat completion using ChatSession
    println!("\n=== Example 2: Single Message Chat with ChatSession ===\n");
    let mut chat_session = model.create_chat_session(&model_params)?;
    chat_session.add_user_message("What is Rust programming language?");
    let chat_output = chat_session.generate(256)?;
    println!("User: What is Rust programming language?");
    println!("Assistant: {}\n", chat_output);

    // Example 3: Multi-turn conversation with default template
    println!("\n=== Example 3: Multi-turn Conversation (Default Template) ===\n");
    let mut chat_session = model.create_chat_session(&model_params)?;

    // First turn
    chat_session.add_user_message("Hello, can you introduce yourself?");
    let response1 = chat_session.generate(128)?;
    println!("User: Hello, can you introduce yourself?");
    println!("Assistant: {}\n", response1);

    // Second turn
    chat_session.add_user_message("What can you help me with?");
    let response2 = chat_session.generate(128)?;
    println!("User: What can you help me with?");
    println!("Assistant: {}\n", response2);

    Ok(())
}
