use ezllama::Result;
use ezllama::{ContextParams, Model, ModelParams};
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

    // Initialize context parameters
    let context_params = ContextParams::default();

    let model = Model::new(&model_params)?;

    // Example 1: Simple text completion using TextSession
    println!("\n=== Example 1: Text Completion with TextSession ===\n");
    let mut text_session = model.create_text_session(&context_params)?;
    print!("7 times 7 is ");
    for token in text_session.prompt("7 times 7 is")?.take(128) {
        print!("{}", token);
    }
    println!();

    // Example 2: Single message chat completion using ChatSession
    println!("\n=== Example 2: Single Message Chat with ChatSession ===\n");
    let mut chat_session = model.create_chat_session(&context_params)?;
    println!("User: What is Rust programming language?");
    print!("Assistant: ");
    for token in chat_session.prompt("What is Rust programming language?")? {
        print!("{}", token);
    }
    println!();

    // Example 3: Multi-turn conversation with default template
    println!("\n=== Example 3: Multi-turn Conversation (Default Template) ===\n");
    let mut chat_session = model.create_chat_session(&context_params)?;

    // First turn
    chat_session.add_user_message("Hello, can you introduce yourself?");
    let response1 = chat_session.generate()?.join();
    println!("User: Hello, can you introduce yourself?");
    println!("Assistant: {}\n", response1);

    // Second turn
    let response2 = chat_session.prompt("What can you help me with?")?.join();
    println!("User: What can you help me with?");
    println!("Assistant: {}\n", response2);

    Ok(())
}
