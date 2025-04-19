use ezllama::{Model, ModelParams, Result};
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

    // Example 1: Using system message with chat session
    println!("\n=== Example 1: Chat with System Message ===\n");
    let mut chat_session = model.create_chat_session_with_system(
        "You are a helpful assistant that specializes in Rust programming.",
        &model_params,
    )?;
    
    chat_session.add_user_message("What are the benefits of using the Result type in Rust?");
    let response = chat_session.generate(256)?;
    
    println!("System: You are a helpful assistant that specializes in Rust programming.");
    println!("User: What are the benefits of using the Result type in Rust?");
    println!("Assistant: {}\n", response);

    // Example 2: Multi-turn conversation with system message
    println!("\n=== Example 2: Multi-turn Conversation with System Message ===\n");
    let mut chat_session = model.create_chat_session(&model_params)?;
    
    // Add system message
    chat_session.add_system_message("You are a friendly assistant that speaks in a casual, conversational tone.");
    
    // First turn
    chat_session.add_user_message("Tell me about the weather today.");
    let response1 = chat_session.generate(128)?;
    
    println!("System: You are a friendly assistant that speaks in a casual, conversational tone.");
    println!("User: Tell me about the weather today.");
    println!("Assistant: {}\n", response1);
    
    // Second turn
    chat_session.add_user_message("What should I wear?");
    let response2 = chat_session.generate(128)?;
    
    println!("User: What should I wear?");
    println!("Assistant: {}\n", response2);

    // Example 3: Using the convenience method for chat completion with system message
    println!("\n=== Example 3: Chat Completion with System Message ===\n");
    
    let system_message = "You are a concise assistant that provides brief, to-the-point answers.";
    let user_message = "Explain quantum computing.";
    
    let response = model.chat_completion_with_system(
        system_message,
        user_message,
        256,
        &model_params,
    )?;
    
    println!("System: {}", system_message);
    println!("User: {}", user_message);
    println!("Assistant: {}\n", response);

    Ok(())
}
