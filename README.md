# ezllama

[![Crates.io](https://img.shields.io/crates/v/ezllama.svg)](https://crates.io/crates/ezllama)
[![Documentation](https://docs.rs/ezllama/badge.svg)](https://docs.rs/ezllama)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

An opinionated, simple Rust interface for local LLMs, powered by [llama-cpp-2](https://github.com/rustformers/llama-cpp-rs).

## Features

- **Simple API**: Designed for ease of use with a clean, intuitive interface
- **Text and Chat Completion**: Support for both text and chat completion tasks
- **Tracing Integration**: Built-in logging via the tracing ecosystem

Right now it only supports the basics, but I might add more features in the future
as I need them.

## Installation

Add ezllama to your Cargo.toml:

```toml
[dependencies]
ezllama = "0.1.0"
```

For GPU acceleration, enable the appropriate feature:

```toml
[dependencies]
ezllama = { version = "0.1.0", features = ["cuda"] }  # For CUDA support
# or
ezllama = { version = "0.1.0", features = ["metal"] }  # For Metal support (macOS)
# or
ezllama = { version = "0.1.0", features = ["vulkan"] }  # For Vulkan support
```

## Quick Start

```rust
use ezllama::{Model, ModelParams, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Initialize the model
    let model_params = ModelParams {
        model_path: PathBuf::from("path/to/your/model.gguf"),
        ..Default::default()
    };

    let mut model = Model::new(&model_params)?;

    // Simple text completion using TextSession
    let mut text_session = model.create_text_session_with_prompt("Once upon a time");
    let output = text_session.generate(128)?;
    println!("Text completion: {}", output);

    // Single message chat completion using ChatSession
    let mut chat_session = model.create_chat_session();
    chat_session.add_user_message("What is Rust programming language?");
    let chat_output = chat_session.generate(256)?;
    println!("Chat response: {}", chat_output);

    // Multi-turn conversation
    let mut chat_session = model.create_chat_session();

    // First turn
    chat_session.add_user_message("Hello, can you introduce yourself?");
    let response1 = chat_session.generate(128)?;
    println!("Assistant: {}", response1);

    // Second turn
    chat_session.add_user_message("What can you help me with?");
    let response2 = chat_session.generate(128)?;
    println!("Assistant: {}", response2);

    Ok(())
}
```

## Advanced Usage

### Session Types

```rust
// Create a text session for text completion
let mut text_session = model.create_text_session();
text_session.set_prompt("Once upon a time");
let output = text_session.generate(128)?;

// Append to the existing prompt for continuation
text_session.append_prompt(" and then");
let more_output = text_session.generate(128)?;
```

### System Messages

```rust
// Create a chat session with a system message
let mut chat_session = model.create_chat_session_with_system(
    "You are a helpful assistant that specializes in Rust programming.",
    &model_params
)?;

// Or add a system message to an existing session
let mut chat_session = model.create_chat_session(&model_params)?;
chat_session.add_system_message("You are a helpful assistant.");

// One-shot completion with system message
let response = model.chat_completion_with_system(
    "You are a concise assistant.",
    "Explain quantum computing.",
    256,
    &model_params
)?;
```

### Custom Chat Templates

```rust
// Create a chat session with a custom template
let template = "{{0_role}}: {{0_content}}\n{{1_role}}: {{1_content}}";
let mut chat_session = model.create_chat_session_with_template(template.to_string(), &model_params)?;
```

### Model Parameters

```rust
let model_params = ModelParams {
    model_path: PathBuf::from("path/to/your/model.gguf"),
    threads: Some(4),                // Number of threads for generation
    ctx_size: Some(NonZeroU32::new(4096).unwrap()), // Context size
    ..Default::default()
};
```

## License

Licensed under MIT.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual licensed as above, without any additional terms or conditions.
