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

    let model = Model::new(&model_params)?;

    let mut chat_session = model.create_chat_session(&model_params)?;

    // First turn
    chat_session.add_user_message("Hello, can you introduce yourself?");
    let response1 = chat_session.generate(128)?;
    println!("Assistant: {}", response1);

    // Second turn (can do the same in one step)
    let response2 = chat_session.prompt("What can you help me with?", 128)?;
    println!("Assistant: {}", response2);

    Ok(())
}
```

## Advanced Usage

### Text completion

**Note:** Many models struggle with continuing from a second prompt
in the same context.

```rust
// Create a text session for text completion
let mut text_session = model.create_text_session(&model_params)?;
let output = text_session.prompt("Once upon a time", 128)?;

// Continue generating from the existing context
let more_output = text_session.prompt(" and then", 128)?;
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

## License

Dual-licensed under MIT and Apache 2.0.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual licensed as above, without any additional terms or conditions.
