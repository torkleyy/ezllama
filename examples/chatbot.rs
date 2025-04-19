use ezllama::{Model, ModelParams, Result};
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::{EnvFilter, fmt};

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Check for verbose flag early
    let verbose = args.iter().any(|arg| arg == "--verbose" || arg == "-v");

    // Initialize tracing for logging
    let env_filter = if verbose {
        // Enable detailed logging when verbose flag is present
        EnvFilter::builder()
            .with_default_directive(Level::DEBUG.into())
            .from_env_lossy()
    } else {
        // Disable most logging by default
        EnvFilter::builder()
            .with_default_directive(Level::ERROR.into())
            .from_env_lossy()
    };

    // Initialize the subscriber with the configured filter
    fmt::Subscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(std::io::stdout)
        .init();

    // We've already parsed the args above

    if args.len() < 2 {
        eprintln!("Usage: {} <model_path> [options]", args[0]);
        eprintln!("Options:");
        eprintln!("  --system \"<system message>\"  Set a system message");
        eprintln!("  --tokens <number>           Number of tokens to generate (default: 256)");
        eprintln!("  --ctx-size <number>         Context size (default: 2048)");
        eprintln!("  --disable-gpu               Disable GPU acceleration");
        eprintln!("  --verbose, -v               Enable verbose logging");
        return Ok(());
    }

    // Extract model path (first argument)
    let model_path = PathBuf::from(&args[1]);

    // Parse optional arguments
    let mut system_message = None;
    let mut num_tokens = 2048;
    let mut ctx_size = 2048;
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    let mut disable_gpu = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--system" => {
                if i + 1 < args.len() {
                    system_message = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --system requires a message");
                    return Ok(());
                }
            }
            "--tokens" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<usize>() {
                        Ok(n) => {
                            num_tokens = n;
                            i += 2;
                        }
                        Err(_) => {
                            eprintln!("Error: --tokens requires a number");
                            return Ok(());
                        }
                    }
                } else {
                    eprintln!("Error: --tokens requires a number");
                    return Ok(());
                }
            }
            "--ctx-size" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<u32>() {
                        Ok(n) => {
                            ctx_size = n;
                            i += 2;
                        }
                        Err(_) => {
                            eprintln!("Error: --ctx-size requires a number");
                            return Ok(());
                        }
                    }
                } else {
                    eprintln!("Error: --ctx-size requires a number");
                    return Ok(());
                }
            }
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            "--disable-gpu" => {
                disable_gpu = true;
                i += 1;
            }
            "--verbose" | "-v" => {
                // Already handled earlier
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                return Ok(());
            }
        }
    }

    // Initialize model parameters
    let model_params = ModelParams {
        model_path,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
        ctx_size: NonZeroU32::new(ctx_size),
        ..Default::default()
    };

    // Load the model
    println!(
        "Loading model from {}...",
        model_params.model_path.display()
    );
    let model = match Model::new(&model_params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            return Ok(());
        }
    };
    println!("Model loaded successfully!");

    // Create a chat session
    let mut chat_session = if let Some(ref system) = system_message {
        println!("Using system message: {}", system);
        model.create_chat_session_with_system(system, &model_params)?
    } else {
        model.create_chat_session(&model_params)?
    };

    // Welcome message
    println!("\n=== Chat Bot ===");
    println!("Type your messages and press Enter to chat with the assistant.");
    println!("Type 'exit' or 'quit' to end the conversation.");
    println!("Type 'clear' to start a new conversation.");
    println!("Type 'system: <message>' to set a system message.");
    println!("===========================\n");

    // Main chat loop
    loop {
        // Display prompt
        print!("You: ");
        io::stdout().flush().unwrap();

        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        // Check for exit command
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        // Check for clear command
        if input.eq_ignore_ascii_case("clear") {
            // Create a new chat session
            chat_session = if let Some(ref system) = system_message {
                model.create_chat_session_with_system(system, &model_params)?
            } else {
                model.create_chat_session(&model_params)?
            };
            println!("Conversation cleared!");
            continue;
        }

        // Check for system message command
        if input.starts_with("system:") {
            let system_msg = input.trim_start_matches("system:").trim();
            chat_session = model.create_chat_session_with_system(system_msg, &model_params)?;
            println!(
                "Created new chat session with system message: {}",
                system_msg
            );
            continue;
        }

        // Generate response
        match chat_session.prompt(input) {
            Ok(response) => {
                let mut stdio = io::stderr().lock();
                write!(stdio, "Assistant: ")?;
                stdio.flush()?;
                for token in response.take(num_tokens) {
                    write!(stdio, "{}", token)?;
                    stdio.flush()?;
                }
                writeln!(stdio)?;
            }
            Err(e) => {
                eprintln!("Error generating response: {}", e);
            }
        }
    }

    Ok(())
}
