//! This is a translation of simple.cpp and simple-chat.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result, anyhow, bail};
// use clap::Parser;
// use hf_hub::api::sync::ApiBuilder;
use lazy_static::lazy_static;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, ggml_time_us, send_logs_to_tracing};
use tracing::{debug, info, trace};

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;

use std::time::Duration;

/// Role for a chat message (user or assistant)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    /// User message
    User,
    /// Assistant message
    Assistant,
}

impl ChatRole {
    /// Convert the role to a string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// A single message in a chat conversation
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of the message sender (user or assistant)
    pub role: ChatRole,
    /// The content of the message
    pub content: String,
}

/// Chat template format for different models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateFormat {
    /// Use the template embedded in the model
    ModelDefault,
    Default,
    /// Custom template
    Custom(String),
}

/// A chat session containing a sequence of messages
pub struct ChatSession<'a> {
    /// The messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// The chat template format to use
    pub template_format: ChatTemplateFormat,
    /// Reference to the model
    model: &'a mut Model,
}

impl<'a> ChatSession<'a> {
    /// Add a user message to the chat session
    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: content.to_string(),
        });
    }

    /// Add an assistant message to the chat session
    pub fn add_assistant_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: content.to_string(),
        });
    }

    /// Get the last message in the chat session
    pub fn last_message(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    /// Generate a response to the conversation
    pub fn prompt(&mut self, n_len: i32) -> Result<String> {
        // Format the chat messages into a prompt
        let formatted_prompt = match &self.template_format {
            ChatTemplateFormat::ModelDefault => {
                // Use the model's default template
                let template = self.model.model.get_chat_template()?;
                let chat: Vec<_> = self
                    .messages
                    .iter()
                    .map(|message| {
                        LlamaChatMessage::new(
                            message.role.as_str().to_string(),
                            message.content.clone(),
                        )
                        .unwrap()
                    })
                    .collect();
                self.model
                    .model
                    .apply_chat_template(&template, &chat, true)?
            }
            ChatTemplateFormat::Default => {
                // Create a simple prompt format
                let mut prompt = String::new();
                for message in &self.messages {
                    prompt.push_str(&format!(
                        "<|{}|>\n{}",
                        message.role.as_str(),
                        message.content
                    ));
                    prompt.push_str("\n");
                }
                prompt.push_str("<|assistant|>\n");
                prompt
            }
            ChatTemplateFormat::Custom(template) => {
                // Use the custom template
                let mut prompt = template.clone();
                for (i, message) in self.messages.iter().enumerate() {
                    let role_placeholder = format!("{{{}_role}}", i);
                    let content_placeholder = format!("{{{}_content}}", i);

                    prompt = prompt.replace(&role_placeholder, message.role.as_str());
                    prompt = prompt.replace(&content_placeholder, &message.content);
                }
                prompt
            }
        };

        debug!("Chat prompt: {}", formatted_prompt);

        // Generate the response
        let response = self.model.generate(&formatted_prompt, n_len)?;

        // Add the response to the chat session
        self.add_assistant_message(&response);

        Ok(response)
    }
}

// Initialize the LlamaBackend globally
lazy_static! {
    static ref LLAMA_BACKEND: LlamaBackend = {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));
        LlamaBackend::init().expect("Failed to initialize LlamaBackend")
    };
}

/// A wrapper around LlamaModel and its context for efficient repeated use
pub struct Model {
    model: LlamaModel,
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    batch: LlamaBatch,
    decoder: encoding_rs::Decoder,
}

impl Model {
    /// Create a new LlamaModelWrapper with the given parameters
    pub fn new(params: &ModelParams) -> Result<Self> {
        // Create model parameters
        let model_params = {
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            let params = if !params.disable_gpu {
                LlamaModelParams::default().with_n_gpu_layers(1000)
            } else {
                LlamaModelParams::default()
            };
            #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
            let params = LlamaModelParams::default();
            params
        };

        let mut model_params = pin!(model_params);

        // Apply key-value overrides
        for (k, v) in &params.key_value_overrides {
            let k = CString::new(k.as_bytes()).map_err(|e| anyhow!("invalid key {}: {}", k, e))?;
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }

        // Load the model
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, &params.model_path, &model_params)
            .map_err(|e| anyhow!("unable to load model: {}", e))?;

        // Initialize the context
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(params.ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));

        if let Some(threads) = params.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = params.threads_batch.or(params.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // Create the context
        // Note: We're using 'static lifetime because LLAMA_BACKEND is a static reference
        // This is safe because the context will not outlive the model or the backend
        let ctx_result = model.new_context(&LLAMA_BACKEND, ctx_params);
        let ctx_with_lifetime = ctx_result.with_context(|| "unable to create the llama_context")?;

        // This is safe because LLAMA_BACKEND is 'static and model lives as long as the context
        let ctx = unsafe {
            std::mem::transmute::<
                llama_cpp_2::context::LlamaContext<'_>,
                llama_cpp_2::context::LlamaContext<'static>,
            >(ctx_with_lifetime)
        };

        // Create a batch with size 512
        let batch = LlamaBatch::new(512, 1);

        // Create a UTF-8 decoder
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(Self {
            model,
            ctx,
            batch,
            decoder,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, n_len: i32) -> Result<String> {
        let add_bos = self.ctx.get_kv_cache_used_cells() == 0;
        // Tokenize the prompt
        let tokens_list = self
            .model
            .str_to_token(
                prompt,
                if add_bos {
                    AddBos::Always
                } else {
                    AddBos::Never
                },
            )
            .with_context(|| format!("failed to tokenize {prompt}"))?;

        let n_cxt = self.ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

        info!(
            "n_len = {}, n_ctx = {}, k_kv_req = {}",
            n_len, n_cxt, n_kv_req
        );

        // Make sure the KV cache is big enough
        if n_kv_req > n_cxt {
            bail!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough\n\
                either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(n_len)? {
            bail!("the prompt is too long, it has more tokens than n_len")
        }

        // Clear the batch and add tokens
        self.batch.clear();

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            self.batch.add(token, i, &[0], is_last)?;
        }

        self.ctx
            .decode(&mut self.batch)
            .with_context(|| "llama_decode() failed")?;

        // Main generation loop
        let mut n_cur = self.batch.n_tokens();
        let mut n_decode = 0;
        let mut output = String::new();

        let t_main_start = ggml_time_us();

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

        let mut token_text = String::with_capacity(93);

        while n_cur <= n_len {
            // Sample the next token
            let token = sampler.sample(&self.ctx, self.batch.n_tokens() - 1);

            sampler.accept(token);

            // Check for end of generation token
            if self.model.is_eog_token(token) {
                debug!("End of generation token detected");
                break;
            }

            // Convert token to text and add to output
            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            token_text.clear();
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut token_text, false);
            trace!(name: "token-gen", "Generated token: {}", token_text);
            output.push_str(&token_text);

            // Prepare for next token
            self.batch.clear();
            self.batch.add(token, n_cur, &[0], true)?;

            n_cur += 1;

            self.ctx
                .decode(&mut self.batch)
                .with_context(|| "failed to eval")?;

            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        info!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        debug!("Timings: {}", self.ctx.timings());

        Ok(output)
    }

    /// Create a new chat session with this model
    pub fn create_chat_session(&mut self) -> ChatSession {
        ChatSession {
            messages: Vec::new(),
            template_format: ChatTemplateFormat::ModelDefault,
            model: self,
        }
    }

    /// Create a new chat session with a custom template
    pub fn create_chat_session_with_template(&mut self, template: String) -> ChatSession {
        ChatSession {
            messages: Vec::new(),
            template_format: ChatTemplateFormat::Custom(template),
            model: self,
        }
    }

    /// Generate a chat response for a single user message
    /// This is a convenience method that creates a new chat session with a single user message
    pub fn chat_completion(&mut self, user_message: &str, n_len: i32) -> Result<String> {
        let mut session = self.create_chat_session();
        session.add_user_message(user_message);
        session.prompt(n_len)
    }
}

#[derive(Debug, Clone)]
pub struct ModelParams {
    /// The path to the model
    pub model_path: PathBuf,
    /// override some parameters of the model
    pub key_value_overrides: Vec<(String, ParamOverrideValue)>,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    pub disable_gpu: bool,
    /// RNG seed (default: 1234)
    pub seed: Option<u32>,
    /// number of threads to use during generation (default: use all available threads)
    pub threads: Option<i32>,
    /// number of threads to use during batch and prompt processing (default: use all available threads)
    pub threads_batch: Option<i32>,
    /// size of the prompt context (default: loaded from the model)
    pub ctx_size: Option<NonZeroU32>,
    // do not put verbose flag bc the binary should init logger
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            key_value_overrides: Vec::new(),
            #[cfg(any(feature = "cuda", feature = "vulkan"))]
            disable_gpu: false,
            seed: None,
            threads: None,
            threads_batch: None,
            ctx_size: None,
        }
    }
}

// Helper function to parse key-value pairs for model parameters
pub fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}
