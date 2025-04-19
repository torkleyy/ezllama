use crate::error::{Error, Result};
use lazy_static::lazy_static;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;

use crate::TextSession;
use crate::chat::{ChatSession, ChatTemplateFormat};

// Initialize the LlamaBackend globally
lazy_static! {
    pub static ref LLAMA_BACKEND: LlamaBackend = {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));
        LlamaBackend::init()
            .map_err(|e| Error::BackendInitError(e.to_string()))
            .expect("Failed to initialize LlamaBackend")
    };
}

/// A wrapper around LlamaModel for efficient repeated use
pub struct Model {
    pub(crate) model: LlamaModel,
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
            let k = CString::new(k.as_bytes())
                .map_err(|e| Error::ParseError(format!("invalid key {}: {}", k, e)))?;
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }

        // Load the model
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, &params.model_path, &model_params)
            .map_err(|e| Error::ModelLoadError(format!("unable to load model: {}", e)))?;

        Ok(Self { model })
    }

    /// Create a new chat session with this model
    pub fn create_chat_session<'a>(&'a self, params: &ModelParams) -> Result<ChatSession<'a>> {
        // Initialize the context
        let mut ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(params.ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));

        if let Some(threads) = params.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = params.threads_batch.or(params.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // Create the context
        let ctx_result = self.model.new_context(&LLAMA_BACKEND, ctx_params);
        let ctx = ctx_result.map_err(|e| {
            Error::ContextCreationError(format!("unable to create the llama_context: {}", e))
        })?;

        // Create a batch with size 512
        let batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);

        // Create a UTF-8 decoder
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(ChatSession {
            messages: Vec::new(),
            template_format: ChatTemplateFormat::ModelDefault,
            session: TextSession::new_with_context(self, ctx, batch, decoder),
        })
    }

    /// Create a new chat session with a custom template
    pub fn create_chat_session_with_template(
        &self,
        template: String,
        params: &ModelParams,
    ) -> Result<ChatSession> {
        let mut session = self.create_chat_session(params)?;
        session.template_format = ChatTemplateFormat::Custom(template);
        Ok(session)
    }

    /// Generate a chat response for a single user message
    /// This is a convenience method that creates a new chat session with a single user message
    pub fn chat_completion(
        &self,
        user_message: &str,
        num_tokens: i32,
        params: &ModelParams,
    ) -> Result<String> {
        let mut session = self.create_chat_session(params)?;
        session.prompt(user_message, num_tokens)
    }

    /// Create a new text session
    pub fn create_text_session(&self, params: &crate::model::ModelParams) -> Result<TextSession> {
        // Initialize the context
        let mut ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(
                params
                    .ctx_size
                    .or(Some(std::num::NonZeroU32::new(2048).unwrap())),
            );

        if let Some(threads) = params.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = params.threads_batch.or(params.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // Create the context
        let ctx_result = self
            .model
            .new_context(&crate::model::LLAMA_BACKEND, ctx_params);
        let ctx_with_lifetime = ctx_result.map_err(|e| {
            Error::ContextCreationError(format!("unable to create the llama_context: {}", e))
        })?;

        // This is safe because LLAMA_BACKEND is 'static and model lives as long as the context
        let ctx = unsafe {
            std::mem::transmute::<
                llama_cpp_2::context::LlamaContext<'_>,
                llama_cpp_2::context::LlamaContext<'static>,
            >(ctx_with_lifetime)
        };

        // Create a batch with size 512
        let batch = llama_cpp_2::llama_batch::LlamaBatch::new(512, 1);

        // Create a UTF-8 decoder
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(TextSession::new_with_context(self, ctx, batch, decoder))
    }

    /// Generate text from a prompt using a one-time text session
    /// This is a convenience method that creates a new text session with a prompt
    pub fn text_completion(
        &mut self,
        prompt: &str,
        num_tokens: i32,
        params: &crate::model::ModelParams,
    ) -> Result<String> {
        let mut session = self.create_text_session(params)?;
        session.prompt(prompt, num_tokens)
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
        .ok_or_else(|| Error::ParseError(format!("invalid KEY=value: no `=` found in `{}`", s)))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| Error::ParseError("must be one of i64, f64, or bool".to_string()))?;

    Ok((key, value))
}
