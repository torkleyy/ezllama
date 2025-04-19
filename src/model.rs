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

use crate::TextSession;
use crate::chat::{ChatSession, ChatTemplateFormat};

// Initialize the LlamaBackend globally
lazy_static! {
    pub static ref LLAMA_BACKEND: LlamaBackend = {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
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
            let v = param_override(v).map_err(|e| {
                Error::ParseError(format!("invalid value {} for key {k:?}: {}", v, e))
            })?;
            model_params.as_mut().append_kv_override(k.as_c_str(), v);
        }

        // Load the model
        let model = LlamaModel::load_from_file(&LLAMA_BACKEND, &params.model_path, &model_params)
            .map_err(|e| Error::ModelLoadError(format!("unable to load model: {}", e)))?;

        Ok(Self { model })
    }

    /// Create a new chat session with this model
    pub fn create_chat_session<'a>(
        &'a self,
        context_params: &ContextParams,
    ) -> Result<ChatSession<'a>> {
        Ok(ChatSession {
            messages: Vec::new(),
            template_format: ChatTemplateFormat::ModelDefault,
            session: self.create_text_session(context_params)?,
            start_index: 0,
        })
    }

    /// Create a new chat session with a custom template
    pub fn create_chat_session_with_template(
        &self,
        template: String,
        context_params: &ContextParams,
    ) -> Result<ChatSession> {
        let mut session = self.create_chat_session(context_params)?;
        session.template_format = ChatTemplateFormat::Custom(template);
        Ok(session)
    }

    /// Create a new chat session with a system message
    pub fn create_chat_session_with_system(
        &self,
        system_message: &str,
        context_params: &ContextParams,
    ) -> Result<ChatSession> {
        let mut session = self.create_chat_session(context_params)?;
        session.add_system_message(system_message);
        Ok(session)
    }

    /// Generate a chat response for a single user message
    /// This is a convenience method that creates a new chat session with a single user message
    pub fn chat_completion(
        &self,
        user_message: &str,
        context_params: &ContextParams,
    ) -> Result<String> {
        self.create_chat_session(context_params)?
            .prompt(user_message)
            .map(|ts| ts.join())
    }

    /// Generate a chat response for a single user message with a system message
    /// This is a convenience method that creates a new chat session with a system message and a user message
    pub fn chat_completion_with_system(
        &self,
        system_message: &str,
        user_message: &str,
        context_params: &ContextParams,
    ) -> Result<String> {
        self.create_chat_session_with_system(system_message, context_params)?
            .prompt(user_message)
            .map(|ts| ts.join())
    }

    /// Create a new text session
    pub fn create_text_session(&self, context_params: &ContextParams) -> Result<TextSession> {
        // Initialize the context
        let mut ctx_params = llama_cpp_2::context::params::LlamaContextParams::default()
            .with_n_ctx(
                context_params
                    .ctx_size
                    .map(|n| n as u32)
                    .and_then(NonZeroU32::new),
            );

        if let Some(threads) = context_params.threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        if let Some(threads_batch) = context_params.threads_batch.or(context_params.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch as i32);
        }

        // Create the context
        let ctx_result = self
            .model
            .new_context(&crate::model::LLAMA_BACKEND, ctx_params);
        let ctx = ctx_result.map_err(|e| {
            Error::ContextCreationError(format!("unable to create the llama_context: {}", e))
        })?;

        Ok(TextSession::new_with_context(self, ctx))
    }

    /// Generate text from a prompt using a one-time text session
    /// This is a convenience method that creates a new text session with a prompt
    pub fn text_completion(
        &mut self,
        prompt: &str,
        context_params: &ContextParams,
    ) -> Result<String> {
        let mut session = self.create_text_session(context_params)?;
        session.prompt(prompt).map(|ts| ts.join())
    }
}

#[derive(Debug, Default, Clone)]
pub struct ModelParams {
    /// The path to the model
    pub model_path: PathBuf,
    /// override some parameters of the model
    pub key_value_overrides: Vec<(String, String)>,
    /// Disable offloading layers to the gpu
    pub disable_gpu: bool,
    #[doc(hidden)]
    pub _non_exhaustive: (),
}

/// Parameters for creating a context
#[derive(Debug, Default, Clone)]
pub struct ContextParams {
    /// RNG seed (default: 1234)
    pub seed: Option<u32>,
    /// number of threads to use during generation (default: use all available threads)
    pub threads: Option<usize>,
    /// number of threads to use during batch and prompt processing (default: use all available threads)
    pub threads_batch: Option<usize>,
    /// size of the prompt context (default: loaded from the model)
    pub ctx_size: Option<usize>,
    #[doc(hidden)]
    pub _non_exhaustive: (),
}

// Helper function to parse key-value pairs for model parameters
fn param_override(s: &str) -> Result<ParamOverrideValue> {
    use std::str::FromStr;

    i64::from_str(s)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(s).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(s).map(ParamOverrideValue::Bool))
        .map_err(|_| Error::ParseError("must be one of i64, f64, or bool".to_string()))
}
