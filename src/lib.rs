//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
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
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, ggml_time_us, send_logs_to_tracing};
use tracing::{debug, info, trace};

use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;

use std::sync::Arc;
use std::time::Duration;

// Initialize the LlamaBackend globally
lazy_static! {
    static ref LLAMA_BACKEND: LlamaBackend = {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        LlamaBackend::init().expect("Failed to initialize LlamaBackend")
    };
}

/// A wrapper around LlamaModel and its context for efficient repeated use
pub struct Model {
    model: Arc<LlamaModel>,
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
        let model = Arc::new(
            LlamaModel::load_from_file(&LLAMA_BACKEND, &params.model_path, &model_params)
                .map_err(|e| anyhow!("unable to load model: {}", e))?,
        );

        // Initialize the context
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(params.ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));

        if let Some(threads) = params.threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = params.threads_batch.or(params.threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // This is a workaround for the lifetime issue
        // We're using 'static here because the model is wrapped in an Arc
        // and the backend is a global static
        let ctx = unsafe {
            std::mem::transmute::<
                llama_cpp_2::context::LlamaContext<'_>,
                llama_cpp_2::context::LlamaContext<'static>,
            >(
                model
                    .new_context(&LLAMA_BACKEND, ctx_params)
                    .with_context(|| "unable to create the llama_context")?,
            )
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
    pub fn generate(&mut self, prompt: &str, n_len: i32, seed: Option<u32>) -> Result<String> {
        // Tokenize the prompt
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
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

        // Log the prompt token-by-token
        let mut prompt_text = String::new();
        for token in &tokens_list {
            prompt_text.push_str(&self.model.token_to_str(*token, Special::Tokenize)?);
        }
        debug!("Prompt: {}", prompt_text);

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

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(seed.unwrap_or(1234)),
            LlamaSampler::greedy(),
        ]);

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
            let mut token_text = String::with_capacity(93);
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut token_text, false);
            trace!("Generated token: {}", token_text);
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

#[derive(Debug, Clone)]
pub struct LlamaParams {
    /// The path to the model
    pub model_path: PathBuf,
    /// The prompt
    pub prompt: String,
    /// set the length of the prompt + output in tokens
    pub n_len: i32,
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

impl Default for LlamaParams {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            prompt: "Hello!".to_owned(),
            n_len: 32,
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
