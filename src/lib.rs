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
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct LlamaParams {
    /// The path to the model
    pub model_path: PathBuf,
    /// The prompt
    pub prompt: Option<String>,
    /// Read the prompt from a file
    pub file: Option<String>,
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
            prompt: None,
            file: None,
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

#[allow(clippy::too_many_lines)]
pub fn run_llama(params: LlamaParams) -> Result<()> {
    let LlamaParams {
        n_len,
        model_path,
        prompt,
        file,
        key_value_overrides,
        seed,
        threads,
        threads_batch,
        ctx_size,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
    } = params;

    send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        let params = if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        };
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        let params = LlamaModelParams::default();
        params
    };

    let prompt = if let Some(str) = prompt {
        if file.is_some() {
            bail!("either prompt or file must be specified, but not both")
        }
        str
    } else if let Some(file) = file {
        std::fs::read_to_string(&file).map_err(|e| anyhow!("unable to read {}: {}", file, e))?
    } else {
        "Hello my name is".to_string()
    };

    let mut model_params = pin!(model_params);

    for (k, v) in &key_value_overrides {
        let k = CString::new(k.as_bytes()).map_err(|e| anyhow!("invalid key {}: {}", k, e))?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    // Use the model path directly

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .map_err(|e| anyhow!("unable to load model: {}", e))?;

    // initialize the context
    let mut ctx_params =
        LlamaContextParams::default().with_n_ctx(ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));

    if let Some(threads) = threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = threads_batch.or(threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // tokenize the prompt

    let tokens_list = model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    info!(
        "n_len = {}, n_ctx = {}, k_kv_req = {}",
        n_len, n_cxt, n_kv_req
    );

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_cxt {
        bail!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
        )
    }

    if tokens_list.len() >= usize::try_from(n_len)? {
        bail!("the prompt is too long, it has more tokens than n_len")
    }

    // log the prompt token-by-token
    let mut prompt_text = String::new();
    for token in &tokens_list {
        prompt_text.push_str(&model.token_to_str(*token, Special::Tokenize)?);
    }
    debug!("Prompt: {}", prompt_text);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(512, 1);

    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // main loop

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let t_main_start = ggml_time_us();

    // The `Decoder`
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed.unwrap_or(1234)),
        LlamaSampler::greedy(),
    ]);

    while n_cur <= n_len {
        // sample the next token
        {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            sampler.accept(token);

            // is it an end of stream?
            if model.is_eog_token(token) {
                debug!("End of generation token detected");
                break;
            }

            let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(93);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            trace!("Generated token: {}", output_string);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
        }

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;

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

    debug!("Timings: {}", ctx.timings());

    Ok(())
}
