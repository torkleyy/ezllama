use crate::error::{Error, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use std::time::Instant;
use tracing::{debug, info, trace};

use crate::model::Model;

/// A session for text completion
pub struct TextSession<'a> {
    /// The current prompt
    pub prompt: String,
    /// Reference to the model
    pub(crate) model: &'a Model,
    /// The llama context for this session
    pub(crate) ctx: LlamaContext<'a>,
    /// UTF-8 decoder
    pub(crate) decoder: encoding_rs::Decoder,
    /// The llama sampler for this session
    pub(crate) sampler: LlamaSampler,
}

impl<'a> TextSession<'a> {
    /// Create a new text session with context
    pub(crate) fn new_with_context(model: &'a Model, ctx: LlamaContext<'a>) -> Self {
        Self {
            prompt: String::new(),
            model,
            ctx,
            decoder: encoding_rs::UTF_8.new_decoder(),
            sampler: llama_cpp_2::sampling::LlamaSampler::chain_simple([
                llama_cpp_2::sampling::LlamaSampler::dist(1234),
                llama_cpp_2::sampling::LlamaSampler::greedy(),
            ]),
        }
    }

    /// Add to the prompt and generate tokens
    pub fn prompt(&mut self, prompt: &str, num_tokens: i32) -> Result<String> {
        self.prompt.clear();
        self.prompt.push_str(prompt);
        self.generate(num_tokens)
    }

    /// Generate more tokens without adding to the prompt
    pub fn generate(&mut self, num_tokens: i32) -> Result<String> {
        debug!("Text prompt: {}", self.prompt);

        // Generate the response
        let add_bos = self.ctx.get_kv_cache_used_cells() == 0;
        // Tokenize the prompt
        let tokens_list = self
            .model
            .model
            .str_to_token(
                &self.prompt,
                if add_bos {
                    AddBos::Always
                } else {
                    AddBos::Never
                },
            )
            .map_err(|e| Error::TokenizationError(format!("failed to tokenize prompt: {}", e)))?;

        let n_past = self.ctx.get_kv_cache_used_cells() as i32;
        let n_prompt = tokens_list.len() as i32;
        let n_ctx = self.ctx.n_ctx() as i32;
        let n_kv_req = n_past + n_prompt + num_tokens;

        info!(
            "num_tokens_to_generate = {}, n_prompt = {}, n_ctx = {}, k_kv_req = {}, n_past = {}",
            num_tokens, n_prompt, n_ctx, n_kv_req, n_past
        );

        // Make sure the KV cache is big enough
        if n_kv_req > n_ctx {
            log::info!("KV cache is full, shifting context");

            let keep = n_ctx / 4;
            let left = n_past - keep;
            let discard = left / 2;

            if !self
                .ctx
                .clear_kv_cache_seq(
                    Some(0),
                    Some(keep as u32),
                    Some(keep as u32 + discard as u32),
                )
                .unwrap()
            {
                return Err(Error::KVCacheSizeError(
                    "failed to clear KV cache".to_string(),
                ));
            }

            log::info!("cleared from {} to {}", keep, keep + discard);

            self.ctx
                .kv_cache_seq_add(
                    0,
                    Some(keep as u32 + discard as u32),
                    Some(n_past as u32),
                    -discard as i32,
                )
                .unwrap();
        }

        // Add prompt tokens to batch
        let mut batch = LlamaBatch::get_one(&tokens_list).map_err(|e| {
            Error::BatchError(format!("Failed to create batch from prompt tokens: {}", e))
        })?;

        let mut n_decode = 0;
        let mut output = String::new();

        let sampling_start = Instant::now();

        let mut token_text = String::with_capacity(93);

        while n_decode <= num_tokens {
            self.ctx
                .decode(&mut batch)
                .map_err(|e| Error::DecodingError(format!("llama_decode() failed: {}", e)))?;

            // Sample the next token
            let token = self.sampler.sample(&self.ctx, -1);
            self.sampler.accept(token);

            // Check for end of generation token
            if self.model.model.is_eog_token(token) {
                trace!("End of generation token detected: {token}");
                break;
            }

            // Convert token to text and add to output
            let output_bytes = self
                .model
                .model
                .token_to_bytes(token, Special::Tokenize)
                .map_err(|e| {
                    Error::TokenizationError(format!("Failed to convert token to bytes: {}", e))
                })?;
            token_text.clear();
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut token_text, false);
            trace!(name: "token-gen", "Generated token: {}", token_text);
            output.push_str(&token_text);

            batch = LlamaBatch::get_one(&[token]).map_err(|e| {
                Error::BatchError(format!(
                    "Failed to create batch from generated token: {}",
                    e
                ))
            })?;

            n_decode += 1;
        }

        let duration = sampling_start.elapsed();

        info!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        debug!("Timings: {}", self.ctx.timings());

        // Append the generated text to the prompt for continuity
        self.prompt.push_str(&output);

        Ok(output)
    }
}
