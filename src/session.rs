use crate::error::{Error, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, Special};
use std::time::Duration;
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
    /// The batch for token processing
    pub(crate) batch: LlamaBatch,
    /// UTF-8 decoder
    pub(crate) decoder: encoding_rs::Decoder,
}

impl<'a> TextSession<'a> {
    /// Create a new text session with context
    pub fn new_with_context(
        model: &'a Model,
        ctx: LlamaContext<'a>,
        batch: LlamaBatch,
        decoder: encoding_rs::Decoder,
    ) -> Self {
        Self {
            prompt: String::new(),
            model,
            ctx,
            batch,
            decoder,
        }
    }

    /// Add to the prompt and generate tokens
    pub fn prompt(&mut self, prompt: &str, num_tokens: i32) -> Result<String> {
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

        let n_prompt_tokens = tokens_list.len() as i32;
        let n_ctx = self.ctx.n_ctx() as i32;
        let n_kv_req = n_prompt_tokens + num_tokens;

        info!(
            "num_tokens_to_generate = {}, n_ctx = {}, k_kv_req = {}",
            num_tokens, n_ctx, n_kv_req
        );

        // Make sure the KV cache is big enough
        if n_kv_req > n_ctx {
            return Err(Error::KVCacheSizeError(
                "n_kv_req > n_ctx, the required kv cache size is not big enough\n\
                either reduce n_len or increase n_ctx"
                    .to_string(),
            ));
        }

        // Clear the batch and add tokens
        self.batch.clear();

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            if i > 0 && i % 512 == 0 {
                self.ctx
                    .decode(&mut self.batch)
                    .map_err(|e| Error::DecodingError(format!("llama_decode() failed: {}", e)))?;
                self.batch.clear();
            }

            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            self.batch
                .add(token, i, &[0], is_last)
                .map_err(|e| Error::BatchError(format!("Failed to add token to batch: {}", e)))?;
        }

        self.ctx
            .decode(&mut self.batch)
            .map_err(|e| Error::DecodingError(format!("llama_decode() failed: {}", e)))?;

        // Main generation loop
        let mut n_cur = self.batch.n_tokens();
        let mut n_decode = 0;
        let mut output = String::new();

        let t_main_start = ggml_time_us();

        let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain_simple([
            llama_cpp_2::sampling::LlamaSampler::dist(1234),
            llama_cpp_2::sampling::LlamaSampler::greedy(),
        ]);

        let mut token_text = String::with_capacity(93);

        let n_len = n_prompt_tokens + num_tokens;

        while n_cur <= n_len {
            // Sample the next token
            let token = sampler.sample(&self.ctx, self.batch.n_tokens() - 1);

            sampler.accept(token);

            // Check for end of generation token
            if self.model.model.is_eog_token(token) {
                debug!("End of generation token detected");
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

            // Prepare for next token
            self.batch.clear();
            self.batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| Error::BatchError(format!("Failed to add token to batch: {}", e)))?;

            n_cur += 1;

            self.ctx
                .decode(&mut self.batch)
                .map_err(|e| Error::DecodingError(format!("failed to eval: {}", e)))?;

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

        // Append the generated text to the prompt for continuity
        self.prompt.push_str(&output);

        Ok(output)
    }
}
