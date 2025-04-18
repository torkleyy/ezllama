use anyhow::Result;
use tracing::debug;

use crate::model::Model;

/// A trait representing a session with a language model
pub trait Session {
    /// Generate a response based on the current session state
    fn generate(&mut self, n_len: i32) -> Result<String>;
}

/// A session for text completion
pub struct TextSession<'a> {
    /// The current prompt
    pub prompt: String,
    /// Reference to the model
    model: &'a mut Model,
}

impl<'a> TextSession<'a> {
    /// Create a new text session
    pub fn new(model: &'a mut Model) -> Self {
        Self {
            prompt: String::new(),
            model,
        }
    }

    pub fn prompt(&mut self, prompt: &str, num_tokens: i32) -> Result<String> {
        self.prompt.push_str(prompt);
        self.generate(num_tokens)
    }

    fn generate(&mut self, n_len: i32) -> Result<String> {
        debug!("Text prompt: {}", self.prompt);
        let response = self.model.generate(&self.prompt, n_len)?;

        // Append the generated text to the prompt for continuity
        self.prompt.push_str(&response);

        Ok(response)
    }
}

// Extension methods for Model to create sessions
impl Model {
    /// Create a new text session
    pub fn create_text_session(&mut self) -> TextSession {
        TextSession::new(self)
    }

    /// Generate text from a prompt using a one-time text session
    /// This is a convenience method that creates a new text session with a prompt
    pub fn text_completion(&mut self, prompt: &str, num_tokens: i32) -> Result<String> {
        self.create_text_session().prompt(prompt, num_tokens)
    }
}
