use anyhow::Result;
use llama_cpp_2::model::LlamaChatMessage;
use tracing::debug;

use crate::model::Model;

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
    /// Use a simple default template
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
    pub(crate) model: &'a mut Model,
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

    pub fn prompt(&mut self, user_message: &str, num_tokens: i32) -> Result<String> {
        self.add_user_message(user_message);
        self.generate(num_tokens)
    }

    /// Generate a response to the conversation
    pub fn generate(&mut self, num_tokens: i32) -> Result<String> {
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
        let response = self.model.generate(&formatted_prompt, num_tokens)?;

        // Add the response to the chat session
        self.add_assistant_message(&response);

        Ok(response)
    }
}
