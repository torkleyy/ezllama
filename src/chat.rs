use crate::{
    error::{Error, Result},
    session::TokenStream,
};
use llama_cpp_2::model::LlamaChatMessage;

use crate::TextSession;

/// Role for a chat message (user or assistant)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// System message
    System,
}

impl ChatRole {
    /// Convert the role to a string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::System => "system",
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
    /// The index of the first message in the conversation
    /// that is not yet in the context
    pub(crate) start_index: usize,
    /// Reference to the model
    pub(crate) session: TextSession<'a>,
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

    /// Add a system message to the chat session
    pub fn add_system_message(&mut self, content: &str) {
        self.messages.push(ChatMessage {
            role: ChatRole::System,
            content: content.to_string(),
        });
    }

    /// Get the last message in the chat session
    pub fn last_message(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    pub fn prompt(&mut self, user_message: &str) -> Result<ChatTokenStream<'_, 'a>> {
        self.add_user_message(user_message);
        self.generate()
    }

    /// Generate a response to the conversation
    pub fn generate(&mut self) -> Result<ChatTokenStream<'_, 'a>> {
        if self.messages.is_empty() {
            return Err(Error::ChatTemplateError(
                "No messages in the chat session".to_string(),
            ));
        }

        let start_index = self.start_index.min(self.messages.len());
        let messages = &self.messages[start_index..];

        log::debug!("Formatting {} messages", messages.len());

        // Format the chat messages into a prompt
        let formatted_prompt = match &self.template_format {
            ChatTemplateFormat::ModelDefault => {
                // Use the model's default template
                let template = self.session.model.model.get_chat_template().map_err(|e| {
                    Error::ChatTemplateError(format!("Failed to get chat template: {}", e))
                })?;
                let chat: Vec<_> = messages
                    .iter()
                    .map(|message| {
                        LlamaChatMessage::new(
                            message.role.as_str().to_string(),
                            message.content.clone(),
                        )
                        .unwrap()
                    })
                    .collect();

                let add_assistant = messages.last().map(|m| m.role) == Some(ChatRole::User);
                self.session
                    .model
                    .model
                    .apply_chat_template(&template, &chat, add_assistant)
                    .map_err(|e| {
                        Error::ChatTemplateError(format!("Failed to apply chat template: {}", e))
                    })?
            }
            ChatTemplateFormat::Default => {
                // Create a simple prompt format
                let mut prompt = String::new();

                for message in messages {
                    prompt.push_str(&format!(
                        "<|{}|>\n{}",
                        message.role.as_str(),
                        message.content
                    ));
                    prompt.push('\n');
                }

                prompt.push_str("<|assistant|>\n");
                prompt
            }
            ChatTemplateFormat::Custom(template) => {
                // Use the custom template
                let mut prompt = template.clone();
                for (i, message) in messages.iter().enumerate() {
                    let role_placeholder = format!("{{{}_role}}", i);
                    let content_placeholder = format!("{{{}_content}}", i);

                    prompt = prompt.replace(&role_placeholder, message.role.as_str());
                    prompt = prompt.replace(&content_placeholder, &message.content);
                }
                prompt
            }
        };

        Ok(ChatTokenStream {
            ts: self.session.prompt(&formatted_prompt)?,
            messages: &mut self.messages,
            start_index: &mut self.start_index,
        })
    }
}

pub struct ChatTokenStream<'s, 'a> {
    pub(crate) ts: TokenStream<'s, 'a>,
    pub(crate) messages: &'s mut Vec<ChatMessage>,
    pub(crate) start_index: &'s mut usize,
}

impl<'s, 'a> ChatTokenStream<'s, 'a> {
    pub fn join(self) -> String {
        self.collect::<String>()
    }
}

impl<'s, 'a> Iterator for ChatTokenStream<'s, 'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.ts.next();
        if self.ts.n_decode() == 1 {
            self.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: String::new(),
            });
            *self.start_index = self.messages.len();
        }
        if let Some(token) = &item {
            self.messages.last_mut().unwrap().content.push_str(token);
        }
        item
    }
}
