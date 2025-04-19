//! An opinionated, simple Rust interface for local LLMs, powered by llama-cpp-2.
//!
//! This library provides a simple interface to access LLMs using llama-cpp-2 under the hood.
//! It supports both text completion and chat functionality through a Session-based API.

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod chat;
mod error;
mod model;
mod session;

// Re-export the public API
pub use chat::{ChatMessage, ChatRole, ChatSession, ChatTemplateFormat};
pub use error::{Error, Result};
pub use model::{Model, ModelParams};
pub use session::TextSession;
