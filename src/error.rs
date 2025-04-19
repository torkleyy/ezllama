use std::convert::Infallible;
use std::fmt;
use std::io;
use std::num::ParseIntError;
use std::string::FromUtf8Error;

/// Custom error type for ezllama
#[derive(Debug)]
pub enum Error {
    /// Error initializing the LLM backend
    BackendInitError(String),
    /// Error loading the model
    ModelLoadError(String),
    /// Error creating the context
    ContextCreationError(String),
    /// Error tokenizing input
    TokenizationError(String),
    /// Error decoding tokens
    DecodingError(String),
    /// Error with batch operations
    BatchError(String),
    /// Error parsing parameters
    ParseError(String),
    /// Error with KV cache size
    KVCacheSizeError(String),
    /// Error with chat template
    ChatTemplateError(String),
    /// IO error
    IoError(io::Error),
    /// String conversion error
    StringConversionError(String),
    /// Other error
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BackendInitError(msg) => write!(f, "Backend initialization error: {}", msg),
            Error::ModelLoadError(msg) => write!(f, "Model loading error: {}", msg),
            Error::ContextCreationError(msg) => write!(f, "Context creation error: {}", msg),
            Error::TokenizationError(msg) => write!(f, "Tokenization error: {}", msg),
            Error::DecodingError(msg) => write!(f, "Decoding error: {}", msg),
            Error::BatchError(msg) => write!(f, "Batch operation error: {}", msg),
            Error::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Error::KVCacheSizeError(msg) => write!(f, "KV cache size error: {}", msg),
            Error::ChatTemplateError(msg) => write!(f, "Chat template error: {}", msg),
            Error::IoError(err) => write!(f, "IO error: {}", err),
            Error::StringConversionError(msg) => write!(f, "String conversion error: {}", msg),
            Error::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<FromUtf8Error> for Error {
    fn from(err: FromUtf8Error) -> Self {
        Error::StringConversionError(err.to_string())
    }
}

impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Error::ParseError(err.to_string())
    }
}

impl From<&str> for Error {
    fn from(msg: &str) -> Self {
        Error::Other(msg.to_string())
    }
}

impl From<String> for Error {
    fn from(msg: String) -> Self {
        Error::Other(msg)
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        unreachable!("Infallible error should never happen")
    }
}

/// A specialized Result type for ezllama operations
pub type Result<T> = std::result::Result<T, Error>;
