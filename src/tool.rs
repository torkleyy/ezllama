use crate::error::{Error, Result};
use schemars::JsonSchema;
use schemars::schema::RootSchema;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// A trait for tools that can be called by an agent.
///
/// This trait is implemented for functions that take a single argument `T` where `T`
/// implements `JsonSchema` and `DeserializeOwned`.
pub trait Tool: Send + Sync {
    /// The name of the tool.
    fn name(&self) -> &str;

    /// A description of what the tool does.
    fn description(&self) -> &str;

    /// The JSON schema for the tool's parameters.
    fn parameters_schema(&self) -> RootSchema;

    /// Call the tool with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to pass to the tool as a JSON value.
    ///
    /// # Returns
    ///
    /// The result of calling the tool, or an error if the parameters are invalid
    /// or the tool fails to execute.
    fn call(&self, params: Value) -> Result<Value>;
}

/// A wrapper for a function that implements `Tool`.
///
/// This struct wraps a function that takes a single argument `T` where `T`
/// implements `JsonSchema` and `DeserializeOwned`, and implements the `Tool` trait
/// for it.
#[derive(Debug)]
pub struct FunctionTool<F, T> {
    /// The name of the tool.
    name: String,
    /// A description of what the tool does.
    description: String,
    /// The function to call.
    function: F,
    /// Phantom data for the function's parameter type.
    _marker: std::marker::PhantomData<T>,
}

impl<F, T, R> FunctionTool<F, T>
where
    F: Fn(T) -> R + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: serde::Serialize,
{
    /// Create a new `FunctionTool` from a function.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool.
    /// * `description` - A description of what the tool does.
    /// * `function` - The function to call.
    ///
    /// # Returns
    ///
    /// A new `FunctionTool` that wraps the given function.
    pub fn new(name: impl Into<String>, description: impl Into<String>, function: F) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            function,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, T, R> Tool for FunctionTool<F, T>
where
    F: Fn(T) -> R + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: serde::Serialize,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> RootSchema {
        schemars::schema_for!(T)
    }

    fn call(&self, params: Value) -> Result<Value> {
        // Parse the parameters
        let params: T = serde_json::from_value(params)
            .map_err(|e| Error::ParseError(format!("Failed to parse parameters: {}", e)))?;

        // Call the function
        let result = (self.function)(params);

        // Serialize the result
        let result = serde_json::to_value(result)
            .map_err(|e| Error::Other(format!("Failed to serialize result: {}", e)))?;

        Ok(result)
    }
}

/// A collection of tools that can be called by an agent.
#[derive(Default)]
pub struct ToolCollection {
    tools: Vec<Box<dyn Tool>>,
}

impl std::fmt::Debug for ToolCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCollection")
            .field("tool_count", &self.tools.len())
            .finish()
    }
}

impl ToolCollection {
    /// Create a new empty `ToolCollection`.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool to the collection.
    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Get a tool by name.
    pub fn get_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    /// Get all tools in the collection.
    pub fn tools(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    /// Get the JSON schema for all tools in the collection.
    pub fn schema(&self) -> Value {
        let tools = self
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema(),
                    }
                })
            })
            .collect::<Vec<_>>();

        json!({ "tools": tools })
    }
}

/// Create a new `FunctionTool` from a function.
///
/// This is a convenience function for creating a `FunctionTool` from a function.
///
/// # Arguments
///
/// * `name` - The name of the tool.
/// * `description` - A description of what the tool does.
/// * `function` - The function to call.
///
/// # Returns
///
/// A new `FunctionTool` that wraps the given function.
pub fn function_tool<F, T, R>(
    name: impl Into<String>,
    description: impl Into<String>,
    function: F,
) -> FunctionTool<F, T>
where
    F: Fn(T) -> R + Send + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync,
    R: serde::Serialize,
{
    FunctionTool::new(name, description, function)
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct AddParams {
        a: i32,
        b: i32,
    }

    fn add(params: AddParams) -> i32 {
        params.a + params.b
    }

    #[test]
    fn test_function_tool() {
        let tool = function_tool("add", "Add two numbers", add);

        assert_eq!(tool.name(), "add");
        assert_eq!(tool.description(), "Add two numbers");

        let params = json!({ "a": 1, "b": 2 });
        let result = tool.call(params).unwrap();

        assert_eq!(result, json!(3));
    }

    #[test]
    fn test_tool_collection() {
        let mut tools = ToolCollection::new();
        tools.add_tool(function_tool("add", "Add two numbers", add));

        let tool = tools.get_tool("add").unwrap();
        assert_eq!(tool.name(), "add");

        let params = json!({ "a": 1, "b": 2 });
        let result = tool.call(params).unwrap();

        assert_eq!(result, json!(3));
    }
}
