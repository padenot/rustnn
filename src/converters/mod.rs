use std::collections::HashMap;
use std::path::Path;

use crate::error::GraphError;
use crate::graph::GraphInfo;

mod coreml_mlprogram;
#[cfg(feature = "litert-runtime")]
pub mod litert;
#[cfg(feature = "onnx-converter")]
pub mod onnx;
#[cfg(any(
    feature = "onnx-converter",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime"
))]
mod pool2d_shared;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
mod trtx;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
mod trtx_gru;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
mod trtx_lstm;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
mod trtx_rnn;
mod weight_file_builder;

pub use coreml_mlprogram::CoremlMlProgramConverter;
#[cfg(feature = "litert-runtime")]
pub use litert::LiteRtConverter;
#[cfg(feature = "onnx-converter")]
pub use onnx::OnnxConverter;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub use trtx::TrtxConverter;
pub(crate) use weight_file_builder::WeightFileBuilder;

/// Filename (relative to the `.onnx` file directory) for ONNX external initializer data produced by the ONNX converter.
///
/// When [`ConvertedGraph::weights_data`] is `Some`, write those bytes next to the model using this exact name so
/// `external_data` `location` entries resolve correctly for ONNX Runtime and other tools.
pub const ONNX_EXTERNAL_WEIGHTS_FILENAME: &str = "rustnn_external_weights.data";

/// Get operand name for an operand ID, or generate a default name
///
/// This is a shared helper used by all converters to ensure consistent
/// operand naming across different backend formats.
pub(crate) fn operand_name(graph: &GraphInfo, id: u32) -> String {
    graph
        .operand(id)
        .and_then(|op| op.name.clone())
        .unwrap_or_else(|| format!("operand_{}", id))
}

#[derive(Debug, Clone)]
pub struct ConvertedGraph {
    pub format: &'static str,
    pub content_type: &'static str,
    pub data: Vec<u8>,
    /// Optional weight file data for formats that require external weights (e.g., CoreML Float16)
    pub weights_data: Option<Vec<u8>>,
}

/// Write a converted Core ML graph as a deterministic `.mlpackage` source artifact.
///
/// This does not invoke Core ML compilation. Build tooling can pass the resulting package to
/// `coremlcompiler compile` ahead of time and ship the generated `.mlmodelc` directory.
pub fn save_coreml_package(
    converted: &ConvertedGraph,
    package_path: &Path,
) -> Result<(), GraphError> {
    if converted.format != "coreml" {
        return Err(GraphError::ConversionFailed {
            format: converted.format.to_owned(),
            reason: "only Core ML conversions can be written as .mlpackage".to_owned(),
        });
    }
    if package_path
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("mlpackage")
    {
        return Err(GraphError::ConversionFailed {
            format: "coreml".to_owned(),
            reason: format!(
                "Core ML package path must end in .mlpackage: {}",
                package_path.display()
            ),
        });
    }
    if package_path.exists() {
        let mut entries = std::fs::read_dir(package_path)
            .map_err(|error| GraphError::export(package_path, error))?;
        if entries
            .next()
            .transpose()
            .map_err(|error| GraphError::export(package_path, error))?
            .is_some()
        {
            return Err(GraphError::ConversionFailed {
                format: "coreml".to_owned(),
                reason: format!(
                    "refusing to overwrite non-empty Core ML package {}",
                    package_path.display()
                ),
            });
        }
    }

    let data_dir = package_path.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data_dir).map_err(|error| GraphError::export(&data_dir, error))?;
    let model_path = data_dir.join("model.mlmodel");
    std::fs::write(&model_path, &converted.data)
        .map_err(|error| GraphError::export(&model_path, error))?;

    let model_id = "00000000-0000-0000-0000-0000000000AA";
    let weights_id = "00000000-0000-0000-0000-0000000000BB";
    let weights_entry = if let Some(weights) = &converted.weights_data {
        let weights_dir = data_dir.join("weights");
        std::fs::create_dir_all(&weights_dir)
            .map_err(|error| GraphError::export(&weights_dir, error))?;
        let weights_path = weights_dir.join("weights.bin");
        std::fs::write(&weights_path, weights)
            .map_err(|error| GraphError::export(&weights_path, error))?;
        format!(
            r#",
    "{weights_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Weights",
      "name": "weights",
      "path": "com.apple.CoreML/weights"
    }}"#
        )
    } else {
        String::new()
    };
    let manifest = format!(
        r#"{{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {{
    "{model_id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel"
    }}{weights_entry}
  }},
  "rootModelIdentifier": "{model_id}"
}}
"#
    );
    let manifest_path = package_path.join("Manifest.json");
    std::fs::write(&manifest_path, manifest)
        .map_err(|error| GraphError::export(&manifest_path, error))
}

pub trait GraphConverter {
    fn format(&self) -> &'static str;
    fn convert(&self, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError>;
}

pub struct ConverterRegistry {
    converters: HashMap<&'static str, Box<dyn GraphConverter + Send + Sync>>,
}

impl ConverterRegistry {
    pub fn with_defaults() -> Self {
        let mut registry = Self {
            converters: HashMap::new(),
        };
        #[cfg(feature = "onnx-converter")]
        registry.register(Box::new(OnnxConverter));
        registry.register(Box::new(CoremlMlProgramConverter));
        #[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
        registry.register(Box::new(TrtxConverter::new()));
        #[cfg(feature = "litert-runtime")]
        registry.register(Box::new(LiteRtConverter::new()));
        registry
    }

    pub fn register(&mut self, converter: Box<dyn GraphConverter + Send + Sync>) {
        self.converters.insert(converter.format(), converter);
    }

    pub fn available_formats(&self) -> Vec<&'static str> {
        let mut keys: Vec<_> = self.converters.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    pub fn convert(&self, format: &str, graph: &GraphInfo) -> Result<ConvertedGraph, GraphError> {
        let key = format.to_ascii_lowercase();
        let Some(converter) = self.converters.get(key.as_str()) else {
            return Err(GraphError::UnknownConverter {
                requested: format.to_string(),
                available: self.available_formats(),
            });
        };
        converter.convert(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConverterRegistry, GraphConverter};
    use crate::error::GraphError;
    use crate::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind};

    fn s(shape: &[u32]) -> Vec<crate::graph::Dimension> {
        crate::graph::to_dimension_vector(shape)
    }

    struct DummyConverter;

    impl GraphConverter for DummyConverter {
        fn format(&self) -> &'static str {
            "dummy"
        }

        fn convert(&self, _graph: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
            Ok(super::ConvertedGraph {
                format: "dummy",
                content_type: "application/octet-stream",
                data: vec![1, 2, 3],
                weights_data: None,
            })
        }
    }

    #[test]
    fn converts_via_registry() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverter));

        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: s(&[]),
                    pending_permutation: vec![],
                },
                name: Some("x".to_string()),
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let converted = registry.convert("dummy", &graph).unwrap();
        assert_eq!(converted.format, "dummy");
        assert_eq!(converted.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_with_defaults_registers_converters() {
        let registry = ConverterRegistry::with_defaults();
        let formats = registry.available_formats();

        assert!(formats.contains(&"coreml"));
        #[cfg(feature = "onnx-converter")]
        assert!(formats.contains(&"onnx"));
        #[cfg(not(feature = "onnx-converter"))]
        assert!(!formats.contains(&"onnx"));
    }

    #[test]
    fn test_available_formats_sorted() {
        let registry = ConverterRegistry::with_defaults();
        let formats = registry.available_formats();

        // Verify formats are sorted
        let mut sorted_formats = formats.clone();
        sorted_formats.sort_unstable();
        assert_eq!(formats, sorted_formats);
    }

    #[test]
    fn test_convert_unknown_format_returns_error() {
        let registry = ConverterRegistry::with_defaults();
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let result = registry.convert("unknown_format", &graph);
        assert!(result.is_err());

        match result.unwrap_err() {
            GraphError::UnknownConverter {
                requested,
                available,
            } => {
                assert_eq!(requested, "unknown_format");
                assert!(!available.is_empty());
            }
            _ => panic!("Expected UnknownConverter error"),
        }
    }

    #[test]
    fn test_convert_case_insensitive() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverter));

        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        // Should work with different cases
        registry.convert("dummy", &graph).unwrap();
        registry.convert("DUMMY", &graph).unwrap();
        registry.convert("Dummy", &graph).unwrap();
    }

    #[test]
    fn test_operand_name_with_named_operand() {
        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: s(&[1, 2]),
                    pending_permutation: vec![],
                },
                name: Some("input_tensor".to_string()),
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 0);
        assert_eq!(name, "input_tensor");
    }

    #[test]
    fn test_operand_name_with_unnamed_operand() {
        let graph = GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Input,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: s(&[1, 2]),
                    pending_permutation: vec![],
                },
                name: None,
            }],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 0);
        assert_eq!(name, "operand_0");
    }

    #[test]
    fn test_operand_name_invalid_id() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let name = super::operand_name(&graph, 999);
        assert_eq!(name, "operand_999");
    }

    struct DummyConverterWithWeights;

    impl GraphConverter for DummyConverterWithWeights {
        fn format(&self) -> &'static str {
            "dummy_with_weights"
        }

        fn convert(&self, _graph: &GraphInfo) -> Result<super::ConvertedGraph, GraphError> {
            Ok(super::ConvertedGraph {
                format: "dummy_with_weights",
                content_type: "application/octet-stream",
                data: vec![1, 2, 3],
                weights_data: Some(vec![4, 5, 6, 7, 8]),
            })
        }
    }

    #[test]
    fn test_converted_graph_with_weights() {
        let mut registry = ConverterRegistry {
            converters: Default::default(),
        };
        registry.register(Box::new(DummyConverterWithWeights));

        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: Default::default(),
            id_to_constant_tensor_operand_map: Default::default(),
            quantized: false,
        };

        let converted = registry.convert("dummy_with_weights", &graph).unwrap();
        assert_eq!(converted.format, "dummy_with_weights");
        assert_eq!(converted.data, vec![1, 2, 3]);
        assert_eq!(converted.weights_data, Some(vec![4, 5, 6, 7, 8]));
    }

    #[test]
    fn test_converted_graph_clone() {
        let original = super::ConvertedGraph {
            format: "test",
            content_type: "application/octet-stream",
            data: vec![1, 2, 3],
            weights_data: Some(vec![4, 5]),
        };

        let cloned = original.clone();
        assert_eq!(cloned.format, original.format);
        assert_eq!(cloned.content_type, original.content_type);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.weights_data, original.weights_data);
    }
}
