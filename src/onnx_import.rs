use prost::Message;
use webnn_graph::onnx::convert::{ConvertOptions, OnnxConverter};
use webnn_graph::protos::onnx::{
    ModelProto, ValueInfoProto, tensor_shape_proto::dimension::Value as DimensionValue,
};

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::webnn_json;

#[derive(Debug, Clone, Copy)]
pub struct OnnxImportOptions {
    pub unknown_dimension_value: u32,
}

impl Default for OnnxImportOptions {
    fn default() -> Self {
        Self {
            unknown_dimension_value: 1,
        }
    }
}

pub fn load_onnx_graph_from_bytes(
    onnx_bytes: &[u8],
    options: OnnxImportOptions,
) -> Result<GraphInfo, GraphError> {
    let mut model =
        ModelProto::decode(onnx_bytes).map_err(|error| GraphError::ConversionFailed {
            format: "onnx-import".to_string(),
            reason: error.to_string(),
        })?;
    concretize_unknown_dimensions(&mut model, options.unknown_dimension_value);

    let converter = OnnxConverter::new(model).map_err(map_onnx_error)?;
    converter.extract_metadata().map_err(map_onnx_error)?;
    let graph_json = converter
        .convert(&ConvertOptions {
            extract_weights: false,
            output_path: "imported.webnn".to_string(),
            weights_path: None,
            manifest_path: None,
            free_dim_overrides: Default::default(),
            optimize: false,
            experimental_dynamic_inputs: false,
        })
        .map_err(map_onnx_error)?;
    webnn_json::from_graph_json(&graph_json)
}

fn map_onnx_error(error: webnn_graph::onnx::convert::OnnxError) -> GraphError {
    GraphError::ConversionFailed {
        format: "onnx-import".to_string(),
        reason: error.to_string(),
    }
}

fn concretize_unknown_dimensions(model: &mut ModelProto, unknown_dimension_value: u32) {
    let Some(graph) = model.graph.as_mut() else {
        return;
    };
    for value in graph
        .input
        .iter_mut()
        .chain(graph.value_info.iter_mut())
        .chain(graph.output.iter_mut())
    {
        concretize_value_info_dimensions(value, unknown_dimension_value);
    }
}

fn concretize_value_info_dimensions(value: &mut ValueInfoProto, unknown_dimension_value: u32) {
    let Some(type_proto) = value.r#type.as_mut() else {
        return;
    };
    let Some(webnn_graph::protos::onnx::type_proto::Value::TensorType(tensor_type)) =
        type_proto.value.as_mut()
    else {
        return;
    };
    let Some(shape) = tensor_type.shape.as_mut() else {
        return;
    };
    let fallback = i64::from(unknown_dimension_value);
    for dim in &mut shape.dim {
        match dim.value {
            Some(DimensionValue::DimValue(value)) if value > 0 => {}
            _ => {
                dim.value = Some(DimensionValue::DimValue(fallback));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webnn_graph::protos::onnx::{
        GraphProto, TensorShapeProto, TypeProto, tensor_shape_proto, type_proto,
    };

    #[test]
    fn concretize_unknown_and_symbolic_dimensions() {
        let mut model = ModelProto {
            graph: Some(GraphProto {
                input: vec![ValueInfoProto {
                    name: "features".to_string(),
                    r#type: Some(TypeProto {
                        value: Some(type_proto::Value::TensorType(
                            webnn_graph::protos::onnx::type_proto::Tensor {
                                elem_type: 1,
                                shape: Some(TensorShapeProto {
                                    dim: vec![
                                        tensor_shape_proto::Dimension {
                                            value: Some(
                                                tensor_shape_proto::dimension::Value::DimParam(
                                                    "batch".to_string(),
                                                ),
                                            ),
                                            ..Default::default()
                                        },
                                        tensor_shape_proto::Dimension {
                                            value: None,
                                            ..Default::default()
                                        },
                                        tensor_shape_proto::Dimension {
                                            value: Some(
                                                tensor_shape_proto::dimension::Value::DimValue(40),
                                            ),
                                            ..Default::default()
                                        },
                                    ],
                                }),
                            },
                        )),
                        denotation: String::new(),
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };

        concretize_unknown_dimensions(&mut model, 64);
        let type_value = model.graph.as_ref().unwrap().input[0]
            .r#type
            .as_ref()
            .unwrap()
            .value
            .as_ref()
            .unwrap();
        let type_proto::Value::TensorType(tensor_type) = type_value else {
            panic!("test input should be a tensor type");
        };
        let dims = &tensor_type.shape.as_ref().unwrap().dim;
        assert_eq!(
            dims.iter()
                .map(|dim| match dim.value {
                    Some(tensor_shape_proto::dimension::Value::DimValue(value)) => value,
                    _ => -1,
                })
                .collect::<Vec<_>>(),
            vec![64, 64, 40]
        );
    }
}
