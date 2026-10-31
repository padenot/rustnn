//! Experimental graph rewrite with explicit precision boundaries.
use anyhow::{Context, Result, ensure};
use half::f16;
use rustnn::graph::{ConstantData, OperandKind};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{DataType, GraphInfo, Operation};
use serde_json::json;
use std::collections::{HashMap, HashSet};

fn floating(dtype: DataType) -> bool {
    matches!(dtype, DataType::Float16 | DataType::Float32)
}
struct Rewrite {
    graph: GraphInfo,
    casts: HashMap<(u32, DataType), u32>,
    constant_overflows: usize,
    constant_underflows: usize,
    allow_overflow: bool,
}
impl Rewrite {
    fn cast(&mut self, input: u32, target: DataType) -> Result<u32> {
        let source = self.graph.operands[input as usize].descriptor.data_type;
        if source == target || !floating(source) {
            return Ok(input);
        }
        if let Some(&id) = self.casts.get(&(input, target)) {
            return Ok(id);
        }
        let id = u32::try_from(self.graph.operands.len())?;
        let mut operand = self.graph.operands[input as usize].clone();
        operand.descriptor.data_type = target;
        operand.name = Some(format!(
            "{}_as_{target:?}_{id}",
            operand.name.as_deref().unwrap_or("operand")
        ));
        if let Some(value) = self.graph.constant_operand_ids_to_handles.get(&input) {
            let data = if target == DataType::Float16 {
                let mut bytes = Vec::with_capacity(value.data.len() / 2);
                for chunk in value.data.chunks_exact(4) {
                    let x = f32::from_le_bytes(chunk.try_into()?);
                    let y = f16::from_f32(x);
                    self.constant_overflows += usize::from(x.is_finite() && !y.is_finite());
                    self.constant_underflows +=
                        usize::from(x != 0. && x.is_finite() && y.to_f32() == 0.);
                    ensure!(
                        self.allow_overflow || !x.is_finite() || y.is_finite(),
                        "constant operand {input} overflows float16"
                    );
                    bytes.extend_from_slice(&y.to_le_bytes());
                }
                bytes
            } else {
                value
                    .data
                    .chunks_exact(2)
                    .flat_map(|c| {
                        f16::from_le_bytes(c.try_into().unwrap())
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect()
            };
            let value = ConstantData {
                data,
                label: value.label.clone(),
            };
            operand.kind = OperandKind::Constant;
            self.graph.constant_operand_ids_to_handles.insert(id, value);
        } else {
            operand.kind = OperandKind::Intermediate;
            self.graph.operations.push(Operation::Cast {
                input,
                data_type: target.try_into()?,
                options: None,
                outputs: vec![id],
            });
        }
        self.graph.operands.push(operand);
        self.casts.insert((input, target), id);
        Ok(id)
    }
}

pub(super) fn convert(
    graph: GraphInfo,
    keep: &[String],
    prefixes: &[String],
    allow_overflow: bool,
) -> Result<(GraphInfo, serde_json::Value)> {
    let mut rewrite = Rewrite {
        graph,
        casts: HashMap::new(),
        constant_overflows: 0,
        constant_underflows: 0,
        allow_overflow,
    };
    let operations = std::mem::take(&mut rewrite.graph.operations);
    let keep = keep.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut constants = rewrite
        .graph
        .constant_operand_ids_to_handles
        .keys()
        .copied()
        .collect::<HashSet<_>>();
    let mut wide_operations = 0;
    let mut half_operations = 0;
    for operation in operations {
        let kind = operation.op_type();
        let inputs = operation.input_operands();
        let constant = !inputs.is_empty()
            && operation
                .all_input_operands()
                .iter()
                .all(|id| constants.contains(id));
        let constant_arithmetic = constant
            && !matches!(
                kind,
                "cast"
                    | "identity"
                    | "reshape"
                    | "squeeze"
                    | "unsqueeze"
                    | "transpose"
                    | "split"
                    | "concat"
                    | "slice"
                    | "expand"
                    | "tile"
                    | "pad"
                    | "gather"
            )
            && operation.outputs().iter().any(|&id| {
                rewrite.graph.operands[id as usize].descriptor.data_type == DataType::Float32
            });
        // Preserve integer-to-float conversions: shape-derived sample counts can
        // exceed 65504 even when the audio and learned weights fit float16.
        let wide_integer_cast = kind == "cast"
            && inputs.first().is_some_and(|&id| {
                !floating(rewrite.graph.operands[id as usize].descriptor.data_type)
            })
            && operation.outputs().iter().any(|&id| {
                rewrite.graph.operands[id as usize].descriptor.data_type == DataType::Float32
            });
        let wide = wide_integer_cast
            || constant_arithmetic
            || keep.contains(kind)
            || prefixes.iter().any(|p| operation.label().starts_with(p));
        let movement = matches!(
            kind,
            "reshape"
                | "squeeze"
                | "unsqueeze"
                | "transpose"
                | "split"
                | "concat"
                | "slice"
                | "expand"
                | "tile"
                | "pad"
                | "gather"
                | "identity"
                | "scatterND"
                | "scatterElements"
        );
        let dtype = if wide {
            DataType::Float32
        } else if movement {
            inputs
                .first()
                .map(|&id| rewrite.graph.operands[id as usize].descriptor.data_type)
                .filter(|&d| floating(d))
                .unwrap_or(DataType::Float16)
        } else {
            DataType::Float16
        };
        let mut attributes = operation.attributes_json_value();
        if kind == "cast" {
            if operation
                .outputs()
                .iter()
                .any(|&id| floating(rewrite.graph.operands[id as usize].descriptor.data_type))
            {
                attributes["to"] = serde_json::to_value(MLOperandDataType::try_from(dtype)?)?;
            }
        } else {
            // Operand-valued options are part of the documented interchange representation.
            for key in [
                "c",
                "bias",
                "scale",
                "recurrentBias",
                "peepholeWeight",
                "initialHiddenState",
                "initialCellState",
            ] {
                if let Some(id) = attributes.get(key).and_then(|v| v.as_u64()) {
                    attributes[key] = json!(rewrite.cast(u32::try_from(id)?, dtype)?);
                }
            }
        }
        let inputs = inputs
            .into_iter()
            .map(|id| {
                if kind == "cast" {
                    Ok(id)
                } else {
                    rewrite.cast(id, dtype)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut changed = false;
        for &output in operation.outputs() {
            let descriptor = &mut rewrite.graph.operands[output as usize].descriptor;
            if floating(descriptor.data_type) {
                descriptor.data_type = dtype;
                changed = true;
            }
        }
        if changed {
            if dtype == DataType::Float32 {
                wide_operations += 1;
            } else {
                half_operations += 1;
            }
        }
        let mapped =
            Operation::from_json_attributes(kind, &inputs, operation.outputs(), &attributes)
                .with_context(|| {
                    format!("cannot reconstruct {kind} after explicit precision conversion")
                })?;
        if constant {
            constants.extend(operation.outputs());
        }
        rewrite.graph.operations.push(mapped);
    }
    let outputs = std::mem::take(&mut rewrite.graph.output_operands);
    for output in outputs {
        if floating(rewrite.graph.operands[output as usize].descriptor.data_type) {
            let name = rewrite.graph.operands[output as usize].name.clone();
            let converted = rewrite.cast(output, DataType::Float32)?;
            if converted != output {
                rewrite.graph.operands[output as usize].kind = OperandKind::Intermediate;
                rewrite.graph.operands[output as usize].name =
                    Some(format!("{}_compute", name.as_deref().unwrap_or("output")));
                rewrite.graph.operands[converted as usize].kind = OperandKind::Output;
                rewrite.graph.operands[converted as usize].name = name;
            }
            rewrite.graph.output_operands.push(converted);
        } else {
            rewrite.graph.output_operands.push(output);
        }
    }
    rewrite.graph.output_operands.sort_unstable();
    let used = rewrite
        .graph
        .operations
        .iter()
        .flat_map(Operation::all_input_operands)
        .chain(rewrite.graph.output_operands.iter().copied())
        .collect::<HashSet<_>>();
    rewrite
        .graph
        .constant_operand_ids_to_handles
        .retain(|id, _| used.contains(id));
    rewrite.graph.validate_io_operand_lists()?;
    let report = json!({"fp32Operations":wide_operations,"fp16Operations":half_operations,"precisionBoundaries":rewrite.casts.len(),"constantOverflows":rewrite.constant_overflows,"constantUnderflows":rewrite.constant_underflows,"io":"original input types, float32 floating outputs"});
    Ok((rewrite.graph, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustnn::graph::{Dimension, Operand, OperandDescriptor};
    use rustnn::operator_options::MLClampOptions;
    fn operand(name: &str, kind: OperandKind) -> Operand {
        Operand {
            kind,
            name: Some(name.into()),
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: vec![Dimension::Static(2)],
                pending_permutation: vec![],
            },
        }
    }
    #[test]
    fn retains_normalization_threshold_and_inserts_typed_boundaries() {
        let graph = GraphInfo {
            operands: vec![
                operand("input", OperandKind::Input),
                operand("norm", OperandKind::Intermediate),
                operand("weight", OperandKind::Constant),
                operand("out", OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![3],
            constant_operand_ids_to_handles: [(
                2,
                ConstantData {
                    data: [1f32, 2f32]
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                    label: None,
                },
            )]
            .into(),
            operations: vec![
                Operation::Clamp {
                    input: 0,
                    options: Some(MLClampOptions {
                        min_value: Some(json!(1e-12)),
                        max_value: None,
                        label: "norm".into(),
                    }),
                    outputs: vec![1],
                },
                Operation::Mul {
                    a: 1,
                    b: 2,
                    options: None,
                    outputs: vec![3],
                },
            ],
            ..GraphInfo::default()
        };
        let (graph, report) = convert(graph, &["clamp".into()], &[], false).unwrap();
        assert_eq!(graph.operands[0].descriptor.data_type, DataType::Float32);
        assert_eq!(graph.operands[1].descriptor.data_type, DataType::Float32);
        assert_eq!(graph.operands[3].descriptor.data_type, DataType::Float16);
        assert_eq!(report["constantUnderflows"], 0);
        let output = graph.output_operands[0];
        assert_eq!(
            graph.operands[output as usize].descriptor.data_type,
            DataType::Float32
        );
        assert_eq!(graph.operands[output as usize].name.as_deref(), Some("out"));
        assert!(
            matches!(&graph.operations[0],Operation::Clamp{options:Some(o),..} if o.min_value==Some(json!(1e-12)))
        );
        let multiply = graph
            .operations
            .iter()
            .find(|op| op.op_type() == "mul")
            .unwrap();
        for id in multiply.inputs() {
            assert_eq!(
                graph.operands[id as usize].descriptor.data_type,
                DataType::Float16
            );
        }
        assert!(!graph.constant_operand_ids_to_handles.contains_key(&2));
        graph.validate_io_operand_lists().unwrap();
    }
    #[test]
    fn converts_gemm_optional_c_with_its_matrix_inputs() {
        use rustnn::operator_options::MLGemmOptions;
        let graph = GraphInfo {
            operands: vec![
                operand("a", OperandKind::Input),
                operand("b", OperandKind::Input),
                operand("c", OperandKind::Input),
                operand("out", OperandKind::Output),
            ],
            input_operands: vec![0, 1, 2],
            output_operands: vec![3],
            operations: vec![Operation::Gemm {
                a: 0,
                b: 1,
                options: Some(MLGemmOptions {
                    c: Some(2),
                    ..Default::default()
                }),
                outputs: vec![3],
            }],
            ..Default::default()
        };
        let (graph, _) = convert(graph, &[], &[], false).unwrap();
        let gemm = graph
            .operations
            .iter()
            .find(|op| op.op_type() == "gemm")
            .unwrap();
        for id in gemm.all_input_operands() {
            assert_eq!(
                graph.operands[id as usize].descriptor.data_type,
                DataType::Float16
            );
        }
        assert_eq!(gemm.all_input_operands().len(), 3);
    }
    #[test]
    fn unsqueeze_preserves_float32_audio_input_before_normalization() {
        let graph = GraphInfo {
            operands: vec![
                operand("input", OperandKind::Input),
                operand("out", OperandKind::Output),
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![
                Operation::from_json_attributes("unsqueeze", &[0], &[1], &json!({"axes":[0]}))
                    .unwrap(),
            ],
            ..Default::default()
        };
        let (graph, _) = convert(graph, &[], &[], false).unwrap();
        assert_eq!(graph.operations.len(), 1);
        assert_eq!(graph.operands[1].descriptor.data_type, DataType::Float32);
    }
    #[test]
    fn keeps_large_integer_sample_count_cast_in_float32() {
        let mut input = operand("count", OperandKind::Input);
        input.descriptor.data_type = DataType::Int64;
        let graph = GraphInfo {
            operands: vec![input, operand("float_count", OperandKind::Output)],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![Operation::Cast {
                input: 0,
                data_type: MLOperandDataType::Float32,
                options: None,
                outputs: vec![1],
            }],
            ..Default::default()
        };
        let (graph, _) = convert(graph, &[], &[], false).unwrap();
        assert!(matches!(
            graph.operations[0],
            Operation::Cast {
                data_type: MLOperandDataType::Float32,
                ..
            }
        ));
        assert_eq!(graph.operands[1].descriptor.data_type, DataType::Float32);
    }
}
