//! Explicit FP16 graph-conversion and CoreML execution experiment.
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use half::f16;
use rustnn::backend_selection::{BackendDevice, DeviceType};
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{DataType, GraphInfo, Operation};
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};
use serde_json::json;

#[path = "coreml_precision_probe/mixed.rs"]
mod mixed;

#[derive(Parser)]
struct Args {
    /// Source .webnn graph, with external weights beside it.
    model: PathBuf,
    /// Float32 reference inputs, as named safetensors.
    #[arg(long)]
    inputs: PathBuf,
    /// Destination for results, typed outputs and a copy of the compiled model.
    #[arg(long)]
    output: PathBuf,
    /// Explicitly rewrite float32 operands, constants and casts to float16.
    #[arg(long)]
    fp16: bool,
    /// Preserve these operation types in float32; movement follows its input precision.
    #[arg(long, value_delimiter = ',', requires = "fp16")]
    fp32_ops: Vec<String>,
    /// Preserve operations whose labels begin with these model-specific prefixes.
    #[arg(long, value_delimiter = ',', requires = "fp16")]
    fp32_prefixes: Vec<String>,
    /// Requested CoreML policy; a successful policy can still schedule on CPU.
    #[arg(long, default_value="cpu", value_parser=["cpu","gpu","npu"])]
    device: String,
    /// Measured dispatches; the first is reported separately.
    #[arg(long, default_value_t = 3)]
    trials: usize,
    /// Permit finite constants to overflow during this lossy conversion experiment.
    #[arg(long)]
    allow_constant_overflow: bool,
}

fn narrow(graph: &mut GraphInfo, allow_overflow: bool) -> Result<serde_json::Value> {
    let mut constants = 0;
    let mut overflow = 0;
    let mut underflow = 0;
    for (&id, value) in &mut graph.constant_operand_ids_to_handles {
        if graph.operands[id as usize].descriptor.data_type != DataType::Float32 {
            continue;
        }
        ensure!(
            value.data.len().is_multiple_of(4),
            "invalid float32 constant {id}"
        );
        let mut bytes = Vec::with_capacity(value.data.len() / 2);
        for chunk in value.data.chunks_exact(4) {
            let x = f32::from_le_bytes(chunk.try_into()?);
            let y = f16::from_f32(x);
            overflow += usize::from(x.is_finite() && !y.is_finite());
            underflow += usize::from(x != 0. && x.is_finite() && y.to_f32() == 0.);
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        value.data = bytes;
        constants += 1;
    }
    ensure!(
        allow_overflow || overflow == 0,
        "{overflow} finite constant values overflow FP16; conversion refused"
    );
    let mut operands = 0;
    for value in &mut graph.operands {
        if value.descriptor.data_type == DataType::Float32 {
            value.descriptor.data_type = DataType::Float16;
            operands += 1;
        }
    }
    for operation in &mut graph.operations {
        if let Operation::Cast { data_type, .. } = operation
            && *data_type == MLOperandDataType::Float32
        {
            *data_type = MLOperandDataType::Float16;
        }
    }
    Ok(
        json!({"convertedOperands":operands,"convertedConstants":constants,"constantOverflows":overflow,"constantUnderflows":underflow}),
    )
}
fn bindings(values: &BTreeMap<String, MLTensor>) -> MLNamedTensors<'_> {
    values.iter().map(|(n, t)| (n.as_str(), t)).collect()
}
fn compiled_paths() -> HashSet<PathBuf> {
    fs::read_dir(std::env::temp_dir())
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "mlmodelc")
                && p.file_name()
                    .is_some_and(|x| x.to_string_lossy().starts_with("rustnn_coreml_"))
        })
        .collect()
}
fn copy_dir(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let output = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&path, &output)?;
        } else {
            fs::copy(path, output)?;
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.trials > 0, "at least one trial is required");
    fs::create_dir_all(&args.output)?;
    let mut graph = rustnn::load_graph_from_path(&args.model)?;
    let conversion = if args.fp16 && (!args.fp32_ops.is_empty() || !args.fp32_prefixes.is_empty()) {
        let (converted, report) = mixed::convert(
            graph,
            &args.fp32_ops,
            &args.fp32_prefixes,
            args.allow_constant_overflow,
        )?;
        graph = converted;
        report
    } else if args.fp16 {
        narrow(&mut graph, args.allow_constant_overflow)?
    } else {
        json!(null)
    };
    eprintln!("conversion: {conversion}");
    let device = match args.device.as_str() {
        "gpu" => DeviceType::Gpu,
        "npu" => DeviceType::Npu,
        _ => DeviceType::Cpu,
    };
    let options = MLContextOptions::new(MLPowerPreference::Default, device != DeviceType::Cpu)
        .with_rustnn_device_hint(BackendDevice::Coreml {
            device_type: device,
        });
    let mut context = MLContext::create(&options)?;
    let before = compiled_paths();
    let start = Instant::now();
    let mut built = match context.rustnn_build_graph(graph.clone()) {
        Ok(built) => built,
        Err(error) => {
            dump_failed_program(&graph, &args.output)?;
            return Err(error).context("compile converted graph");
        }
    };
    drop(graph);
    let compile_ms = start.elapsed().as_secs_f64() * 1000.;
    let diagnostic = built.rustnn_load_diagnostics();
    let Some(rustnn::mlcontext::LoadDiagnostics::Coreml(ref policy)) = diagnostic else {
        anyhow::bail!("missing CoreML diagnostics");
    };
    ensure!(
        policy.requested_compute_units == policy.loaded_compute_units,
        "CoreML load fallback rejected: {policy:?}"
    );
    let diagnostics = format!("{diagnostic:?}");
    eprintln!("compile: {compile_ms:.2} ms; {diagnostics}");
    let data = fs::read(&args.inputs)?;
    let supplied = SafeTensors::deserialize(&data)?;
    let mut inputs = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    for (name, desc) in &built.input_descriptors {
        let view = supplied
            .tensor(name)
            .with_context(|| format!("missing input {name}"))?;
        ensure!(
            view.dtype() == Dtype::F32,
            "probe reference inputs must be float32"
        );
        ensure!(
            view.shape()
                == desc
                    .static_or_max_shape()
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>(),
            "input shape mismatch: {name}"
        );
        let descriptor = MLTensorDescriptor::new(
            desc.data_type.try_into()?,
            desc.static_or_max_shape()
                .into_iter()
                .map(u64::from)
                .collect(),
        )
        .to_writable();
        let tensor = context.create_tensor(&descriptor)?;
        let bytes = if desc.data_type == DataType::Float16 {
            view.data()
                .chunks_exact(4)
                .flat_map(|c| {
                    f16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()
                })
                .collect::<Vec<_>>()
        } else {
            ensure!(
                desc.data_type == DataType::Float32,
                "unsupported input type"
            );
            view.data().to_vec()
        };
        context.write_tensor(&tensor, &bytes)?;
        inputs.insert(name.clone(), tensor);
    }
    for (name, desc) in &built.output_descriptors {
        let descriptor = MLTensorDescriptor::new(
            desc.data_type.try_into()?,
            desc.static_or_max_shape()
                .into_iter()
                .map(u64::from)
                .collect(),
        )
        .to_readable();
        outputs.insert(name.clone(), context.create_tensor(&descriptor)?);
    }
    let mut timings = Vec::new();
    let mut values = BTreeMap::new();
    for trial in 0..args.trials {
        let start = Instant::now();
        context.dispatch(&mut built, &bindings(&inputs), &bindings(&outputs))?;
        let dispatch_ms = start.elapsed().as_secs_f64() * 1000.;
        for (name, tensor) in &outputs {
            let mut bytes = vec![0; tensor.rustnn_required_bytes()];
            context.read_tensor(tensor, &mut bytes)?;
            if trial == 0 {
                let desc = &built.output_descriptors[name];
                let floats = match desc.data_type {
                    DataType::Float16 => bytes
                        .chunks_exact(2)
                        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                        .collect::<Vec<_>>(),
                    DataType::Float32 => bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                    _ => anyhow::bail!("unsupported output type"),
                };
                values.insert(
                    name.clone(),
                    (
                        desc.static_or_max_shape()
                            .iter()
                            .map(|&x| x as usize)
                            .collect::<Vec<_>>(),
                        floats,
                    ),
                );
            }
        }
        timings.push(json!({"dispatchMs":dispatch_ms,"dispatchAndReadMs":start.elapsed().as_secs_f64()*1000.}));
    }
    let buffers = values
        .iter()
        .map(|(name, (shape, values))| {
            (
                name.clone(),
                shape.clone(),
                values
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let tensors = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    serialize_to_file(tensors, None, &args.output.join("outputs.safetensors"))?;
    let stats=values.iter().map(|(name,(_,values))|(name.clone(),json!({"elements":values.len(),"nonfinite":values.iter().filter(|x|!x.is_finite()).count(),"maxAbs":values.iter().filter(|x|x.is_finite()).map(|x|x.abs()).fold(0.,f32::max)}))).collect::<BTreeMap<_,_>>();
    let report = json!({"model":args.model,"fp16":args.fp16,"fp32Ops":args.fp32_ops,"fp32Prefixes":args.fp32_prefixes,"requestedDevice":args.device,"placementVerified":false,"conversion":conversion,"compileMs":compile_ms,"loadDiagnostics":diagnostics,"trials":timings,"outputs":stats,"backendStatistics":format!("{:?}",context.rustnn_backend_statistics())});
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    let candidates = compiled_paths()
        .difference(&before)
        .cloned()
        .collect::<Vec<_>>();
    if let [path] = candidates.as_slice() {
        copy_dir(path, &args.output.join("model.mlmodelc"))?;
    } else {
        eprintln!(
            "Skipping compiled artifact capture: expected one new model, found {}. Run probes sequentially.",
            candidates.len()
        );
    }
    ensure!(
        stats.values().all(|s| s["nonfinite"] == 0),
        "model produced nonfinite outputs; see report.json"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustnn::graph::{ConstantData, Operand, OperandDescriptor, OperandKind};
    fn constant(values: &[f32]) -> GraphInfo {
        GraphInfo {
            operands: vec![Operand {
                kind: OperandKind::Constant,
                descriptor: OperandDescriptor {
                    data_type: DataType::Float32,
                    shape: vec![rustnn::graph::Dimension::Static(values.len() as u32)],
                    pending_permutation: vec![],
                },
                name: Some("test".into()),
            }],
            constant_operand_ids_to_handles: [(
                0,
                ConstantData {
                    data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    label: None,
                },
            )]
            .into(),
            ..GraphInfo::default()
        }
    }
    #[test]
    fn rejects_finite_overflow_but_preserves_infinities() {
        assert!(narrow(&mut constant(&[65520.]), false).is_err());
        let mut graph = constant(&[f32::INFINITY, f32::NEG_INFINITY]);
        let report = narrow(&mut graph, false).unwrap();
        assert_eq!(report["constantOverflows"], 0);
        assert_eq!(
            graph.constant_operand_ids_to_handles[&0].data,
            [0x00, 0x7c, 0x00, 0xfc]
        );
    }
    #[test]
    fn reports_underflow_without_losing_representable_subnormals() {
        let mut graph = constant(&[1e-12, f16::from_bits(1).to_f32(), 1.00048828125]);
        let report = narrow(&mut graph, false).unwrap();
        assert_eq!(report["constantUnderflows"], 1);
        assert_eq!(
            graph.constant_operand_ids_to_handles[&0].data,
            [0, 0, 1, 0, 0, 0x3c]
        );
        assert_eq!(graph.operands[0].descriptor.data_type, DataType::Float16);
    }
    #[test]
    fn cast_parameter_agrees_with_the_converted_output_type() {
        let mut graph = constant(&[1.]);
        let mut output = graph.operands[0].clone();
        output.kind = OperandKind::Intermediate;
        output.name = Some("cast_output".into());
        graph.operands.push(output);
        graph.operations.push(Operation::Cast {
            input: 0,
            data_type: MLOperandDataType::Float32,
            options: None,
            outputs: vec![1],
        });
        narrow(&mut graph, false).unwrap();
        assert!(matches!(
            graph.operations[0],
            Operation::Cast {
                data_type: MLOperandDataType::Float16,
                ..
            }
        ));
    }
}

fn dump_failed_program(graph: &GraphInfo, output: &Path) -> Result<()> {
    use prost::Message;
    use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
    use rustnn::protos::coreml::{
        mil_spec::{self, argument::binding::Binding},
        specification,
    };
    let converted = CoremlMlProgramConverter.convert(graph)?;
    fs::write(output.join("failed.mlmodel"), &converted.data)?;
    fn dtype(value: &mil_spec::ValueType) -> Option<i32> {
        match &value.r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => Some(t.data_type),
            _ => None,
        }
    }
    fn inspect(model: &specification::Model) {
        match &model.r#type {
            Some(specification::model::Type::MlProgram(program)) => {
                for function in program.functions.values() {
                    for block in function.block_specializations.values() {
                        let types = function
                            .inputs
                            .iter()
                            .chain(block.operations.iter().flat_map(|op| op.outputs.iter()))
                            .filter_map(|v| Some((v.name.as_str(), dtype(v.r#type.as_ref()?)?)))
                            .collect::<BTreeMap<_, _>>();
                        for op in &block.operations {
                            if op.r#type != "add" {
                                continue;
                            }
                            let inputs = op
                                .inputs
                                .iter()
                                .flat_map(|(key, arg)| {
                                    let types = &types;
                                    arg.arguments.iter().filter_map(move |v| {
                                        let (name, typ) = match &v.binding {
                                            Some(Binding::Name(name)) => {
                                                (name.clone(), types.get(name.as_str()).copied())
                                            }
                                            Some(Binding::Value(value)) => (
                                                "immediate".into(),
                                                value.r#type.as_ref().and_then(dtype),
                                            ),
                                            _ => return None,
                                        };
                                        Some((key, name, typ))
                                    })
                                })
                                .collect::<Vec<_>>();
                            let outputs = op
                                .outputs
                                .iter()
                                .map(|v| (&v.name, v.r#type.as_ref().and_then(dtype)))
                                .collect::<Vec<_>>();
                            if let Some((_, Some(expected))) = outputs.first()
                                && inputs
                                    .iter()
                                    .any(|(_, _, typ)| typ.is_some_and(|d| d != *expected))
                            {
                                eprintln!(
                                    "MIL ADD TYPE MISMATCH inputs={inputs:?} outputs={outputs:?}"
                                );
                            }
                        }
                    }
                }
            }
            Some(specification::model::Type::Pipeline(pipeline)) => {
                for child in &pipeline.models {
                    inspect(child);
                }
            }
            _ => {}
        }
    }
    inspect(&specification::Model::decode(converted.data.as_slice())?);
    Ok(())
}
