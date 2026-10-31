//! Resolve static positional embeddings before BNNS compiles their broadcast/trig/concat chain.
#![cfg(feature = "coreml-runtime")]
use rustnn::converters::{CoremlMlProgramConverter, GraphConverter};
use rustnn::executors::coreml::{CoremlInput, run_coreml_with_inputs_with_weights};
use rustnn::mlcontext::{MLNamedOperands, MLOperandDescriptor};
use rustnn::mlgraphbuilder::MLGraphBuilder;
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLTransposeOptions;

#[test]
fn constant_positional_embedding_compiles_and_matches_reference() {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let positions: Vec<f32> = (0..1344).map(|i| i as f32).collect();
    let divisors: Vec<f32> = (0..256).map(|i| 10000f32.powf(i as f32 / 256.)).collect();
    let descriptor = |shape| MLOperandDescriptor::new(MLOperandDataType::Float32, shape);
    let position = builder
        .constant_from_bytes(
            &descriptor(vec![1344, 1, 1]),
            positions.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
    let exponent = builder
        .constant_from_bytes(
            &descriptor(vec![1, 1, 256]),
            (0..256)
                .map(|i| i as f32 / 256.)
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
    let base = builder
        .constant_from_bytes(&descriptor(vec![]), 10000f32.to_le_bytes().to_vec())
        .unwrap();
    let divisor = builder.pow(base, exponent).unwrap();
    let angles = builder.div(position, divisor).unwrap();
    let cosine = builder.cos(angles).unwrap();
    let sine = builder.sin(angles).unwrap();
    let joined = builder.concat(&[cosine, sine], 2).unwrap();
    let embedding = builder
        .transpose_with_options(
            joined,
            MLTransposeOptions {
                permutation: vec![1, 0, 2],
                ..Default::default()
            },
        )
        .unwrap();
    let input = builder.input("x", &descriptor(vec![1])).unwrap();
    let result = builder.add(input, embedding).unwrap();
    let mut outputs = MLNamedOperands::new();
    outputs.insert("result", result);
    let graph = builder.finish_graph_info(&outputs).unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    // This chain previously faults inside BNNS during CPU model loading on macOS 26.6.2.
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "x".into(),
            shape: vec![1],
            data: vec![2.],
        }],
    )
    .unwrap()
    {
        let actual = attempt.result.unwrap();
        assert_eq!(actual[0].data.len(), 1344 * 512);
        for (row, &position) in positions.iter().enumerate() {
            for (column, &divisor) in divisors.iter().enumerate() {
                let angle = position / divisor;
                for (offset, expected) in
                    [(column, 2. + angle.cos()), (256 + column, 2. + angle.sin())]
                {
                    assert!(
                        (actual[0].data[row * 512 + offset] - expected).abs() < 3e-5,
                        "{} [{row},{offset}]",
                        attempt.compute_unit
                    );
                }
            }
        }
    }
}

#[test]
fn constant_half_rotary_outer_product_through_identity_matches_reference() {
    use half::f16;
    use prost::Message;
    use rustnn::protos::coreml::specification;
    let mut b = MLGraphBuilder::new_uncompiled();
    let half = |shape| MLOperandDescriptor::new(MLOperandDataType::Float16, shape);
    let positions = (0..345)
        .map(|i| f16::from_f32(i as f32))
        .collect::<Vec<_>>();
    let frequencies = (0..32)
        .map(|i| f16::from_f32((i as f32 + 1.) / 64.))
        .collect::<Vec<_>>();
    let position = b
        .constant_from_bytes(
            &half(vec![1, 345, 1]),
            positions.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
    let frequency = b
        .constant_from_bytes(
            &half(vec![1, 1, 32]),
            frequencies.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
    let alias = b.identity(frequency).unwrap();
    let angles = b.matmul(position, alias).unwrap();
    let cosine = b.cos(angles).unwrap();
    let sine = b.sin(angles).unwrap();
    let joined = b.concat(&[cosine, sine], 2).unwrap();
    let wide = b.cast(joined, MLOperandDataType::Float32).unwrap();
    let input = b
        .input(
            "x",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]),
        )
        .unwrap();
    let result = b.add(input, wide).unwrap();
    let graph = b
        .finish_graph_info(&MLNamedOperands::from([("result", result)]))
        .unwrap();
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let model = specification::Model::decode(converted.data.as_slice()).unwrap();
    let specification::model::Type::MlProgram(program) = model.r#type.unwrap() else {
        panic!("expected one program")
    };
    let operations = &program.functions["main"].block_specializations["CoreML7"].operations;
    assert!(
        operations
            .iter()
            .all(|op| !matches!(op.r#type.as_str(), "matmul" | "sin" | "cos" | "concat")),
        "rotary constants must not reach the ANE matmul compiler"
    );
    for attempt in run_coreml_with_inputs_with_weights(
        &converted.data,
        converted.weights_data.as_deref(),
        vec![CoremlInput {
            name: "x".into(),
            shape: vec![1],
            data: vec![0.25],
        }],
    )
    .unwrap()
    {
        let output = attempt.result.unwrap();
        let actual = &output[0].data;
        assert_eq!(actual.len(), 345 * 64);
        for (row, p) in positions.iter().enumerate() {
            for (col, f) in frequencies.iter().enumerate() {
                let angle = f16::from_f32(p.to_f32() * f.to_f32()).to_f32();
                for (offset, expected) in [
                    (col, f16::from_f32(angle.cos()).to_f32() + 0.25),
                    (32 + col, f16::from_f32(angle.sin()).to_f32() + 0.25),
                ] {
                    assert!(
                        (actual[row * 64 + offset] - expected).abs() < 1e-6,
                        "{} [{row},{offset}]",
                        attempt.compute_unit
                    );
                }
            }
        }
    }
}
