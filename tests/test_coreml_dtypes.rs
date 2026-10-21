//! Numeric boundary regressions for both CoreML execution APIs.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use std::collections::HashMap;

use rustnn::converters::{CoremlMlProgramConverter, GraphConverter, coreml_feature_name};
use rustnn::executors::coreml::{
    CoremlInput, CoremlRunAttempt, run_coreml_with_inputs_checked, run_coreml_zeroed,
};
use rustnn::graph::{
    DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, to_dimension_vector,
};
use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operators::Operation;

fn cast_graph(input_type: DataType, output_type: DataType, shape: &[u32]) -> GraphInfo {
    let descriptor = |data_type| OperandDescriptor {
        data_type,
        shape: to_dimension_vector(shape),
        pending_permutation: vec![],
    };
    GraphInfo {
        operands: vec![
            Operand {
                kind: OperandKind::Input,
                name: Some("input".into()),
                descriptor: descriptor(input_type),
            },
            Operand {
                kind: OperandKind::Output,
                name: Some("result".into()),
                descriptor: descriptor(output_type),
            },
        ],
        input_operands: vec![0],
        output_operands: vec![1],
        operations: vec![
            Operation::from_json_attributes(
                "cast",
                &[0],
                &[1],
                &serde_json::json!({"to": output_type}),
            )
            .unwrap(),
        ],
        ..Default::default()
    }
}

fn check_attempts(attempts: &[CoremlRunAttempt], expected: &[f32]) {
    // CPU is required; also check every successful accelerator-enabled policy.
    // A policy request is not evidence of accelerator placement.
    assert!(
        attempts
            .iter()
            .find(|attempt| attempt.compute_unit == "CPU_ONLY")
            .unwrap()
            .result
            .is_ok(),
        "{attempts:?}"
    );
    for attempt in attempts {
        if let Ok(outputs) = &attempt.result {
            assert_eq!(outputs[0].data, expected, "{}", attempt.compute_unit);
        }
    }
}

fn check_convenience(
    input_type: DataType,
    output_type: DataType,
    shape: &[u32],
    values: &[f32],
    expected: &[f32],
) {
    let graph = cast_graph(input_type, output_type, shape);
    let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
    let attempts = run_coreml_with_inputs_checked(
        &converted.data,
        vec![CoremlInput {
            name: coreml_feature_name("input"),
            shape: shape.iter().map(|&size| size as usize).collect(),
            data: values.to_vec(),
        }],
        &HashMap::from([(
            coreml_feature_name("input"),
            graph.operands[0].descriptor.clone(),
        )]),
        &HashMap::from([(
            coreml_feature_name("result"),
            graph.operands[1].descriptor.clone(),
        )]),
    )
    .unwrap();
    check_attempts(&attempts, expected);
}

#[test]
fn coreml_dtypes_convenience_int32_input_is_converted_by_value() {
    let values = [0., 1., -2., 31., 62., 123.];
    check_convenience(
        DataType::Int32,
        DataType::Float32,
        &[2, 3],
        &values,
        &values,
    );
}

#[test]
fn coreml_dtypes_convenience_half_and_integer_output_widths() {
    let values = [-2., 0., 0.5, 1.25, 31., 1000.];
    for (input, output) in [
        (DataType::Float16, DataType::Float32),
        (DataType::Float32, DataType::Float16),
        (DataType::Float16, DataType::Float16),
    ] {
        check_convenience(input, output, &[2, 3], &values, &values);
    }
    check_convenience(
        DataType::Float32,
        DataType::Int32,
        &[2, 3],
        &values,
        &[-2., 0., 0., 1., 31., 1000.],
    );
}

#[test]
fn coreml_dtypes_existing_wpt_cast_vector_exercises_both_apis() {
    // Existing upstream WPT: "cast int32 4D tensor to float32". The WPT
    // harness dispatches typed tensors; this also covers our older f32 adapter.
    let values = [
        45i32, 55, 11, 21, 78, 104, 102, 66, 41, 110, 92, 69, 48, 23, 58, 12, 33, 24, 101, 87, 49,
        118, 1, 77,
    ];
    let floats = values.map(|value| value as f32);
    check_convenience(
        DataType::Int32,
        DataType::Float32,
        &[2, 2, 2, 3],
        &floats,
        &floats,
    );
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_backend_hint(Backend::Coreml),
    )
    .unwrap();
    let mut graph = context
        .rustnn_build_graph(cast_graph(
            DataType::Int32,
            DataType::Float32,
            &[2, 2, 2, 3],
        ))
        .unwrap();
    let input = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Int32, vec![2, 2, 2, 3]).to_writable(),
        )
        .unwrap();
    let output = context
        .create_tensor(
            &MLTensorDescriptor::new(MLOperandDataType::Float32, vec![2, 2, 2, 3]).to_readable(),
        )
        .unwrap();
    context.write_tensor(&input, &values).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("result", &output)]),
        )
        .unwrap();
    let mut actual = [0f32; 24];
    context.read_tensor(&output, &mut actual).unwrap();
    assert_eq!(actual, floats);
}

#[test]
fn coreml_dtypes_dispatch_widens_int32_proxies_exactly_once() {
    for (dtype, tensor_dtype) in [
        (DataType::Int64, MLOperandDataType::Int64),
        (DataType::Uint64, MLOperandDataType::Uint64),
    ] {
        let mut context = MLContext::create(
            &MLContextOptions::new(MLPowerPreference::Default, false)
                .with_rustnn_backend_hint(Backend::Coreml),
        )
        .unwrap();
        let mut graph = context
            .rustnn_build_graph(cast_graph(DataType::Int32, dtype, &[3]))
            .unwrap();
        assert_eq!(graph.output_descriptors["result"].data_type, dtype);
        let input = context
            .create_tensor(
                &MLTensorDescriptor::new(MLOperandDataType::Int32, vec![3]).to_writable(),
            )
            .unwrap();
        let output = context
            .create_tensor(&MLTensorDescriptor::new(tensor_dtype, vec![3]).to_readable())
            .unwrap();
        context.write_tensor(&input, &[-1i32, 0, 123]).unwrap();
        context
            .dispatch(
                &mut graph,
                &MLNamedTensors::from([("input", &input)]),
                &MLNamedTensors::from([("result", &output)]),
            )
            .unwrap();
        if dtype == DataType::Int64 {
            let mut actual = [0i64; 3];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, [-1, 0, 123]);
        } else {
            let mut actual = [0u64; 3];
            context.read_tensor(&output, &mut actual).unwrap();
            assert_eq!(actual, [u32::MAX as u64, 0, 123]);
        }
    }
}

#[test]
fn coreml_dtypes_zeroed_inputs_use_native_allocation_width() {
    for dtype in [
        DataType::Float16,
        DataType::Float32,
        DataType::Int32,
        DataType::Uint8,
        DataType::Int64,
    ] {
        let graph = cast_graph(dtype, DataType::Float32, &[2, 3]);
        let converted = CoremlMlProgramConverter.convert(&graph).unwrap();
        let attempts = run_coreml_zeroed(
            &converted.data,
            &HashMap::from([(
                coreml_feature_name("input"),
                graph.operands[0].descriptor.clone(),
            )]),
        )
        .unwrap();
        check_attempts(&attempts, &[0.; 6]);
    }
}
