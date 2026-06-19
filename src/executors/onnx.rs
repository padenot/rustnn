#![cfg(feature = "onnx-runtime")]

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Once;

use log::{info, warn};
use ndarray::{ArrayD, IxDyn};
use ort::environment::Environment;
use ort::session::SessionInputValue;

use half;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

use crate::converters::ONNX_EXTERNAL_WEIGHTS_FILENAME;
use crate::error::GraphError;
use crate::graph::OperandDescriptor;
use crate::runtime_checks::{RuntimeShapeState, TensorKind, validate_shape_data_length};

static INIT: Once = Once::new();

pub fn ensure_ort_initialized() -> Result<(), GraphError> {
    let mut result = Ok(());
    INIT.call_once(|| {
        info!("Loading onnxruntime");
        let _is_initial_load = ort::init()
            .with_name("rustnn")
            .with_execution_providers([ort::ep::CPUExecutionProvider::default().build()])
            .with_telemetry(false)
            .commit();

        let env = Environment::current().map_err(|e| {
            warn!("Error loading onnxruntime: {e}");
            GraphError::OnnxRuntimeUnavailable
        });
        if let Err(e) = env {
            result = Err(e);
            return;
        }
        let env = env.unwrap();
        env.set_log_level(ort::logging::LogLevel::Verbose);
        info!("Loaded");
    });
    result
}

#[derive(Debug, Clone)]
pub struct OnnxOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: String,
}

/// Tensor data for different types
pub enum TensorData {
    Float32(Vec<f32>),
    Float16(Vec<u16>), // f16 stored as u16 bits
    Int8(Vec<i8>),
    Uint8(Vec<u8>),
    Int32(Vec<i32>),
    Uint32(Vec<u32>),
    Int64(Vec<i64>),
    Uint64(Vec<u64>),
}

impl TensorData {
    fn len(&self) -> usize {
        match self {
            TensorData::Float32(v) => v.len(),
            TensorData::Float16(v) => v.len(),
            TensorData::Int8(v) => v.len(),
            TensorData::Uint8(v) => v.len(),
            TensorData::Int32(v) => v.len(),
            TensorData::Uint32(v) => v.len(),
            TensorData::Int64(v) => v.len(),
            TensorData::Uint64(v) => v.len(),
        }
    }
}

/// Input tensor data for ONNX execution
pub struct OnnxInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Output tensor with actual data
pub struct OnnxOutputWithData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub float32_data: Option<Vec<f32>>,
    pub int64_data: Option<Vec<i64>>,
    pub uint64_data: Option<Vec<u64>>,
}

pub fn run_onnx_zeroed(
    model_bytes: &[u8],
    _inputs: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<OnnxOutput>, GraphError> {
    // Initialize ort global environment (only once per process)
    ensure_ort_initialized()?;

    let mut session = Session::builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?
        .commit_from_memory(model_bytes)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("load model failed: {e}"),
        })?;

    // Build zero-filled inputs
    let mut input_values = Vec::new();
    for input_info in session.inputs().iter() {
        // Get shape from input type
        let shape: Vec<usize> = match input_info.dtype() {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => shape.iter().map(|&d| d.max(1) as usize).collect(),
            _ => {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: format!("input '{}' is not a tensor", input_info.name()),
                });
            }
        };

        let total: usize = shape.iter().product();
        let zeros = vec![0f32; total.max(1)];

        // Convert shape to Vec<i64> for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        let tensor = Value::from_array((shape_i64.as_slice(), zeros)).map_err(|e| {
            GraphError::OnnxRuntimeFailed {
                reason: format!(
                    "failed to create input tensor for {}: {e}",
                    input_info.name()
                ),
            }
        })?;
        input_values.push(tensor.into_dyn());
    }

    // Run inference - convert to Vec of SessionInputValue
    let input_session_values: Vec<SessionInputValue> = input_values
        .into_iter()
        .map(SessionInputValue::from)
        .collect();
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output metadata
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        // Get tensor shape and type
        let (shape, _data) =
            value
                .try_extract_tensor::<f32>()
                .map_err(|e| GraphError::OnnxRuntimeFailed {
                    reason: format!("failed to extract tensor: {e}"),
                })?;

        let shape_vec: Vec<i64> = shape.iter().copied().collect();
        results.push(OnnxOutput {
            name: format!("output_{idx}"),
            shape: shape_vec,
            data_type: "f32".to_string(),
        });
    }
    Ok(results)
}

/// Run ONNX model with actual input tensors and return output tensors with data.
///
/// ONNX Runtime performs its own shape checks. For validation against WebNN operand descriptors
/// (rank, static dimensions, dynamic `maxSize`, linked dynamic names), use
/// [`run_onnx_with_inputs_checked`].
pub fn run_onnx_with_inputs(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    run_onnx_with_inputs_impl(model_bytes, external_weights, inputs, None, None)
}

/// Same as [`run_onnx_with_inputs`], plus optional WebNN operand descriptor validation.
///
/// For each map that is `Some`, validates actual tensor shapes against that map (inputs before the
/// run, outputs after). `None` skips that side. When both maps are `Some`, same-named dynamic
/// dimensions are cross-checked (see `crate::runtime_checks::RuntimeShapeState`).
pub fn run_onnx_with_inputs_checked(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
    input_descriptors: &HashMap<String, OperandDescriptor>,
    output_descriptors: &HashMap<String, OperandDescriptor>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    run_onnx_with_inputs_impl(
        model_bytes,
        external_weights,
        inputs,
        Some(input_descriptors),
        Some(output_descriptors),
    )
}

fn run_onnx_with_inputs_impl(
    model_bytes: &[u8],
    external_weights: Option<&[u8]>,
    inputs: Vec<OnnxInput>,
    input_descriptors: Option<&HashMap<String, OperandDescriptor>>,
    output_descriptors: Option<&HashMap<String, OperandDescriptor>>,
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    // Initialize ort global environment (only once per process)
    ensure_ort_initialized()?;

    let mut builder = Session::builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("session builder failed: {e}"),
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|e| GraphError::OnnxRuntimeFailed {
            reason: format!("set opt level failed: {e}"),
        })?;
    if let Some(weights) = external_weights {
        builder = builder
            .with_external_initializer_file_in_memory(
                ONNX_EXTERNAL_WEIGHTS_FILENAME,
                Cow::Owned(weights.to_vec()),
            )
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("set external initializer failed: {e}"),
            })?;
    }
    let mut session =
        builder
            .commit_from_memory(model_bytes)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("load model failed: {e}"),
            })?;

    // Extract output names for later use
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    let mut runtime_shape_state = RuntimeShapeState::new();
    let mut actual_input_shapes = HashMap::new();
    for input in &inputs {
        validate_shape_data_length(&input.name, &input.shape, input.data.len())?;
        actual_input_shapes.insert(input.name.clone(), input.shape.clone());
    }
    if let Some(descriptors) = input_descriptors {
        runtime_shape_state.validate_named_shapes(
            &actual_input_shapes,
            descriptors,
            TensorKind::Input,
        )?;
    }

    // Build input tensors in the order the ONNX model expects (session.inputs()), looking up by name.
    // Callers may pass inputs in any order (e.g. alphabetical); matching by name avoids wrong mapping.
    let inputs_by_name: HashMap<String, OnnxInput> =
        inputs.into_iter().map(|i| (i.name.clone(), i)).collect();
    let debug_onnx = std::env::var("RUSTNN_DEBUG").as_deref() == Ok("1");
    if debug_onnx {
        eprintln!("[ONNX] session.inputs() order (feed order to runtime):");
        for (idx, input_info) in session.inputs().iter().enumerate() {
            eprintln!("  {}: {}", idx, input_info.name());
        }
    }
    let mut input_session_values: Vec<SessionInputValue> = Vec::new();
    for input_info in session.inputs().iter() {
        let name = input_info.name().to_string();
        let input = inputs_by_name
            .get(&name)
            .ok_or_else(|| GraphError::OnnxRuntimeFailed {
                reason: format!("model expects input '{name}' but it was not provided"),
            })?;
        if debug_onnx {
            let shape_str = format!("{:?}", input.shape);
            let preview = match &input.data {
                TensorData::Float32(d) if d.len() <= 12 => format!("data={:?}", d),
                TensorData::Float32(d) => {
                    format!("len={} first4={:?}", d.len(), &d[..4.min(d.len())])
                }
                _ => "".to_string(),
            };
            eprintln!("  [ONNX] feeding {} shape={} {}", name, shape_str, preview);
        }
        let session_value = match &input.data {
            TensorData::Float32(data) => {
                let value =
                    if input.shape.contains(&0) {
                        // ort tuple-shape API rejects 0-sized dimensions; ndarray path supports them.
                        let array = ArrayD::from_shape_vec(IxDyn(&input.shape), data.clone())
                            .map_err(|e| GraphError::OnnxRuntimeFailed {
                                reason: format!(
                                    "failed to create float32 ndarray input tensor for {}: {e}",
                                    input.name
                                ),
                            })?;
                        Value::from_array(array).map_err(|e| GraphError::OnnxRuntimeFailed {
                            reason: format!(
                                "failed to create float32 input tensor for {} via ndarray: {e}",
                                input.name
                            ),
                        })?
                    } else {
                        // Convert shape to i64 for ort compatibility
                        let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                        Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                            GraphError::OnnxRuntimeFailed {
                                reason: format!(
                                    "failed to create float32 input tensor for {}: {e}",
                                    input.name
                                ),
                            }
                        })?
                    };
                SessionInputValue::from(value)
            }
            TensorData::Float16(data) => {
                // Convert u16 bits to half::f16
                let f16_data: Vec<half::f16> = data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits))
                    .collect();
                // Convert shape to i64 for ort compatibility
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.as_slice(), f16_data)).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create float16 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int8(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int8 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint8(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint8 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int32(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint32(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint32 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Int64(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create int64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
            TensorData::Uint64(data) => {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                let value = Value::from_array((shape_i64.clone(), data.clone())).map_err(|e| {
                    GraphError::OnnxRuntimeFailed {
                        reason: format!(
                            "failed to create uint64 input tensor for {}: {e}",
                            input.name
                        ),
                    }
                })?;
                SessionInputValue::from(value)
            }
        };

        input_session_values.push(session_value);
    }

    // Run inference
    let outputs = session.run(input_session_values.as_slice()).map_err(|e| {
        GraphError::OnnxRuntimeFailed {
            reason: format!("run failed: {e}"),
        }
    })?;

    // Extract output tensors with data
    let mut results = Vec::new();
    for (idx, (_name, value)) in outputs.iter().enumerate() {
        let name = output_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("output_{}", idx));

        // Try to extract tensor with different types
        // The order matches most common types first for performance
        let (shape_vec, data_vec, float32_data, int64_data, uint64_data) =
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, Some(data.to_vec()), None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x.to_f32() as f64).collect();
                (shape_vec, data_vec, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u32>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i8>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u8>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, Some(data.to_vec()), None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u64>() {
                let shape_vec: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let data_vec: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (shape_vec, data_vec, None, None, Some(data.to_vec()))
            } else {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: "failed to extract output tensor: unsupported data type".to_string(),
                });
            };

        results.push(OnnxOutputWithData {
            name,
            shape: shape_vec,
            data: data_vec,
            float32_data,
            int64_data,
            uint64_data,
        });
    }

    if let Some(descriptors) = output_descriptors {
        let mut actual_output_shapes = HashMap::new();
        for output in &results {
            actual_output_shapes.insert(output.name.clone(), output.shape.clone());
        }
        runtime_shape_state.validate_named_shapes(
            &actual_output_shapes,
            descriptors,
            TensorKind::Output,
        )?;
    }

    Ok(results)
}

/// Device-resident tensor implementation for ONNX Runtime
///
/// This enables zero-copy execution by keeping tensors on device (GPU/NPU)
/// across multiple inference steps, eliminating host round-trips for
/// iterative workloads like KV cache in GenAI models.
#[derive(Debug)]
pub struct OrtDeviceTensor {
    /// ONNX Runtime value stored on device
    value: Value,
    /// Session reference to keep it alive
    _session: std::sync::Arc<Session>,
    /// Data type
    dtype: crate::graph::DataType,
    /// Tensor shape
    shape: Vec<usize>,
    /// Device kind (CPU, CUDA, etc.)
    device: crate::tensor::DeviceKind,
}

impl OrtDeviceTensor {
    /// Create a new device tensor with the given shape and data type
    pub fn new(
        session: std::sync::Arc<Session>,
        shape: Vec<usize>,
        dtype: crate::graph::DataType,
        device: crate::tensor::DeviceKind,
    ) -> Result<Self, GraphError> {
        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

        // Create zero-filled tensor on device
        // Currently only supporting f32, will expand to other types
        let total_elements: usize = shape.iter().product();
        let zeros = vec![0.0f32; total_elements.max(1)];

        let value = Value::from_array((shape_i64.as_slice(), zeros))
            .map_err(|e| GraphError::DeviceTensorFailed {
                reason: format!("failed to create device tensor: {e}"),
            })?
            .into();

        Ok(Self {
            value,
            _session: session,
            dtype,
            shape,
            device,
        })
    }

    /// Get a reference to the underlying ORT value
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Get a mutable reference to the underlying ORT value
    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }
}

impl crate::tensor::DeviceTensorBackend for OrtDeviceTensor {
    fn dtype(&self) -> crate::graph::DataType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn device_kind(&self) -> crate::tensor::DeviceKind {
        self.device
    }

    fn backend_kind(&self) -> crate::tensor::BackendKind {
        match self.device {
            crate::tensor::DeviceKind::Cpu => crate::tensor::BackendKind::OnnxCpu,
            crate::tensor::DeviceKind::Cuda => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::DirectML => crate::tensor::BackendKind::OnnxGpu,
            crate::tensor::DeviceKind::CoreML => crate::tensor::BackendKind::OnnxCpu,
        }
    }

    fn read_to_host(&self) -> Result<Vec<f32>, GraphError> {
        // Extract tensor data from device to host
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                let (_, data) = self.value.try_extract_tensor::<f32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f32 tensor from device: {e}"),
                    }
                })?;
                Ok(data.to_vec())
            }
            crate::graph::DataType::Float16 => {
                let (_, data) = self.value.try_extract_tensor::<half::f16>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read f16 tensor from device: {e}"),
                    }
                })?;
                // Convert f16 to f32
                Ok(data.iter().map(|&x| x.to_f32()).collect())
            }
            crate::graph::DataType::Int32 => {
                let (_, data) = self.value.try_extract_tensor::<i32>().map_err(|e| {
                    GraphError::DeviceTensorFailed {
                        reason: format!("failed to read i32 tensor from device: {e}"),
                    }
                })?;
                // Convert i32 to f32
                Ok(data.iter().map(|&x| x as f32).collect())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError> {
        // Write tensor data from host to device
        // Note: ORT doesn't support in-place writes, so we recreate the value
        let expected_size: usize = self.shape.iter().product();
        if data.len() != expected_size {
            return Err(GraphError::DeviceTensorFailed {
                reason: format!(
                    "data size mismatch: expected {} elements, got {}",
                    expected_size,
                    data.len()
                ),
            });
        }

        // Convert shape to i64 for ort compatibility
        let shape_i64: Vec<i64> = self.shape.iter().map(|&d| d as i64).collect();

        // Create new value from host data
        // Currently only supporting f32, will expand to other types
        match self.dtype {
            crate::graph::DataType::Float32 => {
                self.value = Value::from_array((shape_i64.as_slice(), data.to_vec()))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Float16 => {
                // Convert f32 to f16
                let f16_data: Vec<half::f16> =
                    data.iter().map(|&x| half::f16::from_f32(x)).collect();
                self.value = Value::from_array((shape_i64.as_slice(), f16_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write f16 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            crate::graph::DataType::Int32 => {
                // Convert f32 to i32
                let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                self.value = Value::from_array((shape_i64.as_slice(), i32_data))
                    .map_err(|e| GraphError::DeviceTensorFailed {
                        reason: format!("failed to write i32 tensor to device: {e}"),
                    })?
                    .into();
                Ok(())
            }
            _ => Err(GraphError::DeviceTensorFailed {
                reason: format!("unsupported data type for device tensor: {:?}", self.dtype),
            }),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A persistent ONNX Runtime session: model bytes loaded once, reusable across calls.
///
/// Unlike [`run_onnx_with_inputs`] which rebuilds the session on every call,
/// `OrtSession` amortises the startup cost over many inference calls.
pub struct OrtSession {
    session: Session,
    output_names: Vec<String>,
}

impl OrtSession {
    /// Build a session from raw ONNX model bytes.
    ///
    /// `external_weights` is the `.onnx.data` blob when weights are externalised
    /// (as produced by rustnn's ONNX converter for large models).
    pub fn from_model_bytes(
        model_bytes: &[u8],
        external_weights: Option<&[u8]>,
    ) -> Result<Self, GraphError> {
        ensure_ort_initialized()?;
        let mut builder = Session::builder()
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("session builder failed: {e}"),
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("set opt level failed: {e}"),
            })?;
        if let Some(weights) = external_weights {
            builder = builder
                .with_external_initializer_file_in_memory(
                    ONNX_EXTERNAL_WEIGHTS_FILENAME,
                    std::borrow::Cow::Owned(weights.to_vec()),
                )
                .map_err(|e| GraphError::OnnxRuntimeFailed {
                    reason: format!("set external initializer failed: {e}"),
                })?;
        }
        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("load model failed: {e}"),
            })?;
        let output_names = session.outputs().iter().map(|o| o.name().to_string()).collect();
        Ok(Self { session, output_names })
    }

    /// Build a session from a file path, letting ORT resolve external data automatically.
    pub fn from_file(path: &std::path::Path) -> Result<Self, GraphError> {
        ensure_ort_initialized()?;
        let session = Session::builder()
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("session builder failed: {e}"),
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("set opt level failed: {e}"),
            })?
            .commit_from_file(path)
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("load model failed: {e}"),
            })?;
        let output_names = session.outputs().iter().map(|o| o.name().to_string()).collect();
        Ok(Self { session, output_names })
    }

    /// Run one inference pass.
    pub fn run(&mut self, inputs: Vec<OnnxInput>) -> Result<Vec<OnnxOutputWithData>, GraphError> {
        let inputs_by_name: std::collections::HashMap<String, OnnxInput> =
            inputs.into_iter().map(|i| (i.name.clone(), i)).collect();

        let mut input_session_values: Vec<SessionInputValue> = Vec::new();
        for input_info in self.session.inputs().iter() {
            let name = input_info.name().to_string();
            let input = inputs_by_name
                .get(&name)
                .ok_or_else(|| GraphError::OnnxRuntimeFailed {
                    reason: format!("model expects input '{name}' but it was not provided"),
                })?;
            let sv = build_session_input_value(input)?;
            input_session_values.push(sv);
        }

        let outputs = self.session.run(input_session_values.as_slice()).map_err(|e| {
            GraphError::OnnxRuntimeFailed { reason: format!("run failed: {e}") }
        })?;

        extract_output_tensors(outputs, &self.output_names)
    }
}

fn build_session_input_value(input: &OnnxInput) -> Result<SessionInputValue, GraphError> {
    let sv = match &input.data {
        TensorData::Float32(data) => {
            let value = if input.shape.contains(&0) {
                let array = ArrayD::from_shape_vec(IxDyn(&input.shape), data.clone())
                    .map_err(|e| GraphError::OnnxRuntimeFailed {
                        reason: format!("ndarray float32 for {}: {e}", input.name),
                    })?;
                Value::from_array(array)
            } else {
                let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
                Value::from_array((shape_i64.as_slice(), data.clone()))
            }
            .map_err(|e| GraphError::OnnxRuntimeFailed {
                reason: format!("float32 tensor for {}: {e}", input.name),
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Float16(data) => {
            let f16_data: Vec<half::f16> = data.iter().map(|&b| half::f16::from_bits(b)).collect();
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), f16_data)).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("float16 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Int8(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("int8 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Uint8(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("uint8 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Int32(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("int32 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Uint32(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("uint32 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Int64(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("int64 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
        TensorData::Uint64(data) => {
            let shape_i64: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
            let value = Value::from_array((shape_i64.as_slice(), data.clone())).map_err(|e| {
                GraphError::OnnxRuntimeFailed { reason: format!("uint64 tensor for {}: {e}", input.name) }
            })?;
            SessionInputValue::from(value)
        }
    };
    Ok(sv)
}

fn extract_output_tensors(
    outputs: ort::session::SessionOutputs,
    output_names: &[String],
) -> Result<Vec<OnnxOutputWithData>, GraphError> {
    let mut results = Vec::new();
    for (idx, (_key, value)) in outputs.iter().enumerate() {
        let name = output_names.get(idx).cloned().unwrap_or_else(|| format!("output_{idx}"));
        let (shape_vec, data_vec, float32_data, int64_data, uint64_data) =
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                let sv: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let dv: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (sv, dv, Some(data.to_vec()), None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<half::f16>() {
                let sv: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let dv: Vec<f64> = data.iter().map(|&x| x.to_f32() as f64).collect();
                (sv, dv, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                let sv: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let dv: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                (sv, dv, None, None, None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                let sv: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let dv: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let iv = data.to_vec();
                (sv, dv, None, Some(iv), None)
            } else if let Ok((shape, data)) = value.try_extract_tensor::<u64>() {
                let sv: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let dv: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let uv = data.to_vec();
                (sv, dv, None, None, Some(uv))
            } else {
                return Err(GraphError::OnnxRuntimeFailed {
                    reason: format!("unsupported output tensor type for '{name}'"),
                });
            };
        results.push(OnnxOutputWithData { name, shape: shape_vec, data: data_vec, float32_data, int64_data, uint64_data });
    }
    Ok(results)
}

/// Load a `.webnn` graph file, convert to ONNX, and build a persistent [`OrtSession`].
///
/// This is the single entry point for using rustnn's pipeline from external crates: provide a
/// `.webnn` path (with `parakeet.manifest.json` weights alongside), get back a session you
/// can call `run()` on repeatedly.
pub fn load_webnn_as_ort_session(path: &std::path::Path) -> Result<OrtSession, GraphError> {
    let graph = crate::loader::load_graph_from_path(path)?;
    let converted = crate::converters::ConverterRegistry::with_defaults()
        .convert("onnx", &graph)?;
    OrtSession::from_model_bytes(&converted.data, converted.weights_data.as_deref())
}

/// Load a `.webnn` graph file and return the raw ONNX bytes.
///
/// Returns `(onnx_model_bytes, external_weights_bytes)`.  The second element is `Some` when
/// the converter externalised large weight tensors (typical for models > a few hundred MB).
/// Pass both to [`ort::session::Session::builder().commit_from_memory()`] or similar.
///
/// This lets external crates (e.g. `parakeet-rs`) load models through rustnn's graph
/// pipeline while using their own ORT session management.
pub fn load_webnn_as_onnx_bytes(
    path: &std::path::Path,
) -> Result<(Vec<u8>, Option<Vec<u8>>), GraphError> {
    let graph = crate::loader::load_graph_from_path(path)?;
    let converted = crate::converters::ConverterRegistry::with_defaults()
        .convert("onnx", &graph)?;
    Ok((converted.data, converted.weights_data))
}

/// Run ONNX model with device tensor bindings (zero-copy execution)
///
/// This function uses ONNX Runtime IoBinding to execute the model with
/// device-resident tensors, eliminating host-device round-trips.
///
/// Note: This is a placeholder implementation. Full IoBinding support
/// will be added in a future update.
#[allow(dead_code)]
pub fn run_onnx_with_bindings(
    _session: &Session,
    _input_bindings: Vec<(&str, &OrtDeviceTensor)>,
    _output_bindings: Vec<(&str, &mut OrtDeviceTensor)>,
) -> Result<(), GraphError> {
    // Placeholder for future IoBinding implementation
    // Current dispatch() implementation uses regular compute path
    Err(GraphError::DeviceTensorFailed {
        reason: "IoBinding not yet implemented - use dispatch() instead".to_string(),
    })
}

/// Save an MLGraph's ORT backend to disk as an AOT cache for fast subsequent loading.
///
/// Writes the ONNX protobuf to `onnx_path` and the external weights (if any) as
/// `rustnn_external_weights.data` in the same directory. On subsequent runs pass `onnx_path`
/// to `load_ort_graph_cache` — ORT will memory-map the weights rather than reading them into RAM.
pub fn save_ort_graph_cache(
    graph: &crate::MLGraph,
    onnx_path: &std::path::Path,
) -> Result<(), GraphError> {
    use prost::Message;

    let g = graph.backend.as_onnx_session().ok_or_else(|| GraphError::OnnxRuntimeFailed {
        reason: "graph is not an ORT session — cannot save ORT cache".to_string(),
    })?;

    if let Some(weights) = &g.onnx_weights {
        // Use a model-specific weights filename so multiple models can share a directory.
        let stem = onnx_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let weights_filename = format!("{stem}.weights");
        let weights_path = onnx_path.parent().unwrap_or(std::path::Path::new("."))
            .join(&weights_filename);

        // Patch the ONNX protobuf: replace the generic ONNX_EXTERNAL_WEIGHTS_FILENAME
        // with our model-specific name so models in the same directory don't collide.
        let mut model = crate::protos::onnx::ModelProto::decode(g.onnx_data.as_slice())
            .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("parse onnx: {e}") })?;
        if let Some(ref mut graph_proto) = model.graph {
            for init in &mut graph_proto.initializer {
                if init.data_location == 1 {
                    for kv in &mut init.external_data {
                        if kv.key == "location" && kv.value == ONNX_EXTERNAL_WEIGHTS_FILENAME {
                            kv.value = weights_filename.clone();
                        }
                    }
                }
            }
        }
        let patched = model.encode_to_vec();
        std::fs::write(onnx_path, &patched)
            .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("write onnx cache: {e}") })?;
        std::fs::write(&weights_path, weights)
            .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("write weights cache: {e}") })?;
    } else {
        std::fs::write(onnx_path, &g.onnx_data)
            .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("write onnx cache: {e}") })?;
    }
    Ok(())
}

/// Save the compiled (post-optimization) form of the graph for fast subsequent loading.
///
/// For the ORT backend this writes an `.ort` file — subsequent calls to `load_compiled_graph`
/// on that file skip ORT's graph optimization pass entirely, reducing boot time significantly.
/// Other backends will use their own format (CoreML → `.mlmodelc`, RTR-RTX → TBD).
pub fn save_compiled_graph(
    graph: &crate::MLGraph,
    compiled_path: &std::path::Path,
) -> Result<(), GraphError> {
    let g = graph.backend.as_onnx_session().ok_or_else(|| GraphError::OnnxRuntimeFailed {
        reason: "save_compiled_graph: backend is not ORT (CoreML/RTR-RTX not yet implemented)".into(),
    })?;
    // Re-run the session builder with compiled_out so ORT writes the optimized model.
    // We need the original ONNX path — it's the last committed file, which ORT knows internally
    // but doesn't expose. Instead, call with_optimized_model_path on a fresh build from the
    // same data that created this session. Since we stored the ONNX bytes in OrtGraph, use those.
    if g.onnx_data.is_empty() {
        return Err(GraphError::OnnxRuntimeFailed {
            reason: "save_compiled_graph: no ONNX bytes stored (was this loaded from a file cache?)".into(),
        });
    }
    ensure_ort_initialized()?;
    use crate::backends::ort::OrtBuilder;
    // Build a temporary session solely to trigger ORT's optimized model write.
    // Disable EPs that aren't relevant; just need the optimization pass to run.
    let weights_ref = g.onnx_weights.as_deref();
    let mut builder = ort::session::Session::builder()
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("{e}") })?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("{e}") })?
        .with_optimized_model_path(compiled_path)
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("{e}") })?;
    if let Some(w) = weights_ref {
        builder = builder
            .with_external_initializer_file_in_memory(
                ONNX_EXTERNAL_WEIGHTS_FILENAME,
                std::borrow::Cow::Owned(w.to_vec()),
            )
            .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("{e}") })?;
    }
    builder.commit_from_memory(&g.onnx_data)
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("build for compile: {e}") })?;
    Ok(())
}

/// Load a compiled graph produced by `save_compiled_graph`, skipping all compilation steps.
///
/// Detects the backend from the file extension:
/// - `.ort` → ORT compiled model (no graph optimization pass)
/// - `.mlmodelc` → CoreML compiled model (not yet implemented)
///
/// Returns `(MLContext, MLGraph)` ready for `dispatch()`.
pub fn load_compiled_graph(
    compiled_path: &std::path::Path,
    accelerated: bool,
) -> Result<(crate::mlcontext::MLContext<'static>, crate::MLGraph<'static>), GraphError> {
    let name = compiled_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // .opt.onnx = ORT-optimized ONNX (load with Disable to skip re-optimization)
    // .ort      = ORT binary format (if/when supported)
    // .mlmodelc = CoreML compiled (future)
    if name.ends_with(".opt.onnx") || name.ends_with(".ort") {
        load_ort_compiled(compiled_path, accelerated)
    } else {
        Err(GraphError::OnnxRuntimeFailed {
            reason: format!("load_compiled_graph: unsupported format for '{name}' (expected .opt.onnx, .ort, .mlmodelc)"),
        })
    }
}

fn load_ort_compiled(
    compiled_path: &std::path::Path,
    accelerated: bool,
) -> Result<(crate::mlcontext::MLContext<'static>, crate::MLGraph<'static>), GraphError> {
    use crate::backends::ort::{OrtBuilder, OrtGraph};
    use crate::mlcontext::{MLContext, MLContextOptions, MLBackendGraph, MLPowerPreference};
    use crate::graph::{DataType, Dimension, DynamicDimension, GraphInfo, OperandDescriptor};

    ensure_ort_initialized()?;
    let session = OrtBuilder::build_session_from_compiled(compiled_path)
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("load compiled: {e}") })?;

    let (input_descriptors, output_descriptors) = ort_session_io_descriptors(&session);

    let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, accelerated))
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("create context: {e}") })?;

    let mut graph = crate::MLGraph::new(
        MLBackendGraph::OnnxSession(
            OrtGraph { session, onnx_data: Vec::new(), onnx_weights: None },
            std::marker::PhantomData,
        ),
        &GraphInfo::default(),
    ).map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("wrap graph: {e}") })?;

    graph.input_descriptors = input_descriptors;
    graph.output_descriptors = output_descriptors;
    Ok((context, graph))
}

/// Load an MLGraph from an AOT cache file produced by `save_ort_graph_cache`.
///
/// ORT memory-maps the ONNX and discovers `rustnn_external_weights.data` in the same directory
/// — no heap copy of the weight blob. Returns `(MLContext, MLGraph)` ready for `dispatch()`.
pub fn load_ort_graph_cache(
    onnx_path: &std::path::Path,
    accelerated: bool,
) -> Result<(crate::mlcontext::MLContext<'static>, crate::MLGraph<'static>), GraphError> {
    use crate::backends::ort::{OrtBuilder, OrtGraph};
    use crate::mlcontext::{MLContext, MLContextOptions, MLBackendGraph, MLPowerPreference};
    use crate::graph::GraphInfo;

    ensure_ort_initialized()?;

    // If no optimized model exists yet, produce it so the next load skips graph optimization.
    // Use `.opt.onnx` suffix — ORT saves optimized ONNX (smaller search space on reload).
    let compiled_path = {
        let stem = onnx_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        onnx_path.with_file_name(format!("{stem}.opt.onnx"))
    };
    let compiled_out = if compiled_path.exists() { None } else { Some(compiled_path.as_path()) };

    let session = OrtBuilder::build_session_from_file(onnx_path, compiled_out)
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("build session: {e}") })?;

    let (input_descriptors, output_descriptors) = ort_session_io_descriptors(&session);

    let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, accelerated))
        .map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("create context: {e}") })?;

    let mut graph = crate::MLGraph::new(
        MLBackendGraph::OnnxSession(
            OrtGraph { session, onnx_data: Vec::new(), onnx_weights: None },
            std::marker::PhantomData,
        ),
        &GraphInfo::default(),
    ).map_err(|e| GraphError::OnnxRuntimeFailed { reason: format!("wrap graph: {e}") })?;

    graph.input_descriptors = input_descriptors;
    graph.output_descriptors = output_descriptors;
    Ok((context, graph))
}

/// Extract input/output descriptors from an ORT session's type info.
fn ort_session_io_descriptors(
    session: &ort::session::Session,
) -> (std::collections::HashMap<String, crate::graph::OperandDescriptor>,
      std::collections::HashMap<String, crate::graph::OperandDescriptor>)
{
    use crate::graph::{DataType, Dimension, DynamicDimension, OperandDescriptor};
    fn ort_ty(ty: ort::value::TensorElementType) -> DataType {
        use ort::value::TensorElementType;
        match ty {
            TensorElementType::Float32 => DataType::Float32,
            TensorElementType::Float16 => DataType::Float16,
            TensorElementType::Int32   => DataType::Int32,
            TensorElementType::Uint32  => DataType::Uint32,
            TensorElementType::Int64   => DataType::Int64,
            TensorElementType::Uint64  => DataType::Uint64,
            TensorElementType::Int8    => DataType::Int8,
            TensorElementType::Uint8   => DataType::Uint8,
            _                          => DataType::Float32,
        }
    }
    fn ort_dims(shape: &[i64]) -> Vec<Dimension> {
        shape.iter().map(|&d| if d >= 0 {
            Dimension::Static(d as u32)
        } else {
            Dimension::Dynamic(DynamicDimension { max_size: 0, name: String::new() })
        }).collect()
    }
    let mut inputs = std::collections::HashMap::new();
    let mut outputs = std::collections::HashMap::new();
    for o in session.inputs().iter() {
        if let ort::value::ValueType::Tensor { ty, shape, .. } = o.dtype() {
            inputs.insert(o.name().to_string(), OperandDescriptor {
                data_type: ort_ty(*ty), shape: ort_dims(shape), pending_permutation: vec![],
            });
        }
    }
    for o in session.outputs().iter() {
        if let ort::value::ValueType::Tensor { ty, shape, .. } = o.dtype() {
            outputs.insert(o.name().to_string(), OperandDescriptor {
                data_type: ort_ty(*ty), shape: ort_dims(shape), pending_permutation: vec![],
            });
        }
    }
    (inputs, outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_device_tensor_lifecycle() {
        // This test requires ONNX Runtime to be initialized
        // Skip if initialization fails
        if ensure_ort_initialized().is_err() {
            return;
        }

        // Create a simple ONNX model (add operation: y = x + 1)
        // For now, we'll skip this test as it requires a real ONNX model
        // Will be added when we have a test model available
    }
}
