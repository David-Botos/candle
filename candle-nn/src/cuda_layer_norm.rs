// src/cuda_layer_norm.rs
use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_ln(
        x: *const c_void,
        residual: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        dst_add: *const c_void,
        dst: *const c_void,
        mu: *const c_void,
        rsigma: *const c_void,

        epsilon: f32,

        hidden_size_rounded: u32,
        rows: u32,
        cols: u32,
        multi_processor_count: i32,

        wtype: u32,
        itype: u32,
        rtype: u32,
        otype: u32,
        ctype: u32,

        is_rms_norm: c_int,
    );
}

pub(crate) fn layer_norm_internal_type(dtype: candle::DType) -> candle::Result<u32> {
    let internal_type = match dtype {
        candle::DType::F16 => 0,
        candle::DType::BF16 => 1,
        candle::DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    Ok(internal_type)
}

pub(crate) fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}
