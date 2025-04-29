use std::path::PathBuf;
use anyhow::Context;

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=cuda-kernels/layer_norm/ln_api.cu");
        println!("cargo:rerun-if-changed=cuda-kernels/layer_norm/*.cuh");
        println!("cargo:rerun-if-changed=cuda-kernels/layer_norm/*.h");
        compile_cuda_kernels()?;
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() -> anyhow::Result<()> {
    use std::process::Command;
    
    // Set CUDA include directory
    set_cuda_include_dir()?;
    
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").with_context(|| "OUT_DIR not set")?);
    let compute_cap = compute_cap()?;
    
    // Compile ln_api.cu
    let source_file = "cuda-kernels/layer_norm/ln_api.cu";
    let obj_file = out_dir.join("ln_api.o");
    
    let mut command = Command::new("nvcc");
    command
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("-U__CUDA_NO_BFLOAT162_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT162_CONVERSIONS__")
        .arg(format!("--gpu-architecture=sm_{compute_cap}"))
        .arg("-c")
        .args(["-o", obj_file.to_str().unwrap()])
        .args(["--default-stream", "per-thread"])
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg(source_file);
    
    let output = command
        .spawn()
        .with_context(|| "failed spawning nvcc")?
        .wait_with_output()?;
    
    if !output.status.success() {
        anyhow::bail!(
            "nvcc error while compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
            &command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    
    // Create static library
    let lib_file = out_dir.join("liblayernorm.a");
    let mut command = Command::new("nvcc");
    command
        .arg("--lib")
        .args(["-o", lib_file.to_str().unwrap()])
        .arg(&obj_file);
    
    let output = command
        .spawn()
        .with_context(|| "failed spawning nvcc")?
        .wait_with_output()?;
    
    if !output.status.success() {
        anyhow::bail!(
            "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
            &command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=layernorm");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn set_cuda_include_dir() -> anyhow::Result<()> {
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];
    let roots = roots.into_iter().map(Into::<PathBuf>::into);
    let root = env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
        .with_context(|| "cannot find include/cuda.h")?;
    println!(
        "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
        root.join("include").display()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn compute_cap() -> anyhow::Result<usize> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let mut compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .with_context(|| "Could not parse code")?
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=compute_cap")
            .arg("--format=csv")
            .output()
            .with_context(|| "`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.")?;
        let out = std::str::from_utf8(&out.stdout).with_context(|| "stdout is not a utf8 string")?;
        let mut lines = out.lines();
        assert_eq!(
            lines.next().with_context(|| "missing line in stdout")?,
            "compute_cap"
        );
        let cap = lines
            .next()
            .with_context(|| "missing line in stdout")?
            .replace('.', "");
        let cap = cap
            .parse::<usize>()
            .with_context(|| format!("cannot parse as int {cap}"))?;
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
            .arg("--list-gpu-code")
            .output()
            .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).unwrap();

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().unwrap();
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute cap
    if !supported_nvcc_codes.contains(&compute_cap) {
        anyhow::bail!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        anyhow::bail!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
