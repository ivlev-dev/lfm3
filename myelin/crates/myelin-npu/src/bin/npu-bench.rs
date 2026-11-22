//! Host-side FastRPC cost benchmark.
//!
//! On the device this opens `libcdsprpc.so`, enables unsigned PD and measures
//! an empty RPC (`team_info`) plus `load_buf` with 1 KiB and 1 MiB payloads.
//! On a host build it uses [`MockTransport`]; the numbers then describe Rust
//! call overhead, not FastRPC, and the tool says so.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use myelin_npu::{HvxClient, MockTransport, Transport, TransportError};

struct Options {
    mock: bool,
    iterations: usize,
    module: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mock: false,
            iterations: 31,
            module: "lfmgemv".to_owned(),
        }
    }
}

fn main() -> ExitCode {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mock" => options.mock = true,
            "--iterations" | "-n" => {
                let Some(value) = args.next() else {
                    eprintln!("--iterations needs a value");
                    return ExitCode::from(2);
                };
                match value.parse::<usize>() {
                    Ok(value) if value > 0 => options.iterations = value,
                    _ => {
                        eprintln!("invalid iteration count: {value}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--module" => {
                let Some(value) = args.next() else {
                    eprintln!("--module needs a value");
                    return ExitCode::from(2);
                };
                options.module = value;
            }
            "--help" | "-h" => {
                usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                usage();
                return ExitCode::from(2);
            }
        }
    }

    let transport = select_transport(&options);
    let mut client = match HvxClient::open(transport, &options.module) {
        Ok(client) => client,
        Err(error) => {
            eprintln!("open {} failed: {error}", options.module);
            return ExitCode::FAILURE;
        }
    };

    println!(
        "module={} iterations={} transport={}",
        options.module,
        options.iterations,
        if options.mock { "mock" } else { "auto" }
    );

    let mut failure = false;
    if let Err(error) = bench("empty RPC (team_info)", options.iterations, || {
        client.team_info(8).map(|_| ())
    }) {
        eprintln!("empty RPC failed: {error}");
        failure = true;
    }

    let resident = match client.alloc_buf(1 << 20) {
        Ok(handle) => Some(handle),
        Err(error) => {
            eprintln!("alloc_buf(1 MiB) failed: {error}");
            None
        }
    };
    if let Some(handle) = resident {
        let payload_1k = vec![0x5au8; 1 << 10];
        let payload_1m = vec![0xa5u8; 1 << 20];
        for (label, payload) in [
            ("load_buf 1 KiB", &payload_1k),
            ("load_buf 1 MiB", &payload_1m),
        ] {
            if let Err(error) = bench(label, options.iterations, || {
                client.load_buf(handle, 0, payload)
            }) {
                eprintln!("{label} failed: {error}");
                failure = true;
            }
        }
        if let Err(error) = client.release_buf(handle) {
            eprintln!("release_buf failed: {error}");
            failure = true;
        }
    }
    if let Err(error) = client.close() {
        eprintln!("close failed: {error}");
        failure = true;
    }

    if failure {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn bench(
    label: &str,
    iterations: usize,
    mut call: impl FnMut() -> Result<(), TransportError>,
) -> Result<(), TransportError> {
    for _ in 0..3 {
        call()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        call()?;
        samples.push(start.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    println!("{label:<24} min {min:9.1} us  median {median:9.1} us  p95 {p95:9.1} us");
    Ok(())
}

#[cfg(target_os = "android")]
fn select_transport(options: &Options) -> Arc<dyn Transport> {
    use myelin_npu::{Cdsprpc, CdsprpcConfig};

    if options.mock {
        return Arc::new(MockTransport::new());
    }
    let config = CdsprpcConfig::default();
    match Cdsprpc::load(config) {
        Ok(client) => Arc::new(client),
        Err(error) => {
            eprintln!("FastRPC is unavailable: {error}");
            eprintln!("falling back to MockTransport");
            Arc::new(MockTransport::new())
        }
    }
}

#[cfg(not(target_os = "android"))]
fn select_transport(options: &Options) -> Arc<dyn Transport> {
    if !options.mock {
        eprintln!(
            "host build: libcdsprpc.so is Android-only, using MockTransport \
             (numbers are not device RPC cost)"
        );
    }
    Arc::new(MockTransport::new())
}

fn usage() {
    println!(
        "usage: npu-bench [--mock] [--iterations N] [--module NAME]\n\
         \n\
         Runs on the device with libcdsprpc.so and liblfmgemv_skel.so deployed.\n\
         Set ADSP_LIBRARY_PATH to the directory holding the skeleton."
    );
}
