//! Host-side FastRPC cost benchmark.
//!
//! On the device this opens `libcdsprpc.so`, enables unsigned PD and measures
//! an empty RPC (`team_info`) plus `load_buf` with 1 KiB, 256 KiB, 1 MiB and
//! 4 MiB payloads. The last two bracket the weight-loader chunk choice.
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
    let mut client = match HvxClient::open(Arc::clone(&transport), &options.module) {
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

    let resident = match client.alloc_buf(8 << 20) {
        Ok(handle) => Some(handle),
        Err(error) => {
            eprintln!("alloc_buf(8 MiB) failed: {error}");
            None
        }
    };
    if let Some(handle) = resident {
        let payload_1k = vec![0x5au8; 1 << 10];
        let payload_256k = vec![0x33u8; 256 << 10];
        let payload_1m = vec![0xa5u8; 1 << 20];
        let payload_4m = vec![0x77u8; 4 << 20];
        for (label, payload) in [
            ("load_buf 1 KiB", &payload_1k),
            ("load_buf 256 KiB", &payload_256k),
            ("load_buf 1 MiB", &payload_1m),
            ("load_buf 4 MiB", &payload_4m),
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

    // O1 measurement: the Rust `exec` path with a stable command list. The
    // difference between wall and `times[0]` is everything the host and the
    // FastRPC driver spend per call around the DSP commands.
    if let Some(handle) = client.handle() {
        if let Err(error) = bench_exec(&transport, handle, options.iterations) {
            eprintln!("exec bench failed: {error}");
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

/// One-command `exec` through the Rust executor: the cheapest possible list.
///
/// Prints wall min/median/p95 and the DSP microseconds reported by the
/// skeleton, so the difference is the per-call host/driver cost.
fn bench_exec(
    transport: &Arc<dyn Transport>,
    handle: myelin_core::backend::dsp::DspHandle,
    iterations: usize,
) -> Result<(), String> {
    use myelin_core::backend::dsp::{exec, DspExecutor, ExecRun};

    let executor = myelin_npu::Executor::new(Arc::clone(transport));
    executor
        .reserve_scratch(handle, 1 << 20)
        .map_err(|error| format!("reserve_scratch: {error}"))?;
    let rt = [0i32; 4];
    // 1, 8 and 32 commands: the overhead should stay constant if it is
    // per-call, and grow if the uncounted part is DSP-time dependent.
    for commands in [1usize, 8, 32] {
        let mut list = exec::List::new(1 << 20);
        for index in 0..commands {
            list.push(
                exec::Command::quant(0, index as i32 * 256, 256, 4096)
                    .with_dst(exec::dst::OUT, index as i32 * 256),
            );
        }
        let encoded = list.to_bytes();
        let act = vec![0u8; 256 * 4];
        let out_len = commands * 256;
        for (label, rt) in [
            (format!("exec QUANT x{commands} +rt"), &rt[..]),
            (format!("exec QUANT x{commands}"), &rt[..0]),
        ] {
            let mut wall = Vec::with_capacity(iterations);
            let mut dsp = Vec::with_capacity(iterations);
            for _ in 0..3 {
                executor
                    .exec(
                        handle,
                        ExecRun {
                            list: &encoded,
                            commands: list.commands.len(),
                            scratch_bytes: list.scratch_bytes,
                            act: &act,
                            rt,
                            out_len,
                        },
                    )
                    .map_err(|error| format!("exec: {error}"))?;
            }
            for _ in 0..iterations {
                let start = Instant::now();
                let result = executor
                    .exec(
                        handle,
                        ExecRun {
                            list: &encoded,
                            commands: list.commands.len(),
                            scratch_bytes: list.scratch_bytes,
                            act: &act,
                            rt,
                            out_len,
                        },
                    )
                    .map_err(|error| format!("exec: {error}"))?;
                wall.push(start.elapsed().as_secs_f64() * 1e6);
                dsp.push(f64::from(result.times[0]));
            }
            wall.sort_by(f64::total_cmp);
            dsp.sort_by(f64::total_cmp);
            let median = wall[wall.len() / 2];
            let dsp_median = dsp[dsp.len() / 2];
            let p95 = wall[(wall.len() * 95 / 100).min(wall.len() - 1)];
            println!(
                "{label:<24} min {:9.1} us  median {median:9.1} us  p95 {p95:9.1} us  \
             dsp {dsp_median:7.1} us  overhead {:7.1} us",
                wall[0],
                median - dsp_median
            );
        }
    }

    // Does the host gap between two execs cost extra? The model does 50-200
    // microseconds of host work between calls; `rpc_test`/`exec_test` run
    // back-to-back. A 1 ms sleep makes the difference visible.
    {
        let mut list = exec::List::new(1 << 20);
        list.push(exec::Command::quant(0, 0, 256, 4096).with_dst(exec::dst::OUT, 0));
        let encoded = list.to_bytes();
        let act = vec![0u8; 256 * 4];
        let mut wall = Vec::with_capacity(iterations);
        let mut dsp = Vec::with_capacity(iterations);
        for _ in 0..3 {
            executor
                .exec(
                    handle,
                    ExecRun {
                        list: &encoded,
                        commands: 1,
                        scratch_bytes: list.scratch_bytes,
                        act: &act,
                        rt: &[],
                        out_len: 256,
                    },
                )
                .map_err(|error| format!("exec: {error}"))?;
        }
        for _ in 0..iterations {
            std::thread::sleep(std::time::Duration::from_millis(1));
            let start = Instant::now();
            let result = executor
                .exec(
                    handle,
                    ExecRun {
                        list: &encoded,
                        commands: 1,
                        scratch_bytes: list.scratch_bytes,
                        act: &act,
                        rt: &[],
                        out_len: 256,
                    },
                )
                .map_err(|error| format!("exec: {error}"))?;
            wall.push(start.elapsed().as_secs_f64() * 1e6);
            dsp.push(f64::from(result.times[0]));
        }
        wall.sort_by(f64::total_cmp);
        dsp.sort_by(f64::total_cmp);
        let median = wall[wall.len() / 2];
        println!(
            "{:<24} median {median:9.1} us  dsp {:7.1} us  (1 ms sleep between calls, not counted)",
            "exec QUANT gap 1ms",
            dsp[dsp.len() / 2]
        );
    }
    Ok(())
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
         Set ADSP_LIBRARY_PATH to exactly one directory holding the skeleton;\n\
         a colon-separated list fails with AEE_EUNABLETOLOAD on the XQ-BC72."
    );
}
