//! First real end-to-end generation on the Hexagon DSP.
//!
//! ```text
//! adb push model-pack /data/local/tmp/model-pack
//! adb shell ADSP_LIBRARY_PATH=/data/local/tmp/lfmgemv \
//!     /data/local/tmp/lfmgemv/dsp-generate /data/local/tmp/model-pack \
//!     --tokens /data/local/tmp/model-pack/prompt.txt --max-tokens 24
//! ```
//!
//! The binary reads a pack produced by `examples/make_model_pack.rs`, stages
//! every int4 weight into ION (`alloc_in`, zero-copy), attaches the host
//! embedding/norms, and runs a greedy prefill + decode loop. The prompt tokens
//! are produced on the host by `host_generate --tokens-out`.
//!
//! Diagnostics that matter for the first-text milestone:
//!
//! * `FIRST_TOKEN` is the argmax after the prompt, compared on the host with
//!   the CPU backend;
//! * `--dump-first-layers` writes the residual stream after every layer of the
//!   first token, so a divergence can be bisected layer by layer;
//! * `--host-head` switches the vocabulary projection to the exact f32 path,
//!   isolating a DSP-layer error from int4-head quantisation.

#[cfg(target_os = "android")]
fn main() {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use myelin_core::backend::dsp::DspExecutor;
    use myelin_core::backend::HostWeights;
    use myelin_core::ops::argmax;
    use myelin_core::{DspBackend, Model, ModelSpec};
    use myelin_npu::model_pack::ModelPack;
    use myelin_npu::weights::{ResidentWeights, WeightBudget};
    use myelin_npu::{Cdsprpc, CdsprpcConfig, Executor, HvxClient, Transport};

    struct Args {
        pack: PathBuf,
        tokens: PathBuf,
        max_tokens: usize,
        context: usize,
        budget_mb: usize,
        host_head: bool,
        eos: Option<u32>,
        dump_layers: Option<PathBuf>,
        no_reset_check: bool,
    }

    fn parse_args() -> Result<Args, String> {
        let mut args = std::env::args().skip(1);
        let pack = PathBuf::from(
            args.next()
                .ok_or("usage: dsp-generate PACK_DIR --tokens FILE")?,
        );
        let mut tokens = None;
        let mut out = Args {
            pack,
            tokens: PathBuf::new(),
            max_tokens: 24,
            context: 256,
            budget_mb: 1024,
            host_head: false,
            eos: None,
            dump_layers: None,
            no_reset_check: false,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--tokens" => tokens = Some(PathBuf::from(value()?)),
                "--max-tokens" => {
                    out.max_tokens = value()?
                        .parse()
                        .map_err(|_| "--max-tokens must be an integer".to_owned())?;
                }
                "--context" => {
                    out.context = value()?
                        .parse()
                        .map_err(|_| "--context must be an integer".to_owned())?;
                }
                "--budget-mb" => {
                    out.budget_mb = value()?
                        .parse()
                        .map_err(|_| "--budget-mb must be an integer".to_owned())?;
                }
                "--host-head" => out.host_head = true,
                "--eos" => {
                    out.eos = Some(
                        value()?
                            .parse()
                            .map_err(|_| "--eos must be a token id".to_owned())?,
                    );
                }
                "--dump-first-layers" => out.dump_layers = Some(PathBuf::from(value()?)),
                "--no-reset-check" => out.no_reset_check = true,
                other => return Err(format!("unknown flag {other}")),
            }
        }
        if out.max_tokens == 0 {
            return Err("--max-tokens must be positive".to_owned());
        }
        out.tokens = tokens.ok_or("--tokens is required")?;
        Ok(out)
    }

    fn read_tokens(path: &Path) -> Vec<u32> {
        let text = std::fs::read_to_string(path).unwrap_or_else(|error| {
            panic!("{}: {error}", path.display());
        });
        text.split(|character: char| !character.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().expect("token id"))
            .collect()
    }

    fn run(args: Args) -> Result<(), String> {
        let pack = ModelPack::read(&args.pack).map_err(|error| error.to_string())?;
        let spec: ModelSpec = pack.spec();
        spec.validate().map_err(|error| error.to_string())?;
        let prompt = read_tokens(&args.tokens);
        if prompt.is_empty() {
            return Err("prompt token file is empty".to_owned());
        }
        if prompt.len() > args.context {
            return Err(format!(
                "prompt has {} tokens, context is {}",
                prompt.len(),
                args.context
            ));
        }
        let budget = WeightBudget::new(args.budget_mb << 20);
        let started = Instant::now();
        let plan = pack
            .weight_plan(&args.pack, args.context, budget)
            .map_err(|error| error.to_string())?;
        println!(
            "plan: {:.1} MiB resident (budget {:.1} MiB), {} layers, context {} ({} pages)",
            plan.total_bytes() as f64 / (1u64 << 20) as f64,
            budget.bytes() as f64 / (1u64 << 20) as f64,
            spec.dims.layers,
            args.context,
            args.context.div_ceil(myelin_core::kv::PAGE_TOKENS)
        );
        let transport: Arc<dyn Transport> =
            Arc::new(Cdsprpc::load(CdsprpcConfig::default()).map_err(|error| error.to_string())?);
        let client = HvxClient::open(Arc::clone(&transport), "lfmgemv")
            .map_err(|error| error.to_string())?;
        let handle = client.handle().expect("open handle");
        // The backend owns the handle from here; the client must not close it.
        std::mem::forget(client);

        let executor = Arc::new(Executor::new(Arc::clone(&transport)));
        let exec_dyn: Arc<dyn DspExecutor> = executor;
        let provider =
            ResidentWeights::with_ion(Arc::clone(&exec_dyn), handle, plan, Arc::clone(&transport));
        let mut backend =
            DspBackend::with_handle(spec.clone(), Arc::clone(&exec_dyn), handle, "lfmgemv");
        let host = pack
            .host_weights(&args.pack)
            .map_err(|error| error.to_string())?;
        let host: HostWeights = if args.host_head {
            let mut host = host;
            host.set_head_mode(myelin_core::backend::HeadMode::Host);
            host
        } else {
            host
        };
        backend
            .attach_host(host)
            .map_err(|error| error.to_string())?;
        backend.attach_weights(Box::new(provider));
        let mut model = Model::from_backend(spec.clone(), Box::new(backend))
            .map_err(|error| error.to_string())?;
        let backend_name = model.backend_name();
        let mut session = model
            .session(args.context)
            .map_err(|error| error.to_string())?;
        println!(
            "model ready in {:.2}s (backend {}), prompt {} tokens",
            started.elapsed().as_secs_f64(),
            backend_name,
            prompt.len()
        );

        let mut dump_file = args.dump_layers.as_ref().map(|path| {
            std::fs::File::create(path).unwrap_or_else(|error| {
                panic!("{}: {error}", path.display());
            })
        });
        let mut layer_gaps: Vec<f64> = Vec::with_capacity(spec.dims.layers * prompt.len());
        let prefill_started = Instant::now();
        let mut logits = {
            let mut last = Instant::now();
            let mut hook = |layer: usize, values: &[f32]| {
                let now = Instant::now();
                layer_gaps.push(now.duration_since(last).as_secs_f64() * 1e3);
                last = now;
                if let Some(file) = dump_file.as_mut() {
                    let mut bytes = Vec::with_capacity(values.len() * 4);
                    for value in values {
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    file.write_all(&bytes).expect("write layer dump");
                }
                let _ = layer;
            };
            match session.prefill_with(&prompt, &mut hook) {
                Ok(logits) => logits,
                Err(error) => {
                    if let Some(detail) = session.backend_diagnostic() {
                        eprintln!("backend: {detail}");
                    }
                    return Err(error.to_string());
                }
            }
        };
        let prefill = prefill_started.elapsed().as_secs_f64();
        if let Some(file) = dump_file.as_mut() {
            file.flush().expect("flush layer dump");
        }
        let first = argmax(&logits) as u32;
        println!("FIRST_TOKEN {first}");
        let mean_layer = if layer_gaps.is_empty() {
            0.0
        } else {
            layer_gaps.iter().sum::<f64>() / layer_gaps.len() as f64
        };
        println!(
            "PREFILL {:.1} ms ({} tokens, {:.1} ms/token, mean layer {:.2} ms)",
            prefill * 1e3,
            prompt.len(),
            prefill * 1e3 / prompt.len() as f64,
            mean_layer
        );

        let decode_started = Instant::now();
        let mut generated = Vec::with_capacity(args.max_tokens);
        let mut token = first;
        let mut stop = false;
        let mut decode_times = Vec::with_capacity(args.max_tokens);
        for _ in 0..args.max_tokens {
            generated.push(token);
            if args.eos == Some(token) {
                stop = true;
                break;
            }
            let step_started = Instant::now();
            let mut noop = |_: usize, _: &[f32]| {};
            logits = match session.decode_with(token, &mut noop) {
                Ok(logits) => logits,
                Err(error) => {
                    if let Some(detail) = session.backend_diagnostic() {
                        eprintln!("backend: {detail}");
                    }
                    return Err(error.to_string());
                }
            };
            decode_times.push(step_started.elapsed().as_secs_f64() * 1e3);
            token = argmax(&logits) as u32;
        }
        let decode = decode_started.elapsed().as_secs_f64();
        if let Some(file) = dump_file.as_mut() {
            file.flush().expect("flush layer dump");
        }

        println!(
            "TOKENS {}",
            generated
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!(
            "DECODE {:.1} ms for {} generated tokens ({:.2} ms/token, {:.2} tok/s)",
            decode * 1e3,
            generated.len(),
            decode * 1e3 / generated.len().max(1) as f64,
            generated.len() as f64 / decode.max(1e-9)
        );
        if !decode_times.is_empty() {
            println!(
                "DECODE_STEPS {}",
                decode_times
                    .iter()
                    .map(|time| format!("{time:.1}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        println!(
            "TOTAL {:.2} tok/s over {} tokens",
            (prompt.len() + generated.len()) as f64 / (prefill + decode).max(1e-9),
            prompt.len() + generated.len()
        );
        println!("STOP {}", if stop { "eos" } else { "length" });

        if !args.no_reset_check {
            session.reset().map_err(|error| error.to_string())?;
            let mut noop = |_: usize, _: &[f32]| {};
            let logits = session
                .prefill_with(&prompt, &mut noop)
                .map_err(|error| error.to_string())?;
            let again = argmax(&logits) as u32;
            println!(
                "RESET_CHECK {} (first {first}, after reset {again})",
                if again == first { "PASS" } else { "FAIL" }
            );
        }
        Ok(())
    }

    if let Err(error) = run(parse_args().expect("arguments")) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "android"))]
fn main() {
    eprintln!("dsp-generate runs on Android only (needs libcdsprpc.so)");
}
