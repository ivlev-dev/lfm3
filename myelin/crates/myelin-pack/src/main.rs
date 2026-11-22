//! `myelin-pack` command line front end.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use half::f16;

use myelin_pack::kvarn::{self, KvarnFormat, KvarnManifest};
use myelin_pack::kvpage::{self, KvLayout, KvPackParams, VScaleMode};
use myelin_pack::shortconv::{self, ActivationScales, ConvManifest};
use myelin_pack::{
    gemv, gguf_source, metrics, pack_gguf, unpack, DType, Manifest, PackOptions, QualityStats,
    Result,
};

#[derive(Debug, Parser)]
#[command(
    name = "myelin-pack",
    version,
    about = "Pack GGUF tensors into DSP weight formats and verify the accuracy"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List tensors of a GGUF file.
    List {
        /// Path to the GGUF file.
        #[arg(long)]
        model: PathBuf,
        /// Only print tensors whose name contains this substring.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Quantize one tensor, write the blob and its manifest, report quality.
    Pack {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: String,
        /// Binary crownstone blob to write.
        #[arg(long)]
        out: PathBuf,
        /// JSON manifest to write.
        #[arg(long)]
        manifest: PathBuf,
        /// Optional JSON quality report with per-channel statistics.
        #[arg(long)]
        quality_json: Option<PathBuf>,
        /// Read only the first N output channels (spot checks on big tensors).
        #[arg(long)]
        rows: Option<usize>,
        /// Minimum acceptable cosine similarity of the whole tensor.
        #[arg(long, default_value_t = 0.999)]
        min_cos: f64,
        /// Optional floor for the worst output channel (off by default: a
        /// channel dominated by a few outliers can sit below the tensor mean).
        #[arg(long)]
        min_channel_cos: Option<f64>,
    },
    /// Re-check a blob against its manifest and the original GGUF tensor.
    Verify {
        #[arg(long)]
        blob: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 0.999)]
        min_cos: f64,
        #[arg(long)]
        min_channel_cos: Option<f64>,
    },
    /// Pack one LFM3 shortconv layer into the DSP `CONV` blob.
    Conv {
        /// GGUF file; with `--layer` selects `blk.N.shortconv.conv.weight`.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Layer index whose shortconv taps are packed.
        #[arg(long)]
        layer: Option<usize>,
        /// Explicit GGUF tensor name instead of `blk.<layer>.shortconv.conv.weight`.
        #[arg(long)]
        tensor: Option<String>,
        /// Raw little-endian f32 taps (`H * 7` values), an alternative to GGUF.
        #[arg(long)]
        taps: Option<PathBuf>,
        /// Blob to write (`8*H*8 + 4*H` bytes).
        #[arg(long)]
        out: PathBuf,
        /// JSON manifest to write.
        #[arg(long)]
        manifest: PathBuf,
        /// Calibrated `s_bc` (in_proj/int8 activation scale).
        #[arg(long)]
        s_bc: Option<f32>,
        /// Calibrated `s_y` (conv output int8 scale).
        #[arg(long)]
        s_y: Option<f32>,
        /// Write the f32 taps that were packed (input of the C reference).
        #[arg(long)]
        taps_out: Option<PathBuf>,
        /// Optional JSON quality report with per-channel statistics.
        #[arg(long)]
        quality_json: Option<PathBuf>,
        /// Minimum acceptable cosine similarity of the tap table.
        #[arg(long, default_value_t = 0.99)]
        min_cos: f64,
    },
    /// Re-check a shortconv blob against its manifest and (optionally) a source.
    ConvVerify {
        #[arg(long)]
        blob: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        layer: Option<usize>,
        #[arg(long)]
        tensor: Option<String>,
        #[arg(long)]
        taps: Option<PathBuf>,
        #[arg(long, default_value_t = 0.99)]
        min_cos: f64,
    },
    /// Pack KV pages into the DSP `ATTN` images or the kvarn reference codec.
    Kv {
        /// Raw little-endian f32 values, `[head][page][token][dim]`.
        #[arg(long = "in")]
        input: Option<PathBuf>,
        /// GGUF file whose tensor supplies the values (prefix of the tensor).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Tensor used with `--model`.
        #[arg(long)]
        tensor: Option<String>,
        /// `dsp` (the kernel format) or `kvarn-k4v2` / `kvarn-k4v4`.
        #[arg(long, default_value = "dsp")]
        format: String,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 1)]
        pages: usize,
        /// Valid tokens in the last page (the rest is zero padded).
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        /// `row` (head-grouped, what `exec` uses) or `page` (token-major).
        #[arg(long, default_value = "row")]
        layout: String,
        /// `global` (one scalar) or `channel` (per head and dim).
        #[arg(long, default_value = "channel")]
        v_scale: String,
        /// K image (`dsp` format only).
        #[arg(long)]
        out_k: Option<PathBuf>,
        /// Per-token K scales (`dsp` format only).
        #[arg(long)]
        out_kt: Option<PathBuf>,
        /// V image (`dsp` format only).
        #[arg(long)]
        out_v: Option<PathBuf>,
        /// Encoded page (`kvarn` formats only).
        #[arg(long)]
        out: Option<PathBuf>,
        /// JSON manifest to write.
        #[arg(long)]
        manifest: PathBuf,
        /// Write the f32 values that were packed (input of `kv-verify --in`).
        #[arg(long)]
        values_out: Option<PathBuf>,
        /// kvarn: input is already rotated (`lfm_tq_rot_fwd` was applied).
        #[arg(long)]
        no_rotate: bool,
        #[arg(long)]
        quality_json: Option<PathBuf>,
        #[arg(long, default_value_t = 0.99)]
        min_cos: f64,
    },
    /// Re-check KV images against their manifest and (optionally) a source.
    KvVerify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        k: Option<PathBuf>,
        #[arg(long)]
        kt: Option<PathBuf>,
        #[arg(long)]
        v: Option<PathBuf>,
        /// kvarn: encoded page file.
        #[arg(long)]
        page: Option<PathBuf>,
        /// Raw f32 source to re-pack and compare byte for byte.
        #[arg(long = "in")]
        input: Option<PathBuf>,
        #[arg(long, default_value_t = 0.99)]
        min_cos: f64,
    },
    /// Decode KV images back to raw f32.
    KvUnpack {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        k: Option<PathBuf>,
        #[arg(long)]
        kt: Option<PathBuf>,
        #[arg(long)]
        v: Option<PathBuf>,
        #[arg(long)]
        page: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Pack a GGUF tensor into the int4 `GEMV` weight format.
    Gemv {
        #[arg(long)]
        model: PathBuf,
        /// Explicit GGUF tensor name.
        #[arg(long)]
        tensor: Option<String>,
        /// Layer index for `--role` (e.g. `2` for `blk.2.attn_q.weight`).
        #[arg(long)]
        layer: Option<usize>,
        /// `in_proj`, `out_proj`, `attn_q`, `attn_k`, `attn_v`, `attn_o`,
        /// `attn_gate`.
        #[arg(long)]
        role: Option<String>,
        /// Binary int4 blob to write.
        #[arg(long)]
        out: PathBuf,
        /// JSON manifest to write.
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        quality_json: Option<PathBuf>,
        /// Read only the first N output rows (spot checks on big tensors).
        #[arg(long)]
        rows: Option<usize>,
        #[arg(long, default_value_t = 0.98)]
        min_cos: f64,
        /// Optional floor for the worst output row.
        #[arg(long)]
        min_row_cos: Option<f64>,
    },
    /// Re-check an int4 GEMV blob against its manifest and the GGUF tensor.
    GemvVerify {
        #[arg(long)]
        blob: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tensor: Option<String>,
        #[arg(long, default_value_t = 0.98)]
        min_cos: f64,
        #[arg(long)]
        min_row_cos: Option<f64>,
    },
    /// Decode an int4 GEMV blob back to raw f32 (`[n][k]`).
    GemvUnpack {
        #[arg(long)]
        blob: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::List { model, filter } => list(&model, filter.as_deref()),
        Command::Pack {
            model,
            tensor,
            out,
            manifest,
            quality_json,
            rows,
            min_cos,
            min_channel_cos,
        } => pack(&PackArgs {
            model,
            tensor,
            out,
            manifest,
            quality_json,
            rows,
            min_cos,
            min_channel_cos,
        }),
        Command::Verify {
            blob,
            manifest,
            model,
            min_cos,
            min_channel_cos,
        } => verify(&blob, &manifest, &model, min_cos, min_channel_cos),
        Command::Conv {
            model,
            layer,
            tensor,
            taps,
            out,
            manifest,
            s_bc,
            s_y,
            taps_out,
            quality_json,
            min_cos,
        } => conv_pack(ConvArgs {
            model,
            layer,
            tensor,
            taps,
            out,
            manifest,
            s_bc,
            s_y,
            taps_out,
            quality_json,
            min_cos,
        }),
        Command::ConvVerify {
            blob,
            manifest,
            model,
            layer,
            tensor,
            taps,
            min_cos,
        } => conv_verify(&ConvVerifyArgs {
            blob,
            manifest,
            model,
            layer,
            tensor,
            taps,
            min_cos,
        }),
        Command::Kv {
            input,
            model,
            tensor,
            format,
            heads,
            pages,
            tokens,
            layout,
            v_scale,
            out_k,
            out_kt,
            out_v,
            out,
            manifest,
            values_out,
            no_rotate,
            quality_json,
            min_cos,
        } => kv_pack(KvArgs {
            input,
            model,
            tensor,
            format,
            heads,
            pages,
            tokens,
            layout,
            v_scale,
            out_k,
            out_kt,
            out_v,
            out,
            manifest,
            values_out,
            no_rotate,
            quality_json,
            min_cos,
        }),
        Command::KvVerify {
            manifest,
            k,
            kt,
            v,
            page,
            input,
            min_cos,
        } => kv_verify(&KvVerifyArgs {
            manifest,
            k,
            kt,
            v,
            page,
            input,
            min_cos,
        }),
        Command::KvUnpack {
            manifest,
            k,
            kt,
            v,
            page,
            out,
        } => kv_unpack(
            &manifest,
            k.as_deref(),
            kt.as_deref(),
            v.as_deref(),
            page.as_deref(),
            &out,
        ),
        Command::Gemv {
            model,
            tensor,
            layer,
            role,
            out,
            manifest,
            quality_json,
            rows,
            min_cos,
            min_row_cos,
        } => gemv_pack(GemvArgs {
            model,
            tensor,
            layer,
            role,
            out,
            manifest,
            quality_json,
            rows,
            min_cos,
            min_row_cos,
        }),
        Command::GemvVerify {
            blob,
            manifest,
            model,
            tensor,
            min_cos,
            min_row_cos,
        } => gemv_verify(
            &blob,
            &manifest,
            &model,
            tensor.as_deref(),
            min_cos,
            min_row_cos,
        ),
        Command::GemvUnpack {
            blob,
            manifest,
            out,
        } => gemv_unpack(&blob, &manifest, &out),
    }
}

fn list(model: &Path, filter: Option<&str>) -> Result<()> {
    let tensors = gguf_source::list_tensors(model)?;
    let mut shown = 0usize;
    for tensor in &tensors {
        if let Some(needle) = filter {
            if !tensor.name.contains(needle) {
                continue;
            }
        }
        let shape = if tensor.dims.len() == 2 {
            format!("K={} N={}", tensor.dims[0], tensor.dims[1])
        } else {
            format!("{:?}", tensor.dims)
        };
        println!("{:<40} {:<8} {}", tensor.name, tensor.ggml_type, shape);
        shown += 1;
    }
    println!("{shown} of {} tensors", tensors.len());
    Ok(())
}

struct PackArgs {
    model: PathBuf,
    tensor: String,
    out: PathBuf,
    manifest: PathBuf,
    quality_json: Option<PathBuf>,
    rows: Option<usize>,
    min_cos: f64,
    min_channel_cos: Option<f64>,
}

fn pack(args: &PackArgs) -> Result<()> {
    let options = PackOptions { dtype: DType::I8 };
    let (result, _weights) = pack_gguf(&args.model, &args.tensor, args.rows, options)?;

    write_bytes(&args.out, &result.blob)?;
    result.manifest.write(&args.manifest)?;
    if let Some(path) = args.quality_json.as_deref() {
        write_bytes(
            path,
            serde_json::to_string_pretty(&result.quality)?.as_bytes(),
        )?;
    }

    print_summary(&args.tensor, &result.quality, result.blob.len(), &args.out);
    println!(
        "manifest          {} ({} tiles, {} scale blocks)",
        args.manifest.display(),
        result.manifest.tiles.len(),
        result.manifest.n_tiles
    );
    metrics::check_threshold(&result.quality, args.min_cos, args.min_channel_cos)
}

fn verify(
    blob_path: &Path,
    manifest_path: &Path,
    model: &Path,
    min_cos: f64,
    min_channel_cos: Option<f64>,
) -> Result<()> {
    let manifest = Manifest::read(manifest_path)?;
    let blob = std::fs::read(blob_path)?;
    manifest.verify_blob(&blob)?;

    let (weights, _source) = gguf_source::load_matrix(model, &manifest.tensor, Some(manifest.n))?;
    let decoded = unpack(&blob)?;
    let quality = decoded.quality(&weights)?;

    print_summary(&manifest.tensor, &quality, blob.len(), blob_path);
    metrics::check_threshold(&quality, min_cos, min_channel_cos)
}

fn print_summary(tensor: &str, quality: &QualityStats, blob_bytes: usize, path: &Path) {
    println!("tensor            {tensor}");
    println!(
        "samples           {} ({} channels)",
        quality.samples,
        quality.per_channel.len()
    );
    println!(
        "blob              {} ({})",
        human_bytes(blob_bytes),
        path.display()
    );
    println!("cos total         {:.8}", quality.cos_total);
    println!(
        "cos per channel   min {:.8}, mean {:.8}",
        quality.cos_min_channel, quality.cos_mean_channel
    );
    println!(
        "max |delta|       {:.6e} ({:.4} x scale)",
        quality.max_abs_err, quality.max_abs_err_over_scale
    );
    println!(
        "saturated         {} ({:.4}% of samples)",
        quality.saturated,
        quality.saturated_fraction * 100.0
    );
}

// ---------------------------------------------------------------- int4 GEMV

struct GemvArgs {
    model: PathBuf,
    tensor: Option<String>,
    layer: Option<usize>,
    role: Option<String>,
    out: PathBuf,
    manifest: PathBuf,
    quality_json: Option<PathBuf>,
    rows: Option<usize>,
    min_cos: f64,
    min_row_cos: Option<f64>,
}

fn gemv_tensor(tensor: Option<&str>, layer: Option<usize>, role: Option<&str>) -> Result<String> {
    match (tensor, layer, role) {
        (Some(name), None, None) => Ok(name.to_string()),
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => Err(myelin_pack::PackError::RawInput(
            "use either --tensor or --layer/--role, not both".to_string(),
        )),
        (None, Some(layer), Some(role)) => gemv::tensor_for_role(layer, role),
        (None, Some(_), None) => Err(myelin_pack::PackError::RawInput(
            "--layer also needs --role".to_string(),
        )),
        (None, None, Some(_)) => Err(myelin_pack::PackError::RawInput(
            "--role also needs --layer".to_string(),
        )),
        (None, None, None) => Err(myelin_pack::PackError::RawInput(
            "either --tensor or --layer/--role is required".to_string(),
        )),
    }
}

fn gemv_pack(args: GemvArgs) -> Result<()> {
    let tensor = gemv_tensor(args.tensor.as_deref(), args.layer, args.role.as_deref())?;
    let (matrix, source) = gguf_source::load_matrix(&args.model, &tensor, args.rows)?;
    let meta = myelin_pack::PackMeta {
        tensor: tensor.clone(),
        source: Some(source),
    };
    let packed = gemv::build(&matrix, &meta)?;
    let manifest = myelin_pack::GemvManifest::from_blob(&meta, &packed);

    write_bytes(&args.out, &packed.blob)?;
    manifest.write(&args.manifest)?;
    if let Some(path) = args.quality_json.as_deref() {
        write_bytes(
            path,
            serde_json::to_string_pretty(&packed.quality)?.as_bytes(),
        )?;
    }
    print_gemv_summary(&tensor, &packed, &args.out, &args.manifest);
    packed.quality.check(args.min_cos, args.min_row_cos)
}

fn gemv_verify(
    blob_path: &Path,
    manifest_path: &Path,
    model: &Path,
    tensor: Option<&str>,
    min_cos: f64,
    min_row_cos: Option<f64>,
) -> Result<()> {
    let manifest = myelin_pack::GemvManifest::read(manifest_path)?;
    let blob = std::fs::read(blob_path)?;
    let header = manifest.verify_blob(&blob)?;
    let name = tensor.unwrap_or(&manifest.tensor);
    let (weights, _) = gguf_source::load_matrix(model, name, Some(header.n))?;
    if weights.k() != header.k || weights.n() != header.n {
        return Err(myelin_pack::PackError::ManifestMismatch(format!(
            "{name} is {}x{}, manifest is {}x{}",
            weights.n(),
            weights.k(),
            header.n,
            header.k
        )));
    }
    let packed = myelin_pack::GemvBlob::new(header, blob.clone(), &weights)?;
    print_gemv_summary(name, &packed, blob_path, manifest_path);
    packed.quality.check(min_cos, min_row_cos)
}

fn gemv_unpack(blob_path: &Path, manifest_path: &Path, out: &Path) -> Result<()> {
    let manifest = myelin_pack::GemvManifest::read(manifest_path)?;
    let blob = std::fs::read(blob_path)?;
    let header = manifest.verify_blob(&blob)?;
    let values = gemv::unpack(&blob)?;
    write_f32(out, &values)?;
    println!(
        "decoded           {} x {}: {} values -> {}",
        header.n,
        header.k,
        values.len(),
        out.display()
    );
    Ok(())
}

fn print_gemv_summary(tensor: &str, packed: &myelin_pack::GemvBlob, path: &Path, manifest: &Path) {
    let header = &packed.header;
    let quality = &packed.quality;
    println!("tensor            {tensor}");
    println!(
        "shape             k={} n={} (padded {} x {})",
        header.k, header.n, header.k_padded, header.n_padded
    );
    println!(
        "blob              {} ({}), {} rows of {} B",
        human_bytes(packed.blob.len()),
        path.display(),
        header.n_padded,
        header.row_bytes
    );
    println!("manifest          {}", manifest.display());
    println!("build id          {}", gemv::format_build_id_hex());
    println!("cos total         {:.8}", quality.cos_total);
    println!(
        "cos per row       min {:.8}, mean {:.8}",
        quality.cos_min_row, quality.cos_mean_row
    );
    println!(
        "max |delta|/s     {:.4} (int4 limit 0.5)",
        quality.max_abs_err_over_scale
    );
    println!(
        "saturated         {} ({:.4}% of samples)",
        quality.saturated,
        quality.saturated_fraction * 100.0
    );
}

// ---------------------------------------------------------------- shortconv

struct ConvArgs {
    model: Option<PathBuf>,
    layer: Option<usize>,
    tensor: Option<String>,
    taps: Option<PathBuf>,
    out: PathBuf,
    manifest: PathBuf,
    s_bc: Option<f32>,
    s_y: Option<f32>,
    taps_out: Option<PathBuf>,
    quality_json: Option<PathBuf>,
    min_cos: f64,
}

struct ConvVerifyArgs {
    blob: PathBuf,
    manifest: PathBuf,
    model: Option<PathBuf>,
    layer: Option<usize>,
    tensor: Option<String>,
    taps: Option<PathBuf>,
    min_cos: f64,
}

struct ConvSource {
    taps: Vec<f32>,
    h: usize,
    tensor: String,
    source: Option<myelin_pack::SourceInfo>,
}

fn conv_source(
    model: Option<&Path>,
    layer: Option<usize>,
    tensor: Option<&str>,
    taps: Option<&Path>,
) -> Result<ConvSource> {
    if let Some(path) = taps {
        if model.is_some() || layer.is_some() {
            return Err(myelin_pack::PackError::Shortconv(
                "use either --taps or --model/--layer, not both".to_string(),
            ));
        }
        let taps = read_f32(path)?;
        if taps.is_empty() || taps.len() % shortconv::TAPS != 0 {
            return Err(myelin_pack::PackError::Shortconv(format!(
                "{} values in {}, expected a multiple of {}",
                taps.len(),
                path.display(),
                shortconv::TAPS
            )));
        }
        return Ok(ConvSource {
            h: taps.len() / shortconv::TAPS,
            taps,
            tensor: tensor.unwrap_or("raw:f32").to_string(),
            source: None,
        });
    }
    let (model, layer) = match (model, layer) {
        (Some(model), Some(layer)) => (model, layer),
        _ => {
            return Err(myelin_pack::PackError::Shortconv(
                "pass --model and --layer, or --taps".to_string(),
            ))
        }
    };
    let name = tensor
        .map(str::to_string)
        .unwrap_or_else(|| format!("blk.{layer}.shortconv.conv.weight"));
    let (matrix, source) = gguf_source::load_matrix(model, &name, None)?;
    if matrix.k() != shortconv::TAPS {
        return Err(myelin_pack::PackError::Shortconv(format!(
            "tensor `{name}` has K={}, expected {}; `--layer` selects \
             blk.N.shortconv.conv.weight, not in_proj/out_proj",
            matrix.k(),
            shortconv::TAPS
        )));
    }
    Ok(ConvSource {
        taps: matrix.data().to_vec(),
        h: matrix.n(),
        tensor: name,
        source: Some(source),
    })
}

fn conv_scales(s_bc: Option<f32>, s_y: Option<f32>) -> Result<ActivationScales> {
    match (s_bc, s_y) {
        (Some(s_bc), Some(s_y)) => Ok(ActivationScales::cli(s_bc, s_y)),
        (None, None) => Ok(ActivationScales::stand_default()),
        _ => Err(myelin_pack::PackError::Shortconv(
            "--s-bc and --s-y must be given together".to_string(),
        )),
    }
}

fn conv_pack(args: ConvArgs) -> Result<()> {
    let source = conv_source(
        args.model.as_deref(),
        args.layer,
        args.tensor.as_deref(),
        args.taps.as_deref(),
    )?;
    let scales = conv_scales(args.s_bc, args.s_y)?;
    let blob = shortconv::build(&source.taps, source.h, scales.clone())?;
    let quality = shortconv::measure(&source.taps, source.h, &blob.blob)?;
    let manifest = ConvManifest::new(&blob, source.tensor.clone(), source.source.clone());
    manifest.verify_blob(&blob.blob)?;

    write_bytes(&args.out, &blob.blob)?;
    manifest.write(&args.manifest)?;
    if let Some(path) = args.taps_out.as_deref() {
        write_f32(path, &source.taps)?;
    }
    if let Some(path) = args.quality_json.as_deref() {
        write_bytes(path, serde_json::to_string_pretty(&quality)?.as_bytes())?;
    }

    println!("tensor            {}", source.tensor);
    println!("channels          {}", source.h);
    println!(
        "activation scales s_bc={} s_y={} ({})",
        scales.s_bc,
        scales.s_y,
        match scales.source.as_str() {
            "cli" => "from --s-bc/--s-y",
            _ => "stand defaults; pass --s-bc/--s-y with the runtime calibration",
        }
    );
    println!("kshift            {}", blob.ksh);
    println!(
        "blob              {} ({})",
        human_bytes(blob.blob.len()),
        args.out.display()
    );
    println!("cos total         {:.8}", quality.cos_total);
    println!(
        "cos per channel   min {:.8}, mean {:.8}",
        quality.cos_min_channel, quality.cos_mean_channel
    );
    println!(
        "max |delta|       {:.6e} ({:.4} x tap scale)",
        quality.max_abs_err, quality.max_abs_err_over_scale
    );
    println!(
        "saturated         {} ({:.4}% of taps)",
        quality.saturated,
        quality.saturated_fraction * 100.0
    );
    println!("manifest          {}", args.manifest.display());
    if quality.cos_total < args.min_cos {
        return Err(myelin_pack::PackError::QualityBelowThreshold {
            metric: "cos_total".to_string(),
            actual: quality.cos_total,
            threshold: args.min_cos,
        });
    }
    Ok(())
}

fn conv_verify(args: &ConvVerifyArgs) -> Result<()> {
    let manifest = ConvManifest::read(&args.manifest)?;
    let blob = std::fs::read(&args.blob)?;
    manifest.verify_blob(&blob)?;
    println!(
        "manifest ok       h={} ksh={} bytes={} checksum={}",
        manifest.h,
        manifest.ksh,
        blob.len(),
        manifest.blob_fnv1a64
    );

    let source = conv_source(
        args.model.as_deref(),
        args.layer,
        args.tensor.as_deref(),
        args.taps.as_deref(),
    );
    if args.model.is_none() && args.layer.is_none() && args.taps.is_none() {
        println!("source            not checked (pass --model/--layer or --taps)");
        return Ok(());
    }
    let source = source?;
    if source.h != manifest.h {
        return Err(myelin_pack::PackError::ManifestMismatch(format!(
            "source has {} channels, manifest says {}",
            source.h, manifest.h
        )));
    }
    let rebuilt = shortconv::build(&source.taps, source.h, manifest.activation_scales.clone())?;
    if rebuilt.blob != blob {
        let mismatch = rebuilt
            .blob
            .iter()
            .zip(blob.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        return Err(myelin_pack::PackError::ManifestMismatch(format!(
            "rebuild differs from the blob at byte {mismatch}"
        )));
    }
    if rebuilt.ksh != manifest.ksh {
        return Err(myelin_pack::PackError::ManifestMismatch(format!(
            "rebuild ksh {} != manifest {}",
            rebuilt.ksh, manifest.ksh
        )));
    }
    println!(
        "bit-exact         rebuild of {} bytes matches, ksh={}",
        blob.len(),
        rebuilt.ksh
    );
    let quality = shortconv::measure(&source.taps, source.h, &blob)?;
    println!("cos total         {:.8}", quality.cos_total);
    println!(
        "max |delta|       {:.6e} ({:.4} x tap scale)",
        quality.max_abs_err, quality.max_abs_err_over_scale
    );
    if quality.cos_total < args.min_cos {
        return Err(myelin_pack::PackError::QualityBelowThreshold {
            metric: "cos_total".to_string(),
            actual: quality.cos_total,
            threshold: args.min_cos,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------- kv

struct KvArgs {
    input: Option<PathBuf>,
    model: Option<PathBuf>,
    tensor: Option<String>,
    format: String,
    heads: usize,
    pages: usize,
    tokens: usize,
    layout: String,
    v_scale: String,
    out_k: Option<PathBuf>,
    out_kt: Option<PathBuf>,
    out_v: Option<PathBuf>,
    out: Option<PathBuf>,
    manifest: PathBuf,
    values_out: Option<PathBuf>,
    no_rotate: bool,
    quality_json: Option<PathBuf>,
    min_cos: f64,
}

struct KvVerifyArgs {
    manifest: PathBuf,
    k: Option<PathBuf>,
    kt: Option<PathBuf>,
    v: Option<PathBuf>,
    page: Option<PathBuf>,
    input: Option<PathBuf>,
    min_cos: f64,
}

struct KvSource {
    values: Vec<f32>,
    source: Option<myelin_pack::SourceInfo>,
}

fn kv_values(args: &KvArgs, count: usize) -> Result<KvSource> {
    match (args.input.as_deref(), args.model.as_deref()) {
        (Some(_), Some(_)) => Err(myelin_pack::PackError::KvPage(
            "use either --in or --model/--tensor, not both".to_string(),
        )),
        (Some(path), None) => {
            let values = read_f32(path)?;
            if values.len() != count {
                return Err(myelin_pack::PackError::KvPage(format!(
                    "{} has {} f32 values, expected {count}",
                    path.display(),
                    values.len()
                )));
            }
            Ok(KvSource {
                values,
                source: Some(myelin_pack::SourceInfo {
                    file: path.display().to_string(),
                    ggml_type: "f32".to_string(),
                }),
            })
        }
        (None, Some(model)) => {
            let tensor = args.tensor.as_deref().ok_or_else(|| {
                myelin_pack::PackError::KvPage("--model needs --tensor".to_string())
            })?;
            let (values, source) = gguf_source::load_f32_prefix(model, tensor, count)?;
            Ok(KvSource {
                values,
                source: Some(source),
            })
        }
        (None, None) => Err(myelin_pack::PackError::KvPage(
            "pass --in FILE or --model/--tensor".to_string(),
        )),
    }
}

fn kv_pack(args: KvArgs) -> Result<()> {
    let format = if args.format == "dsp" {
        None
    } else {
        Some(KvarnFormat::parse(&args.format)?)
    };
    if args.heads == 0 {
        return Err(myelin_pack::PackError::KvPage(
            "--heads must be positive".to_string(),
        ));
    }
    let valid_tokens = (args.pages - 1)
        .checked_mul(kvpage::PAGE_TOKENS)
        .and_then(|n| n.checked_add(args.tokens))
        .ok_or_else(|| myelin_pack::PackError::KvPage("token count overflows".to_string()))?;
    let count = args
        .heads
        .checked_mul(valid_tokens)
        .and_then(|n| n.checked_mul(kvpage::HEAD_DIM))
        .ok_or_else(|| myelin_pack::PackError::KvPage("input size overflows".to_string()))?;
    let source = kv_values(&args, count)?;
    if let Some(path) = args.values_out.as_deref() {
        write_f32(path, &source.values)?;
    }

    match format {
        None => {
            let params = KvPackParams {
                heads: args.heads,
                pages: args.pages,
                tokens_in_last_page: args.tokens,
                layout: KvLayout::parse(&args.layout)?,
                v_scale_mode: VScaleMode::parse(&args.v_scale)?,
            };
            let image = kvpage::build(&source.values, params)?;
            let quality = kvpage::measure(&source.values, &image)?;
            let manifest = kvpage::KvManifest::new(&image, source.source.clone());
            manifest.verify_images(&image.k, &image.kt, &image.v)?;

            let (out_k, out_kt, out_v) = match (args.out_k, args.out_kt, args.out_v) {
                (Some(k), Some(kt), Some(v)) => (k, kt, v),
                _ => {
                    return Err(myelin_pack::PackError::KvPage(
                        "dsp format needs --out-k, --out-kt and --out-v".to_string(),
                    ))
                }
            };
            write_bytes(&out_k, &image.k)?;
            write_bytes(&out_kt, &image.kt)?;
            write_bytes(&out_v, &image.v)?;
            manifest.write(&args.manifest)?;
            if let Some(path) = args.quality_json.as_deref() {
                write_bytes(path, serde_json::to_string_pretty(&quality)?.as_bytes())?;
            }

            println!(
                "format            {} ({})",
                manifest.format, manifest.layout
            );
            println!(
                "shape             heads={} pages={} tokens={} head_dim={}",
                manifest.heads, manifest.pages, manifest.tokens, manifest.head_dim
            );
            if manifest.heads != kvpage::DSP_KV_HEADS {
                println!(
                    "note              the current DSP kernel is compiled for {} KV heads; {} is valid only for another kernel",
                    kvpage::DSP_KV_HEADS, manifest.heads
                );
            }
            println!(
                "kt                {} (Q16 per token, what the kernel reads)",
                human_bytes(manifest.kt.bytes)
            );
            println!(
                "images            k={} kt={} v={}",
                human_bytes(image.k.len()),
                human_bytes(image.kt.len()),
                human_bytes(image.v.len())
            );
            println!(
                "K cos             total {:.6}, min/max per token {:.6}/{:.6}",
                quality.k_cos_total, quality.k_cos_min_token, quality.k_cos_mean_token
            );
            println!(
                "K error           max {:.6e} ({:.3} x token scale), saturated {} ({:.3}%)",
                quality.k_max_abs_err,
                quality.k_max_abs_err_over_scale,
                quality.k_saturated,
                100.0 * quality.k_saturated as f64 / quality.k_samples as f64
            );
            println!(
                "V cos             total {:.6}, min/max per token {:.6}/{:.6}",
                quality.v_cos_total, quality.v_cos_min_token, quality.v_cos_mean_token
            );
            println!(
                "V error           max {:.6e} ({:.3} x scale), saturated {} ({:.3}%)",
                quality.v_max_abs_err,
                quality.v_max_abs_err_over_scale,
                quality.v_saturated,
                100.0 * quality.v_saturated as f64 / quality.v_samples as f64
            );
            println!("v scale mode      {}", manifest.v_scale_mode);
            println!("out scale rule    {}", manifest.command.out_scale_rule);
            println!("manifest          {}", args.manifest.display());
            if quality.k_cos_total < args.min_cos {
                return Err(myelin_pack::PackError::QualityBelowThreshold {
                    metric: "k_cos_total".to_string(),
                    actual: quality.k_cos_total,
                    threshold: args.min_cos,
                });
            }
            if quality.v_cos_total < args.min_cos {
                return Err(myelin_pack::PackError::QualityBelowThreshold {
                    metric: "v_cos_total".to_string(),
                    actual: quality.v_cos_total,
                    threshold: args.min_cos,
                });
            }
            Ok(())
        }
        Some(format) => pack_kvarn(&args, format, &source),
    }
}

fn pack_kvarn(args: &KvArgs, format: KvarnFormat, source: &KvSource) -> Result<()> {
    if args.pages != 1 || args.tokens != kvpage::PAGE_TOKENS {
        return Err(myelin_pack::PackError::Kvarn(
            "kvarn encodes exactly one 32-token page per head".to_string(),
        ));
    }
    if args.layout != "row" || args.v_scale != "channel" {
        return Err(myelin_pack::PackError::Kvarn(
            "--layout/--v-scale describe the DSP format and do not apply to kvarn".to_string(),
        ));
    }
    let out = args
        .out
        .as_deref()
        .ok_or_else(|| myelin_pack::PackError::Kvarn("kvarn format needs --out".to_string()))?;
    let page_bytes = format.page_bytes();
    let mut encoded = Vec::with_capacity(args.heads * page_bytes);
    let mut page = vec![f16::ZERO; kvarn::PAGE_TOKENS * kvarn::HEAD_DIM];

    for head in 0..args.heads {
        let base = head * kvarn::PAGE_TOKENS * kvarn::HEAD_DIM;
        for (index, slot) in page.iter_mut().enumerate() {
            *slot = f16::from_f32(source.values[base + index]);
        }
        if !args.no_rotate {
            kvarn::rot_fwd_page(&mut page)?;
        }
        let codes = kvarn::commit(format, &page)?;
        encoded.extend_from_slice(&codes.encode());
    }

    let manifest = KvarnManifest::new(
        format,
        args.heads,
        !args.no_rotate,
        source.source.clone(),
        &encoded,
    );
    write_bytes(out, &encoded)?;
    manifest.write(&args.manifest)?;
    println!(
        "format            {} (reference codec, not read by the DSP kernel)",
        format.name()
    );
    println!(
        "shape             heads={} page={}x{}",
        args.heads,
        kvarn::PAGE_TOKENS,
        kvarn::HEAD_DIM
    );
    println!(
        "rotation          {}",
        if args.no_rotate {
            "off (input already rotated)"
        } else {
            "lfm_tq_rot_fwd"
        }
    );
    println!(
        "encoded           {} ({})",
        human_bytes(encoded.len()),
        out.display()
    );
    println!("manifest          {}", args.manifest.display());
    Ok(())
}

fn kv_verify(args: &KvVerifyArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.manifest)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let format = value
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    match format.as_str() {
        kvpage::FORMAT_NAME => {
            let manifest = kvpage::KvManifest::read(&args.manifest)?;
            let (k, kt, v) = match (args.k.as_deref(), args.kt.as_deref(), args.v.as_deref()) {
                (Some(k), Some(kt), Some(v)) => (k, kt, v),
                _ => {
                    return Err(myelin_pack::PackError::KvPage(
                        "dsp manifest needs --k, --kt and --v".to_string(),
                    ))
                }
            };
            let k_bytes = std::fs::read(k)?;
            let kt_bytes = std::fs::read(kt)?;
            let v_bytes = std::fs::read(v)?;
            let _image = manifest.verify_images(&k_bytes, &kt_bytes, &v_bytes)?;
            println!(
                "manifest ok       heads={} pages={} tokens={} layout={}",
                manifest.heads, manifest.pages, manifest.tokens, manifest.layout
            );
            println!(
                "checksums         k={} kt={} v={}",
                manifest.k_fnv1a64, manifest.kt_fnv1a64, manifest.v_fnv1a64
            );
            if let Some(path) = args.input.as_deref() {
                let values = read_f32(path)?;
                let params = manifest.params()?;
                if values.len() != manifest.heads * manifest.tokens * kvpage::HEAD_DIM {
                    return Err(myelin_pack::PackError::ManifestMismatch(format!(
                        "{} has {} values, manifest describes {}",
                        path.display(),
                        values.len(),
                        manifest.heads * manifest.tokens * kvpage::HEAD_DIM
                    )));
                }
                let rebuilt = kvpage::build(&values, params)?;
                if rebuilt.k != k_bytes || rebuilt.kt != kt_bytes || rebuilt.v != v_bytes {
                    return Err(myelin_pack::PackError::ManifestMismatch(
                        "re-pack of the source differs from the images".to_string(),
                    ));
                }
                println!(
                    "bit-exact         re-pack of the {} source values matches",
                    values.len()
                );
                let quality = kvpage::measure(&values, &rebuilt)?;
                println!(
                    "K cos {:.6} (min {:.6})  V cos {:.6} (min {:.6})",
                    quality.k_cos_total,
                    quality.k_cos_min_token,
                    quality.v_cos_total,
                    quality.v_cos_min_token
                );
                if quality.k_cos_total < args.min_cos || quality.v_cos_total < args.min_cos {
                    return Err(myelin_pack::PackError::QualityBelowThreshold {
                        metric: "kv_cos_total".to_string(),
                        actual: quality.k_cos_total.min(quality.v_cos_total),
                        threshold: args.min_cos,
                    });
                }
            }
            Ok(())
        }
        "kvarn-page" => {
            let manifest = KvarnManifest::read(&args.manifest)?;
            let page_path = args.page.as_deref().ok_or_else(|| {
                myelin_pack::PackError::Kvarn("kvarn manifest needs --page".to_string())
            })?;
            let bytes = std::fs::read(page_path)?;
            manifest.verify(&bytes)?;
            println!(
                "manifest ok       {} heads, {} bytes/page, rotate={}",
                manifest.heads, manifest.page_bytes, manifest.rotate
            );
            println!("checksum          {}", manifest.fnv1a64);
            if let Some(path) = args.input.as_deref() {
                let values = read_f32(path)?;
                let manifest_values = manifest.heads * kvarn::PAGE_TOKENS * kvarn::HEAD_DIM;
                if values.len() != manifest_values {
                    return Err(myelin_pack::PackError::ManifestMismatch(format!(
                        "{} has {} values, manifest describes {manifest_values}",
                        path.display(),
                        values.len()
                    )));
                }
                let mut rotated = vec![f16::ZERO; kvarn::PAGE_TOKENS * kvarn::HEAD_DIM];
                let mut rebuilt = Vec::with_capacity(bytes.len());
                for head in 0..manifest.heads {
                    let base = head * kvarn::PAGE_TOKENS * kvarn::HEAD_DIM;
                    for (index, slot) in rotated.iter_mut().enumerate() {
                        *slot = f16::from_f32(values[base + index]);
                    }
                    if manifest.rotate {
                        kvarn::rot_fwd_page(&mut rotated)?;
                    }
                    rebuilt
                        .extend_from_slice(&kvarn::commit(manifest.format()?, &rotated)?.encode());
                }
                if rebuilt != bytes {
                    return Err(myelin_pack::PackError::ManifestMismatch(
                        "re-pack of the source differs from the page file".to_string(),
                    ));
                }
                println!(
                    "bit-exact         re-pack of the {} source values matches",
                    values.len()
                );
            }
            Ok(())
        }
        other => Err(myelin_pack::PackError::ManifestMismatch(format!(
            "unknown manifest format `{other}`"
        ))),
    }
}

fn kv_unpack(
    manifest_path: &Path,
    k: Option<&Path>,
    kt: Option<&Path>,
    v: Option<&Path>,
    page: Option<&Path>,
    out: &Path,
) -> Result<()> {
    let text = std::fs::read_to_string(manifest_path)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let format = value
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    match format.as_str() {
        kvpage::FORMAT_NAME => {
            let manifest = kvpage::KvManifest::read(manifest_path)?;
            let (k, kt, v) = match (k, kt, v) {
                (Some(k), Some(kt), Some(v)) => (k, kt, v),
                _ => {
                    return Err(myelin_pack::PackError::KvPage(
                        "dsp manifest needs --k, --kt and --v".to_string(),
                    ))
                }
            };
            let image = manifest.verify_images(
                &std::fs::read(k)?,
                &std::fs::read(kt)?,
                &std::fs::read(v)?,
            )?;
            let (keys, values) = image.decode();
            let slots = image.params.slots();
            let valid = image.params.valid_tokens();
            let mut out_values =
                Vec::with_capacity(image.params.heads * valid * kvpage::HEAD_DIM * 2);
            for head in 0..image.params.heads {
                for slot in 0..valid {
                    let base = (head * slots + slot) * kvpage::HEAD_DIM;
                    out_values.extend_from_slice(&keys[base..base + kvpage::HEAD_DIM]);
                }
            }
            for head in 0..image.params.heads {
                for slot in 0..valid {
                    let base = (head * slots + slot) * kvpage::HEAD_DIM;
                    out_values.extend_from_slice(&values[base..base + kvpage::HEAD_DIM]);
                }
            }
            write_f32(out, &out_values)?;
            println!(
                "decoded           {} heads x {} tokens, keys then values, {} values ({})",
                image.params.heads,
                valid,
                out_values.len(),
                out.display()
            );
            Ok(())
        }
        "kvarn-page" => {
            let manifest = KvarnManifest::read(manifest_path)?;
            let page_path = page.ok_or_else(|| {
                myelin_pack::PackError::Kvarn("kvarn manifest needs --page".to_string())
            })?;
            let bytes = std::fs::read(page_path)?;
            manifest.verify(&bytes)?;
            let mut out_values =
                Vec::with_capacity(manifest.heads * kvarn::PAGE_TOKENS * kvarn::HEAD_DIM);
            for head in 0..manifest.heads {
                let base = head * manifest.page_bytes;
                let decoded = kvarn::KvarnPage::decode(
                    manifest.format()?,
                    &bytes[base..base + manifest.page_bytes],
                )?;
                for token in 0..kvarn::PAGE_TOKENS {
                    let mut key = decoded.decode_k(token);
                    let mut val = decoded.decode_v(token);
                    if manifest.rotate {
                        kvarn::rot_inv(&mut key);
                        kvarn::rot_inv(&mut val);
                    }
                    out_values.extend_from_slice(&key);
                    out_values.extend_from_slice(&val);
                }
            }
            write_f32(out, &out_values)?;
            println!(
                "decoded           {} heads x {} tokens, {} values ({})",
                manifest.heads,
                kvarn::PAGE_TOKENS,
                out_values.len(),
                out.display()
            );
            Ok(())
        }
        other => Err(myelin_pack::PackError::ManifestMismatch(format!(
            "unknown manifest format `{other}`"
        ))),
    }
}

// ------------------------------------------------------------------ helpers

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(myelin_pack::PackError::RawInput(format!(
            "{} has {} bytes, expected a multiple of 4",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes(path, &bytes)
}

fn write_bytes(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, data)?;
    Ok(())
}

fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}
