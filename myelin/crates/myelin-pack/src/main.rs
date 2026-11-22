//! `myelin-pack` command line front end.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use myelin_pack::{
    gguf_source, metrics, pack_gguf, unpack, DType, Manifest, PackOptions, QualityStats, Result,
};

#[derive(Debug, Parser)]
#[command(
    name = "myelin-pack",
    version,
    about = "Pack GGUF tensors into int8 HMX crownstones and verify the accuracy"
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
