//! Dump the layer-0 input the DSP path feeds to the device.
//!
//! ```text
//! cargo run --release -p myelin-npu --example pack_first_layer -- \
//!     /tmp/opencode/model-pack 597 5205 302 3980 355
//! ```
//!
//! Writes `embed.bin` and `input.bin` (f32, little endian) next to the pack so
//! a device dump of the first layer can be compared against the exact vector
//! the backend used.

use std::path::PathBuf;

use myelin_core::backend::{NormSlot, NormWeights};
use myelin_npu::model_pack::ModelPack;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("pack dir"));
    let tokens: Vec<u32> = args.map(|value| value.parse().expect("token id")).collect();
    let token = *tokens.first().expect("at least one token");

    let pack = ModelPack::read(&dir).expect("pack");
    let spec = pack.spec();
    let host = pack.host_weights(&dir).expect("host weights");
    let hidden = spec.dims.hidden;

    let mut embed = vec![0f32; hidden];
    host.embed(token, &mut embed).expect("embed");

    let mut input = vec![0f32; hidden];
    host.norm(
        NormWeights::Layer {
            layer: 0,
            slot: NormSlot::Attention,
        },
        &embed,
        &mut input,
    )
    .expect("norm");

    let dump = |name: &str, values: &[f32]| {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(dir.join(name), bytes).expect("write");
    };
    dump("embed.bin", &embed);
    dump("input.bin", &input);
    println!(
        "token {token}: embed rms {:.4}, input rms {:.4}",
        (embed.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt(),
        (input.iter().map(|v| v * v).sum::<f32>() / hidden as f32).sqrt()
    );
}
