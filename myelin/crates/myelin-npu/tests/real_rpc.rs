//! Real-device FastRPC smoke tests.
//!
//! Ignored by default: they need a phone with `libcdsprpc.so`, unsigned PD
//! enabled and `liblfmgemv_skel.so` reachable through `ADSP_LIBRARY_PATH`.
//!
//! ```text
//! adb push liblfmgemv_skel.so /data/local/tmp/myelin/
//! adb shell ADSP_LIBRARY_PATH=/data/local/tmp/myelin \
//!     /data/local/tmp/myelin/real_rpc --ignored --test-threads=1
//! ```
//!
//! `ADSP_LIBRARY_PATH` must hold exactly one directory: this device fails with
//! `AEE_EUNABLETOLOAD` when the value contains a colon.

#[cfg(target_os = "android")]
mod android {
    use std::sync::Arc;

    use myelin_npu::{fastrpc_error_name, Cdsprpc, CdsprpcConfig, HvxClient, Transport};

    const MODULE: &str = "lfmgemv";

    fn transport() -> Arc<Cdsprpc> {
        Arc::new(Cdsprpc::load(CdsprpcConfig::default()).expect("load libcdsprpc.so"))
    }

    fn check<T>(label: &str, result: Result<T, myelin_npu::TransportError>) -> T {
        match result {
            Ok(value) => {
                println!("{label:<40} ok");
                value
            }
            Err(error) => {
                let code = error.code().unwrap_or(0);
                println!(
                    "{label:<40} FAIL 0x{code:08x} ({})  {error}",
                    fastrpc_error_name(code)
                );
                panic!("{label} failed: {error}");
            }
        }
    }

    #[test]
    #[ignore = "requires a phone with libcdsprpc.so and liblfmgemv_skel.so"]
    fn real_fastrpc_alloc_load_256k_and_4mib() {
        let transport = transport();
        let mut client = HvxClient::open(transport, MODULE).expect("open lfmgemv");
        println!("open {MODULE}: handle={:?}", client.handle());

        let info = check("team_info(8)", client.team_info(8));
        println!("team_info: {info:?}");

        let handle = check("alloc_buf(8 MiB)", client.alloc_buf(8 << 20));
        println!("resident handle = {handle}");
        assert!(handle >= 0, "alloc_buf returned {handle}");

        let small = vec![0x11u8; 256 << 10];
        let large = vec![0x22u8; 4 << 20];
        check("load_buf 256 KiB @ 0", client.load_buf(handle, 0, &small));
        check(
            "load_buf 4 MiB @ 256 KiB",
            client.load_buf(handle, 256 << 10, &large),
        );

        check("release_buf", client.release_buf(handle));
        check("close", client.close());
    }

    #[test]
    #[ignore = "requires a phone with libcdsprpc.so and liblfmgemv_skel.so"]
    fn real_fastrpc_attach_out_zero_copy() {
        let transport = transport();
        let mut client = HvxClient::open(Arc::clone(&transport) as Arc<dyn Transport>, MODULE)
            .expect("open lfmgemv");

        // N=512, K=256 int4 -> 64 KiB of resident weights.
        const N: i32 = 512;
        const K: i32 = 256;
        let handle = check("alloc_buf(64 KiB)", client.alloc_buf((N * K / 2) as usize));
        // `with_scale=1` allocates the per-row dequant tables that the
        // resident output path needs; bit 2 would select int8 weights.
        check(
            "set_shape(512, 256, with_scale=1)",
            client.set_shape(handle, N, K, 1),
        );
        // Deterministic non-zero int4 nibbles: (i % 7) + 4 keeps the value
        // inside the signed 4-bit range without relying on the dequant table.
        let weights: Vec<u8> = (0..(N * K / 2) as usize)
            .map(|index| {
                let low = ((index % 7) + 4) as u8;
                let high = (((index + 3) % 7) + 4) as u8;
                (high << 4) | low
            })
            .collect();
        check("load_buf weights", client.load_buf(handle, 0, &weights));
        let mut scales = Vec::with_capacity(16 + 4 * N as usize);
        scales.extend_from_slice(&N.to_le_bytes());
        scales.extend_from_slice(&K.to_le_bytes());
        scales.extend_from_slice(&1.0f32.to_le_bytes());
        scales.extend_from_slice(&1.0f32.to_le_bytes());
        scales.resize(16 + 4 * N as usize, 0);
        for scale in scales[16..].chunks_exact_mut(4) {
            scale.copy_from_slice(&1.0f32.to_le_bytes());
        }
        check("set_scale", client.set_scale(handle, &scales));

        let act = vec![1u8; K as usize];
        // Reference through the ordinary rout path, before the shared buffer
        // is touched by the CPU (ION is cached; a pre-fill would make the
        // host read its own stale cache line instead of the DSP result).
        let reference = check(
            "gemv (reference dst)",
            client.gemv(myelin_npu::GemvRequest {
                scalars: myelin_npu::GemvScalars {
                    h: handle,
                    m: 1,
                    reps: 1,
                },
                act: &act,
                dst_len: N as usize,
            }),
        )
        .dst;

        let mut out = check("alloc_shared(4 KiB)", transport.alloc_shared(4096));
        check(
            "attach_out(fd, 512)",
            client.attach_out(handle, out.fd(), N),
        );

        let times = check("gemv_resident", client.gemv_resident(handle, 1, &act, 1));
        println!("gemv_resident times: {times:?}");

        // The DSP writes into the mapped ION pages; on this device the host
        // mapping only observes them after `detach_out` flushes the DSP view,
        // so detach before comparing (the data itself never went through the
        // host).
        check("detach_out", client.detach_out(handle));
        let written = &out.as_slice()[..N as usize];
        println!(
            "attached output: first bytes {:?}",
            &written[..8.min(written.len())]
        );
        assert_eq!(written, reference.as_slice(), "attach_out output mismatch");

        check("release_buf", client.release_buf(handle));
        check("close", client.close());
    }
}
