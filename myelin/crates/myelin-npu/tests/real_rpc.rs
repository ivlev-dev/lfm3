//! Real-device FastRPC smoke test.
//!
//! Ignored by default: it needs a phone with `libcdsprpc.so`, unsigned PD
//! enabled and `liblfmgemv_skel.so` reachable through `ADSP_LIBRARY_PATH`.
//!
//! ```text
//! adb push liblfmgemv_skel.so /data/local/tmp/myelin/
//! adb shell ADSP_LIBRARY_PATH=/data/local/tmp/myelin \
//!     /data/local/tmp/myelin/npu_bench --iterations 11
//! ```
//!
//! The Rust harness can run the same check with:
//! `cargo test -p myelin-npu --target aarch64-linux-android -- --ignored`.

#[cfg(target_os = "android")]
#[test]
#[ignore = "requires a phone with libcdsprpc.so and liblfmgemv_skel.so"]
fn real_fastrpc_opens_allocates_and_reads_team_info() {
    use std::sync::Arc;

    use myelin_npu::{Cdsprpc, CdsprpcConfig, HvxClient};

    let transport = Arc::new(Cdsprpc::load(CdsprpcConfig::default()).expect("load libcdsprpc.so"));
    let mut client = HvxClient::open(transport, "lfmgemv").expect("open lfmgemv");
    let handle = client.alloc_buf(64 * 1024).expect("alloc_buf");
    assert!(handle >= 0, "alloc_buf returned {handle}");
    let info = client.team_info(8).expect("team_info");
    assert_eq!(info.len(), 8);
    client.release_buf(handle).expect("release_buf");
    client.close().expect("close");
}
