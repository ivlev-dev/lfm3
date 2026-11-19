use anyhow::{Context as _, Result, anyhow};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_mem};
use sha2::{Digest, Sha256};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::memory::{Buffer, ClMem};
pub use opencl3::program::Program;

pub const UPLOAD_ASYNC_MIN_BYTES: usize = 1 << 20;
const PROGRAM_CACHE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Caps {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub version: String,
    pub driver: String,
    pub opencl_c: String,
    pub compute_units: u32,
    pub clock_mhz: u32,
    pub global_mem_bytes: u64,
    pub max_work_group_size: usize,
    pub preferred_wg_multiple: usize,
    pub fp16: bool,
    pub integer_dot: bool,
    pub subgroups: bool,
    pub sub_group_size: usize,
    pub extensions: Vec<String>,
}

pub struct Gpu {
    pub device: Device,
    pub context: Context,
    pub queue: CommandQueue,
    pub caps: Caps,
    pub profile: bool,
    pub kernel_times: RefCell<HashMap<String, (u32, f64)>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub median_ms: f64,
    pub min_ms: f64,
    pub mean_ms: f64,
}

pub enum Arg {
    Mem(cl_mem),
    U32(u32),
    U64(u64),
    I32(i32),
    F32(f32),
    Local(usize),
}

impl Gpu {
    pub fn open(index: usize) -> Result<Self> {
        Self::open_with(index, false)
    }

    pub fn open_with(index: usize, profiling: bool) -> Result<Self> {
        let ids = match std::env::var("LFM_PLATFORM") {
            Ok(pat) if !pat.is_empty() => {
                let mut out = Vec::new();
                for plat in opencl3::platform::get_platforms().context("clGetPlatformIDs")? {
                    let name = plat.name().unwrap_or_default();
                    let ver = plat.version().unwrap_or_default();
                    if name.contains(&pat) || ver.contains(&pat) {
                        out.extend(plat.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default());
                    }
                }
                out
            }
            _ => get_all_devices(CL_DEVICE_TYPE_ALL).context("clGetDeviceIDs")?,
        };
        let id = *ids
            .get(index)
            .ok_or_else(|| anyhow!("no OpenCL device at index {index}, found {}", ids.len()))?;
        let device = Device::new(id);
        let context = Context::from_device(&device).context("clCreateContext")?;
        let flags = if profiling {
            CL_QUEUE_PROFILING_ENABLE
        } else {
            0
        };
        let queue = CommandQueue::create_default_with_properties(&context, flags, 0)
            .context("clCreateCommandQueue")?;
        let extensions: Vec<String> = device
            .extensions()
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let caps = Caps {
            name: device.name().unwrap_or_default(),
            vendor: device.vendor().unwrap_or_default(),
            device_type: match device.dev_type().unwrap_or(0) {
                opencl3::device::CL_DEVICE_TYPE_GPU => "gpu".to_string(),
                opencl3::device::CL_DEVICE_TYPE_CPU => "cpu".to_string(),
                opencl3::device::CL_DEVICE_TYPE_ACCELERATOR => "accelerator".to_string(),
                opencl3::device::CL_DEVICE_TYPE_CUSTOM => "custom".to_string(),
                _ => "other".to_string(),
            },
            version: device.version().unwrap_or_default(),
            driver: device.driver_version().unwrap_or_default(),
            opencl_c: device.opencl_c_version().unwrap_or_default(),
            compute_units: device.max_compute_units().unwrap_or_default(),
            clock_mhz: device.max_clock_frequency().unwrap_or_default(),
            global_mem_bytes: device.global_mem_size().unwrap_or_default(),
            max_work_group_size: device.max_work_group_size().unwrap_or_default(),
            preferred_wg_multiple: device
                .preferred_work_group_size_multiple()
                .unwrap_or_default(),
            fp16: extensions.iter().any(|e| e == "cl_khr_fp16"),
            integer_dot: extensions.iter().any(|e| e == "cl_khr_integer_dot_product")
                || device.integer_dot_product_capabilities_khr().unwrap_or(0) != 0,
            subgroups: extensions.iter().any(|e| e.starts_with("cl_khr_subgroup")),
            sub_group_size: 0,
            extensions,
        };
        let sub_group_size = probe_sub_group_size(&context, &queue)?;
        let mut caps = caps;
        caps.sub_group_size = sub_group_size;
        Ok(Self {
            device,
            context,
            queue,
            caps,
            profile: profiling,
            kernel_times: RefCell::new(HashMap::new()),
        })
    }

    pub fn build(&self, source: &str, options: &str) -> Result<Program> {
        self.build_with(source, options, !program_cache_disabled())
    }

    pub fn build_with(&self, source: &str, options: &str, use_cache: bool) -> Result<Program> {
        let cached = (use_cache && self.caps.device_type == "gpu")
            .then(|| self.program_cache_path(source, options))
            .flatten();
        if let Some(path) = &cached
            && let Some(program) = load_program_binary(&self.context, path)
        {
            return Ok(program);
        }
        let program = Program::create_and_build_from_source(&self.context, source, options)
            .map_err(|e| anyhow!("build failed: {e}"))?;
        if let Some(path) = &cached {
            store_program_binary(&program, path);
        }
        Ok(program)
    }

    fn program_cache_path(&self, source: &str, options: &str) -> Option<PathBuf> {
        let root = program_cache_root()?;
        let key = program_cache_key(source, options, &self.caps.name, &self.caps.driver);
        Some(root.join(format!("{key}.bin")))
    }

    pub fn alloc<T>(&self, len: usize) -> Result<Buffer<T>> {
        unsafe { Buffer::create(&self.context, CL_MEM_READ_WRITE, len, std::ptr::null_mut()) }
            .context("clCreateBuffer")
    }

    pub fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Buffer<T>> {
        let mut buf = self.alloc::<T>(data.len())?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])
                .context("clEnqueueWriteBuffer")?;
        }
        Ok(buf)
    }

    pub fn upload_bytes(&self, data: &[u8]) -> Result<Buffer<u8>> {
        let mut buf = self.alloc::<u8>(data.len())?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])
                .context("clEnqueueWriteBuffer")?;
        }
        Ok(buf)
    }

    /// # Safety
    /// `data` must stay alive and unmodified until the next [`Gpu::finish`] call.
    pub unsafe fn upload_async<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Buffer<T>> {
        let mut buf = self.alloc::<T>(data.len())?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buf, CL_NON_BLOCKING, 0, data, &[])
                .context("clEnqueueWriteBuffer")?;
        }
        Ok(buf)
    }

    /// # Safety
    /// `data` must stay alive and unmodified until the next [`Gpu::finish`] call.
    pub unsafe fn upload_bytes_async(&self, data: &[u8]) -> Result<Buffer<u8>> {
        let mut buf = self.alloc::<u8>(data.len())?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buf, CL_NON_BLOCKING, 0, data, &[])
                .context("clEnqueueWriteBuffer")?;
        }
        Ok(buf)
    }

    pub fn finish(&self) -> Result<()> {
        self.queue.finish().context("clFinish")
    }

    pub fn download_at<T: bytemuck::Pod>(
        &self,
        buf: &Buffer<T>,
        offset: usize,
        len: usize,
    ) -> Result<Vec<T>> {
        let mut out = vec![T::zeroed(); len];
        unsafe {
            self.queue
                .enqueue_read_buffer(buf, CL_BLOCKING, offset, &mut out, &[])
                .context("clEnqueueReadBuffer")?;
        }
        Ok(out)
    }

    pub fn download<T: bytemuck::Pod>(&self, buf: &Buffer<T>, len: usize) -> Result<Vec<T>> {
        let mut out = vec![T::zeroed(); len];
        unsafe {
            self.queue
                .enqueue_read_buffer(buf, CL_BLOCKING, 0, &mut out, &[])
                .context("clEnqueueReadBuffer")?;
        }
        Ok(out)
    }

    pub fn readonly<T>(&self, buf: &Buffer<T>) -> cl_mem {
        buf.get()
    }

    /// Запуск ядра в очереди устройства.
    ///
    /// # Safety
    /// Вызывающий гарантирует, что набор `args` соответствует сигнатуре ядра
    /// (порядок, типы и размеры буферов), и что буферы живы до завершения ядра.
    pub unsafe fn launch(
        &self,
        kernel: &Kernel,
        global: usize,
        local: usize,
        args: &[Arg],
    ) -> Result<Event> {
        let mut exec = ExecuteKernel::new(kernel);
        exec.set_global_work_size(global);
        if local > 0 {
            exec.set_local_work_size(local);
        }
        for arg in args {
            match arg {
                Arg::Mem(m) => {
                    unsafe { exec.set_arg(m) };
                }
                Arg::U32(v) => {
                    unsafe { exec.set_arg(v) };
                }
                Arg::U64(v) => {
                    unsafe { exec.set_arg(v) };
                }
                Arg::I32(v) => {
                    unsafe { exec.set_arg(v) };
                }
                Arg::F32(v) => {
                    unsafe { exec.set_arg(v) };
                }
                Arg::Local(n) => {
                    unsafe { exec.set_arg_local_buffer(*n) };
                }
            }
        }
        if std::env::var("LFM_LAUNCH_DEBUG").is_ok() {
            eprintln!(
                "launch {} global={} local={}",
                kernel.function_name().unwrap_or_default(),
                global,
                local
            );
        }
        let event = unsafe { exec.enqueue_nd_range(&self.queue) }.map_err(|e| {
            anyhow!(
                "clEnqueueNDRangeKernel({}, global={global}, local={local}): {e}",
                kernel.function_name().unwrap_or_default()
            )
        })?;
        if self.profile {
            let name = kernel.function_name().unwrap_or_default();
            let started = Instant::now();
            event.wait()?;
            let us = started.elapsed().as_secs_f64() * 1e6;
            let mut map = self.kernel_times.borrow_mut();
            let e = map.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += us;
        }
        Ok(event)
    }

    pub fn profile_report(&self, total_ms: f64) -> Vec<(String, u32, f64, f64)> {
        let map = self.kernel_times.borrow();
        let mut rows: Vec<(String, u32, f64, f64)> = map
            .iter()
            .map(|(k, (n, us))| (k.clone(), *n, *us / *n as f64, *us / 1000.0))
            .collect();
        rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        let _ = total_ms;
        rows
    }

    pub fn reset_profile(&self) {
        self.kernel_times.borrow_mut().clear();
    }

    pub fn elapsed(event: &Event) -> Result<Duration> {
        let start = event.profiling_command_start()?;
        let end = event.profiling_command_end()?;
        Ok(Duration::from_nanos(end.saturating_sub(start)))
    }

    pub fn bench<F>(&self, warmup: usize, reps: usize, mut f: F) -> Result<Stats>
    where
        F: FnMut() -> Result<()>,
    {
        for _ in 0..warmup {
            f()?;
        }
        self.queue.finish()?;
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            f()?;
            self.queue.finish()?;
            samples.push(started.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_ms = *samples.first().unwrap_or(&0.0);
        let median_ms = samples.get(samples.len() / 2).copied().unwrap_or(0.0);
        let mean_ms = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
        Ok(Stats {
            median_ms,
            min_ms,
            mean_ms,
        })
    }
}

pub struct UploadStage<'a> {
    gpu: &'a Gpu,
    host: RefCell<Vec<Box<dyn Any>>>,
}

impl<'a> UploadStage<'a> {
    pub fn new(gpu: &'a Gpu) -> Self {
        Self {
            gpu,
            host: RefCell::new(Vec::new()),
        }
    }

    pub fn upload<T: bytemuck::Pod + Send + 'static>(&self, data: Vec<T>) -> Result<Buffer<T>> {
        if std::mem::size_of_val(data.as_slice()) < UPLOAD_ASYNC_MIN_BYTES {
            return self.gpu.upload(&data);
        }
        let buf = unsafe { self.gpu.upload_async(data.as_slice()) }?;
        self.host.borrow_mut().push(Box::new(data));
        Ok(buf)
    }

    pub fn upload_bytes(&self, data: Vec<u8>) -> Result<Buffer<u8>> {
        if data.len() < UPLOAD_ASYNC_MIN_BYTES {
            return self.gpu.upload_bytes(&data);
        }
        let buf = unsafe { self.gpu.upload_bytes_async(&data) }?;
        self.host.borrow_mut().push(Box::new(data));
        Ok(buf)
    }

    pub fn upload_borrowed<T: bytemuck::Pod>(&self, data: &'a [T]) -> Result<Buffer<T>> {
        unsafe { self.gpu.upload_async(data) }
    }

    pub fn finish(&self) -> Result<()> {
        if let Err(e) = self.gpu.finish() {
            let keep = self.host.replace(Vec::new());
            std::mem::forget(keep);
            return Err(e);
        }
        self.host.borrow_mut().clear();
        Ok(())
    }
}

impl Drop for UploadStage<'_> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

const PROBE_SRC: &str =
    "__kernel void probe_sgs(__global uint *out) { out[0] = get_sub_group_size(); }";

fn probe_sub_group_size(context: &Context, queue: &CommandQueue) -> Result<usize> {
    let program = Program::create_and_build_from_source(context, PROBE_SRC, "-cl-std=CL3.0")
        .map_err(|e| anyhow!("probe build failed: {e}"))?;
    let kernel = Kernel::create(&program, "probe_sgs")?;
    let mut buffer: Buffer<u32> =
        unsafe { Buffer::create(context, CL_MEM_READ_WRITE, 256, std::ptr::null_mut())? };
    let zeros = [0u32; 256];
    unsafe {
        queue.enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &zeros, &[])?;
    }
    let mut exec = ExecuteKernel::new(&kernel);
    exec.set_global_work_size(256);
    exec.set_local_work_size(256);
    unsafe {
        exec.set_arg(&buffer.get());
    }
    let event = unsafe { exec.enqueue_nd_range(queue)? };
    event.wait()?;
    let mut out = [0u32; 1];
    unsafe {
        queue.enqueue_read_buffer(&buffer, CL_BLOCKING, 0, &mut out[..], &[])?;
    }
    Ok(out[0] as usize)
}

pub fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn program_cache_disabled() -> bool {
    match std::env::var("LFM_PROGRAM_CACHE") {
        Ok(v) => {
            let v = v.trim();
            v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")
        }
        Err(_) => false,
    }
}

pub fn program_cache_root() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("lfm3").join("programs"))
}

pub fn program_cache_key(
    source: &str,
    options: &str,
    device_name: &str,
    driver_version: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PROGRAM_CACHE_VERSION.to_le_bytes());
    for field in [source, options, device_name, driver_version] {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    hex_lower(hasher.finalize().as_slice())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn load_program_binary(context: &Context, path: &Path) -> Option<Program> {
    let binary = std::fs::read(path).ok()?;
    if binary.is_empty() {
        return None;
    }
    Program::create_and_build_from_binary(context, &[binary.as_slice()], "").ok()
}

fn store_program_binary(program: &Program, path: &Path) {
    let Ok(binaries) = program.get_binaries() else {
        return;
    };
    let Some(binary) = binaries.first().filter(|b| !b.is_empty()) else {
        return;
    };
    let Some(root) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(root).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, binary).is_err() {
        return;
    }
    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "__kernel void k(__global int *x) { x[0] = 1; }";
    const OPTIONS: &str = "-cl-std=CL3.0 -DPB=6";
    const DEVICE: &str = "AMD Radeon RX 6400";
    const DRIVER: &str = "23.20.1";

    #[test]
    fn cache_key_is_deterministic() {
        let a = program_cache_key(SOURCE, OPTIONS, DEVICE, DRIVER);
        let b = program_cache_key(SOURCE, OPTIONS, DEVICE, DRIVER);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(
            a.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn cache_key_changes_with_every_input() {
        let base = program_cache_key(SOURCE, OPTIONS, DEVICE, DRIVER);
        assert_ne!(
            base,
            program_cache_key("__kernel void k() {}", OPTIONS, DEVICE, DRIVER)
        );
        assert_ne!(
            base,
            program_cache_key(SOURCE, "-cl-std=CL2.0", DEVICE, DRIVER)
        );
        assert_ne!(
            base,
            program_cache_key(SOURCE, OPTIONS, "Intel UHD", DRIVER)
        );
        assert_ne!(base, program_cache_key(SOURCE, OPTIONS, DEVICE, "24.1.0"));
    }

    #[test]
    fn cache_key_has_no_field_ambiguity() {
        let a = program_cache_key("ab", "c", DEVICE, DRIVER);
        let b = program_cache_key("a", "bc", DEVICE, DRIVER);
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_golden() {
        assert_eq!(
            program_cache_key(SOURCE, OPTIONS, DEVICE, DRIVER),
            "031e352389b61fc8ef1350794a2945dc32cc776279a95f92bbe097a6b800385e"
        );
    }

    #[test]
    fn cache_root_falls_back_to_home() {
        let root = program_cache_root();
        assert!(root.is_none() || root.unwrap().ends_with("lfm3/programs"));
    }
}
