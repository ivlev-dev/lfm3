use anyhow::{Context as _, Result, anyhow};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::CL_MEM_READ_WRITE;
use opencl3::types::{CL_BLOCKING, cl_mem};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub use opencl3::event::Event;
pub use opencl3::kernel::Kernel;
pub use opencl3::memory::{Buffer, ClMem};
pub use opencl3::program::Program;

#[derive(Debug, Clone)]
pub struct Caps {
    pub name: String,
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
        Program::create_and_build_from_source(&self.context, source, options)
            .map_err(|e| anyhow!("build failed: {e}"))
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
            eprintln!("launch {}", kernel.function_name().unwrap_or_default());
        }
        let event = unsafe { exec.enqueue_nd_range(&self.queue) }.map_err(|e| {
            anyhow!(
                "clEnqueueNDRangeKernel({}, global={global}, local={local}): {e}",
                kernel.function_name().unwrap_or_default()
            )
        })?;
        if self.profile {
            let name = kernel.function_name().unwrap_or_default();
            event.wait()?;
            let us = Self::elapsed(&event)?.as_secs_f64() * 1e6;
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
