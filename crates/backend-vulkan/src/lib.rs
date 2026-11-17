use std::ffi::CStr;

use anyhow::{anyhow, Result};
use ash::vk;

const GEMV_Q4_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4.spv"));

pub struct Vk {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    cmd_pool: vk::CommandPool,
}

pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub kind: String,
    pub memory_mb: u64,
}

impl Vk {
    pub fn devices() -> Result<Vec<DeviceInfo>> {
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = unsafe { entry.create_instance(&create, None)? };
        let phys = unsafe { instance.enumerate_physical_devices()? };
        let mut out = Vec::new();
        for (index, &physical) in phys.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            let name = unsafe {
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            let kind = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
                vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
                vk::PhysicalDeviceType::CPU => "cpu",
                _ => "other",
            };
            let mem = unsafe { instance.get_physical_device_memory_properties(physical) };
            let mut total = 0u64;
            for i in 0..mem.memory_heap_count {
                total += mem.memory_heaps[i as usize].size;
            }
            out.push(DeviceInfo {
                index,
                name,
                kind: kind.to_string(),
                memory_mb: total / (1 << 20),
            });
        }
        unsafe { instance.destroy_instance(None) };
        Ok(out)
    }

    pub fn open(preferred: Option<usize>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
        let create = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = unsafe { entry.create_instance(&create, None)? };
        let phys = unsafe { instance.enumerate_physical_devices()? };
        let index = preferred.unwrap_or_else(|| {
            std::env::var("LFM_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        });
        let physical = *phys
            .get(index)
            .ok_or_else(|| anyhow!("no vulkan device at {index} (found {})", phys.len()))?;
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow!("no compute queue"))? as u32;
        let prio = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio);
        let queues = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        let device = unsafe { instance.create_device(physical, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        Ok(Self {
            entry,
            instance,
            device,
            physical,
            queue,
            queue_family,
            cmd_pool,
        })
    }

    pub fn name(&self) -> String {
        let props = unsafe { self.instance.get_physical_device_properties(self.physical) };
        unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        let props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        for i in 0..props.memory_type_count {
            if bits & (1 << i) != 0
                && props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
            {
                return Ok(i);
            }
        }
        Err(anyhow!("no suitable vulkan memory type"))
    }

    fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&info, None)? };
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let index = self.memory_type(
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(index);
        let memory = unsafe { self.device.allocate_memory(&alloc, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        Ok((buffer, memory))
    }

    fn upload(&self, bytes: &[u8]) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let (buffer, memory) = self.create_buffer(
            bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        unsafe {
            let ptr = self
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            self.device.unmap_memory(memory);
        }
        Ok((buffer, memory))
    }

    fn read_buffer(&self, buffer: &(vk::Buffer, vk::DeviceMemory), bytes: usize) -> Result<Vec<u8>> {
        let mut out = vec![0u8; bytes];
        unsafe {
            let ptr = self.device.map_memory(
                buffer.1,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), bytes);
            self.device.unmap_memory(buffer.1);
        }
        Ok(out)
    }

    fn destroy_buf(&self, buffer: &(vk::Buffer, vk::DeviceMemory)) {
        unsafe {
            self.device.destroy_buffer(buffer.0, None);
            self.device.free_memory(buffer.1, None);
        }
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = (&self.entry, &self.queue, self.queue_family);
    }
}

pub struct GemvQ4 {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
}

impl GemvQ4 {
    pub fn new(vk: &Vk) -> Result<Self> {
        let bindings = (0..4)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { vk.device.create_descriptor_set_layout(&set_info, None)? };
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(12);
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(std::slice::from_ref(&range));
        let pipeline_layout = unsafe { vk.device.create_pipeline_layout(&layout_info, None)? };
        let words = ash::util::read_spv(&mut std::io::Cursor::new(GEMV_Q4_SPV))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = unsafe { vk.device.create_shader_module(&module_info, None)? };
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            vk.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| anyhow!("pipeline: {e}"))?[0]
        };
        unsafe { vk.device.destroy_shader_module(module, None) };
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(1);
        let pool = unsafe { vk.device.create_descriptor_pool(&pool_info, None)? };
        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            pool,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        vk: &Vk,
        weights: &[u8],
        xq: &[i8],
        xs: &[f32],
        rows: usize,
        cols: usize,
        tok: usize,
    ) -> Result<Vec<f32>> {
        let w = vk.upload(weights)?;
        let xi: Vec<i32> = xq.iter().map(|&v| i32::from(v)).collect();
        let q = vk.upload(bytemuck::cast_slice(&xi))?;
        let s = vk.upload(bytemuck::cast_slice(xs))?;
        let (y_buf, y_mem) = vk.create_buffer(
            (rows * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let y = (y_buf, y_mem);
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(std::slice::from_ref(&self.set_layout));
        let set = unsafe { vk.device.allocate_descriptor_sets(&set_info)?[0] };
        let infos = [
            vk::DescriptorBufferInfo::default().buffer(w.0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(q.0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(s.0).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(y.0).range(vk::WHOLE_SIZE),
        ];
        let writes = (0..4)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect::<Vec<_>>();
        unsafe { vk.device.update_descriptor_sets(&writes, &[]) };
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(vk.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { vk.device.allocate_command_buffers(&cmd_info)?[0] };
        let begin = vk::CommandBufferBeginInfo::default();
        unsafe {
            vk.device.begin_command_buffer(cmd, &begin)?;
            vk.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            vk.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            let push = [rows as u32, cols as u32, tok as u32];
            vk.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push),
            );
            vk.device.cmd_dispatch(cmd, rows as u32, 1, 1);
            vk.device.end_command_buffer(cmd)?;
        }
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { vk.device.create_fence(&fence_info, None)? };
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            vk.device.queue_submit(vk.queue, &[submit], fence)?;
            vk.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        unsafe {
            vk.device.destroy_fence(fence, None);
            vk.device.free_command_buffers(vk.cmd_pool, &[cmd]);
            vk.device.free_descriptor_sets(self.pool, &[set])?;
        }
        let bytes = vk.read_buffer(&y, rows * 4)?;
        let out = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vk.destroy_buf(&w);
        vk.destroy_buf(&q);
        vk.destroy_buf(&s);
        vk.destroy_buf(&y);
        Ok(out)
    }
}

impl Drop for GemvQ4 {
    fn drop(&mut self) {
        let _ = (self.set_layout, self.pipeline_layout, self.pipeline, self.pool);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_q4_block_matches_cpu() {
        let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/LFM3-1.7B-Q4_K_M.gguf"));
        let path = path.as_path();
        if !path.exists() {
            return;
        }
        let gguf = lfm_gguf::GgufFile::open(path).expect("gguf");
        let bytes = gguf.tensor_bytes("blk.2.attn_q.weight").expect("tensor");
        let mut block = [0u8; 144];
        block.copy_from_slice(&bytes[..144]);
        let deq = lfm_quant::dequant_q4_k(&block);
        let expect: f32 = deq.iter().sum();
        let Ok(vk) = Vk::open(None) else {
            return;
        };
        let gemv = GemvQ4::new(&vk).expect("pipe");
        let xq = vec![127i8; 256];
        let xs = vec![1.0f32 / 127.0; 8];
        let out = gemv.run(&vk, &block, &xq, &xs, 1, 256, 0).expect("run");
        println!("vulkan {} cpu {}", out[0], expect);
        assert!((out[0] - expect).abs() < expect.abs() * 1e-3 + 1e-3);
    }

    #[test]
    fn real_q4_row_matches_cpu() {
        let path = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/LFM3-1.7B-Q4_K_M.gguf"));
        let path = path.as_path();
        if !path.exists() {
            return;
        }
        let gguf = lfm_gguf::GgufFile::open(path).expect("gguf");
        let info = gguf.tensor("blk.2.attn_q.weight").expect("tensor").clone();
        let cols = info.dims[0] as usize;
        let bytes = gguf.tensor_bytes(&info.name).expect("bytes");
        let deq = lfm_quant::dequant_row_q4_k(&bytes[..cols / 256 * 144], cols).expect("deq");
        let expect: f32 = deq.iter().sum();
        let Ok(vk) = Vk::open(None) else {
            return;
        };
        let gemv = GemvQ4::new(&vk).expect("pipe");
        let xq = vec![127i8; cols];
        let xs = vec![1.0f32 / 127.0; cols / 32];
        let out = gemv.run(&vk, &bytes, &xq, &xs, 1, cols, 0).expect("run");
        println!("row vulkan {} cpu {}", out[0], expect);
        assert!((out[0] - expect).abs() < expect.abs() * 1e-3 + 1e-3);
    }

    #[test]
    fn synthetic_q4_gemv() {
        let Ok(vk) = Vk::open(None) else {
            return;
        };
        let gemv = GemvQ4::new(&vk).expect("pipeline");
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0u16.to_le_bytes());
        for i in 4..8 {
            block[i] = 1;
        }
        for i in 12..16 {
            block[i] = 1;
        }
        for i in 16..144 {
            block[i] = 0xFF;
        }
        let xq = vec![127i8; 256];
        let xs = vec![1.0f32 / 127.0; 8];
        let out = gemv.run(&vk, &block, &xq, &xs, 1, 256, 0).expect("run");
        assert!((out[0] - 3840.0).abs() < 1.0, "got {}", out[0]);
    }
}
