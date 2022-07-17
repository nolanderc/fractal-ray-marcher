use anyhow::Context;
use pollster::FutureExt;

pub struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface,
    ) -> anyhow::Result<GpuContext> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await
            .context("could not get a graphics adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: adapter.features() & !wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                    limits: adapter.limits(),
                },
                None,
            )
            .block_on()?;

        Ok(GpuContext {
            adapter,
            device,
            queue,
        })
    }
}
