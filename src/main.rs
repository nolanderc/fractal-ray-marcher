mod frame_counter;
mod gpu;

use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use anyhow::Context;
use frame_counter::FrameCounter;
use gpu::GpuContext;
use pollster::FutureExt as _;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::VirtualKeyCode,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[macro_use]
extern crate tracing;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wgpu=warn".parse().unwrap())
                // silence warnings about leaked objects (bug in wgpu)
                .add_directive("wgpu_hal::auxil::dxgi::exception=error".parse().unwrap()),
        )
        .init();

    let event_loop = EventLoop::new();
    let mut state = State::new(&event_loop).block_on()?;
    let mut last_update = Instant::now();
    let mut modifiers = winit::event::ModifiersState::default();

    let mut held_keys = HashSet::new();
    let mut held_buttons = HashSet::new();

    event_loop.run(move |event, _, flow| {
        use winit::event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent};
        match event {
            Event::WindowEvent { event, .. } => match event {
                // exit when the window closes or the user hits ESC
                WindowEvent::CloseRequested => *flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        if input.state == ElementState::Pressed {
                            held_keys.insert(key);
                        } else {
                            held_keys.remove(&key);
                        }
                    }

                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Escape) => *flow = ControlFlow::Exit,

                        // reset camera focus to default
                        Some(VirtualKeyCode::Key0) => state.camera = Default::default(),

                        _ => {}
                    }
                }

                WindowEvent::MouseInput { button, state, .. } => {
                    if state == ElementState::Pressed {
                        held_buttons.insert(button);
                    } else {
                        held_buttons.remove(&button);
                    }
                }

                // handle changes in window size
                WindowEvent::Resized(new_size) => state.resize(new_size),
                WindowEvent::ScaleFactorChanged {
                    scale_factor: _,
                    new_inner_size,
                } => state.resize(*new_inner_size),

                WindowEvent::ModifiersChanged(new_modifiers) => modifiers = new_modifiers,

                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    if held_buttons.contains(&MouseButton::Left) {
                        let size = state.window.inner_size();
                        let scale = -1.0 / size.width as f32;
                        let dx = dx as f32 * scale;
                        let dy = dy as f32 * scale;

                        if modifiers.shift() {
                            state.camera.translate(3.0 * dx, 3.0 * dy);
                        } else {
                            state.camera.rotate(dx, dy);
                        }
                    }
                }
                DeviceEvent::MouseWheel {
                    delta: winit::event::MouseScrollDelta::LineDelta(_, dy),
                } => {
                    state.camera.zoom(dy);
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                let now = Instant::now();
                let dt = now.saturating_duration_since(last_update);
                if let Err(error) = state.update(dt) {
                    error!("could not update state: {:#}", error);
                }
                last_update = now;

                if let Err(error) = state.render() {
                    error!("could not render frame: {:#}", error);
                }
            }
            _ => {}
        }
    })
}

struct State {
    instance: wgpu::Instance,
    gpu: GpuContext,

    window: Window,
    surface: wgpu::Surface,
    surface_format: wgpu::TextureFormat,

    /// State for rendering
    render_pipeline: RenderPipeline,

    /// Counts the frame timings
    frame_counter: FrameCounter,

    /// Time since the app was started
    time: Duration,

    /// Camera controlled by the user
    camera: Camera,
}

struct RenderPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera: glam::Mat4,
    near: f32,
    far: f32,
    time: f32,
    _padding: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Camera {
    focus: glam::Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            focus: glam::vec3(0.0, 0.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            distance: 3.0,
            fov: 70f32.to_radians(),
            near: 0.1,
            far: 10.0,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> glam::Vec3 {
        let dx = self.yaw.cos() * self.pitch.cos();
        let dy = self.pitch.sin();
        let dz = self.yaw.sin() * self.pitch.cos();
        glam::vec3(dx, dy, dz).normalize_or_zero()
    }

    pub fn transform(&self, size: PhysicalSize<u32>) -> glam::Mat4 {
        let aspect_ratio = size.width as f32 / size.height as f32;
        let perspective = glam::Mat4::perspective_lh(self.fov, aspect_ratio, self.near, self.far);

        let view = glam::Mat4::look_at_lh(
            self.focus - self.distance * self.forward(),
            self.focus,
            glam::Vec3::Y,
        );

        perspective * view
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        use std::f32::consts::{FRAC_PI_2, TAU};

        self.yaw += TAU * dx;
        self.pitch += TAU * dy;

        self.yaw %= TAU;
        self.pitch = self.pitch.clamp(-FRAC_PI_2 + 0.1, FRAC_PI_2 - 0.1);
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        let right = glam::vec3(self.yaw.sin(), 0.0, -self.yaw.cos());
        let up = right.cross(self.forward());
        self.focus += self.distance * (dx * right + dy * up);
    }

    pub fn zoom(&mut self, steps: f32) {
        self.distance *= f32::powf(0.9, steps);
        self.distance = self.distance.max(0.1);
    }
}

impl RenderPipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let vertex = wgpu::VertexState {
            module: &module,
            entry_point: "vertex_main",
            buffers: &[],
        };

        let fragment = wgpu::FragmentState {
            module: &module,
            entry_point: "fragment_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        };

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &Self::bind_group_layout_entries(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &Self::bind_group_entries(&uniform_buffer),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex,
            primitive,
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(fragment),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
        }
    }

    fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 1] {
        fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::all(),
                ty,
                count: None,
            }
        }

        fn uniform_buffer<T>() -> wgpu::BindingType {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<T>() as _),
            }
        }

        [entry(0, uniform_buffer::<Uniforms>())]
    }

    fn bind_group_entries(uniforms: &wgpu::Buffer) -> [wgpu::BindGroupEntry; 1] {
        [wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        }]
    }
}

impl State {
    async fn new(event_loop: &EventLoop<()>) -> anyhow::Result<State> {
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1280, 720))
            .build(event_loop)
            .context("could not create a window")?;

        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
        let gpu = GpuContext::new(&instance, &surface).await?;

        let surface_format = surface
            .get_supported_formats(&gpu.adapter)
            .get(0)
            .copied()
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        surface.configure(
            &gpu.device,
            &Self::surface_configuration(surface_format, window.inner_size()),
        );

        let render_pipeline = RenderPipeline::new(&gpu.device, surface_format);

        Ok(State {
            instance,
            gpu,
            window,
            surface,
            surface_format,
            render_pipeline,
            frame_counter: FrameCounter::new(30),
            time: Duration::from_secs(0),
            camera: Default::default(),
        })
    }

    fn surface_configuration(
        format: wgpu::TextureFormat,
        size: PhysicalSize<u32>,
    ) -> wgpu::SurfaceConfiguration {
        info!(?format, ?size, "configuring surface");
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        dbg!(new_size);
        if new_size.width * new_size.height == 0 {
            return;
        }
        self.surface.configure(
            &self.gpu.device,
            &Self::surface_configuration(self.surface_format, new_size),
        );
        let _ = self.render();
    }

    pub fn update(&mut self, dt: Duration) -> anyhow::Result<()> {
        self.time += dt;
        Ok(())
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        // make sure that the window has some drawable area
        let window_size = self.window.inner_size();
        if window_size.width * window_size.height == 0 {
            return Ok(());
        }

        self.update_buffers();

        self.frame_counter.on_new_frame();
        if self.frame_counter.frame_number() % 30 == 0 {
            let fps = self.frame_counter.frame_rate();
            let title = format!("SDF @ {}", fps.round());
            self.window.set_title(&title);
        }

        let frame = self
            .get_next_frame()
            .context("could not get next texture in swapchain")?;

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let frame_view = frame.texture.create_view(&Default::default());

            let color_attachment = wgpu::RenderPassColorAttachment {
                view: &frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: true,
                },
            };

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.render_pipeline.pipeline);
            rpass.set_bind_group(0, &self.render_pipeline.bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        let command_buffer = encoder.finish();
        self.gpu.queue.submit([command_buffer]);

        frame.present();

        Ok(())
    }

    fn get_next_frame(&mut self) -> anyhow::Result<wgpu::SurfaceTexture> {
        let retries = 3;

        for _ in 0..retries {
            match self.surface.get_current_texture() {
                Ok(frame) => return Ok(frame),
                Err(error @ (wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated)) => {
                    if error == wgpu::SurfaceError::Lost {
                        // recreate the surface
                        self.surface = unsafe { self.instance.create_surface(&self.window) };
                    }

                    dbg!(error);

                    // reconfigure the surface
                    self.surface.configure(
                        &self.gpu.device,
                        &Self::surface_configuration(self.surface_format, self.window.inner_size()),
                    );
                }
                Err(error) => return Err(error.into()),
            }
        }

        Err(anyhow::format_err!("failed after {retries} attempts"))
    }

    fn update_buffers(&mut self) {
        self.gpu.queue.write_buffer(
            &self.render_pipeline.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                camera: self.camera.transform(self.window.inner_size()).inverse(),
                near: self.camera.near,
                far: self.camera.far,
                time: self.time.as_secs_f32(),
                ..bytemuck::Zeroable::zeroed()
            }]),
        )
    }
}
