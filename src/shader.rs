mod validate;

use std::path::Path;

use anyhow::Context;

macro_rules! load_shader {
    ($path:literal) => {{
        use std::sync::atomic::{AtomicBool, Ordering};
        static INITIAL_LOAD: AtomicBool = AtomicBool::new(true);

        if INITIAL_LOAD.swap(false, Ordering::SeqCst) {
            anyhow::Ok(wgpu::include_wgsl!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/shaders/",
                $path
            )))
        } else {
            $crate::shader::read_shader($path)
        }
    }};
}

pub fn read_shader(name: &str) -> anyhow::Result<wgpu::ShaderModuleDescriptor> {
    let path = Path::new("shaders").join(name);
    let source = std::fs::read_to_string(&path)
        .with_context(|| format!("could not open '{}'", path.display()))?;

    validate::validate(&source, name).context("could not validate shader")?;

    Ok(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}
