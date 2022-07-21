#[path = "src/shader/validate.rs"]
mod validate;

use anyhow::Context as _;

fn main() -> anyhow::Result<()> {
    check_shaders()
}

fn check_shaders() -> anyhow::Result<()> {
    for entry in walkdir::WalkDir::new(".") {
        let path = entry?.into_path();
        if path.extension() != Some("wgsl".as_ref()) {
            continue;
        }

        println!("cargo:rerun-if-changed={}", path.display());

        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("could not load shader: {path:?}"))?;

        validate::validate(&source, &path.display().to_string())?;
    }

    Ok(())
}
