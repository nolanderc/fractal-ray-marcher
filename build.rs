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

        let module = match naga::front::wgsl::parse_str(&source) {
            Ok(module) => module,
            Err(error) => {
                error.emit_to_stderr_with_path(&source, &path.display().to_string());
                return Err(anyhow::format_err!("could not parse shader: {path:?}"));
            }
        };

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::default(),
            naga::valid::Capabilities::default(),
        );

        if let Err(error) = validator.validate(&module) {
            emit_validation_error(&error, &source, &path.display().to_string());
            return Err(anyhow::format_err!("failed to validate shader: {path:?}"));
        }
    }

    Ok(())
}

fn emit_validation_error(
    error: &naga::WithSpan<naga::valid::ValidationError>,
    source: &str,
    path: &str,
) {
    use codespan_reporting::{
        diagnostic::{Diagnostic, Label},
        files::SimpleFile,
        term,
    };

    let files = SimpleFile::new(path, source);
    let config = term::Config::default();
    let mut writer = term::termcolor::Ansi::new(std::io::stderr());

    let diagnostic = Diagnostic::error()
        .with_message(error.to_string())
        .with_labels(
            error
                .spans()
                .map(|&(span, ref desc)| {
                    Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
                })
                .collect(),
        )
        .with_notes({
            let mut notes = Vec::new();
            let mut source: &dyn std::error::Error = error;
            while let Some(next) = std::error::Error::source(source) {
                notes.push(next.to_string());
                source = next;
            }
            notes
        });

    term::emit(&mut writer, &config, &files, &diagnostic).expect("cannot write error");
}
