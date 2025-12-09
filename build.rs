use vergen::EmitBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild if Python sources change
    println!("cargo:rerun-if-changed=plugins/");
    println!("cargo:rerun-if-changed=pdf2text/");

    EmitBuilder::builder()
        .build_date()
        .build_timestamp()
        .git_branch()
        .git_commit_date()
        .git_commit_timestamp()
        .git_sha(false) // full SHA, not short
        .git_dirty(false) // don't include untracked files
        .emit()?;

    Ok(())
}
