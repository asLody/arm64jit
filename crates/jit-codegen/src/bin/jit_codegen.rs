use std::env;
use std::fs;
use std::path::PathBuf;

use jit_codegen::{collect_flat_from_generated_rust, generate_encoder_module};
use jit_spec::{flatten_instruction_set, parse_instructions_json_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: jit-codegen <input-json> <output-rs> [instruction-set]")?;
    let output = args
        .next()
        .ok_or("usage: jit-codegen <input-json> <output-rs> [instruction-set]")?;
    let set = args.next().unwrap_or_else(|| String::from("A64"));

    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    let flat = if input.is_dir() {
        collect_flat_from_generated_rust(&input)?
    } else {
        let doc = parse_instructions_json_file(&input)?;
        flatten_instruction_set(&doc, &set)?
    };
    let generated = generate_encoder_module(&flat)?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, generated)?;
    Ok(())
}
