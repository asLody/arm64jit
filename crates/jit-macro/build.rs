use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use jit_codegen::generate_macro_normalization_module;
use jit_spec::{flatten_instruction_set, parse_instructions_json_file};
use serde::Deserialize;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Deserialize)]
struct AliasRuleSpec {
    alias: String,
    canonical: String,
    transform: String,
    #[serde(default)]
    fixed_imms: Option<i16>,
}

fn main() {
    if let Err(err) = run() {
        panic!("failed generating jit macro normalization rules: {err}");
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    let instructions_json = manifest_dir
        .join("..")
        .join("jit")
        .join("spec")
        .join("Instructions.json");
    let alias_rules_json = manifest_dir
        .join("..")
        .join("jit")
        .join("spec")
        .join("alias_rules.json");
    let build_rs = manifest_dir.join("build.rs");
    let codegen_core = manifest_dir
        .join("..")
        .join("jit-codegen")
        .join("src")
        .join("core.rs");
    let spec_lib = manifest_dir
        .join("..")
        .join("jit-spec")
        .join("src")
        .join("lib.rs");

    let tracked_inputs = [
        instructions_json.clone(),
        alias_rules_json.clone(),
        build_rs,
        codegen_core,
        spec_lib,
    ];
    for path in &tracked_inputs {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let cache_key = compute_cache_key(&tracked_inputs)?;
    let cache_dir = build_cache_root(&out_dir, "jit-macro").join(cache_key);
    let output_files = ["generated_macro_rules.rs"];

    if restore_cached_outputs(&cache_dir, &out_dir, &output_files)? {
        return Ok(());
    }

    let doc = parse_instructions_json_file(&instructions_json)?;
    let flat = flatten_instruction_set(&doc, "A64")?;

    let generated = generate_macro_normalization_module(&flat)?;
    let alias_rules = load_alias_rules(&alias_rules_json)?;
    let generated_aliases = generate_alias_rules_module(&alias_rules)?;

    fs::write(
        out_dir.join(output_files[0]),
        format!("{generated}\n{generated_aliases}"),
    )?;
    persist_cached_outputs(&cache_dir, &out_dir, &output_files)?;
    Ok(())
}

fn load_alias_rules(path: &Path) -> Result<Vec<AliasRuleSpec>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut rules: Vec<AliasRuleSpec> = serde_json::from_str(&text)?;
    rules.sort_by(|lhs, rhs| lhs.alias.cmp(&rhs.alias));
    rules.dedup_by(|lhs, rhs| lhs.alias == rhs.alias);
    Ok(rules)
}

fn transform_variant_name(name: &str) -> Result<&'static str, Box<dyn std::error::Error>> {
    match name {
        "pure_rename" => Ok("PureRename"),
        "ret_default" => Ok("RetDefault"),
        "cmp_like" => Ok("CmpLike"),
        "cmn_like" => Ok("CmnLike"),
        "tst_like" => Ok("TstLike"),
        "mov_like" => Ok("MovLike"),
        "mul_like" => Ok("MulLike"),
        "ror_like" => Ok("RorLike"),
        "mvn_like" => Ok("MvnLike"),
        "smull_like" => Ok("SmullLike"),
        "umull_like" => Ok("UmullLike"),
        "cinc_like" => Ok("CincLike"),
        "cset_like" => Ok("CsetLike"),
        "cneg_like" => Ok("CnegLike"),
        "bitfield_bfi" => Ok("BitfieldBfi"),
        "bitfield_bfxil" => Ok("BitfieldBfxil"),
        "bitfield_bfc" => Ok("BitfieldBfc"),
        "bitfield_ubfx" => Ok("BitfieldUbfx"),
        "bitfield_sbfx" => Ok("BitfieldSbfx"),
        "bitfield_sbfiz" => Ok("BitfieldSbfiz"),
        "bitfield_extract_fixed" => Ok("BitfieldExtractFixed"),
        "extend_long_zero" => Ok("ExtendLongZero"),
        "stsetl_like" => Ok("StsetlLike"),
        "dc_like" => Ok("DcLike"),
        _ => Err(format!("unknown alias transform {name:?}").into()),
    }
}

fn generate_alias_rules_module(
    rules: &[AliasRuleSpec],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut out = String::new();
    out.push_str("// @generated alias rules for jit macro. DO NOT EDIT.\n");
    out.push_str(
        "\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) enum AliasTransform {\n\
    PureRename,\n\
    RetDefault,\n\
    CmpLike,\n\
    CmnLike,\n\
    TstLike,\n\
    MovLike,\n\
    MulLike,\n\
    RorLike,\n\
    MvnLike,\n\
    SmullLike,\n\
    UmullLike,\n\
    CincLike,\n\
    CsetLike,\n\
    CnegLike,\n\
    BitfieldBfi,\n\
    BitfieldBfxil,\n\
    BitfieldBfc,\n\
    BitfieldUbfx,\n\
    BitfieldSbfx,\n\
    BitfieldSbfiz,\n\
    BitfieldExtractFixed,\n\
    ExtendLongZero,\n\
    StsetlLike,\n\
    DcLike,\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct AliasRule {\n\
    pub alias: &'static str,\n\
    pub canonical: &'static str,\n\
    pub transform: AliasTransform,\n\
    pub fixed_imms: i16,\n\
}\n\n\
pub(crate) static ALIAS_RULES: &[AliasRule] = &[\n",
    );

    for rule in rules {
        let transform = transform_variant_name(&rule.transform)?;
        out.push_str("    AliasRule {\n");
        out.push_str(&format!("        alias: {:?},\n", rule.alias));
        out.push_str(&format!("        canonical: {:?},\n", rule.canonical));
        out.push_str(&format!(
            "        transform: AliasTransform::{transform},\n"
        ));
        out.push_str(&format!(
            "        fixed_imms: {},\n",
            rule.fixed_imms.unwrap_or(-1)
        ));
        out.push_str("    },\n");
    }
    out.push_str("];\n\n");
    out.push_str(
        "pub(crate) fn lookup_alias_rule(mnemonic: &str) -> Option<&'static AliasRule> {\n",
    );
    out.push_str("    let idx = ALIAS_RULES\n");
    out.push_str("        .binary_search_by(|rule| rule.alias.cmp(mnemonic))\n");
    out.push_str("        .ok()?;\n");
    out.push_str("    Some(&ALIAS_RULES[idx])\n");
    out.push_str("}\n");
    Ok(out)
}

fn fnv1a_update(mut state: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        state ^= u64::from(byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

fn compute_cache_key(paths: &[PathBuf]) -> Result<String, Box<dyn std::error::Error>> {
    let mut state = FNV_OFFSET_BASIS;
    for path in paths {
        state = fnv1a_update(state, path.as_os_str().as_encoded_bytes());
        state = fnv1a_update(state, &[0xff]);
        let bytes = fs::read(path)?;
        state = fnv1a_update(state, &bytes);
        state = fnv1a_update(state, &[0x00]);
    }
    Ok(format!("{state:016x}"))
}

fn build_cache_root(out_dir: &Path, bucket: &str) -> PathBuf {
    for ancestor in out_dir.ancestors() {
        if ancestor.file_name() == Some(OsStr::new("target")) {
            return ancestor.join(".jit-build-cache").join(bucket);
        }
    }
    out_dir.join(".jit-build-cache").join(bucket)
}

fn restore_cached_outputs(
    cache_dir: &Path,
    out_dir: &Path,
    output_files: &[&str],
) -> Result<bool, Box<dyn std::error::Error>> {
    if !cache_dir.is_dir() {
        return Ok(false);
    }

    for name in output_files {
        let src = cache_dir.join(name);
        if !src.is_file() {
            return Ok(false);
        }
    }

    fs::create_dir_all(out_dir)?;
    for name in output_files {
        fs::copy(cache_dir.join(name), out_dir.join(name))?;
    }
    Ok(true)
}

fn persist_cached_outputs(
    cache_dir: &Path,
    out_dir: &Path,
    output_files: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    for name in output_files {
        fs::copy(out_dir.join(name), cache_dir.join(name))?;
    }
    Ok(())
}
