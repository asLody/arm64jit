#![forbid(unsafe_code)]
#![deny(
    missing_docs,
    dead_code,
    nonstandard_style,
    unused_imports,
    unused_mut,
    unused_variables,
    unused_unsafe,
    unreachable_patterns
)]

//! Parser and normalization for Arm AARCHMRS instruction specifications.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

const INSTRUCTION_BITS: usize = 32;

/// Errors produced while reading and normalizing AARCHMRS data.
#[derive(Debug, Error)]
pub enum SpecError {
    /// IO failure while reading a specification file.
    #[error("failed reading specification: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parse failure.
    #[error("failed parsing JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Requested instruction set is not present.
    #[error("instruction set not found: {0}")]
    SetNotFound(String),
    /// Malformed bit value encountered in the source.
    #[error("malformed bit value for {context}: {value}")]
    MalformedBits {
        /// Value context.
        context: &'static str,
        /// Raw value.
        value: String,
    },
    /// Range/bit-string width mismatch.
    #[error("value width mismatch for {context}: expected {expected}, got {got}")]
    WidthMismatch {
        /// Value context.
        context: &'static str,
        /// Expected width.
        expected: usize,
        /// Actual width.
        got: usize,
    },
}

/// Root AARCHMRS document.
#[derive(Debug, Deserialize)]
pub struct InstructionsDoc {
    /// Instruction-set roots (A64/A32/T32).
    pub instructions: Vec<InstructionSet>,
    /// Metadata section.
    pub _meta: Meta,
}

/// Metadata root.
#[derive(Debug, Deserialize)]
pub struct Meta {
    /// License metadata.
    pub license: License,
}

/// License metadata.
#[derive(Debug, Deserialize)]
pub struct License {
    /// Copyright text.
    pub copyright: String,
    /// License detail text.
    pub info: String,
}

/// A top-level instruction set root.
#[derive(Debug, Deserialize)]
pub struct InstructionSet {
    /// Nested instruction groups/instructions.
    pub children: Vec<InstructionGroupOrInstruction>,
    /// Root encoding fragments for this set.
    pub encoding: Encodeset,
    /// Set name (`A64`, `A32`, `T32`).
    pub name: String,
}

/// Nested instruction group.
#[derive(Debug, Deserialize)]
pub struct InstructionGroup {
    /// Nested children.
    pub children: Vec<InstructionGroupOrInstruction>,
    /// Group-level encoding fragments.
    pub encoding: Encodeset,
    /// Group name.
    pub name: String,
}

/// Instruction group child.
#[derive(Debug, Deserialize)]
#[serde(tag = "_type")]
pub enum InstructionGroupOrInstruction {
    /// Nested instruction group.
    #[serde(rename = "Instruction.InstructionGroup")]
    InstructionGroup(InstructionGroup),
    /// Concrete instruction variant.
    #[serde(rename = "Instruction.Instruction")]
    Instruction(Instruction),
    /// Alias entry.
    #[serde(rename = "Instruction.InstructionAlias")]
    InstructionAlias(InstructionAlias),
}

/// Encodeset node.
#[derive(Debug, Deserialize)]
pub struct Encodeset {
    /// Encodeset values.
    pub values: Vec<Encode>,
}

/// Encodeset value.
#[derive(Debug, Deserialize)]
#[serde(tag = "_type")]
pub enum Encode {
    /// Named bitfield.
    #[serde(rename = "Instruction.Encodeset.Field")]
    Field(Field),
    /// Raw fixed bits.
    #[serde(rename = "Instruction.Encodeset.Bits")]
    Bits(Bits),
}

/// Named field definition.
#[derive(Debug, Deserialize)]
pub struct Field {
    /// Field name.
    pub name: String,
    /// Bit range.
    pub range: Range,
    /// Should-be mask in string form.
    pub should_be_mask: Value,
    /// Field value pattern (`x/0/1`).
    pub value: Value,
}

/// Fixed-bit definition.
#[derive(Debug, Deserialize)]
pub struct Bits {
    /// Bit range.
    pub range: Range,
    /// Should-be mask in string form.
    pub should_be_mask: Value,
    /// Fixed pattern (`0/1/x`).
    pub value: Value,
}

/// Bit range definition.
#[derive(Copy, Clone, Debug, Deserialize)]
pub struct Range {
    /// Least-significant bit index.
    pub start: u32,
    /// Number of bits.
    pub width: u32,
}

/// String-encoded value from AARCHMRS JSON.
#[derive(Debug, Deserialize)]
pub struct Value {
    /// Raw string payload.
    pub value: String,
}

impl Value {
    /// Returns the unquoted bit-string payload (e.g. `'010x' -> "010x"`).
    #[must_use]
    pub fn bit_string(&self) -> Option<&str> {
        let raw = self.value.as_str();
        if raw.starts_with('"') && raw.ends_with('"') {
            return Some(&raw[1..raw.len() - 1]);
        }
        if raw.starts_with('\'') && raw.ends_with('\'') {
            return Some(&raw[1..raw.len() - 1]);
        }
        None
    }
}

/// Concrete normalized field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatField {
    /// Field name (deduplicated if split).
    pub name: String,
    /// Least significant bit index.
    pub lsb: u8,
    /// Width in bits.
    pub width: u8,
    /// Signedness inference.
    pub signed: bool,
}

/// Concrete normalized instruction variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatInstruction {
    /// Lower-case mnemonic.
    pub mnemonic: String,
    /// Canonical variant name.
    pub variant: String,
    /// Tree path inside the source dataset.
    pub path: String,
    /// Fixed bit mask.
    pub fixed_mask: u32,
    /// Fixed bit value.
    pub fixed_value: u32,
    /// Ordered operands.
    pub fields: Vec<FlatField>,
    /// Explicit register class hints extracted from `_meta.encoded_in`.
    ///
    /// Keys are normalized semantic field names (for example `rd`, `rn`, `vm`).
    pub register_hints: BTreeMap<String, RegisterClassHint>,
}

/// Register class hint extracted from source metadata.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegisterClassHint {
    /// General-purpose 32-bit register class (`Wn` family).
    Gpr32,
    /// General-purpose 64-bit register class (`Xn` family).
    Gpr64,
    /// SIMD/FP vector register class (`Vn` family).
    Simd,
    /// SVE Z register class (`Zn` family).
    SveZ,
    /// Predicate register class (`Pn` family).
    Predicate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BitCell {
    Fixed(bool),
    Field(String),
}

/// Parses AARCHMRS JSON payload.
///
/// # Errors
///
/// Returns [`SpecError`] when JSON is invalid.
pub fn parse_instructions_json(payload: &str) -> Result<InstructionsDoc, SpecError> {
    Ok(serde_json::from_str(payload)?)
}

/// Parses AARCHMRS JSON file from path.
///
/// # Errors
///
/// Returns [`SpecError`] for IO/JSON failures.
pub fn parse_instructions_json_file(path: &Path) -> Result<InstructionsDoc, SpecError> {
    let payload = fs::read_to_string(path)?;
    parse_instructions_json(&payload)
}

/// Flattens one instruction set (e.g. `A64`) into normalized instruction variants.
///
/// # Errors
///
/// Returns [`SpecError`] on malformed bit patterns or missing set.
pub fn flatten_instruction_set(
    doc: &InstructionsDoc,
    set_name: &str,
) -> Result<Vec<FlatInstruction>, SpecError> {
    let Some(set) = doc.instructions.iter().find(|s| s.name == set_name) else {
        return Err(SpecError::SetNotFound(set_name.to_owned()));
    };

    let mut stack = vec![&set.encoding];
    let mut path = vec![set.name.as_str()];
    let mut out = Vec::new();
    walk_children(&set.children, &mut stack, &mut path, &mut out)?;
    Ok(out)
}

fn walk_children<'a>(
    children: &'a [InstructionGroupOrInstruction],
    stack: &mut Vec<&'a Encodeset>,
    path: &mut Vec<&'a str>,
    out: &mut Vec<FlatInstruction>,
) -> Result<(), SpecError> {
    for child in children {
        match child {
            InstructionGroupOrInstruction::InstructionGroup(group) => {
                path.push(group.name.as_str());
                stack.push(&group.encoding);
                walk_children(&group.children, stack, path, out)?;
                stack.pop();
                path.pop();
            }
            InstructionGroupOrInstruction::Instruction(instruction) => {
                stack.push(&instruction.encoding);
                path.push(instruction.name.as_str());
                let register_hints = extract_register_hints(instruction.meta.as_ref());

                let (mut fixed_mask, mut fixed_value, mut fields) = flatten_stack(stack)?;
                if let Some(condition) = &instruction.condition {
                    apply_condition_constraints(
                        condition,
                        &mut fixed_mask,
                        &mut fixed_value,
                        &mut fields,
                    )?;
                }
                let mnemonic = infer_mnemonic(&instruction.name);
                out.push(FlatInstruction {
                    mnemonic,
                    variant: instruction.name.clone(),
                    path: path.join("/"),
                    fixed_mask,
                    fixed_value,
                    fields,
                    register_hints,
                });

                path.pop();
                stack.pop();
            }
            InstructionGroupOrInstruction::InstructionAlias(_) => {}
        }
    }

    Ok(())
}

fn base_field_name(field_name: &str) -> &str {
    let Some((base, suffix)) = field_name.rsplit_once('_') else {
        return field_name;
    };
    if suffix.chars().all(|ch| ch.is_ascii_digit()) {
        base
    } else {
        field_name
    }
}

fn semantic_field_name(field_name: &str) -> &str {
    if matches!(field_name, "option_13" | "option_15") {
        field_name
    } else {
        base_field_name(field_name)
    }
}

fn has_semantic_field(fields: &[String], name: &str) -> bool {
    fields.iter().any(|field| field == name)
}

fn extract_register_hints(meta: Option<&serde_json::Value>) -> BTreeMap<String, RegisterClassHint> {
    let Some(meta) = meta.and_then(serde_json::Value::as_object) else {
        return BTreeMap::new();
    };
    let Some(encoded_in) = meta
        .get("encoded_in")
        .and_then(serde_json::Value::as_object)
    else {
        return BTreeMap::new();
    };

    let mut hints = BTreeMap::<String, RegisterClassHint>::new();
    for (placeholder, expr) in encoded_in {
        let Some(hint) = register_hint_from_placeholder(placeholder) else {
            continue;
        };
        let mut identifiers = BTreeSet::<String>::new();
        collect_identifier_nodes(expr, &mut identifiers);
        for identifier in identifiers {
            let normalized = identifier.to_ascii_lowercase();
            let semantic = semantic_field_name(&normalized).to_owned();
            if !is_register_semantic_field_name(&semantic) {
                continue;
            }
            hints.insert(semantic, hint);
        }
    }
    hints
}

fn register_hint_from_placeholder(placeholder: &str) -> Option<RegisterClassHint> {
    let trimmed = placeholder
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>');
    let head = trimmed.split('|').next().unwrap_or(trimmed).trim();
    let first = head.chars().next()?;
    if !first.is_ascii_uppercase() {
        return None;
    }
    match first {
        'W' => Some(RegisterClassHint::Gpr32),
        'X' => Some(RegisterClassHint::Gpr64),
        'V' => Some(RegisterClassHint::Simd),
        'Z' => Some(RegisterClassHint::SveZ),
        'P' => Some(RegisterClassHint::Predicate),
        _ => None,
    }
}

fn collect_identifier_nodes(node: &serde_json::Value, out: &mut BTreeSet<String>) {
    match node {
        serde_json::Value::Object(obj) => {
            if obj.get("_type").and_then(serde_json::Value::as_str) == Some("AST.Identifier")
                && let Some(value) = obj.get("value").and_then(serde_json::Value::as_str)
            {
                out.insert(value.to_owned());
            }
            for value in obj.values() {
                collect_identifier_nodes(value, out);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_identifier_nodes(value, out);
            }
        }
        _ => {}
    }
}

fn is_register_semantic_field_name(name: &str) -> bool {
    matches!(
        name,
        "ra" | "rd"
            | "rdn"
            | "rm"
            | "rn"
            | "rs"
            | "rt"
            | "rt2"
            | "rt3"
            | "rt4"
            | "rv"
            | "vd"
            | "vdn"
            | "vm"
            | "vn"
            | "vt"
            | "za"
            | "zad"
            | "zada"
            | "zan"
            | "zat"
            | "zd"
            | "zda"
            | "zdn"
            | "zk"
            | "zm"
            | "zn"
            | "zt"
            | "pd"
            | "pdm"
            | "pdn"
            | "pg"
            | "pm"
            | "pn"
            | "pnd"
            | "png"
            | "pnn"
            | "pnv"
            | "pt"
            | "pv"
    )
}

fn is_fixed_bit_pair(opcode: u32, opcode_mask: u32, hi: u8, lo: u8, accepted: &[(u8, u8)]) -> bool {
    let hi_mask = 1u32 << hi;
    let lo_mask = 1u32 << lo;
    if (opcode_mask & hi_mask) == 0 || (opcode_mask & lo_mask) == 0 {
        return false;
    }

    let hi_value = ((opcode & hi_mask) != 0) as u8;
    let lo_value = ((opcode & lo_mask) != 0) as u8;
    accepted
        .iter()
        .any(|&(accepted_hi, accepted_lo)| accepted_hi == hi_value && accepted_lo == lo_value)
}

fn parse_condition_identifier(node: &serde_json::Value) -> Option<String> {
    let obj = node.as_object()?;
    if obj.get("_type")?.as_str()? != "AST.Identifier" {
        return None;
    }
    Some(obj.get("value")?.as_str()?.to_ascii_lowercase())
}

fn parse_condition_bit_string(node: &serde_json::Value) -> Option<String> {
    let obj = node.as_object()?;
    if obj.get("_type")?.as_str()? != "Values.Value" {
        return None;
    }
    let raw = obj.get("value")?.as_str()?;
    let bits = if raw.starts_with('\'') && raw.ends_with('\'') {
        &raw[1..raw.len() - 1]
    } else if raw.starts_with('"') && raw.ends_with('"') {
        &raw[1..raw.len() - 1]
    } else {
        return None;
    };
    if bits.chars().all(|ch| ch == '0' || ch == '1') {
        Some(bits.to_owned())
    } else {
        None
    }
}

fn parse_condition_equality(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> Option<(String, String)> {
    let name = parse_condition_identifier(left)?;
    let bits = parse_condition_bit_string(right)?;
    Some((name, bits))
}

fn collect_condition_constraints(
    node: &serde_json::Value,
    constraints: &mut HashMap<String, String>,
) -> Result<(), SpecError> {
    let Some(obj) = node.as_object() else {
        return Ok(());
    };
    let Some(kind) = obj.get("_type").and_then(serde_json::Value::as_str) else {
        return Ok(());
    };
    if kind != "AST.BinaryOp" {
        return Ok(());
    }

    let Some(op) = obj.get("op").and_then(serde_json::Value::as_str) else {
        return Ok(());
    };
    let Some(left) = obj.get("left") else {
        return Ok(());
    };
    let Some(right) = obj.get("right") else {
        return Ok(());
    };

    match op {
        "&&" => {
            collect_condition_constraints(left, constraints)?;
            collect_condition_constraints(right, constraints)?;
        }
        "==" => {
            let pair = parse_condition_equality(left, right)
                .or_else(|| parse_condition_equality(right, left));
            let Some((name, bits)) = pair else {
                return Ok(());
            };
            match constraints.get(&name) {
                Some(existing) if existing != &bits => {
                    return Err(SpecError::MalformedBits {
                        context: "instruction.condition",
                        value: format!(
                            "conflicting condition binding for {name}: {existing}/{bits}"
                        ),
                    });
                }
                Some(_) => {}
                None => {
                    constraints.insert(name, bits);
                }
            }
        }
        _ => {}
    }

    Ok(())
}

fn apply_condition_constraints(
    condition: &serde_json::Value,
    fixed_mask: &mut u32,
    fixed_value: &mut u32,
    fields: &mut Vec<FlatField>,
) -> Result<(), SpecError> {
    let mut constraints = HashMap::<String, String>::new();
    collect_condition_constraints(condition, &mut constraints)?;
    if constraints.is_empty() {
        return Ok(());
    }

    let mut remove_indices = Vec::<usize>::new();
    for (idx, field) in fields.iter().enumerate() {
        let normalized = field.name.to_ascii_lowercase();
        let semantic = semantic_field_name(&normalized);
        let Some(bits) = constraints
            .get(&normalized)
            .or_else(|| constraints.get(semantic))
            .cloned()
        else {
            continue;
        };

        if bits.len() != field.width as usize {
            return Err(SpecError::WidthMismatch {
                context: "instruction.condition",
                expected: field.width as usize,
                got: bits.len(),
            });
        }

        for (offset, ch) in bits.chars().enumerate() {
            let bit = usize::from(field.lsb) + (usize::from(field.width) - 1 - offset);
            *fixed_mask |= 1u32 << bit;
            if ch == '1' {
                *fixed_value |= 1u32 << bit;
            } else {
                *fixed_value &= !(1u32 << bit);
            }
        }
        remove_indices.push(idx);
    }

    for idx in remove_indices.into_iter().rev() {
        fields.remove(idx);
    }
    Ok(())
}

fn infer_mnemonic(variant: &str) -> String {
    let head = variant
        .split('_')
        .next()
        .unwrap_or(variant)
        .to_ascii_lowercase();

    if let Some(stripped) = head.strip_suffix(".cond") {
        stripped.to_owned()
    } else {
        head
    }
}

fn infer_signed_field(
    field_name: &str,
    opcode: u32,
    opcode_mask: u32,
    semantic_fields: &[String],
) -> bool {
    let normalized = field_name.to_ascii_lowercase();
    let semantic_name = semantic_field_name(&normalized);

    if semantic_name.starts_with("simm")
        || semantic_name.starts_with("soffset")
        || semantic_name.contains("offset")
    {
        return true;
    }

    if matches!(semantic_name, "imm26" | "imm19" | "imm14") {
        return true;
    }

    if semantic_name == "immhi"
        && has_semantic_field(semantic_fields, "immhi")
        && has_semantic_field(semantic_fields, "immlo")
    {
        return true;
    }

    if semantic_name == "imm7"
        && has_semantic_field(semantic_fields, "rt")
        && has_semantic_field(semantic_fields, "rt2")
        && has_semantic_field(semantic_fields, "rn")
    {
        return true;
    }

    if semantic_name == "imm9"
        && has_semantic_field(semantic_fields, "rt")
        && has_semantic_field(semantic_fields, "rn")
        && is_fixed_bit_pair(opcode, opcode_mask, 11, 10, &[(0, 0), (0, 1), (1, 1)])
    {
        return true;
    }

    false
}

fn flatten_stack(stack: &[&Encodeset]) -> Result<(u32, u32, Vec<FlatField>), SpecError> {
    let mut cells: [Option<BitCell>; INSTRUCTION_BITS] = core::array::from_fn(|_| None);

    for encodeset in stack {
        for enc in &encodeset.values {
            match enc {
                Encode::Field(field) => apply_field(&mut cells, field)?,
                Encode::Bits(bits) => apply_bits(&mut cells, bits)?,
            }
        }
    }

    let mut fixed_mask = 0u32;
    let mut fixed_value = 0u32;
    for (idx, cell) in cells.iter().enumerate() {
        if let Some(BitCell::Fixed(bit)) = cell {
            fixed_mask |= 1u32 << idx;
            if *bit {
                fixed_value |= 1u32 << idx;
            }
        }
    }

    let mut fields = Vec::<FlatField>::new();
    let mut bit = 0usize;
    while bit < INSTRUCTION_BITS {
        let Some(BitCell::Field(name)) = cells[bit].as_ref() else {
            bit += 1;
            continue;
        };

        let mut width = 1usize;
        while bit + width < INSTRUCTION_BITS {
            let Some(BitCell::Field(next)) = cells[bit + width].as_ref() else {
                break;
            };
            if next != name {
                break;
            }
            width += 1;
        }

        fields.push(FlatField {
            name: name.clone(),
            lsb: bit as u8,
            width: width as u8,
            signed: false,
        });

        bit += width;
    }

    let semantic_fields = fields
        .iter()
        .map(|field| {
            let normalized = field.name.to_ascii_lowercase();
            semantic_field_name(&normalized).to_owned()
        })
        .collect::<Vec<_>>();
    for field in &mut fields {
        field.signed = infer_signed_field(&field.name, fixed_value, fixed_mask, &semantic_fields);
    }

    dedup_split_names(&mut fields);
    fields.sort_by(|left, right| right.lsb.cmp(&left.lsb).then(left.name.cmp(&right.name)));

    Ok((fixed_mask, fixed_value, fields))
}

fn dedup_split_names(fields: &mut [FlatField]) {
    let mut counts = HashMap::<String, usize>::new();
    for field in fields.iter() {
        *counts.entry(field.name.clone()).or_default() += 1;
    }

    for field in fields.iter_mut() {
        if counts.get(&field.name).copied().unwrap_or_default() > 1 {
            field.name = format!("{}_{}", field.name, field.lsb);
        }
    }
}

fn apply_bits(
    cells: &mut [Option<BitCell>; INSTRUCTION_BITS],
    bits: &Bits,
) -> Result<(), SpecError> {
    let pattern = parse_pattern(
        bits.value.bit_string(),
        "bits.value",
        bits.range.width as usize,
    )?;

    for (bit_idx, ch) in bit_positions(bits.range).zip(pattern.chars()) {
        match ch {
            '0' => cells[bit_idx as usize] = Some(BitCell::Fixed(false)),
            '1' => cells[bit_idx as usize] = Some(BitCell::Fixed(true)),
            'x' => cells[bit_idx as usize] = None,
            _ => {
                return Err(SpecError::MalformedBits {
                    context: "bits.value",
                    value: pattern.to_owned(),
                });
            }
        }
    }

    Ok(())
}

fn apply_field(
    cells: &mut [Option<BitCell>; INSTRUCTION_BITS],
    field: &Field,
) -> Result<(), SpecError> {
    let pattern = parse_pattern(
        field.value.bit_string(),
        "field.value",
        field.range.width as usize,
    )?;

    for (bit_idx, ch) in bit_positions(field.range).zip(pattern.chars()) {
        match ch {
            'x' => cells[bit_idx as usize] = Some(BitCell::Field(field.name.clone())),
            '0' => cells[bit_idx as usize] = Some(BitCell::Fixed(false)),
            '1' => cells[bit_idx as usize] = Some(BitCell::Fixed(true)),
            _ => {
                return Err(SpecError::MalformedBits {
                    context: "field.value",
                    value: pattern.to_owned(),
                });
            }
        }
    }

    Ok(())
}

fn parse_pattern<'a>(
    value: Option<&'a str>,
    context: &'static str,
    expected_width: usize,
) -> Result<&'a str, SpecError> {
    let Some(bits) = value else {
        return Err(SpecError::MalformedBits {
            context,
            value: String::from("<unquoted>"),
        });
    };

    if bits.len() != expected_width {
        return Err(SpecError::WidthMismatch {
            context,
            expected: expected_width,
            got: bits.len(),
        });
    }

    Ok(bits)
}

fn bit_positions(range: Range) -> impl Iterator<Item = u32> {
    let start = range.start;
    let width = range.width;
    (start..start + width).rev()
}

/// Instruction entry (leaf).
#[derive(Debug, Deserialize)]
pub struct Instruction {
    /// Optional source metadata.
    #[serde(default, rename = "_meta")]
    pub meta: Option<serde_json::Value>,
    /// Leaf encoding.
    pub encoding: Encodeset,
    /// Variant name.
    pub name: String,
    /// Optional condition AST (feature gates and selector field constraints).
    #[serde(default)]
    pub condition: Option<serde_json::Value>,
    /// Operation ID.
    pub operation_id: String,
    /// Nested children (typically aliases).
    #[serde(default)]
    pub children: Vec<InstructionGroupOrInstruction>,
}

/// Alias entry.
#[derive(Debug, Deserialize)]
pub struct InstructionAlias {
    /// Alias name.
    pub name: String,
    /// Operation ID.
    pub operation_id: String,
    /// Nested children.
    #[serde(default)]
    pub children: Vec<InstructionGroupOrInstruction>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
{
  "_meta": { "license": { "copyright": "x", "info": "x" } },
  "instructions": [
    {
      "name": "A64",
      "encoding": { "values": [] },
      "children": [
        {
          "_type": "Instruction.InstructionGroup",
          "name": "dpimm",
          "encoding": { "values": [] },
          "children": [
            {
              "_type": "Instruction.Instruction",
              "name": "ADD_64_addsub_imm",
              "operation_id": "ADD_64_addsub_imm",
              "encoding": {
                "values": [
                  { "_type": "Instruction.Encodeset.Bits", "range": { "start": 23, "width": 9 }, "should_be_mask": { "value": "'000000000'" }, "value": { "value": "'100100010'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "sh", "range": { "start": 22, "width": 1 }, "should_be_mask": { "value": "'0'" }, "value": { "value": "'x'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "imm12", "range": { "start": 10, "width": 12 }, "should_be_mask": { "value": "'000000000000'" }, "value": { "value": "'xxxxxxxxxxxx'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Rn", "range": { "start": 5, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Rd", "range": { "start": 0, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } }
                ]
              }
            }
          ]
        }
      ]
    }
  ]
}
"#;

    #[test]
    fn parse_and_flatten() {
        let doc = parse_instructions_json(SAMPLE).expect("parse should succeed");
        let flat = flatten_instruction_set(&doc, "A64").expect("flatten should succeed");
        assert_eq!(flat.len(), 1);

        let add = &flat[0];
        assert_eq!(add.mnemonic, "add");
        assert_eq!(add.variant, "ADD_64_addsub_imm");
        assert_eq!(add.fixed_value, 0b100100010u32 << 23);
        assert_eq!(add.fields.len(), 4);
        assert_eq!(add.fields[0].name, "sh");
        assert_eq!(add.fields[0].lsb, 22);
        assert_eq!(add.fields[1].name, "imm12");
        assert_eq!(add.fields[1].lsb, 10);
        assert!(add.register_hints.is_empty());
    }

    #[test]
    fn encoded_in_metadata_provides_register_class_hints() {
        let sample = r#"
{
  "_meta": { "license": { "copyright": "x", "info": "x" } },
  "instructions": [
    {
      "name": "A64",
      "encoding": { "values": [] },
      "children": [
        {
          "_type": "Instruction.InstructionGroup",
          "name": "dp3src",
          "encoding": { "values": [] },
          "children": [
            {
              "_type": "Instruction.Instruction",
              "_meta": {
                "encoded_in": {
                  "<Xd>": [{ "_type": "AST.Identifier", "value": "Rd" }],
                  "<Xa>": [{ "_type": "AST.Identifier", "value": "Ra" }],
                  "<Wn>": [{ "_type": "AST.Identifier", "value": "Rn" }],
                  "<Wm>": [{ "_type": "AST.Identifier", "value": "Rm" }]
                }
              },
              "name": "SMADDL_64WA_dp_3src",
              "operation_id": "SMADDL",
              "encoding": {
                "values": [
                  { "_type": "Instruction.Encodeset.Bits", "range": { "start": 21, "width": 3 }, "should_be_mask": { "value": "'000'" }, "value": { "value": "'001'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Rm", "range": { "start": 16, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Ra", "range": { "start": 10, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Rn", "range": { "start": 5, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } },
                  { "_type": "Instruction.Encodeset.Field", "name": "Rd", "range": { "start": 0, "width": 5 }, "should_be_mask": { "value": "'00000'" }, "value": { "value": "'xxxxx'" } }
                ]
              }
            }
          ]
        }
      ]
    }
  ]
}
"#;

        let doc = parse_instructions_json(sample).expect("parse should succeed");
        let flat = flatten_instruction_set(&doc, "A64").expect("flatten should succeed");
        assert_eq!(flat.len(), 1);
        let inst = &flat[0];
        assert_eq!(
            inst.register_hints.get("rd"),
            Some(&RegisterClassHint::Gpr64)
        );
        assert_eq!(
            inst.register_hints.get("ra"),
            Some(&RegisterClassHint::Gpr64)
        );
        assert_eq!(
            inst.register_hints.get("rn"),
            Some(&RegisterClassHint::Gpr32)
        );
        assert_eq!(
            inst.register_hints.get("rm"),
            Some(&RegisterClassHint::Gpr32)
        );
    }
}
