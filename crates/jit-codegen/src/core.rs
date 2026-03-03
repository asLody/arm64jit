use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use jit_spec::{FlatField, FlatInstruction};
use thiserror::Error;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum GeneratedOperandKind {
    GprRegister,
    Gpr32Register,
    Gpr64Register,
    SimdRegister,
    SveZRegister,
    PredicateRegister,
    Immediate,
    Condition,
    ShiftKind,
    ExtendKind,
    SysRegPart,
    Arrangement,
    Lane,
}

impl GeneratedOperandKind {
    fn as_rust(self) -> &'static str {
        match self {
            Self::GprRegister => "OperandConstraintKind::GprRegister",
            Self::Gpr32Register => "OperandConstraintKind::Gpr32Register",
            Self::Gpr64Register => "OperandConstraintKind::Gpr64Register",
            Self::SimdRegister => "OperandConstraintKind::SimdRegister",
            Self::SveZRegister => "OperandConstraintKind::SveZRegister",
            Self::PredicateRegister => "OperandConstraintKind::PredicateRegister",
            Self::Immediate => "OperandConstraintKind::Immediate",
            Self::Condition => "OperandConstraintKind::Condition",
            Self::ShiftKind => "OperandConstraintKind::ShiftKind",
            Self::ExtendKind => "OperandConstraintKind::ExtendKind",
            Self::SysRegPart => "OperandConstraintKind::SysRegPart",
            Self::Arrangement => "OperandConstraintKind::Arrangement",
            Self::Lane => "OperandConstraintKind::Lane",
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum VariantWidthHint {
    W32,
    W64,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct OrderedField {
    index: u8,
    rank: u16,
    kind: GeneratedOperandKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GeneratedSplitImmediatePlan {
    first_slot: u8,
    second_slot: u8,
    kind: GeneratedSplitImmediateKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum GeneratedSplitImmediateKind {
    AdrLike {
        immlo_field_index: u8,
        immhi_field_index: u8,
        scale: i64,
    },
    BitIndex6 {
        b5_field_index: u8,
        b40_field_index: u8,
    },
    LogicalImmRs {
        immr_field_index: u8,
        imms_field_index: u8,
        reg_size: u8,
    },
    LogicalImmNrs {
        n_field_index: u8,
        immr_field_index: u8,
        imms_field_index: u8,
        reg_size: u8,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum GeneratedMemoryAddressingConstraint {
    None,
    NoOffset,
    Offset,
    PreIndex,
    PostIndex,
}

#[derive(Debug, Clone)]
struct PreparedVariant {
    inst: FlatInstruction,
    operand_order: Vec<u8>,
    operand_kinds: Vec<GeneratedOperandKind>,
    user_kinds: Vec<GeneratedOperandKind>,
    implicit_defaults: Vec<(u8, i64)>,
    memory_addressing: GeneratedMemoryAddressingConstraint,
    field_scales: Vec<u16>,
    split_plan: Option<GeneratedSplitImmediatePlan>,
    gpr32_extend_compatibility: u64,
    user_shape_key: u128,
}

impl GeneratedMemoryAddressingConstraint {
    fn as_rust(self) -> &'static str {
        match self {
            Self::None => "MemoryAddressingConstraintSpec::None",
            Self::NoOffset => "MemoryAddressingConstraintSpec::NoOffset",
            Self::Offset => "MemoryAddressingConstraintSpec::Offset",
            Self::PreIndex => "MemoryAddressingConstraintSpec::PreIndex",
            Self::PostIndex => "MemoryAddressingConstraintSpec::PostIndex",
        }
    }
}

#[derive(Debug, Clone)]
struct InstructionContext {
    opcode: u32,
    opcode_mask: u32,
    semantic_fields: Vec<String>,
}

impl InstructionContext {
    fn from_instruction(inst: &FlatInstruction) -> Self {
        let mut semantic_fields = Vec::with_capacity(inst.fields.len());
        for field in &inst.fields {
            let normalized = normalize_field_name(&field.name);
            semantic_fields.push(semantic_field_name(&normalized).to_owned());
        }

        Self {
            opcode: inst.fixed_value,
            opcode_mask: inst.fixed_mask,
            semantic_fields,
        }
    }

    #[cfg(test)]
    fn from_semantic_fields(opcode: u32, opcode_mask: u32, semantic_fields: Vec<String>) -> Self {
        Self {
            opcode,
            opcode_mask,
            semantic_fields,
        }
    }

    fn has_field(&self, field_name: &str) -> bool {
        self.semantic_fields.iter().any(|name| name == field_name)
    }

    fn bit_value(&self, bit: u8) -> Option<u8> {
        let mask = 1u32 << bit;
        if (self.opcode_mask & mask) == 0 {
            return None;
        }
        Some(((self.opcode & mask) != 0) as u8)
    }

    fn has_memory_base_data(&self) -> bool {
        let has_base = self.has_field("rn");
        let has_data = self.has_field("rt")
            || self.has_field("rt2")
            || self.has_field("rt3")
            || self.has_field("rt4");
        has_base && has_data
    }

    fn has_memory_offset_components(&self) -> bool {
        let has_offset = self.has_field("imm7")
            || self.has_field("imm9")
            || self.has_field("imm12")
            || self.has_field("rm")
            || self.has_field("option")
            || self.has_field("s")
            || self.has_field("xs");
        has_offset
    }

    fn memory_like(&self) -> bool {
        self.has_memory_base_data() && self.has_memory_offset_components()
    }
}

#[inline]
fn kind_shape_code(kind: GeneratedOperandKind) -> u8 {
    match kind {
        GeneratedOperandKind::GprRegister => 1,
        GeneratedOperandKind::Gpr32Register => 2,
        GeneratedOperandKind::Gpr64Register => 3,
        GeneratedOperandKind::SimdRegister => 4,
        GeneratedOperandKind::SveZRegister => 5,
        GeneratedOperandKind::PredicateRegister => 6,
        GeneratedOperandKind::Immediate => 7,
        GeneratedOperandKind::Condition => 8,
        GeneratedOperandKind::ShiftKind => 9,
        GeneratedOperandKind::ExtendKind => 10,
        GeneratedOperandKind::SysRegPart => 11,
        GeneratedOperandKind::Arrangement => 12,
        GeneratedOperandKind::Lane => 13,
    }
}

#[inline]
fn memory_shape_code(memory_addressing: GeneratedMemoryAddressingConstraint) -> Option<u8> {
    match memory_addressing {
        // `NoOffset` still omits a dedicated shape token; semantic filtering is
        // enforced by `MemoryAddressingConstraintSpec` during candidate matching.
        GeneratedMemoryAddressingConstraint::None
        | GeneratedMemoryAddressingConstraint::NoOffset => None,
        GeneratedMemoryAddressingConstraint::Offset => Some(14),
        GeneratedMemoryAddressingConstraint::PreIndex => Some(15),
        // 0 stays distinct because encoded length is part of the shape key.
        GeneratedMemoryAddressingConstraint::PostIndex => Some(0),
    }
}

#[inline]
fn encode_operand_shape_key(
    kinds: &[GeneratedOperandKind],
    memory_addressing: GeneratedMemoryAddressingConstraint,
) -> Option<u128> {
    let memory_kind = memory_shape_code(memory_addressing);
    let total_len = kinds.len() + usize::from(memory_kind.is_some());
    if total_len > 30 {
        return None;
    }

    let mut key = total_len as u128;
    for (idx, kind) in kinds.iter().copied().enumerate() {
        let shift = 8 + (idx * 4);
        key |= u128::from(kind_shape_code(kind)) << shift;
    }
    if let Some(memory_kind) = memory_kind {
        let shift = 8 + (kinds.len() * 4);
        key |= u128::from(memory_kind) << shift;
    }
    Some(key)
}

fn expected_user_operand_kinds(
    operand_kinds: &[GeneratedOperandKind],
    split_plan: Option<GeneratedSplitImmediatePlan>,
) -> Vec<GeneratedOperandKind> {
    fn split_input_span(kind: GeneratedSplitImmediateKind) -> usize {
        match kind {
            GeneratedSplitImmediateKind::AdrLike { .. }
            | GeneratedSplitImmediateKind::BitIndex6 { .. }
            | GeneratedSplitImmediateKind::LogicalImmRs { .. } => 2,
            GeneratedSplitImmediateKind::LogicalImmNrs { .. } => 3,
        }
    }

    let mut out = Vec::with_capacity(operand_kinds.len());
    let mut slot = 0usize;
    while slot < operand_kinds.len() {
        if let Some(plan) = split_plan
            && slot == usize::from(plan.first_slot)
        {
            out.push(GeneratedOperandKind::Immediate);
            slot = slot.saturating_add(split_input_span(plan.kind));
            continue;
        }
        out.push(operand_kinds[slot]);
        slot += 1;
    }
    out
}

#[inline]
fn generated_kind_matches(expected: GeneratedOperandKind, actual: GeneratedOperandKind) -> bool {
    if expected == actual {
        return true;
    }

    matches!(
        (expected, actual),
        (
            GeneratedOperandKind::GprRegister,
            GeneratedOperandKind::Gpr32Register
        ) | (
            GeneratedOperandKind::GprRegister,
            GeneratedOperandKind::Gpr64Register
        ) | (
            GeneratedOperandKind::SysRegPart,
            GeneratedOperandKind::Immediate
        )
    )
}

#[inline]
fn generated_kind_matches_for_slot(
    variant: &PreparedVariant,
    slot: usize,
    actual: GeneratedOperandKind,
) -> bool {
    let Some(expected) = variant.user_kinds.get(slot).copied() else {
        return false;
    };
    if generated_kind_matches(expected, actual) {
        return true;
    }
    if !(expected == GeneratedOperandKind::Gpr64Register
        && actual == GeneratedOperandKind::Gpr32Register)
    {
        return false;
    }
    if slot >= u64::BITS as usize {
        return false;
    }
    ((variant.gpr32_extend_compatibility >> slot) & 1) != 0
}

#[inline]
fn generated_kind_specificity(kind: GeneratedOperandKind) -> u16 {
    match kind {
        GeneratedOperandKind::GprRegister => 3,
        GeneratedOperandKind::Gpr32Register
        | GeneratedOperandKind::Gpr64Register
        | GeneratedOperandKind::SimdRegister
        | GeneratedOperandKind::SveZRegister
        | GeneratedOperandKind::PredicateRegister => 4,
        GeneratedOperandKind::Immediate => 1,
        GeneratedOperandKind::Condition
        | GeneratedOperandKind::ShiftKind
        | GeneratedOperandKind::ExtendKind
        | GeneratedOperandKind::SysRegPart
        | GeneratedOperandKind::Arrangement
        | GeneratedOperandKind::Lane => 2,
    }
}

#[inline]
fn generated_variant_rank(variant: &PreparedVariant) -> u64 {
    let fixed_bits = u64::from(
        variant
            .inst
            .fixed_mask
            .count_ones()
            .min(u32::from(u16::MAX)) as u16,
    );

    let mut kind_specificity = 0u16;
    let mut immediate_narrowness = 0u16;
    for (slot, kind) in variant.operand_kinds.iter().copied().enumerate() {
        kind_specificity = kind_specificity.saturating_add(generated_kind_specificity(kind));
        if kind != GeneratedOperandKind::Immediate {
            continue;
        }
        let Some(field_idx) = variant.operand_order.get(slot).copied() else {
            continue;
        };
        let field_idx = usize::from(field_idx);
        if let Some(field) = variant.inst.fields.get(field_idx) {
            immediate_narrowness =
                immediate_narrowness.saturating_add((64u16).saturating_sub(u16::from(field.width)));
        }
    }

    let explicit_operands = variant
        .operand_order
        .len()
        .saturating_sub(variant.implicit_defaults.len())
        .min(usize::from(u8::MAX)) as u8;

    let implicit_penalty = (u8::MAX as usize)
        .saturating_sub(variant.implicit_defaults.len().min(usize::from(u8::MAX)))
        as u8;

    (fixed_bits << 48)
        | (u64::from(kind_specificity) << 32)
        | (u64::from(immediate_narrowness) << 16)
        | (u64::from(explicit_operands) << 8)
        | u64::from(implicit_penalty)
}

fn user_shape_candidates(
    variants: &[PreparedVariant],
    input_kinds: &[GeneratedOperandKind],
    memory_addressing: Option<GeneratedMemoryAddressingConstraint>,
) -> Vec<usize> {
    let mut out = Vec::new();
    for (idx, variant) in variants.iter().enumerate() {
        if let Some(memory_addressing) = memory_addressing
            && variant.memory_addressing != memory_addressing
        {
            continue;
        }
        if variant.user_kinds.len() != input_kinds.len() {
            continue;
        }
        let mut compatible = true;
        for (slot, actual) in input_kinds.iter().copied().enumerate() {
            if !generated_kind_matches_for_slot(variant, slot, actual) {
                compatible = false;
                break;
            }
        }
        if compatible {
            out.push(idx);
        }
    }
    out
}

fn sorted_flat_instructions(flat: &[FlatInstruction]) -> Vec<FlatInstruction> {
    let mut ordered = flat.to_vec();
    ordered.sort_by(|lhs, rhs| {
        lhs.mnemonic
            .cmp(&rhs.mnemonic)
            .then(lhs.variant.cmp(&rhs.variant))
    });
    ordered
}

fn prepare_variants(flat: &[FlatInstruction]) -> Result<Vec<PreparedVariant>, CodegenError> {
    let ordered = sorted_flat_instructions(flat);
    let mut prepared = Vec::with_capacity(ordered.len());

    for inst in ordered {
        let (operand_order, operand_kinds, implicit_defaults) = derive_operand_metadata(&inst)?;
        let memory_addressing = derive_memory_addressing_constraint(&inst);
        let field_scales = derive_field_scales(&inst);
        let split_plan = derive_split_immediate_plan(&inst, &operand_order, &operand_kinds);
        let gpr32_extend_compatibility =
            derive_gpr32_extend_compatibility(&inst, &operand_order, &operand_kinds);
        let user_kinds = expected_user_operand_kinds(&operand_kinds, split_plan);
        let user_shape_key =
            encode_operand_shape_key(&user_kinds, memory_addressing).ok_or_else(|| {
                CodegenError::Parse {
                    path: inst.path.clone(),
                    message: format!(
                        "operand shape for variant {} exceeds encoder key capacity",
                        inst.variant
                    ),
                }
            })?;

        prepared.push(PreparedVariant {
            inst,
            operand_order,
            operand_kinds,
            user_kinds,
            implicit_defaults,
            memory_addressing,
            field_scales,
            split_plan,
            gpr32_extend_compatibility,
            user_shape_key,
        });
    }

    Ok(prepared)
}

#[inline]
fn hash_mnemonic_with_seed(mnemonic: &str, seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for byte in mnemonic.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn next_pow2(value: usize) -> usize {
    if value <= 1 {
        1
    } else {
        value.next_power_of_two()
    }
}

fn build_mnemonic_perfect_hash(
    mnemonics: &[String],
) -> Result<(u64, usize, Vec<u16>), CodegenError> {
    let count = mnemonics.len();
    let mut table_size = next_pow2(count.saturating_mul(2));
    if table_size < 8 {
        table_size = 8;
    }

    for size_attempt in 0..6usize {
        let size = table_size << size_attempt;
        let mask = size - 1;
        for seed in 0u64..200_000u64 {
            let mut table = vec![u16::MAX; size];
            let mut ok = true;
            for (mnemonic_index, mnemonic) in mnemonics.iter().enumerate() {
                let slot = (hash_mnemonic_with_seed(mnemonic, seed) as usize) & mask;
                if table[slot] != u16::MAX {
                    ok = false;
                    break;
                }
                if mnemonic_index > usize::from(u16::MAX) {
                    return Err(CodegenError::Parse {
                        path: String::from("generated_specs"),
                        message: String::from("too many mnemonics to fit in u16 dispatch table"),
                    });
                }
                table[slot] = mnemonic_index as u16;
            }
            if ok {
                return Ok((seed, size, table));
            }
        }
    }

    Err(CodegenError::Parse {
        path: String::from("generated_specs"),
        message: String::from("failed to build collision-free mnemonic hash table"),
    })
}

fn usize_to_u16(value: usize, context: &str) -> Result<u16, CodegenError> {
    u16::try_from(value).map_err(|_| CodegenError::Parse {
        path: String::from("generated_specs"),
        message: format!("{context} exceeds u16 capacity"),
    })
}

fn field_signature(fields: &[FlatField]) -> String {
    let mut sig = String::new();
    for field in fields {
        let _ = write!(
            &mut sig,
            "{}:{}:{}:{}|",
            field.name, field.lsb, field.width, field.signed
        );
    }
    sig
}

fn intern_pool<T: Clone + Ord>(
    pool: &mut Vec<Vec<T>>,
    map: &mut BTreeMap<Vec<T>, usize>,
    value: &[T],
) -> usize {
    if let Some(existing) = map.get(value).copied() {
        existing
    } else {
        let next = pool.len();
        let owned = value.to_vec();
        map.insert(owned.clone(), next);
        pool.push(owned);
        next
    }
}

/// Errors emitted by codegen.
#[derive(Debug, Error)]
pub enum CodegenError {
    /// No instruction variants were provided.
    #[error("no instruction variants to generate")]
    EmptyInput,
    /// IO failure while scanning generated Rust source files.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Parse failure while extracting instruction metadata from Rust source.
    #[error("parse error in {path}: {message}")]
    Parse {
        /// File path where parsing failed.
        path: String,
        /// Human-readable parse detail.
        message: String,
    },
    /// Encountered an unmapped operand field while deriving constraints.
    #[error("unmapped operand field in {variant}: {field} (width={width})")]
    UnmappedOperandField {
        /// Variant name.
        variant: String,
        /// Field name.
        field: String,
        /// Field width.
        width: u8,
    },
}

/// Generates a Rust module that declares encoding specs for all provided variants.
///
/// # Errors
///
/// Returns [`CodegenError`] when input is empty.
pub fn generate_encoder_module(flat: &[FlatInstruction]) -> Result<String, CodegenError> {
    if flat.is_empty() {
        return Err(CodegenError::EmptyInput);
    }

    let prepared = prepare_variants(flat)?;
    let mut field_pool = Vec::<Vec<FlatField>>::new();
    let mut field_pool_map = BTreeMap::<String, usize>::new();
    let mut field_pool_idx = Vec::<usize>::with_capacity(prepared.len());
    let mut operand_order_pool = Vec::<Vec<u8>>::new();
    let mut operand_order_pool_map = BTreeMap::<Vec<u8>, usize>::new();
    let mut operand_order_pool_idx = Vec::<usize>::with_capacity(prepared.len());
    let mut operand_kinds_pool = Vec::<Vec<GeneratedOperandKind>>::new();
    let mut operand_kinds_pool_map = BTreeMap::<Vec<GeneratedOperandKind>, usize>::new();
    let mut operand_kinds_pool_idx = Vec::<usize>::with_capacity(prepared.len());
    let mut implicit_defaults_pool = Vec::<Vec<(u8, i64)>>::new();
    let mut implicit_defaults_pool_map = BTreeMap::<Vec<(u8, i64)>, usize>::new();
    let mut implicit_defaults_pool_idx = Vec::<usize>::with_capacity(prepared.len());
    let mut field_scales_pool = Vec::<Vec<u16>>::new();
    let mut field_scales_pool_map = BTreeMap::<Vec<u16>, usize>::new();
    let mut field_scales_pool_idx = Vec::<usize>::with_capacity(prepared.len());
    let mut split_plan_pool = Vec::<Option<GeneratedSplitImmediatePlan>>::new();
    let mut split_plan_pool_map = BTreeMap::<Option<GeneratedSplitImmediatePlan>, usize>::new();
    let mut split_plan_pool_idx = Vec::<usize>::with_capacity(prepared.len());

    for variant in &prepared {
        let sig = field_signature(&variant.inst.fields);
        let idx = if let Some(existing) = field_pool_map.get(&sig).copied() {
            existing
        } else {
            let next = field_pool.len();
            field_pool_map.insert(sig, next);
            field_pool.push(variant.inst.fields.clone());
            next
        };
        field_pool_idx.push(idx);
        operand_order_pool_idx.push(intern_pool(
            &mut operand_order_pool,
            &mut operand_order_pool_map,
            &variant.operand_order,
        ));
        operand_kinds_pool_idx.push(intern_pool(
            &mut operand_kinds_pool,
            &mut operand_kinds_pool_map,
            &variant.operand_kinds,
        ));
        implicit_defaults_pool_idx.push(intern_pool(
            &mut implicit_defaults_pool,
            &mut implicit_defaults_pool_map,
            &variant.implicit_defaults,
        ));
        field_scales_pool_idx.push(intern_pool(
            &mut field_scales_pool,
            &mut field_scales_pool_map,
            &variant.field_scales,
        ));
        let split_idx =
            if let Some(existing) = split_plan_pool_map.get(&variant.split_plan).copied() {
                existing
            } else {
                let next = split_plan_pool.len();
                split_plan_pool_map.insert(variant.split_plan, next);
                split_plan_pool.push(variant.split_plan);
                next
            };
        split_plan_pool_idx.push(split_idx);
    }

    let mut out = String::new();
    out.push_str("// @generated by jit-codegen. DO NOT EDIT.\n");
    out.push_str(
        "use jit_core::{BitFieldSpec, EncodeError, EncodingSpec, ImplicitField, InstructionCode, MemoryAddressingConstraintSpec, Operand, OperandConstraintKind, SplitImmediateKindSpec, SplitImmediatePlanSpec};\n\n",
    );

    for (idx, fields) in field_pool.iter().enumerate() {
        let fields_ident = format!("FIELDS_{idx}");

        writeln!(&mut out, "const {fields_ident}: &[BitFieldSpec] = &[").expect("write string");

        for field in fields {
            writeln!(
                &mut out,
                "    BitFieldSpec {{ name: {:?}, lsb: {}, width: {}, signed: {} }},",
                field.name, field.lsb, field.width, field.signed
            )
            .expect("write string");
        }

        out.push_str("];\n\n");
    }

    for (idx, order) in operand_order_pool.iter().enumerate() {
        let ident = format!("OPERAND_ORDER_{idx}");
        write!(&mut out, "const {ident}: &[u8] = &[").expect("write string");
        for (slot, value) in order.iter().copied().enumerate() {
            if slot > 0 {
                out.push_str(", ");
            }
            write!(&mut out, "{value}").expect("write string");
        }
        out.push_str("];\n");
    }
    out.push('\n');

    for (idx, kinds) in operand_kinds_pool.iter().enumerate() {
        let ident = format!("OPERAND_KINDS_{idx}");
        write!(&mut out, "const {ident}: &[OperandConstraintKind] = &[").expect("write string");
        for (slot, kind) in kinds.iter().copied().enumerate() {
            if slot > 0 {
                out.push_str(", ");
            }
            out.push_str(kind.as_rust());
        }
        out.push_str("];\n");
    }
    out.push('\n');

    for (idx, defaults) in implicit_defaults_pool.iter().enumerate() {
        let ident = format!("IMPLICIT_DEFAULTS_{idx}");
        write!(&mut out, "const {ident}: &[ImplicitField] = &[").expect("write string");
        for (slot, (field_index, value)) in defaults.iter().copied().enumerate() {
            if slot > 0 {
                out.push_str(", ");
            }
            write!(
                &mut out,
                "ImplicitField {{ field_index: {field_index}, value: {value} }}"
            )
            .expect("write string");
        }
        out.push_str("];\n");
    }
    out.push('\n');

    for (idx, scales) in field_scales_pool.iter().enumerate() {
        let ident = format!("FIELD_SCALES_{idx}");
        write!(&mut out, "const {ident}: &[u16] = &[").expect("write string");
        for (slot, scale) in scales.iter().copied().enumerate() {
            if slot > 0 {
                out.push_str(", ");
            }
            write!(&mut out, "{scale}").expect("write string");
        }
        out.push_str("];\n");
    }
    out.push('\n');

    for (idx, split_plan) in split_plan_pool.iter().copied().enumerate() {
        let ident = format!("SPLIT_PLAN_{idx}");
        let rhs = match split_plan {
            Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind:
                    GeneratedSplitImmediateKind::AdrLike {
                        immlo_field_index,
                        immhi_field_index,
                        scale,
                    },
            }) => format!(
                "Some(SplitImmediatePlanSpec {{ first_slot: {first_slot}, second_slot: {second_slot}, kind: SplitImmediateKindSpec::AdrLike {{ immlo_field_index: {immlo_field_index}, immhi_field_index: {immhi_field_index}, scale: {scale} }} }})"
            ),
            Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind:
                    GeneratedSplitImmediateKind::BitIndex6 {
                        b5_field_index,
                        b40_field_index,
                    },
            }) => format!(
                "Some(SplitImmediatePlanSpec {{ first_slot: {first_slot}, second_slot: {second_slot}, kind: SplitImmediateKindSpec::BitIndex6 {{ b5_field_index: {b5_field_index}, b40_field_index: {b40_field_index} }} }})"
            ),
            Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind:
                    GeneratedSplitImmediateKind::LogicalImmRs {
                        immr_field_index,
                        imms_field_index,
                        reg_size,
                    },
            }) => format!(
                "Some(SplitImmediatePlanSpec {{ first_slot: {first_slot}, second_slot: {second_slot}, kind: SplitImmediateKindSpec::LogicalImmRs {{ immr_field_index: {immr_field_index}, imms_field_index: {imms_field_index}, reg_size: {reg_size} }} }})"
            ),
            Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind:
                    GeneratedSplitImmediateKind::LogicalImmNrs {
                        n_field_index,
                        immr_field_index,
                        imms_field_index,
                        reg_size,
                    },
            }) => format!(
                "Some(SplitImmediatePlanSpec {{ first_slot: {first_slot}, second_slot: {second_slot}, kind: SplitImmediateKindSpec::LogicalImmNrs {{ n_field_index: {n_field_index}, immr_field_index: {immr_field_index}, imms_field_index: {imms_field_index}, reg_size: {reg_size} }} }})"
            ),
            None => String::from("None"),
        };
        writeln!(
            &mut out,
            "const {ident}: Option<SplitImmediatePlanSpec> = {rhs};"
        )
        .expect("write string");
    }
    out.push('\n');

    out.push_str(
        "\
const fn make_spec(\n\
    mnemonic: &'static str,\n\
    variant: &'static str,\n\
    opcode: u32,\n\
    opcode_mask: u32,\n\
    fields: &'static [BitFieldSpec],\n\
    operand_order: &'static [u8],\n\
    operand_kinds: &'static [OperandConstraintKind],\n\
    implicit_defaults: &'static [ImplicitField],\n\
    memory_addressing: MemoryAddressingConstraintSpec,\n\
    field_scales: &'static [u16],\n\
    split_immediate_plan: Option<SplitImmediatePlanSpec>,\n\
    gpr32_extend_compatibility: u64,\n\
) -> EncodingSpec {\n\
    EncodingSpec {\n\
        mnemonic,\n\
        variant,\n\
        opcode,\n\
        opcode_mask,\n\
        fields,\n\
        operand_order,\n\
        operand_kinds,\n\
        implicit_defaults,\n\
        memory_addressing,\n\
        field_scales,\n\
        split_immediate_plan,\n\
        gpr32_extend_compatibility,\n\
    }\n\
}\n\n",
    );

    out.push_str(
        "\
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]\n\
/// Opaque identifier of one canonical instruction variant.\n\
pub struct VariantId(pub u16);\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]\n\
/// Opaque identifier of one canonical mnemonic dispatch bucket.\n\
pub struct MnemonicId(pub u16);\n\n",
    );

    let mut variant_spec_exprs = Vec::<String>::with_capacity(prepared.len());
    for (idx, variant) in prepared.iter().enumerate() {
        let inst = &variant.inst;
        let fields_ident = format!("FIELDS_{}", field_pool_idx[idx]);
        let operand_order_ident = format!("OPERAND_ORDER_{}", operand_order_pool_idx[idx]);
        let operand_kinds_ident = format!("OPERAND_KINDS_{}", operand_kinds_pool_idx[idx]);
        let implicit_defaults_ident =
            format!("IMPLICIT_DEFAULTS_{}", implicit_defaults_pool_idx[idx]);
        let field_scales_ident = format!("FIELD_SCALES_{}", field_scales_pool_idx[idx]);
        let split_plan_ident = format!("SPLIT_PLAN_{}", split_plan_pool_idx[idx]);
        variant_spec_exprs.push(format!(
            "make_spec({:?}, {:?}, 0x{:08x}, 0x{:08x}, {fields_ident}, {operand_order_ident}, {operand_kinds_ident}, {implicit_defaults_ident}, {}, {field_scales_ident}, {split_plan_ident}, {}u64)",
            inst.mnemonic,
            inst.variant,
            inst.fixed_value,
            inst.fixed_mask,
            variant.memory_addressing.as_rust(),
            variant.gpr32_extend_compatibility
        ));
    }

    out.push_str("/// Generated encoding specs.\n");
    out.push_str("pub static SPECS: &[EncodingSpec] = &[\n");
    for expr in &variant_spec_exprs {
        writeln!(&mut out, "    {expr},").expect("write string");
    }
    out.push_str("];\n");

    out.push_str("/// Total number of generated canonical variants.\n");
    writeln!(
        &mut out,
        "pub const VARIANT_COUNT: usize = {};\n",
        prepared.len()
    )
    .expect("write string");

    out.push_str(
        "\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct MnemonicDispatchEntry {\n\
    pub name: &'static str,\n\
    pub spec_start: u16,\n\
    pub spec_len: u16,\n\
    pub shape_start: u16,\n\
    pub shape_len: u16,\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct ShapeDispatchEntry {\n\
    pub key: u128,\n\
    pub start: u16,\n\
    pub len: u16,\n\
}\n\n",
    );

    #[derive(Debug)]
    struct MnemonicEntry {
        name: String,
        spec_start: u16,
        spec_len: u16,
        shape_start: u16,
        shape_len: u16,
    }

    let mut mnemonic_entries = Vec::<MnemonicEntry>::new();
    let mut shape_entries = Vec::<(u128, u16, u16)>::new();
    let mut shape_variant_ids = Vec::<u16>::new();

    let mut idx = 0usize;
    while idx < prepared.len() {
        let start = idx;
        let mnemonic = prepared[idx].inst.mnemonic.clone();
        idx += 1;
        while idx < prepared.len() && prepared[idx].inst.mnemonic == mnemonic {
            idx += 1;
        }
        let end = idx;

        let shape_start = usize_to_u16(shape_entries.len(), "shape entry start")?;
        let mnemonic_variants = &prepared[start..end];
        let mut input_shapes = BTreeMap::<
            u128,
            (
                Vec<GeneratedOperandKind>,
                GeneratedMemoryAddressingConstraint,
            ),
        >::new();
        for variant in mnemonic_variants {
            input_shapes
                .entry(variant.user_shape_key)
                .or_insert_with(|| (variant.user_kinds.clone(), variant.memory_addressing));
        }

        for (shape_key, (input_kinds, memory_addressing)) in input_shapes {
            let mut candidates =
                user_shape_candidates(mnemonic_variants, &input_kinds, Some(memory_addressing));
            if candidates.is_empty() {
                continue;
            }
            candidates.sort_by(|lhs, rhs| {
                let left = generated_variant_rank(&mnemonic_variants[*lhs]);
                let right = generated_variant_rank(&mnemonic_variants[*rhs]);
                right.cmp(&left)
            });
            let start_variant_idx = usize_to_u16(shape_variant_ids.len(), "shape variant start")?;
            let variants_len = usize_to_u16(candidates.len(), "shape variant len")?;
            for local_variant_idx in candidates {
                shape_variant_ids.push(usize_to_u16(start + local_variant_idx, "variant id")?);
            }
            shape_entries.push((shape_key, start_variant_idx, variants_len));
        }
        let shape_end = usize_to_u16(shape_entries.len(), "shape entry end")?;

        mnemonic_entries.push(MnemonicEntry {
            name: mnemonic,
            spec_start: usize_to_u16(start, "mnemonic spec start")?,
            spec_len: usize_to_u16(end - start, "mnemonic spec len")?,
            shape_start,
            shape_len: shape_end - shape_start,
        });
    }

    out.push_str("pub(crate) static SHAPE_VARIANT_IDS: &[VariantId] = &[\n");
    for variant_id in &shape_variant_ids {
        writeln!(&mut out, "    VariantId({variant_id}),").expect("write string");
    }
    out.push_str("];\n");

    out.push_str("pub(crate) static SHAPE_DISPATCH: &[ShapeDispatchEntry] = &[\n");
    for (shape_key, start, len) in &shape_entries {
        writeln!(
            &mut out,
            "    ShapeDispatchEntry {{ key: {shape_key}u128, start: {start}, len: {len} }},"
        )
        .expect("write string");
    }
    out.push_str("];\n");

    out.push_str("pub(crate) static MNEMONIC_DISPATCH: &[MnemonicDispatchEntry] = &[\n");
    for entry in &mnemonic_entries {
        writeln!(
            &mut out,
            "    MnemonicDispatchEntry {{ name: {:?}, spec_start: {}, spec_len: {}, shape_start: {}, shape_len: {} }},",
            entry.name, entry.spec_start, entry.spec_len, entry.shape_start, entry.shape_len
        )
        .expect("write string");
    }
    out.push_str("];\n\n");

    let mnemonic_names = mnemonic_entries
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    let (hash_seed, hash_size, hash_table) = build_mnemonic_perfect_hash(&mnemonic_names)?;

    writeln!(
        &mut out,
        "const MNEMONIC_HASH_SEED: u64 = {hash_seed}u64;\nconst MNEMONIC_HASH_MASK: usize = {};\n",
        hash_size - 1
    )
    .expect("write string");

    out.push_str("pub(crate) static MNEMONIC_HASH_TABLE: &[u16] = &[\n");
    for slot in hash_table {
        writeln!(&mut out, "    {slot},").expect("write string");
    }
    out.push_str("];\n\n");

    out.push_str(
        "\
#[inline]\n\
fn hash_mnemonic(mnemonic: &str) -> u64 {\n\
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ MNEMONIC_HASH_SEED;\n\
    for byte in mnemonic.as_bytes() {\n\
        hash ^= u64::from(*byte);\n\
        hash = hash.wrapping_mul(0x100_0000_01b3);\n\
    }\n\
    hash\n\
}\n\n\
pub(crate) fn mnemonic_id_from_str(mnemonic: &str) -> Option<MnemonicId> {\n\
    let slot = (hash_mnemonic(mnemonic) as usize) & MNEMONIC_HASH_MASK;\n\
    let entry = *MNEMONIC_HASH_TABLE.get(slot)?;\n\
    if entry == u16::MAX {\n\
        return None;\n\
    }\n\
    let index = usize::from(entry);\n\
    let dispatch = *MNEMONIC_DISPATCH.get(index)?;\n\
    if dispatch.name == mnemonic {\n\
        Some(MnemonicId(entry))\n\
    } else {\n\
        None\n\
    }\n\
}\n\n\
pub(crate) fn mnemonic_name(id: MnemonicId) -> Option<&'static str> {\n\
    let entry = *MNEMONIC_DISPATCH.get(usize::from(id.0))?;\n\
    Some(entry.name)\n\
}\n\n\
pub(crate) fn specs_for_mnemonic_id(id: MnemonicId) -> Option<&'static [EncodingSpec]> {\n\
    let entry = *MNEMONIC_DISPATCH.get(usize::from(id.0))?;\n\
    let start = usize::from(entry.spec_start);\n\
    let len = usize::from(entry.spec_len);\n\
    Some(&SPECS[start..start + len])\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) enum ShapeVariantMatch {\n\
    Unique(VariantId),\n\
    Ambiguous,\n\
}\n\n\
#[inline]\n\
fn lookup_shape_dispatch_entry(id: MnemonicId, shape_key: u128) -> Option<ShapeDispatchEntry> {\n\
    let entry = *MNEMONIC_DISPATCH.get(usize::from(id.0))?;\n\
    let shape_start = usize::from(entry.shape_start);\n\
    let shape_end = shape_start + usize::from(entry.shape_len);\n\
    let shapes = &SHAPE_DISPATCH[shape_start..shape_end];\n\
    let mut left = 0usize;\n\
    let mut right = shapes.len();\n\
    while left < right {\n\
        let mid = left + ((right - left) >> 1);\n\
        let probe = shapes[mid].key;\n\
        if probe < shape_key {\n\
            left = mid + 1;\n\
            continue;\n\
        }\n\
        if probe > shape_key {\n\
            right = mid;\n\
            continue;\n\
        }\n\
        return Some(shapes[mid]);\n\
    }\n\
    None\n\
}\n\n\
pub(crate) fn variant_match_for_shape(id: MnemonicId, shape_key: u128) -> Option<ShapeVariantMatch> {\n\
    let entry = lookup_shape_dispatch_entry(id, shape_key)?;\n\
    let start = usize::from(entry.start);\n\
    let len = usize::from(entry.len);\n\
    if len == 1 {\n\
        return Some(ShapeVariantMatch::Unique(SHAPE_VARIANT_IDS[start]));\n\
    }\n\
    Some(ShapeVariantMatch::Ambiguous)\n\
}\n\n\
#[allow(dead_code)]\n\
pub(crate) fn variants_for_shape(id: MnemonicId, shape_key: u128) -> Option<&'static [VariantId]> {\n\
    let entry = lookup_shape_dispatch_entry(id, shape_key)?;\n\
    let start = usize::from(entry.start);\n\
    let len = usize::from(entry.len);\n\
    Some(&SHAPE_VARIANT_IDS[start..start + len])\n\
}\n\n\
pub(crate) fn encode_variant(id: VariantId, operands: &[Operand]) -> Result<InstructionCode, EncodeError> {\n\
    let index = usize::from(id.0);\n\
    let spec = SPECS.get(index).ok_or(EncodeError::NoMatchingVariant)?;\n\
    jit_core::encode_by_spec_operands(spec, operands)\n\
}\n\n\
pub(crate) fn spec_for_variant(id: VariantId) -> Option<&'static EncodingSpec> {\n\
    SPECS.get(usize::from(id.0))\n\
}\n\n\
pub(crate) fn specs_for_mnemonic(mnemonic: &str) -> Option<&'static [EncodingSpec]> {\n\
    let id = mnemonic_id_from_str(mnemonic)?;\n\
    specs_for_mnemonic_id(id)\n\
}\n",
    );

    out.push_str(
        "\
pub(crate) fn mnemonic_name_const<const MNEMONIC: u16>() -> Option<&'static str> {\n\
    let index = MNEMONIC as usize;\n\
    let entry = *MNEMONIC_DISPATCH.get(index)?;\n\
    Some(entry.name)\n\
}\n\n\
pub(crate) fn specs_for_mnemonic_id_const<const MNEMONIC: u16>() -> Option<&'static [EncodingSpec]> {\n\
    let index = MNEMONIC as usize;\n\
    let entry = *MNEMONIC_DISPATCH.get(index)?;\n\
    let start = usize::from(entry.spec_start);\n\
    let len = usize::from(entry.spec_len);\n\
    Some(&SPECS[start..start + len])\n\
}\n\n\
pub(crate) fn spec_for_variant_const<const VARIANT: u16>() -> Option<&'static EncodingSpec> {\n\
    SPECS.get(VARIANT as usize)\n\
}\n\n\
pub(crate) fn encode_variant_const<const VARIANT: u16>(operands: &[Operand]) -> Result<InstructionCode, EncodeError> {\n\
    let spec = spec_for_variant_const::<VARIANT>().ok_or(EncodeError::NoMatchingVariant)?;\n\
    jit_core::encode_by_spec_operands(spec, operands)\n\
}\n",
    );

    Ok(out)
}

/// Generates a Rust module that declares JIT macro normalization rule tables.
///
/// # Errors
///
/// Returns [`CodegenError`] when alias metadata cannot be derived from input variants.
pub fn generate_macro_normalization_module(
    flat: &[FlatInstruction],
) -> Result<String, CodegenError> {
    let mut shift_to_imm = BTreeSet::<String>::new();
    let mut sysreg_gpr_swap = BTreeSet::<String>::new();
    let mut reloc_b26 = BTreeSet::<String>::new();
    let mut reloc_bcond19 = BTreeSet::<String>::new();
    let mut reloc_cbz19 = BTreeSet::<String>::new();
    let mut reloc_imm19 = BTreeSet::<String>::new();
    let mut reloc_tbz14 = BTreeSet::<String>::new();
    let mut reloc_adr21 = BTreeSet::<String>::new();
    let mut reloc_adrp21 = BTreeSet::<String>::new();
    for inst in flat {
        if variant_supports_shift_to_immediate_normalization(inst)? {
            shift_to_imm.insert(inst.mnemonic.clone());
        }
        if variant_supports_sysreg_gpr_swap_normalization(inst)? {
            sysreg_gpr_swap.insert(inst.mnemonic.clone());
        }
        if let Some(kind) = variant_macro_reloc_kind(inst)? {
            match kind {
                MacroRelocKind::B26 => {
                    reloc_b26.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::BCond19 => {
                    reloc_bcond19.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::Cbz19 => {
                    reloc_cbz19.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::Imm19 => {
                    reloc_imm19.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::Tbz14 => {
                    reloc_tbz14.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::Adr21 => {
                    reloc_adr21.insert(inst.mnemonic.clone());
                }
                MacroRelocKind::Adrp21 => {
                    reloc_adrp21.insert(inst.mnemonic.clone());
                }
            }
        }
    }
    let prepared = prepare_variants(flat)?;

    let mut mnemonic_id_rules = Vec::<(String, u16)>::new();
    let mut seen_mnemonics = BTreeSet::<String>::new();
    for variant in &prepared {
        let mnemonic = variant.inst.mnemonic.clone();
        if seen_mnemonics.insert(mnemonic.clone()) {
            let id = usize_to_u16(mnemonic_id_rules.len(), "macro mnemonic id")?;
            mnemonic_id_rules.push((mnemonic, id));
        }
    }

    let mnemonic_id_map = mnemonic_id_rules
        .iter()
        .map(|(mnemonic, id)| (mnemonic.clone(), *id))
        .collect::<BTreeMap<_, _>>();

    let mut direct_variant_rules = BTreeMap::<(u16, u128), Option<u16>>::new();
    let mut start = 0usize;
    while start < prepared.len() {
        let mnemonic = prepared[start].inst.mnemonic.clone();
        let mut end = start + 1;
        while end < prepared.len() && prepared[end].inst.mnemonic == mnemonic {
            end += 1;
        }

        let mnemonic_variants = &prepared[start..end];
        let mut input_shapes = BTreeMap::<
            u128,
            (
                Vec<GeneratedOperandKind>,
                GeneratedMemoryAddressingConstraint,
            ),
        >::new();
        for variant in mnemonic_variants {
            input_shapes
                .entry(variant.user_shape_key)
                .or_insert_with(|| (variant.user_kinds.clone(), variant.memory_addressing));
        }

        for (shape_key, (input_kinds, memory_mode)) in input_shapes {
            let candidates =
                user_shape_candidates(mnemonic_variants, &input_kinds, Some(memory_mode));
            if candidates.is_empty() {
                continue;
            }
            let mnemonic_id = *mnemonic_id_map
                .get(&mnemonic)
                .expect("mnemonic id must exist for prepared mnemonic");
            if candidates.len() != 1 {
                continue;
            }
            let local_variant = candidates[0];
            let global_variant = start + local_variant;
            let variant_id = usize_to_u16(global_variant, "macro direct variant id")?;
            let macro_shape_key = shape_key;

            let entry = direct_variant_rules
                .entry((mnemonic_id, macro_shape_key))
                .or_insert(Some(variant_id));
            if let Some(existing) = *entry
                && existing != variant_id
            {
                *entry = None;
            }
        }
        start = end;
    }
    let mut direct_variant_rules = direct_variant_rules
        .into_iter()
        .filter_map(|((mnemonic_id, shape_key), variant)| {
            variant.map(|variant_id| (mnemonic_id, shape_key, variant_id))
        })
        .collect::<Vec<_>>();
    direct_variant_rules.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));

    const NORM_FLAG_SHIFT_TO_IMMEDIATE: u8 = 1u8 << 0;
    const NORM_FLAG_SYSREG_GPR_SWAP: u8 = 1u8 << 1;

    const RELOC_MASK_B26: u8 = 1u8 << 0;
    const RELOC_MASK_BCOND19: u8 = 1u8 << 1;
    const RELOC_MASK_CBZ19: u8 = 1u8 << 2;
    const RELOC_MASK_IMM19: u8 = 1u8 << 3;
    const RELOC_MASK_TBZ14: u8 = 1u8 << 4;
    const RELOC_MASK_ADR21: u8 = 1u8 << 5;
    const RELOC_MASK_ADRP21: u8 = 1u8 << 6;

    let mut mnemonic_normalization_rules = Vec::<(u16, u8, u8)>::new();
    for (mnemonic, mnemonic_id) in &mnemonic_id_rules {
        let mut flags = 0u8;
        if shift_to_imm.contains(mnemonic) {
            flags |= NORM_FLAG_SHIFT_TO_IMMEDIATE;
        }
        if sysreg_gpr_swap.contains(mnemonic) {
            flags |= NORM_FLAG_SYSREG_GPR_SWAP;
        }

        let mut reloc_mask = 0u8;
        if reloc_b26.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_B26;
        }
        if reloc_bcond19.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_BCOND19;
        }
        if reloc_cbz19.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_CBZ19;
        }
        if reloc_imm19.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_IMM19;
        }
        if reloc_tbz14.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_TBZ14;
        }
        if reloc_adr21.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_ADR21;
        }
        if reloc_adrp21.contains(mnemonic) {
            reloc_mask |= RELOC_MASK_ADRP21;
        }

        if flags != 0 || reloc_mask != 0 {
            mnemonic_normalization_rules.push((*mnemonic_id, flags, reloc_mask));
        }
    }
    mnemonic_normalization_rules.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    let conditional_branch_alias_rules = collect_conditional_branch_alias_rules(flat)?;

    let mut out = String::new();
    out.push_str("// @generated by jit-codegen. DO NOT EDIT.\n");
    out.push_str(
        "\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct MnemonicIdRule {\n\
    pub mnemonic: &'static str,\n\
    pub id: u16,\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct DirectVariantRule {\n\
    pub mnemonic_id: u16,\n\
    pub shape_key: u128,\n\
    pub variant_id: u16,\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct MnemonicNormalizationRule {\n\
    pub mnemonic_id: u16,\n\
    pub flags: u8,\n\
    pub reloc_mask: u8,\n\
}\n\n\
#[derive(Debug, Copy, Clone, PartialEq, Eq)]\n\
pub(crate) struct ConditionalBranchAliasRule {\n\
    pub alias: &'static str,\n\
    pub base_mnemonic: &'static str,\n\
    pub condition_code: u8,\n\
}\n\n\
pub(crate) static MNEMONIC_ID_RULES: &[MnemonicIdRule] = &[\n",
    );
    for (mnemonic, id) in &mnemonic_id_rules {
        writeln!(
            &mut out,
            "    MnemonicIdRule {{ mnemonic: {:?}, id: {} }},",
            mnemonic, id
        )
        .expect("write string");
    }
    out.push_str("];\n");
    out.push_str("pub(crate) static DIRECT_VARIANT_RULES: &[DirectVariantRule] = &[\n");
    for (mnemonic_id, shape_key, variant_id) in &direct_variant_rules {
        writeln!(
            &mut out,
            "    DirectVariantRule {{ mnemonic_id: {}, shape_key: {}u128, variant_id: {} }},",
            mnemonic_id, shape_key, variant_id
        )
        .expect("write string");
    }
    out.push_str("];\n");
    out.push_str("pub(crate) const NORM_FLAG_SHIFT_TO_IMMEDIATE: u8 = 1u8 << 0;\n");
    out.push_str("pub(crate) const NORM_FLAG_SYSREG_GPR_SWAP: u8 = 1u8 << 1;\n");
    out.push_str("pub(crate) const RELOC_MASK_B26: u8 = 1u8 << 0;\n");
    out.push_str("pub(crate) const RELOC_MASK_BCOND19: u8 = 1u8 << 1;\n");
    out.push_str("pub(crate) const RELOC_MASK_CBZ19: u8 = 1u8 << 2;\n");
    out.push_str("pub(crate) const RELOC_MASK_IMM19: u8 = 1u8 << 3;\n");
    out.push_str("pub(crate) const RELOC_MASK_TBZ14: u8 = 1u8 << 4;\n");
    out.push_str("pub(crate) const RELOC_MASK_ADR21: u8 = 1u8 << 5;\n");
    out.push_str("pub(crate) const RELOC_MASK_ADRP21: u8 = 1u8 << 6;\n");

    out.push_str(
        "pub(crate) static MNEMONIC_NORMALIZATION_RULES: &[MnemonicNormalizationRule] = &[\n",
    );
    for (mnemonic_id, flags, reloc_mask) in &mnemonic_normalization_rules {
        writeln!(
            &mut out,
            "    MnemonicNormalizationRule {{ mnemonic_id: {}, flags: {}, reloc_mask: {} }},",
            mnemonic_id, flags, reloc_mask
        )
        .expect("write string");
    }
    out.push_str("];\n");

    out.push_str(
        "pub(crate) static CONDITIONAL_BRANCH_ALIAS_RULES: &[ConditionalBranchAliasRule] = &[\n",
    );
    for (alias, base_mnemonic, condition_code) in &conditional_branch_alias_rules {
        writeln!(
            &mut out,
            "    ConditionalBranchAliasRule {{ alias: {:?}, base_mnemonic: {:?}, condition_code: {} }},",
            alias, base_mnemonic, condition_code
        )
        .expect("write string");
    }
    out.push_str("];\n");
    Ok(out)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MacroRelocKind {
    B26,
    BCond19,
    Cbz19,
    Imm19,
    Tbz14,
    Adr21,
    Adrp21,
}

fn is_gpr_generated_kind(kind: GeneratedOperandKind) -> bool {
    matches!(
        kind,
        GeneratedOperandKind::GprRegister
            | GeneratedOperandKind::Gpr32Register
            | GeneratedOperandKind::Gpr64Register
    )
}

fn variant_supports_shift_to_immediate_normalization(
    inst: &FlatInstruction,
) -> Result<bool, CodegenError> {
    let context = InstructionContext::from_instruction(inst);
    let (_, kinds, implicit_defaults) = derive_operand_metadata(inst)?;
    if kinds.len() != 4
        || !is_gpr_generated_kind(kinds[0])
        || !is_gpr_generated_kind(kinds[1])
        || !is_gpr_generated_kind(kinds[2])
        || kinds[3] != GeneratedOperandKind::Immediate
    {
        return Ok(false);
    }

    let has_shift_default = implicit_defaults.iter().any(|(field_index, value)| {
        if *value != 0 {
            return false;
        }
        let idx = *field_index as usize;
        if idx >= inst.fields.len() {
            return false;
        }
        let normalized = normalize_field_name(&inst.fields[idx].name);
        semantic_field_name(&normalized) == "shift"
    });

    if !has_shift_default {
        return Ok(false);
    }

    // Normalize only register forms (not memory-like variants).
    Ok(!context.memory_like())
}

fn variant_supports_sysreg_gpr_swap_normalization(
    inst: &FlatInstruction,
) -> Result<bool, CodegenError> {
    let (_, kinds, _) = derive_operand_metadata(inst)?;
    let mut gpr_count = 0usize;
    let mut sysreg_part_count = 0usize;
    let mut immediate_count = 0usize;

    for kind in kinds {
        match kind {
            GeneratedOperandKind::GprRegister
            | GeneratedOperandKind::Gpr32Register
            | GeneratedOperandKind::Gpr64Register => {
                gpr_count += 1;
            }
            GeneratedOperandKind::SysRegPart => {
                sysreg_part_count += 1;
            }
            GeneratedOperandKind::Immediate => {
                immediate_count += 1;
            }
            _ => return Ok(false),
        }
    }

    Ok(gpr_count == 1 && sysreg_part_count == 4 && immediate_count == 1)
}

fn semantic_field_exists(inst: &FlatInstruction, semantic_name: &str) -> bool {
    inst.fields.iter().any(|field| {
        let normalized = normalize_field_name(&field.name);
        semantic_field_name(&normalized) == semantic_name
    })
}

fn opcode_pattern_matches(inst: &FlatInstruction, mask: u32, value: u32) -> bool {
    (inst.fixed_mask & mask) == mask && (inst.fixed_value & mask) == value
}

fn variant_macro_reloc_kind(
    inst: &FlatInstruction,
) -> Result<Option<MacroRelocKind>, CodegenError> {
    let (_, kinds, _) = derive_operand_metadata(inst)?;

    let has_imm26 = semantic_field_exists(inst, "imm26");
    let has_imm19 = semantic_field_exists(inst, "imm19");
    let has_imm14 = semantic_field_exists(inst, "imm14");
    let has_cond = semantic_field_exists(inst, "cond");
    let has_rt = semantic_field_exists(inst, "rt");
    let has_b5 = semantic_field_exists(inst, "b5");
    let has_b40 = semantic_field_exists(inst, "b40");
    let has_immlo = semantic_field_exists(inst, "immlo");
    let has_immhi = semantic_field_exists(inst, "immhi");
    let has_rd = semantic_field_exists(inst, "rd");

    if has_imm26
        && kinds.len() == 1
        && kinds[0] == GeneratedOperandKind::Immediate
        && (opcode_pattern_matches(inst, 0xfc00_0000, 0x1400_0000)
            || opcode_pattern_matches(inst, 0xfc00_0000, 0x9400_0000))
    {
        return Ok(Some(MacroRelocKind::B26));
    }

    if has_imm19
        && has_cond
        && kinds.len() == 2
        && kinds[0] == GeneratedOperandKind::Condition
        && kinds[1] == GeneratedOperandKind::Immediate
        && opcode_pattern_matches(inst, 0xff00_0010, 0x5400_0000)
    {
        return Ok(Some(MacroRelocKind::BCond19));
    }

    if has_imm19
        && has_rt
        && !has_cond
        && kinds.len() == 2
        && is_gpr_generated_kind(kinds[0])
        && kinds[1] == GeneratedOperandKind::Immediate
        && (opcode_pattern_matches(inst, 0x7f00_0000, 0x3400_0000)
            || opcode_pattern_matches(inst, 0x7f00_0000, 0x3500_0000))
    {
        return Ok(Some(MacroRelocKind::Cbz19));
    }

    if has_imm19
        && !has_cond
        && kinds.len() == 2
        && matches!(
            kinds[0],
            GeneratedOperandKind::GprRegister
                | GeneratedOperandKind::Gpr32Register
                | GeneratedOperandKind::Gpr64Register
                | GeneratedOperandKind::SimdRegister
                | GeneratedOperandKind::SveZRegister
                | GeneratedOperandKind::PredicateRegister
                | GeneratedOperandKind::Immediate
        )
        && kinds[1] == GeneratedOperandKind::Immediate
    {
        return Ok(Some(MacroRelocKind::Imm19));
    }

    if has_imm14
        && has_rt
        && has_b5
        && has_b40
        && kinds.len() == 4
        && is_gpr_generated_kind(kinds[0])
        && kinds[1] == GeneratedOperandKind::Immediate
        && kinds[2] == GeneratedOperandKind::Immediate
        && kinds[3] == GeneratedOperandKind::Immediate
        && (opcode_pattern_matches(inst, 0x7f00_0000, 0x3600_0000)
            || opcode_pattern_matches(inst, 0x7f00_0000, 0x3700_0000))
    {
        return Ok(Some(MacroRelocKind::Tbz14));
    }

    if has_immlo
        && has_immhi
        && has_rd
        && kinds.len() == 3
        && is_gpr_generated_kind(kinds[0])
        && kinds[1] == GeneratedOperandKind::Immediate
        && kinds[2] == GeneratedOperandKind::Immediate
        && opcode_pattern_matches(inst, 0x9f00_0000, 0x1000_0000)
    {
        return Ok(Some(MacroRelocKind::Adr21));
    }

    if has_immlo
        && has_immhi
        && has_rd
        && kinds.len() == 3
        && is_gpr_generated_kind(kinds[0])
        && kinds[1] == GeneratedOperandKind::Immediate
        && kinds[2] == GeneratedOperandKind::Immediate
        && opcode_pattern_matches(inst, 0x9f00_0000, 0x9000_0000)
    {
        return Ok(Some(MacroRelocKind::Adrp21));
    }

    Ok(None)
}

const CONDITIONAL_BRANCH_SUFFIX_CODES: [(&str, u8); 18] = [
    ("eq", 0),
    ("ne", 1),
    ("cs", 2),
    ("hs", 2),
    ("cc", 3),
    ("lo", 3),
    ("mi", 4),
    ("pl", 5),
    ("vs", 6),
    ("vc", 7),
    ("hi", 8),
    ("ls", 9),
    ("ge", 10),
    ("lt", 11),
    ("gt", 12),
    ("le", 13),
    ("al", 14),
    ("nv", 15),
];

fn register_conditional_alias(
    aliases: &mut BTreeMap<String, (String, u8)>,
    ambiguous: &mut BTreeSet<String>,
    alias: String,
    base_mnemonic: &str,
    condition_code: u8,
) {
    if ambiguous.contains(&alias) {
        return;
    }

    match aliases.get(&alias) {
        None => {
            aliases.insert(alias, (base_mnemonic.to_owned(), condition_code));
        }
        Some((existing_base, existing_condition))
            if existing_base == base_mnemonic && *existing_condition == condition_code => {}
        Some(_) => {
            aliases.remove(&alias);
            ambiguous.insert(alias);
        }
    }
}

/// Collects generated conditional-branch alias spellings and their canonical mnemonic mappings.
///
/// The returned tuples are `(alias, canonical_mnemonic, condition_code)` where
/// `condition_code` is the architectural 4-bit condition encoding (`0..=15`).
///
/// # Errors
///
/// Returns [`CodegenError`] when operand metadata derivation fails for input variants.
pub fn collect_conditional_branch_alias_rules(
    flat: &[FlatInstruction],
) -> Result<Vec<(String, String, u8)>, CodegenError> {
    let mut base_mnemonics = BTreeSet::<String>::new();
    for inst in flat {
        if matches!(
            variant_macro_reloc_kind(inst)?,
            Some(MacroRelocKind::BCond19)
        ) {
            base_mnemonics.insert(inst.mnemonic.clone());
        }
    }

    let mut aliases = BTreeMap::<String, (String, u8)>::new();
    let mut ambiguous = BTreeSet::<String>::new();

    for base in &base_mnemonics {
        for (suffix, condition_code) in CONDITIONAL_BRANCH_SUFFIX_CODES {
            register_conditional_alias(
                &mut aliases,
                &mut ambiguous,
                format!("{base}.{suffix}"),
                base,
                condition_code,
            );
            register_conditional_alias(
                &mut aliases,
                &mut ambiguous,
                format!("{base}{suffix}"),
                base,
                condition_code,
            );
        }
    }

    for alias in ambiguous {
        aliases.remove(&alias);
    }

    Ok(aliases
        .into_iter()
        .map(|(alias, (base_mnemonic, condition_code))| (alias, base_mnemonic, condition_code))
        .collect())
}

fn derive_operand_metadata(
    inst: &FlatInstruction,
) -> Result<(Vec<u8>, Vec<GeneratedOperandKind>, Vec<(u8, i64)>), CodegenError> {
    let width_hint = variant_width_hint(&inst.variant);
    let context = InstructionContext::from_instruction(inst);
    let mut ordered = Vec::<OrderedField>::new();
    let mut implicit_defaults = Vec::<(u8, i64)>::new();

    for (idx, field) in inst.fields.iter().enumerate() {
        if idx > u8::MAX as usize {
            break;
        }

        let field_name = normalize_field_name(&field.name);
        let semantic_name = semantic_field_name(&field_name);
        let kind = infer_operand_kind(semantic_name, field, width_hint, &context, &inst.variant)?;
        if let Some(value) = implicit_default_value(semantic_name, kind) {
            implicit_defaults.push((idx as u8, value));
            continue;
        }

        ordered.push(OrderedField {
            index: idx as u8,
            rank: field_rank(semantic_name, kind),
            kind,
        });
    }

    ordered.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then(left.index.cmp(&right.index))
    });
    implicit_defaults.sort_by(|left, right| left.0.cmp(&right.0));

    let order = ordered.iter().map(|slot| slot.index).collect::<Vec<_>>();
    let kinds = ordered.iter().map(|slot| slot.kind).collect::<Vec<_>>();
    Ok((order, kinds, implicit_defaults))
}

fn normalized_semantic_fields(inst: &FlatInstruction) -> Vec<String> {
    let mut out = Vec::with_capacity(inst.fields.len());
    for field in &inst.fields {
        let normalized = normalize_field_name(&field.name);
        out.push(semantic_field_name(&normalized).to_owned());
    }
    out
}

fn unique_semantic_field_index(semantic_fields: &[String], name: &str) -> Option<usize> {
    let mut found = None;
    for (idx, semantic) in semantic_fields.iter().enumerate() {
        if semantic != name {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(idx);
    }
    found
}

fn operand_slot_for_field_index(operand_order: &[u8], field_index: usize) -> Option<usize> {
    let mut found = None;
    for (slot, encoded_index) in operand_order.iter().copied().enumerate() {
        if usize::from(encoded_index) != field_index {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(slot);
    }
    found
}

fn has_adjacent_immediate_runs(kinds: &[GeneratedOperandKind]) -> bool {
    kinds.windows(2).any(|pair| {
        pair[0] == GeneratedOperandKind::Immediate && pair[1] == GeneratedOperandKind::Immediate
    }) || kinds.windows(3).any(|triple| {
        triple[0] == GeneratedOperandKind::Immediate
            && triple[1] == GeneratedOperandKind::Immediate
            && triple[2] == GeneratedOperandKind::Immediate
    })
}

fn fixed_bit_value(inst: &FlatInstruction, bit: u8) -> Option<u8> {
    let mask = 1u32 << bit;
    if (inst.fixed_mask & mask) == 0 {
        return None;
    }
    Some(((inst.fixed_value & mask) != 0) as u8)
}

fn has_semantic_field(semantic_fields: &[String], field_name: &str) -> bool {
    semantic_fields.iter().any(|name| name == field_name)
}

fn is_writeback_variant_name(variant: &str) -> bool {
    variant
        .split('_')
        .any(|segment| segment.eq_ignore_ascii_case("writeback"))
}

fn derive_memory_addressing_constraint(
    inst: &FlatInstruction,
) -> GeneratedMemoryAddressingConstraint {
    let semantic_fields = normalized_semantic_fields(inst);
    let has_rn = has_semantic_field(&semantic_fields, "rn");
    let has_rt = has_semantic_field(&semantic_fields, "rt");
    let has_rt2 = has_semantic_field(&semantic_fields, "rt2");
    let has_memory_offset_fields = has_semantic_field(&semantic_fields, "imm7")
        || has_semantic_field(&semantic_fields, "imm9")
        || has_semantic_field(&semantic_fields, "imm12")
        || has_semantic_field(&semantic_fields, "rm")
        || has_semantic_field(&semantic_fields, "option")
        || has_semantic_field(&semantic_fields, "s")
        || has_semantic_field(&semantic_fields, "xs");

    if has_rn && has_rt && has_rt2 && has_semantic_field(&semantic_fields, "imm7") {
        let bit24 = fixed_bit_value(inst, 24);
        let bit23 = fixed_bit_value(inst, 23);
        return match (bit24, bit23) {
            (Some(1), Some(0)) => GeneratedMemoryAddressingConstraint::Offset,
            (Some(0), Some(1)) => GeneratedMemoryAddressingConstraint::PostIndex,
            (Some(1), Some(1)) => GeneratedMemoryAddressingConstraint::PreIndex,
            _ => GeneratedMemoryAddressingConstraint::None,
        };
    }

    if has_rn && has_rt && has_semantic_field(&semantic_fields, "imm9") {
        let bit11 = fixed_bit_value(inst, 11);
        let bit10 = fixed_bit_value(inst, 10);
        return match (bit11, bit10) {
            (Some(0), Some(0)) => GeneratedMemoryAddressingConstraint::Offset,
            (Some(0), Some(1)) => GeneratedMemoryAddressingConstraint::PostIndex,
            (Some(1), Some(1)) => GeneratedMemoryAddressingConstraint::PreIndex,
            _ => GeneratedMemoryAddressingConstraint::None,
        };
    }

    if has_rn && has_rt && has_semantic_field(&semantic_fields, "imm12") {
        return GeneratedMemoryAddressingConstraint::Offset;
    }

    if has_rn && (has_rt || has_rt2) && !has_memory_offset_fields {
        if is_writeback_variant_name(&inst.variant) {
            return GeneratedMemoryAddressingConstraint::PreIndex;
        }
        return GeneratedMemoryAddressingConstraint::NoOffset;
    }

    GeneratedMemoryAddressingConstraint::None
}

fn derive_memory_pair_offset_scale(
    inst: &FlatInstruction,
    semantic_fields: &[String],
) -> Option<u16> {
    if !(has_semantic_field(semantic_fields, "rn")
        && has_semantic_field(semantic_fields, "rt")
        && has_semantic_field(semantic_fields, "rt2")
        && has_semantic_field(semantic_fields, "imm7"))
    {
        return None;
    }

    let v = fixed_bit_value(inst, 26)?;
    let b31 = fixed_bit_value(inst, 31)?;
    let b30 = fixed_bit_value(inst, 30)?;
    match (v, b31, b30) {
        (0, 0, 0) => Some(4),
        (0, 1, 0) => Some(8),
        (1, 0, 0) => Some(4),
        (1, 0, 1) => Some(8),
        (1, 1, 0) => Some(16),
        _ => None,
    }
}

fn derive_memory_imm12_scale(inst: &FlatInstruction, semantic_fields: &[String]) -> Option<u16> {
    if !(has_semantic_field(semantic_fields, "rn")
        && has_semantic_field(semantic_fields, "rt")
        && has_semantic_field(semantic_fields, "imm12"))
    {
        return None;
    }

    let b31 = fixed_bit_value(inst, 31)?;
    let b30 = fixed_bit_value(inst, 30)?;
    let size = (b31 << 1) | b30;
    if fixed_bit_value(inst, 23) == Some(1) && size == 0 {
        return Some(16);
    }
    Some(1u16 << size)
}

fn derive_field_scales(inst: &FlatInstruction) -> Vec<u16> {
    let semantic_fields = normalized_semantic_fields(inst);
    let pair_scale = derive_memory_pair_offset_scale(inst, &semantic_fields);
    let imm12_scale = derive_memory_imm12_scale(inst, &semantic_fields);

    let mut scales = vec![1u16; inst.fields.len()];
    for (idx, field) in inst.fields.iter().enumerate() {
        let semantic = semantic_fields
            .get(idx)
            .map(String::as_str)
            .unwrap_or_default();
        if semantic == "imm26" && field.lsb == 0 {
            scales[idx] = 4;
            continue;
        }
        if semantic == "imm19" && field.lsb == 5 {
            scales[idx] = 4;
            continue;
        }
        if semantic == "imm14" && field.lsb == 5 {
            scales[idx] = 4;
            continue;
        }
        if semantic == "imm7"
            && let Some(scale) = pair_scale
        {
            scales[idx] = scale;
            continue;
        }
        if semantic == "imm12"
            && let Some(scale) = imm12_scale
        {
            scales[idx] = scale;
        }
    }
    scales
}

fn derive_split_immediate_plan(
    inst: &FlatInstruction,
    operand_order: &[u8],
    operand_kinds: &[GeneratedOperandKind],
) -> Option<GeneratedSplitImmediatePlan> {
    if !has_adjacent_immediate_runs(operand_kinds) {
        return None;
    }

    let semantic_fields = normalized_semantic_fields(inst);
    let is_logical_immediate_mnemonic =
        matches!(inst.mnemonic.as_str(), "and" | "ands" | "eor" | "orr");
    let n_field_index = unique_semantic_field_index(&semantic_fields, "n");
    let immr_field_index = unique_semantic_field_index(&semantic_fields, "immr");
    let imms_field_index = unique_semantic_field_index(&semantic_fields, "imms");
    if is_logical_immediate_mnemonic
        && let (Some(n_field_index), Some(immr_field_index), Some(imms_field_index)) =
            (n_field_index, immr_field_index, imms_field_index)
    {
        let n_slot = operand_slot_for_field_index(operand_order, n_field_index)?;
        let immr_slot = operand_slot_for_field_index(operand_order, immr_field_index)?;
        let imms_slot = operand_slot_for_field_index(operand_order, imms_field_index)?;
        let mut slots = [n_slot, immr_slot, imms_slot];
        slots.sort_unstable();
        let first_slot = slots[0];
        let second_slot = slots[2];
        if second_slot == first_slot + 2
            && operand_kinds.get(first_slot) == Some(&GeneratedOperandKind::Immediate)
            && operand_kinds.get(first_slot + 1) == Some(&GeneratedOperandKind::Immediate)
            && operand_kinds.get(second_slot) == Some(&GeneratedOperandKind::Immediate)
        {
            let rd_field_index = unique_semantic_field_index(&semantic_fields, "rd")
                .or_else(|| unique_semantic_field_index(&semantic_fields, "rt"));
            let reg_size = if let Some(rd_field_index) = rd_field_index {
                let rd_slot = operand_slot_for_field_index(operand_order, rd_field_index)?;
                match operand_kinds.get(rd_slot).copied() {
                    Some(GeneratedOperandKind::Gpr32Register) => 32u8,
                    Some(GeneratedOperandKind::Gpr64Register) => 64u8,
                    _ => return None,
                }
            } else {
                return None;
            };

            let first_slot = u8::try_from(first_slot).ok()?;
            let second_slot = u8::try_from(second_slot).ok()?;
            let n_field_index = u8::try_from(n_field_index).ok()?;
            let immr_field_index = u8::try_from(immr_field_index).ok()?;
            let imms_field_index = u8::try_from(imms_field_index).ok()?;
            return Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind: GeneratedSplitImmediateKind::LogicalImmNrs {
                    n_field_index,
                    immr_field_index,
                    imms_field_index,
                    reg_size,
                },
            });
        }
    }

    if is_logical_immediate_mnemonic
        && let (Some(immr_field_index), Some(imms_field_index)) =
            (immr_field_index, imms_field_index)
    {
        let immr_slot = operand_slot_for_field_index(operand_order, immr_field_index)?;
        let imms_slot = operand_slot_for_field_index(operand_order, imms_field_index)?;
        let first_slot = immr_slot.min(imms_slot);
        let second_slot = immr_slot.max(imms_slot);
        if second_slot == first_slot + 1
            && operand_kinds.get(first_slot) == Some(&GeneratedOperandKind::Immediate)
            && operand_kinds.get(second_slot) == Some(&GeneratedOperandKind::Immediate)
        {
            let rd_field_index = unique_semantic_field_index(&semantic_fields, "rd")
                .or_else(|| unique_semantic_field_index(&semantic_fields, "rt"));
            let reg_size = if let Some(rd_field_index) = rd_field_index {
                let rd_slot = operand_slot_for_field_index(operand_order, rd_field_index)?;
                match operand_kinds.get(rd_slot).copied() {
                    Some(GeneratedOperandKind::Gpr32Register) => 32u8,
                    Some(GeneratedOperandKind::Gpr64Register) => 64u8,
                    _ => return None,
                }
            } else {
                return None;
            };

            let first_slot = u8::try_from(first_slot).ok()?;
            let second_slot = u8::try_from(second_slot).ok()?;
            let immr_field_index = u8::try_from(immr_field_index).ok()?;
            let imms_field_index = u8::try_from(imms_field_index).ok()?;
            return Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind: GeneratedSplitImmediateKind::LogicalImmRs {
                    immr_field_index,
                    imms_field_index,
                    reg_size,
                },
            });
        }
    }

    let immlo_field_index = unique_semantic_field_index(&semantic_fields, "immlo");
    let immhi_field_index = unique_semantic_field_index(&semantic_fields, "immhi");
    if let (Some(immlo_field_index), Some(immhi_field_index)) =
        (immlo_field_index, immhi_field_index)
    {
        let immlo_slot = operand_slot_for_field_index(operand_order, immlo_field_index)?;
        let immhi_slot = operand_slot_for_field_index(operand_order, immhi_field_index)?;
        let first_slot = immlo_slot.min(immhi_slot);
        let second_slot = immlo_slot.max(immhi_slot);
        if second_slot == first_slot + 1
            && operand_kinds.get(first_slot) == Some(&GeneratedOperandKind::Immediate)
            && operand_kinds.get(second_slot) == Some(&GeneratedOperandKind::Immediate)
        {
            let scale = if fixed_bit_value(inst, 31) == Some(1) {
                4096
            } else {
                1
            };
            let first_slot = u8::try_from(first_slot).ok()?;
            let second_slot = u8::try_from(second_slot).ok()?;
            let immlo_field_index = u8::try_from(immlo_field_index).ok()?;
            let immhi_field_index = u8::try_from(immhi_field_index).ok()?;
            return Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind: GeneratedSplitImmediateKind::AdrLike {
                    immlo_field_index,
                    immhi_field_index,
                    scale,
                },
            });
        }
    }

    let b5_field_index = unique_semantic_field_index(&semantic_fields, "b5");
    let b40_field_index = unique_semantic_field_index(&semantic_fields, "b40");
    if let (Some(b5_field_index), Some(b40_field_index)) = (b5_field_index, b40_field_index) {
        let b5_slot = operand_slot_for_field_index(operand_order, b5_field_index)?;
        let b40_slot = operand_slot_for_field_index(operand_order, b40_field_index)?;
        let first_slot = b5_slot.min(b40_slot);
        let second_slot = b5_slot.max(b40_slot);
        if second_slot == first_slot + 1
            && operand_kinds.get(first_slot) == Some(&GeneratedOperandKind::Immediate)
            && operand_kinds.get(second_slot) == Some(&GeneratedOperandKind::Immediate)
        {
            let first_slot = u8::try_from(first_slot).ok()?;
            let second_slot = u8::try_from(second_slot).ok()?;
            let b5_field_index = u8::try_from(b5_field_index).ok()?;
            let b40_field_index = u8::try_from(b40_field_index).ok()?;
            return Some(GeneratedSplitImmediatePlan {
                first_slot,
                second_slot,
                kind: GeneratedSplitImmediateKind::BitIndex6 {
                    b5_field_index,
                    b40_field_index,
                },
            });
        }
    }

    None
}

fn derive_gpr32_extend_compatibility(
    inst: &FlatInstruction,
    operand_order: &[u8],
    operand_kinds: &[GeneratedOperandKind],
) -> u64 {
    let semantic_fields = normalized_semantic_fields(inst);
    let mut bitset = 0u64;
    for slot in 0..operand_kinds.len().saturating_sub(1) {
        if slot >= u64::BITS as usize {
            break;
        }
        if operand_kinds[slot] != GeneratedOperandKind::Gpr64Register
            || operand_kinds[slot + 1] != GeneratedOperandKind::ExtendKind
        {
            continue;
        }
        let field_idx = usize::from(operand_order[slot]);
        let next_field_idx = usize::from(operand_order[slot + 1]);
        if field_idx >= semantic_fields.len() || next_field_idx >= semantic_fields.len() {
            continue;
        }
        if semantic_fields[field_idx] == "rm" && semantic_fields[next_field_idx] == "option" {
            bitset |= 1u64 << slot;
        }
    }
    bitset
}

fn variant_width_hint(variant: &str) -> VariantWidthHint {
    fn token_width_hint(token: &str) -> VariantWidthHint {
        let lower = token.to_ascii_lowercase();
        for (prefix, hint) in [("32", VariantWidthHint::W32), ("64", VariantWidthHint::W64)] {
            if let Some(rest) = lower.strip_prefix(prefix)
                && (rest.is_empty() || rest.chars().all(|ch| ch.is_ascii_alphabetic()))
            {
                return hint;
            }
        }

        // Some AArch64 variants encode width as suffix with a semantic prefix
        // (for example, `LR32` / `LR64` in load-acquire families).
        for (suffix, hint) in [("32", VariantWidthHint::W32), ("64", VariantWidthHint::W64)] {
            if let Some(prefix) = lower.strip_suffix(suffix) {
                if prefix.is_empty() || !prefix.chars().all(|ch| ch.is_ascii_alphabetic()) {
                    continue;
                }
                if matches!(prefix, "lr" | "sl" | "r" | "x" | "w") {
                    return hint;
                }
            }
        }

        VariantWidthHint::Unknown
    }

    let mut has_32 = false;
    let mut has_64 = false;
    for token in variant.split('_') {
        match token_width_hint(token) {
            VariantWidthHint::W32 => has_32 = true,
            VariantWidthHint::W64 => has_64 = true,
            VariantWidthHint::Unknown => {}
        }
    }

    match (has_32, has_64) {
        (true, false) => VariantWidthHint::W32,
        (false, true) => VariantWidthHint::W64,
        _ => VariantWidthHint::Unknown,
    }
}

fn normalize_field_name(name: &str) -> String {
    name.to_ascii_lowercase()
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

fn infer_operand_kind(
    semantic_name: &str,
    field: &FlatField,
    width_hint: VariantWidthHint,
    context: &InstructionContext,
    variant: &str,
) -> Result<GeneratedOperandKind, CodegenError> {
    if matches!(semantic_name, "op0" | "op1" | "op2" | "crn" | "crm") {
        return Ok(GeneratedOperandKind::SysRegPart);
    }
    if semantic_name == "cond" {
        return Ok(GeneratedOperandKind::Condition);
    }
    if semantic_name == "shift" {
        return Ok(GeneratedOperandKind::ShiftKind);
    }
    if semantic_name == "option" {
        return Ok(GeneratedOperandKind::ExtendKind);
    }
    if is_predicate_register_field(semantic_name) {
        return Ok(GeneratedOperandKind::PredicateRegister);
    }
    if is_sve_z_register_field(semantic_name) {
        return Ok(GeneratedOperandKind::SveZRegister);
    }
    if is_simd_register_field(semantic_name) {
        return Ok(GeneratedOperandKind::SimdRegister);
    }
    if is_gpr_register_field(semantic_name) {
        return infer_gpr_kind(semantic_name, field, width_hint, context, variant);
    }
    if is_arrangement_field(semantic_name) {
        return Ok(GeneratedOperandKind::Arrangement);
    }
    if is_lane_field(semantic_name) {
        return Ok(GeneratedOperandKind::Lane);
    }
    if is_known_immediate_field(semantic_name) {
        return Ok(GeneratedOperandKind::Immediate);
    }
    Err(CodegenError::UnmappedOperandField {
        variant: variant.to_owned(),
        field: field.name.clone(),
        width: field.width,
    })
}

fn is_predicate_register_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "pd" | "pdm" | "pdn" | "pg" | "pm" | "pn" | "pnd" | "png" | "pnn" | "pnv" | "pt" | "pv"
    )
}

fn is_sve_z_register_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "za" | "zad" | "zada" | "zan" | "zat" | "zd" | "zda" | "zdn" | "zk" | "zm" | "zn" | "zt"
    )
}

fn is_simd_register_field(field_name: &str) -> bool {
    matches!(field_name, "vd" | "vdn" | "vm" | "vn" | "vt")
}

fn is_gpr_register_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "ra" | "rd" | "rdn" | "rm" | "rn" | "rs" | "rt" | "rt2" | "rv"
    )
}

fn infer_gpr_kind(
    field_name: &str,
    field: &FlatField,
    width_hint: VariantWidthHint,
    context: &InstructionContext,
    variant: &str,
) -> Result<GeneratedOperandKind, CodegenError> {
    match (field_name, field.width) {
        ("rt", 3) => return Ok(GeneratedOperandKind::Immediate),
        ("rm", 4) => return Ok(GeneratedOperandKind::SimdRegister),
        ("rs", 2) | ("rv", 2) => return Ok(GeneratedOperandKind::Immediate),
        (_, 5) => {}
        _ => {
            return Err(CodegenError::UnmappedOperandField {
                variant: variant.to_owned(),
                field: field.name.clone(),
                width: field.width,
            });
        }
    }

    let memory_form = context.has_memory_base_data();

    if field_name == "rn" && memory_form {
        return Ok(GeneratedOperandKind::Gpr64Register);
    }

    if matches!(field_name, "rt" | "rt2" | "rt3" | "rt4") && memory_form {
        if width_hint == VariantWidthHint::Unknown && context.bit_value(26) == Some(1) {
            return Ok(GeneratedOperandKind::SimdRegister);
        }

        if width_hint == VariantWidthHint::Unknown {
            if let Some(bit30) = context.bit_value(30) {
                return if bit30 == 0 {
                    Ok(GeneratedOperandKind::Gpr32Register)
                } else {
                    Ok(GeneratedOperandKind::Gpr64Register)
                };
            }
        }

        return match width_hint {
            VariantWidthHint::W32 => Ok(GeneratedOperandKind::Gpr32Register),
            VariantWidthHint::W64 => Ok(GeneratedOperandKind::Gpr64Register),
            VariantWidthHint::Unknown => Ok(GeneratedOperandKind::GprRegister),
        };
    }

    if width_hint == VariantWidthHint::Unknown
        && !context.memory_like()
        && context
            .semantic_fields
            .iter()
            .any(|name| is_arrangement_field(name))
        && matches!(
            field_name,
            "ra" | "rd" | "rdn" | "rm" | "rn" | "rs" | "rt" | "rt2"
        )
    {
        return Ok(GeneratedOperandKind::SimdRegister);
    }

    match width_hint {
        VariantWidthHint::W32 => Ok(GeneratedOperandKind::Gpr32Register),
        VariantWidthHint::W64 => Ok(GeneratedOperandKind::Gpr64Register),
        VariantWidthHint::Unknown => Ok(GeneratedOperandKind::GprRegister),
    }
}

fn is_arrangement_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "q" | "size" | "sz" | "tsize" | "tsz" | "tszh" | "tszl" | "len" | "vl"
    )
}

fn is_lane_field(field_name: &str) -> bool {
    matches!(field_name, "lane" | "index" | "idx")
}

fn is_known_immediate_field(field_name: &str) -> bool {
    matches!(
        field_name,
        "a" | "b"
            | "b40"
            | "b5"
            | "c"
            | "cmode"
            | "d"
            | "e"
            | "f"
            | "g"
            | "h"
            | "hw"
            | "i1"
            | "i2"
            | "i2h"
            | "i2l"
            | "i3"
            | "i3h"
            | "i3l"
            | "i4"
            | "i4a"
            | "i4b"
            | "i4c"
            | "i4h"
            | "i4l"
            | "imm12"
            | "imm13"
            | "imm14"
            | "imm16"
            | "imm19"
            | "imm2"
            | "imm26"
            | "imm3"
            | "imm4"
            | "imm5"
            | "imm5b"
            | "imm6"
            | "imm7"
            | "imm8"
            | "imm8h"
            | "imm8l"
            | "imm9"
            | "imm9h"
            | "imm9l"
            | "immb"
            | "immh"
            | "immhi"
            | "immlo"
            | "immr"
            | "imms"
            | "k"
            | "l"
            | "m"
            | "mask"
            | "msz"
            | "n"
            | "nzcv"
            | "o0"
            | "o1"
            | "off2"
            | "off3"
            | "off4"
            | "op"
            | "option_13"
            | "option_15"
            | "pattern"
            | "prfop"
            | "rot"
            | "s"
            | "scale"
            | "sf"
            | "sh"
            | "simm7"
            | "t"
            | "u"
            | "v"
            | "xs"
    )
}

fn implicit_default_value(field_name: &str, kind: GeneratedOperandKind) -> Option<i64> {
    if field_name == "sh" {
        return Some(0);
    }
    if kind == GeneratedOperandKind::ShiftKind && field_name == "shift" {
        return Some(0);
    }
    None
}

fn field_rank(field_name: &str, kind: GeneratedOperandKind) -> u16 {
    match kind {
        GeneratedOperandKind::GprRegister
        | GeneratedOperandKind::Gpr32Register
        | GeneratedOperandKind::Gpr64Register
        | GeneratedOperandKind::SimdRegister
        | GeneratedOperandKind::SveZRegister
        | GeneratedOperandKind::PredicateRegister => register_rank(field_name),
        GeneratedOperandKind::Condition => 400,
        GeneratedOperandKind::ShiftKind | GeneratedOperandKind::ExtendKind => 500,
        GeneratedOperandKind::Arrangement => 550,
        GeneratedOperandKind::Lane => 575,
        GeneratedOperandKind::SysRegPart => match field_name {
            "op0" => 100,
            "op1" => 200,
            "crn" => 300,
            "crm" => 400,
            "op2" => 500,
            _ => 550,
        },
        GeneratedOperandKind::Immediate => 700,
    }
}

fn register_rank(field_name: &str) -> u16 {
    match field_name {
        "pd" | "pdm" | "pdn" | "pnd" | "rd" | "rdn" | "rt" | "vd" | "vdn" | "zd" | "zda"
        | "zdn" | "zt" | "za" | "zad" | "zada" | "zat" => 100,
        "rt2" => 110,
        "pg" | "png" | "rn" | "vn" | "zn" | "zan" => 200,
        "ra" | "pn" | "pnn" => 250,
        "pm" | "pnv" | "rm" | "vm" | "zm" => 300,
        "rs" | "pt" | "pv" => 350,
        _ => 375,
    }
}

/// Collects normalized instruction variants from generated `aarchmrs-instructions` Rust source.
///
/// The `root` must point to an ISA folder such as `.../aarchmrs-instructions/src/A64`.
///
/// # Errors
///
/// Returns [`CodegenError`] when filesystem traversal or source parsing fails.
pub fn collect_flat_from_generated_rust(root: &Path) -> Result<Vec<FlatInstruction>, CodegenError> {
    let mut files = Vec::new();
    collect_rs_files(root, &mut files)?;
    files.sort();

    let mut out = Vec::new();
    for file in files {
        let source = fs::read_to_string(&file)?;
        let mut file_items = parse_instruction_modules(&file, &source)?;
        out.append(&mut file_items);
    }

    out.sort_by(|a, b| a.mnemonic.cmp(&b.mnemonic).then(a.variant.cmp(&b.variant)));
    Ok(out)
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), CodegenError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            collect_rs_files(&path, out)?;
        } else if ty.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn parse_instruction_modules(
    path: &Path,
    source: &str,
) -> Result<Vec<FlatInstruction>, CodegenError> {
    let mut result = Vec::new();
    let mut cursor = 0usize;

    while let Some(offset) = source[cursor..].find("pub mod ") {
        let start = cursor + offset;
        let name_start = start + "pub mod ".len();
        let Some(name_end) = find_ident_end(source, name_start) else {
            return Err(parse_error(path, "invalid module name"));
        };
        let mod_name = &source[name_start..name_end];

        let idx = skip_ws(source, name_end);
        if idx >= source.len() {
            break;
        }

        if source.as_bytes()[idx] == b';' {
            cursor = idx + 1;
            continue;
        }

        if source.as_bytes()[idx] != b'{' {
            return Err(parse_error(path, "expected '{' after module declaration"));
        }

        let Some(end_brace) = find_matching_brace(source, idx) else {
            return Err(parse_error(path, "unbalanced module braces"));
        };

        let body = &source[idx + 1..end_brace];
        if let Some(inst) = parse_instruction_module(path, mod_name, body)? {
            result.push(inst);
        }

        cursor = end_brace + 1;
    }

    Ok(result)
}

fn parse_instruction_module(
    path: &Path,
    mod_name: &str,
    body: &str,
) -> Result<Option<FlatInstruction>, CodegenError> {
    let Some(mask) = parse_named_const_u32(body, "OPCODE_MASK") else {
        return Ok(None);
    };
    let Some(opcode) = parse_named_const_u32(body, "OPCODE") else {
        return Ok(None);
    };

    let Some((params, expr_body)) = parse_function_signature_and_body(mod_name, body) else {
        return Ok(None);
    };

    let semantic_fields = params
        .iter()
        .map(|param| {
            let normalized = normalize_field_name(&param.name);
            semantic_field_name(&normalized).to_owned()
        })
        .collect::<Vec<_>>();

    let mut fields = Vec::with_capacity(params.len());
    for param in params {
        let Some(lsb) = find_param_shift(expr_body, &param.name) else {
            return Err(parse_error(
                path,
                &format!("cannot find shift for field {} in {}", param.name, mod_name),
            ));
        };

        let name = param.name;
        let signed = infer_signed_field(&name, opcode, mask, &semantic_fields);
        fields.push(FlatField {
            name,
            lsb,
            width: param.width,
            signed,
        });
    }

    let mnemonic = infer_mnemonic(mod_name);
    Ok(Some(FlatInstruction {
        mnemonic,
        variant: mod_name.to_owned(),
        path: format!("{}::{}", path.display(), mod_name),
        fixed_mask: mask,
        fixed_value: opcode,
        fields,
    }))
}

#[derive(Debug)]
struct Param {
    name: String,
    width: u8,
}

fn parse_function_signature_and_body<'a>(
    mod_name: &str,
    body: &'a str,
) -> Option<(Vec<Param>, &'a str)> {
    let needle = format!("pub const fn {mod_name}");
    let start = body.find(&needle)?;
    let sig_open = body[start..].find('(')? + start;
    let sig_close = find_matching_paren(body, sig_open)?;

    let params_text = &body[sig_open + 1..sig_close];
    let params = parse_params(params_text);

    let from_u32_pos = body[sig_close..].find("InstructionCode::from_u32(")? + sig_close;
    let open = body[from_u32_pos..].find('(')? + from_u32_pos;
    let close = find_matching_paren(body, open)?;
    let expr = &body[open + 1..close];

    Some((params, expr))
}

fn parse_params(text: &str) -> Vec<Param> {
    let mut params = Vec::new();

    for raw in text.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }

        let Some((name_raw, ty_raw)) = part.split_once(':') else {
            continue;
        };
        let name = name_raw.trim().to_owned();
        let ty = ty_raw.trim();

        let width = parse_bitvalue_width(ty).unwrap_or(0);
        params.push(Param { name, width });
    }

    params
}

fn parse_bitvalue_width(ty: &str) -> Option<u8> {
    let start = ty.find("BitValue<")? + "BitValue<".len();
    let rest = &ty[start..];
    let end = rest.find('>')?;
    rest[..end].trim().parse::<u8>().ok()
}

fn parse_named_const_u32(body: &str, name: &str) -> Option<u32> {
    let needle = format!("pub const {name}: u32 =");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let end = rest.find(';')?;
    parse_u32_literal(rest[..end].trim())
}

fn parse_u32_literal(text: &str) -> Option<u32> {
    let mut s = text.trim().replace('_', "");
    if let Some(stripped) = s.strip_suffix("u32") {
        s = stripped.to_owned();
    }

    if let Some(bits) = s.strip_prefix("0b") {
        u32::from_str_radix(bits, 2).ok()
    } else if let Some(hex) = s.strip_prefix("0x") {
        u32::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u32>().ok()
    }
}

fn find_param_shift(expr: &str, param: &str) -> Option<u8> {
    let needle = format!("{param}.into_inner()");
    let mut cursor = 0usize;

    while let Some(off) = expr[cursor..].find(&needle) {
        let start = cursor + off + needle.len();
        let tail = &expr[start..];
        let tail = tail.trim_start();
        if !tail.starts_with("<<") {
            cursor = start;
            continue;
        }

        let tail = tail[2..].trim_start();
        let digits_len = tail.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits_len == 0 {
            cursor = start;
            continue;
        }

        if let Ok(value) = tail[..digits_len].parse::<u8>() {
            return Some(value);
        }
        cursor = start;
    }

    None
}

fn find_ident_end(source: &str, start: usize) -> Option<usize> {
    let mut end = start;
    for ch in source[start..].chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            end += ch.len_utf8();
        } else {
            break;
        }
    }
    if end == start { None } else { Some(end) }
}

fn skip_ws(source: &str, mut idx: usize) -> usize {
    while idx < source.len() && source.as_bytes()[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx
}

fn find_matching_brace(source: &str, open_idx: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open_idx..].iter().enumerate() {
        match *byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_idx + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_matching_paren(source: &str, open_idx: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open_idx..].iter().enumerate() {
        match *byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_idx + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_error(path: &Path, message: &str) -> CodegenError {
    CodegenError::Parse {
        path: path.display().to_string(),
        message: message.to_owned(),
    }
}

fn infer_mnemonic(variant: &str) -> String {
    variant
        .split('_')
        .next()
        .unwrap_or(variant)
        .to_ascii_lowercase()
}

fn infer_signed_field(
    field_name: &str,
    opcode: u32,
    opcode_mask: u32,
    semantic_fields: &[String],
) -> bool {
    let field = normalize_field_name(field_name);
    let semantic_name = semantic_field_name(&field);

    if semantic_name.starts_with("simm") || semantic_name.contains("offset") {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_context() -> InstructionContext {
        InstructionContext::from_semantic_fields(0, 0, Vec::new())
    }

    #[test]
    fn generate_non_empty() {
        let input = vec![FlatInstruction {
            mnemonic: "add".to_string(),
            variant: "ADD_64_addsub_imm".to_string(),
            path: "A64/dpimm/ADD_64_addsub_imm".to_string(),
            fixed_mask: 0xffff_ffff,
            fixed_value: 0x9100_0000,
            fields: vec![FlatField {
                name: "Rd".to_string(),
                lsb: 0,
                width: 5,
                signed: false,
            }],
        }];

        let text = generate_encoder_module(&input).expect("codegen should succeed");
        assert!(text.contains("SPECS"));
        assert!(text.contains("ADD_64_addsub_imm"));
    }

    #[test]
    fn macro_normalization_module_includes_generated_rules() {
        let input = vec![
            FlatInstruction {
                mnemonic: "msr".to_string(),
                variant: "MSR_SR_systemmove".to_string(),
                path: "A64/control/MSR_SR_systemmove".to_string(),
                fixed_mask: 0xfff0_0000,
                fixed_value: 0xd510_0000,
                fields: vec![
                    FlatField {
                        name: "o0".to_string(),
                        lsb: 19,
                        width: 1,
                        signed: false,
                    },
                    FlatField {
                        name: "op1".to_string(),
                        lsb: 16,
                        width: 3,
                        signed: false,
                    },
                    FlatField {
                        name: "CRn".to_string(),
                        lsb: 12,
                        width: 4,
                        signed: false,
                    },
                    FlatField {
                        name: "CRm".to_string(),
                        lsb: 8,
                        width: 4,
                        signed: false,
                    },
                    FlatField {
                        name: "op2".to_string(),
                        lsb: 5,
                        width: 3,
                        signed: false,
                    },
                    FlatField {
                        name: "Rt".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "add".to_string(),
                variant: "ADD_64_addsub_imm".to_string(),
                path: "A64/dpimm/ADD_64_addsub_imm".to_string(),
                fixed_mask: 0xff80_0000,
                fixed_value: 0x9100_0000,
                fields: vec![
                    FlatField {
                        name: "sh".to_string(),
                        lsb: 22,
                        width: 1,
                        signed: false,
                    },
                    FlatField {
                        name: "imm12".to_string(),
                        lsb: 10,
                        width: 12,
                        signed: false,
                    },
                    FlatField {
                        name: "Rn".to_string(),
                        lsb: 5,
                        width: 5,
                        signed: false,
                    },
                    FlatField {
                        name: "Rd".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "b".to_string(),
                variant: "B_only_branch_imm".to_string(),
                path: "A64/control/B_only_branch_imm".to_string(),
                fixed_mask: 0xfc00_0000,
                fixed_value: 0x1400_0000,
                fields: vec![FlatField {
                    name: "imm26".to_string(),
                    lsb: 0,
                    width: 26,
                    signed: true,
                }],
            },
            FlatInstruction {
                mnemonic: "b".to_string(),
                variant: "B_only_condbranch".to_string(),
                path: "A64/control/B_only_condbranch".to_string(),
                fixed_mask: 0xff00_0010,
                fixed_value: 0x5400_0000,
                fields: vec![
                    FlatField {
                        name: "imm19".to_string(),
                        lsb: 5,
                        width: 19,
                        signed: true,
                    },
                    FlatField {
                        name: "cond".to_string(),
                        lsb: 0,
                        width: 4,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "cbz".to_string(),
                variant: "CBZ_64_compbranch".to_string(),
                path: "A64/control/CBZ_64_compbranch".to_string(),
                fixed_mask: 0x7f00_0000,
                fixed_value: 0x3400_0000,
                fields: vec![
                    FlatField {
                        name: "imm19".to_string(),
                        lsb: 5,
                        width: 19,
                        signed: true,
                    },
                    FlatField {
                        name: "Rt".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "tbz".to_string(),
                variant: "TBZ_only_testbranch".to_string(),
                path: "A64/control/TBZ_only_testbranch".to_string(),
                fixed_mask: 0x7f00_0000,
                fixed_value: 0x3600_0000,
                fields: vec![
                    FlatField {
                        name: "b5".to_string(),
                        lsb: 31,
                        width: 1,
                        signed: false,
                    },
                    FlatField {
                        name: "b40".to_string(),
                        lsb: 19,
                        width: 5,
                        signed: false,
                    },
                    FlatField {
                        name: "imm14".to_string(),
                        lsb: 5,
                        width: 14,
                        signed: true,
                    },
                    FlatField {
                        name: "Rt".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "adr".to_string(),
                variant: "ADR_only_pcreladdr".to_string(),
                path: "A64/control/ADR_only_pcreladdr".to_string(),
                fixed_mask: 0x9f00_0000,
                fixed_value: 0x1000_0000,
                fields: vec![
                    FlatField {
                        name: "immlo".to_string(),
                        lsb: 29,
                        width: 2,
                        signed: false,
                    },
                    FlatField {
                        name: "immhi".to_string(),
                        lsb: 5,
                        width: 19,
                        signed: true,
                    },
                    FlatField {
                        name: "Rd".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
            FlatInstruction {
                mnemonic: "adrp".to_string(),
                variant: "ADRP_only_pcreladdr".to_string(),
                path: "A64/control/ADRP_only_pcreladdr".to_string(),
                fixed_mask: 0x9f00_0000,
                fixed_value: 0x9000_0000,
                fields: vec![
                    FlatField {
                        name: "immlo".to_string(),
                        lsb: 29,
                        width: 2,
                        signed: false,
                    },
                    FlatField {
                        name: "immhi".to_string(),
                        lsb: 5,
                        width: 19,
                        signed: true,
                    },
                    FlatField {
                        name: "Rd".to_string(),
                        lsb: 0,
                        width: 5,
                        signed: false,
                    },
                ],
            },
        ];

        let text = generate_macro_normalization_module(&input).expect("module generation");
        assert!(text.contains("MNEMONIC_NORMALIZATION_RULES"));
        assert!(text.contains("CONDITIONAL_BRANCH_ALIAS_RULES"));
        assert!(text.contains("NORM_FLAG_SHIFT_TO_IMMEDIATE"));
        assert!(text.contains("NORM_FLAG_SYSREG_GPR_SWAP"));
        assert!(text.contains("RELOC_MASK_B26"));
        assert!(text.contains("RELOC_MASK_BCOND19"));
        assert!(text.contains("RELOC_MASK_CBZ19"));
        assert!(text.contains("RELOC_MASK_IMM19"));
        assert!(text.contains("RELOC_MASK_TBZ14"));
        assert!(text.contains("RELOC_MASK_ADR21"));
        assert!(text.contains("RELOC_MASK_ADRP21"));
        assert!(text.contains("\"beq\""));
        assert!(text.contains("\"b.eq\""));
        assert!(text.contains("\"msr\""));
        assert!(text.contains("\"b\""));
        assert!(text.contains("\"cbz\""));
        assert!(text.contains("\"tbz\""));
        assert!(text.contains("\"adr\""));
        assert!(text.contains("\"adrp\""));
    }

    #[test]
    fn source_name_mapping_is_deterministic_for_core_kinds() {
        let variant = "CSEL_64_condsel";
        let width_hint = variant_width_hint(variant);
        let context = empty_context();

        let rd = infer_operand_kind(
            semantic_field_name("rd"),
            &FlatField {
                name: "Rd".to_string(),
                lsb: 0,
                width: 5,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("rd must map");
        let cond = infer_operand_kind(
            semantic_field_name("cond"),
            &FlatField {
                name: "cond".to_string(),
                lsb: 12,
                width: 4,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("cond must map");
        let shift = infer_operand_kind(
            semantic_field_name("shift"),
            &FlatField {
                name: "shift".to_string(),
                lsb: 22,
                width: 2,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("shift must map");
        let extend = infer_operand_kind(
            semantic_field_name("option"),
            &FlatField {
                name: "option".to_string(),
                lsb: 13,
                width: 3,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("extend must map");
        let sysreg = infer_operand_kind(
            semantic_field_name("crn"),
            &FlatField {
                name: "CRn".to_string(),
                lsb: 12,
                width: 4,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("sysreg part must map");
        let p_reg = infer_operand_kind(
            semantic_field_name("pd"),
            &FlatField {
                name: "Pd".to_string(),
                lsb: 0,
                width: 4,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("predicate must map");
        let z_reg = infer_operand_kind(
            semantic_field_name("zd"),
            &FlatField {
                name: "Zd".to_string(),
                lsb: 0,
                width: 5,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("z must map");
        let v_reg = infer_operand_kind(
            semantic_field_name("vn"),
            &FlatField {
                name: "Vn".to_string(),
                lsb: 5,
                width: 5,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("simd must map");
        let split_option = infer_operand_kind(
            semantic_field_name("option_13"),
            &FlatField {
                name: "option_13".to_string(),
                lsb: 13,
                width: 1,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("split option must map");

        assert_eq!(rd, GeneratedOperandKind::Gpr64Register);
        assert_eq!(cond, GeneratedOperandKind::Condition);
        assert_eq!(shift, GeneratedOperandKind::ShiftKind);
        assert_eq!(extend, GeneratedOperandKind::ExtendKind);
        assert_eq!(sysreg, GeneratedOperandKind::SysRegPart);
        assert_eq!(p_reg, GeneratedOperandKind::PredicateRegister);
        assert_eq!(z_reg, GeneratedOperandKind::SveZRegister);
        assert_eq!(v_reg, GeneratedOperandKind::SimdRegister);
        assert_eq!(split_option, GeneratedOperandKind::Immediate);
    }

    #[test]
    fn variant_width_hint_recognizes_suffix_tokens() {
        assert_eq!(
            variant_width_hint("SUBS_32S_addsub_imm"),
            VariantWidthHint::W32
        );
        assert_eq!(
            variant_width_hint("SUBS_64S_addsub_imm"),
            VariantWidthHint::W64
        );
        assert_eq!(
            variant_width_hint("ADD_64_addsub_imm"),
            VariantWidthHint::W64
        );
        assert_eq!(
            variant_width_hint("LDAR_LR32_ldstord"),
            VariantWidthHint::W32
        );
        assert_eq!(
            variant_width_hint("LDAR_LR64_ldstord"),
            VariantWidthHint::W64
        );
        assert_eq!(
            variant_width_hint("STLRH_SL32_ldstord"),
            VariantWidthHint::W32
        );
        assert_eq!(
            variant_width_hint("STLR_SL64_ldstord"),
            VariantWidthHint::W64
        );
        assert_eq!(variant_width_hint("FOO_U32_bar"), VariantWidthHint::Unknown);
        assert_eq!(variant_width_hint("foo_bar"), VariantWidthHint::Unknown);
    }

    #[test]
    fn infer_signed_field_uses_structural_context() {
        let pair_fields = vec![
            "rt".to_string(),
            "rt2".to_string(),
            "rn".to_string(),
            "imm7".to_string(),
        ];
        let imm9_fields = vec!["rt".to_string(), "rn".to_string(), "imm9".to_string()];
        let pcrel_fields = vec!["rd".to_string(), "immlo".to_string(), "immhi".to_string()];

        assert!(infer_signed_field(
            "imm7",
            0xa9800000,
            0xffc00000,
            &pair_fields
        ));
        assert!(infer_signed_field(
            "imm9",
            0xf8000400,
            0xffe00c00,
            &imm9_fields
        ));
        assert!(!infer_signed_field(
            "imm12",
            0xf9000000,
            0xffc00000,
            &imm9_fields
        ));
        assert!(infer_signed_field(
            "immhi",
            0x10000000,
            0x9f000000,
            &pcrel_fields
        ));
        assert!(!infer_signed_field(
            "immlo",
            0x10000000,
            0x9f000000,
            &pcrel_fields
        ));
    }

    #[test]
    fn source_name_mapping_handles_known_outliers() {
        let variant = "RPRFM_R_ldst_regoff";
        let width_hint = variant_width_hint(variant);
        let context = empty_context();
        let rt = infer_operand_kind(
            semantic_field_name("rt"),
            &FlatField {
                name: "Rt".to_string(),
                lsb: 0,
                width: 3,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("rt outlier must map");
        let rm = infer_operand_kind(
            semantic_field_name("rm"),
            &FlatField {
                name: "Rm".to_string(),
                lsb: 16,
                width: 4,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect("rm outlier must map");

        assert_eq!(rt, GeneratedOperandKind::Immediate);
        assert_eq!(rm, GeneratedOperandKind::SimdRegister);
    }

    #[test]
    fn memory_rt_operands_become_simd_when_v_bit_is_fixed() {
        let variant = "TEST_ldstpair";
        let context = InstructionContext::from_semantic_fields(
            0x6d80_0000,
            0xffc0_0000,
            vec![
                "imm7".to_string(),
                "rt2".to_string(),
                "rn".to_string(),
                "rt".to_string(),
            ],
        );

        let rt = infer_operand_kind(
            semantic_field_name("rt"),
            &FlatField {
                name: "Rt".to_string(),
                lsb: 0,
                width: 5,
                signed: false,
            },
            VariantWidthHint::Unknown,
            &context,
            variant,
        )
        .expect("rt must map");

        let rn = infer_operand_kind(
            semantic_field_name("rn"),
            &FlatField {
                name: "Rn".to_string(),
                lsb: 5,
                width: 5,
                signed: false,
            },
            VariantWidthHint::Unknown,
            &context,
            variant,
        )
        .expect("rn must map");

        assert_eq!(rt, GeneratedOperandKind::SimdRegister);
        assert_eq!(rn, GeneratedOperandKind::Gpr64Register);
    }

    #[test]
    fn memory_rt_operands_follow_lr_width_suffix_hint() {
        let context = InstructionContext::from_semantic_fields(
            0x88df_fc00,
            0xffff_fc00,
            vec!["rn".to_string(), "rt".to_string()],
        );
        let rn_field = FlatField {
            name: "Rn".to_string(),
            lsb: 5,
            width: 5,
            signed: false,
        };
        let rt_field = FlatField {
            name: "Rt".to_string(),
            lsb: 0,
            width: 5,
            signed: false,
        };
        let rn = infer_operand_kind(
            semantic_field_name("rn"),
            &rn_field,
            variant_width_hint("LDAR_LR32_ldstord"),
            &context,
            "LDAR_LR32_ldstord",
        )
        .expect("rn must map");

        let rt32 = infer_operand_kind(
            semantic_field_name("rt"),
            &rt_field,
            variant_width_hint("LDAR_LR32_ldstord"),
            &context,
            "LDAR_LR32_ldstord",
        )
        .expect("rt32 must map");
        let rt64 = infer_operand_kind(
            semantic_field_name("rt"),
            &rt_field,
            variant_width_hint("LDAR_LR64_ldstord"),
            &context,
            "LDAR_LR64_ldstord",
        )
        .expect("rt64 must map");

        assert_eq!(rn, GeneratedOperandKind::Gpr64Register);
        assert_eq!(rt32, GeneratedOperandKind::Gpr32Register);
        assert_eq!(rt64, GeneratedOperandKind::Gpr64Register);
    }

    #[test]
    fn memory_rt_operands_fall_back_to_bit30_width_for_stlr_sl() {
        let context32 = InstructionContext::from_semantic_fields(
            0x889f_fc00,
            0xffff_fc00,
            vec!["rn".to_string(), "rt".to_string()],
        );
        let context64 = InstructionContext::from_semantic_fields(
            0xc89f_fc00,
            0xffff_fc00,
            vec!["rn".to_string(), "rt".to_string()],
        );
        let rt_field = FlatField {
            name: "Rt".to_string(),
            lsb: 0,
            width: 5,
            signed: false,
        };
        let rn_field = FlatField {
            name: "Rn".to_string(),
            lsb: 5,
            width: 5,
            signed: false,
        };

        let rt32 = infer_operand_kind(
            semantic_field_name("rt"),
            &rt_field,
            VariantWidthHint::Unknown,
            &context32,
            "STLR_SL32_ldstord",
        )
        .expect("rt32 must map");
        let rt64 = infer_operand_kind(
            semantic_field_name("rt"),
            &rt_field,
            VariantWidthHint::Unknown,
            &context64,
            "STLR_SL64_ldstord",
        )
        .expect("rt64 must map");
        let rn32 = infer_operand_kind(
            semantic_field_name("rn"),
            &rn_field,
            VariantWidthHint::Unknown,
            &context32,
            "STLR_SL32_ldstord",
        )
        .expect("rn32 must map");
        let rn64 = infer_operand_kind(
            semantic_field_name("rn"),
            &rn_field,
            VariantWidthHint::Unknown,
            &context64,
            "STLR_SL64_ldstord",
        )
        .expect("rn64 must map");

        assert_eq!(rt32, GeneratedOperandKind::Gpr32Register);
        assert_eq!(rt64, GeneratedOperandKind::Gpr64Register);
        assert_eq!(rn32, GeneratedOperandKind::Gpr64Register);
        assert_eq!(rn64, GeneratedOperandKind::Gpr64Register);
    }

    #[test]
    fn no_offset_memory_forms_are_tagged_explicitly() {
        let inst = FlatInstruction {
            mnemonic: "stlr".to_string(),
            variant: "STLR_SL64_ldstord".to_string(),
            path: "A64/ldst/STLR_SL64_ldstord".to_string(),
            fixed_mask: 0xffff_fc00,
            fixed_value: 0xc89f_fc00,
            fields: vec![
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rt".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        assert_eq!(
            derive_memory_addressing_constraint(&inst),
            GeneratedMemoryAddressingConstraint::NoOffset
        );
    }

    #[test]
    fn writeback_memory_forms_are_tagged_as_preindex() {
        let inst = FlatInstruction {
            mnemonic: "stlr".to_string(),
            variant: "STLR_64S_ldapstl_writeback".to_string(),
            path: "A64/ldst/STLR_64S_ldapstl_writeback".to_string(),
            fixed_mask: 0xffff_fc00,
            fixed_value: 0xd980_0800,
            fields: vec![
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rt".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        assert_eq!(
            derive_memory_addressing_constraint(&inst),
            GeneratedMemoryAddressingConstraint::PreIndex
        );
    }

    #[test]
    fn unknown_field_mapping_fails_generation() {
        let input = vec![FlatInstruction {
            mnemonic: "foo".to_string(),
            variant: "FOO_only_test".to_string(),
            path: "A64/test/FOO_only_test".to_string(),
            fixed_mask: 0xffff_ffff,
            fixed_value: 0,
            fields: vec![FlatField {
                name: "mystery".to_string(),
                lsb: 0,
                width: 7,
                signed: false,
            }],
        }];

        let err = generate_encoder_module(&input).expect_err("must fail for unknown field");
        match err {
            CodegenError::UnmappedOperandField {
                variant,
                field,
                width,
            } => {
                assert_eq!(variant, "FOO_only_test");
                assert_eq!(field, "mystery");
                assert_eq!(width, 7);
            }
            other => panic!("unexpected error kind: {other}"),
        }
    }

    #[test]
    fn i_prefix_is_not_accepted_without_explicit_mapping() {
        let variant = "FOO_only_test";
        let width_hint = variant_width_hint(variant);
        let context = empty_context();
        let err = infer_operand_kind(
            semantic_field_name("ix"),
            &FlatField {
                name: "ix".to_string(),
                lsb: 10,
                width: 3,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect_err("unknown i* field must fail");

        match err {
            CodegenError::UnmappedOperandField {
                variant,
                field,
                width,
            } => {
                assert_eq!(variant, "FOO_only_test");
                assert_eq!(field, "ix");
                assert_eq!(width, 3);
            }
            other => panic!("unexpected error kind: {other}"),
        }
    }

    #[test]
    fn unsupported_gpr_width_fails_with_unmapped_error() {
        let variant = "FOO_64_test";
        let width_hint = variant_width_hint(variant);
        let context = empty_context();
        let err = infer_operand_kind(
            semantic_field_name("rn"),
            &FlatField {
                name: "Rn".to_string(),
                lsb: 5,
                width: 4,
                signed: false,
            },
            width_hint,
            &context,
            variant,
        )
        .expect_err("unsupported width must fail");

        match err {
            CodegenError::UnmappedOperandField {
                variant,
                field,
                width,
            } => {
                assert_eq!(variant, "FOO_64_test");
                assert_eq!(field, "Rn");
                assert_eq!(width, 4);
            }
            other => panic!("unexpected error kind: {other}"),
        }
    }

    #[test]
    fn pair_memory_order_is_rt_then_rt2() {
        let inst = FlatInstruction {
            mnemonic: "stp".to_string(),
            variant: "STP_64_ldstpair_pre".to_string(),
            path: "A64/ldst/STP_64_ldstpair_pre".to_string(),
            fixed_mask: 0xffc0_0000,
            fixed_value: 0xa980_0000,
            fields: vec![
                FlatField {
                    name: "imm7".to_string(),
                    lsb: 15,
                    width: 7,
                    signed: true,
                },
                FlatField {
                    name: "Rt2".to_string(),
                    lsb: 10,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rt".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        let (order, kinds, defaults) = derive_operand_metadata(&inst).expect("metadata");
        assert_eq!(order, vec![3, 1, 2, 0]);
        assert_eq!(
            kinds,
            vec![
                GeneratedOperandKind::Gpr64Register,
                GeneratedOperandKind::Gpr64Register,
                GeneratedOperandKind::Gpr64Register,
                GeneratedOperandKind::Immediate
            ]
        );
        assert!(defaults.is_empty());
    }

    #[test]
    fn derives_split_immediate_plan_for_adr_like_variants() {
        let inst = FlatInstruction {
            mnemonic: "adr".to_string(),
            variant: "ADR_only_pcreladdr".to_string(),
            path: "A64/adr/ADR_only_pcreladdr".to_string(),
            fixed_mask: 0x9f00_0000,
            fixed_value: 0x1000_0000,
            fields: vec![
                FlatField {
                    name: "immlo".to_string(),
                    lsb: 29,
                    width: 2,
                    signed: false,
                },
                FlatField {
                    name: "immhi".to_string(),
                    lsb: 5,
                    width: 19,
                    signed: true,
                },
                FlatField {
                    name: "Rd".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        let (order, kinds, _) = derive_operand_metadata(&inst).expect("metadata");
        let plan = derive_split_immediate_plan(&inst, &order, &kinds).expect("split plan");
        assert_eq!(
            plan,
            GeneratedSplitImmediatePlan {
                first_slot: 1,
                second_slot: 2,
                kind: GeneratedSplitImmediateKind::AdrLike {
                    immlo_field_index: 0,
                    immhi_field_index: 1,
                    scale: 1,
                },
            }
        );
    }

    #[test]
    fn derives_split_immediate_plan_for_logical_immediate_variants() {
        let inst = FlatInstruction {
            mnemonic: "eor".to_string(),
            variant: "EOR_64_log_imm".to_string(),
            path: "A64/eor/EOR_64_log_imm".to_string(),
            fixed_mask: 0xff80_0000,
            fixed_value: 0xd200_0000,
            fields: vec![
                FlatField {
                    name: "N".to_string(),
                    lsb: 22,
                    width: 1,
                    signed: false,
                },
                FlatField {
                    name: "immr".to_string(),
                    lsb: 16,
                    width: 6,
                    signed: false,
                },
                FlatField {
                    name: "imms".to_string(),
                    lsb: 10,
                    width: 6,
                    signed: false,
                },
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rd".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        let (order, kinds, _) = derive_operand_metadata(&inst).expect("metadata");
        let plan = derive_split_immediate_plan(&inst, &order, &kinds).expect("split plan");
        assert_eq!(
            plan,
            GeneratedSplitImmediatePlan {
                first_slot: 2,
                second_slot: 4,
                kind: GeneratedSplitImmediateKind::LogicalImmNrs {
                    n_field_index: 0,
                    immr_field_index: 1,
                    imms_field_index: 2,
                    reg_size: 64,
                },
            }
        );
        assert_eq!(
            expected_user_operand_kinds(&kinds, Some(plan)),
            vec![
                GeneratedOperandKind::Gpr64Register,
                GeneratedOperandKind::Gpr64Register,
                GeneratedOperandKind::Immediate
            ]
        );
    }

    #[test]
    fn derives_split_immediate_plan_for_32bit_logical_immediate_variants() {
        let inst = FlatInstruction {
            mnemonic: "eor".to_string(),
            variant: "EOR_32_log_imm".to_string(),
            path: "A64/eor/EOR_32_log_imm".to_string(),
            fixed_mask: 0xffc0_0000,
            fixed_value: 0x5200_0000,
            fields: vec![
                FlatField {
                    name: "immr".to_string(),
                    lsb: 16,
                    width: 6,
                    signed: false,
                },
                FlatField {
                    name: "imms".to_string(),
                    lsb: 10,
                    width: 6,
                    signed: false,
                },
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rd".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        let (order, kinds, _) = derive_operand_metadata(&inst).expect("metadata");
        let plan = derive_split_immediate_plan(&inst, &order, &kinds).expect("split plan");
        assert_eq!(
            plan,
            GeneratedSplitImmediatePlan {
                first_slot: 2,
                second_slot: 3,
                kind: GeneratedSplitImmediateKind::LogicalImmRs {
                    immr_field_index: 0,
                    imms_field_index: 1,
                    reg_size: 32,
                },
            }
        );
        assert_eq!(
            expected_user_operand_kinds(&kinds, Some(plan)),
            vec![
                GeneratedOperandKind::Gpr32Register,
                GeneratedOperandKind::Gpr32Register,
                GeneratedOperandKind::Immediate
            ]
        );
    }

    #[test]
    fn derives_gpr32_extend_compatibility_for_add_ext() {
        let inst = FlatInstruction {
            mnemonic: "add".to_string(),
            variant: "ADD_64_addsub_ext".to_string(),
            path: "A64/add/ADD_64_addsub_ext".to_string(),
            fixed_mask: 0xffe0_0000,
            fixed_value: 0x8b20_0000,
            fields: vec![
                FlatField {
                    name: "Rm".to_string(),
                    lsb: 16,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "option".to_string(),
                    lsb: 13,
                    width: 3,
                    signed: false,
                },
                FlatField {
                    name: "imm3".to_string(),
                    lsb: 10,
                    width: 3,
                    signed: false,
                },
                FlatField {
                    name: "Rn".to_string(),
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                FlatField {
                    name: "Rd".to_string(),
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
        };

        let (order, kinds, _) = derive_operand_metadata(&inst).expect("metadata");
        let bitset = derive_gpr32_extend_compatibility(&inst, &order, &kinds);
        assert_eq!(bitset, 0b100);
    }
}
