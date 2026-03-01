//! Runtime variant matcher and field encoder.
//!
//! High-level flow:
//! 1. Flatten structured operands into a linear kind/value stream.
//! 2. Materialize optional fields (for example bare `[base]` vs explicit `#0`).
//! 3. Select candidate variants by shape and validate constraints.
//! 4. Encode normalized values into bitfields with scale/range checks.
//! 5. Return precise diagnostics when no unique variant can be chosen.

use alloc::vec::Vec;

use crate::types::*;

fn field_mask(width: u8) -> u32 {
    if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct FlatOperand {
    /// Operand kind token used by the selector.
    kind: OperandConstraintKind,
    /// Canonical immediate/register index value after flattening.
    value: i64,
    /// Optional tokens can be omitted when matching shorter forms.
    optional: bool,
}

#[inline]
fn reg_constraint_for_class(class: RegClass) -> OperandConstraintKind {
    match class {
        RegClass::W | RegClass::Wsp => OperandConstraintKind::Gpr32Register,
        RegClass::X | RegClass::Xsp => OperandConstraintKind::Gpr64Register,
        RegClass::V | RegClass::B | RegClass::H | RegClass::S | RegClass::D | RegClass::Q => {
            OperandConstraintKind::SimdRegister
        }
        RegClass::Z => OperandConstraintKind::SveZRegister,
        RegClass::P => OperandConstraintKind::PredicateRegister,
    }
}

#[inline]
fn condition_code_value(code: ConditionCode) -> i64 {
    match code {
        ConditionCode::Eq => 0,
        ConditionCode::Ne => 1,
        ConditionCode::Cs => 2,
        ConditionCode::Cc => 3,
        ConditionCode::Mi => 4,
        ConditionCode::Pl => 5,
        ConditionCode::Vs => 6,
        ConditionCode::Vc => 7,
        ConditionCode::Hi => 8,
        ConditionCode::Ls => 9,
        ConditionCode::Ge => 10,
        ConditionCode::Lt => 11,
        ConditionCode::Gt => 12,
        ConditionCode::Le => 13,
        ConditionCode::Al => 14,
        ConditionCode::Nv => 15,
    }
}

#[inline]
fn shift_kind_value(kind: ShiftKind) -> i64 {
    match kind {
        ShiftKind::Lsl => 0,
        ShiftKind::Lsr => 1,
        ShiftKind::Asr => 2,
        ShiftKind::Ror => 3,
        ShiftKind::Msl => 4,
    }
}

#[inline]
fn extend_kind_value(kind: ExtendKind) -> i64 {
    match kind {
        ExtendKind::Uxtb => 0,
        ExtendKind::Uxth => 1,
        ExtendKind::Uxtw => 2,
        ExtendKind::Uxtx => 3,
        ExtendKind::Sxtb => 4,
        ExtendKind::Sxth => 5,
        ExtendKind::Sxtw => 6,
        ExtendKind::Sxtx => 7,
    }
}

#[inline]
fn arrangement_value(arrangement: VectorArrangement) -> i64 {
    match arrangement {
        VectorArrangement::B8 => 0,
        VectorArrangement::B16 => 1,
        VectorArrangement::H4 => 2,
        VectorArrangement::H8 => 3,
        VectorArrangement::S2 => 4,
        VectorArrangement::S4 => 5,
        VectorArrangement::D1 => 6,
        VectorArrangement::D2 => 7,
        VectorArrangement::Q1 => 8,
    }
}

#[inline]
fn push_flat(
    out: &mut [FlatOperand; 64],
    len: &mut usize,
    operand: FlatOperand,
) -> Result<(), EncodeError> {
    if *len >= out.len() {
        return Err(EncodeError::OperandCountMismatch);
    }
    out[*len] = operand;
    *len += 1;
    Ok(())
}

#[inline]
fn push_flat_optional(
    out: &mut [FlatOperand; 64],
    len: &mut usize,
    kind: OperandConstraintKind,
    value: i64,
) -> Result<(), EncodeError> {
    push_flat(
        out,
        len,
        FlatOperand {
            kind,
            value,
            optional: true,
        },
    )
}

fn flatten_register(
    reg: RegisterOperand,
    out: &mut [FlatOperand; 64],
    len: &mut usize,
) -> Result<(), EncodeError> {
    push_flat(
        out,
        len,
        FlatOperand {
            kind: reg_constraint_for_class(reg.class),
            value: i64::from(reg.code),
            optional: false,
        },
    )?;

    if let Some(arrangement) = reg.arrangement {
        push_flat(
            out,
            len,
            FlatOperand {
                kind: OperandConstraintKind::Arrangement,
                value: arrangement_value(arrangement),
                optional: false,
            },
        )?;
    }

    if let Some(lane) = reg.lane {
        push_flat(
            out,
            len,
            FlatOperand {
                kind: OperandConstraintKind::Lane,
                value: i64::from(lane),
                optional: false,
            },
        )?;
    }

    Ok(())
}

fn flatten_modifier(
    modifier: Modifier,
    out: &mut [FlatOperand; 64],
    len: &mut usize,
) -> Result<(), EncodeError> {
    match modifier {
        Modifier::Shift(shift) => {
            push_flat(
                out,
                len,
                FlatOperand {
                    kind: OperandConstraintKind::ShiftKind,
                    value: shift_kind_value(shift.kind),
                    optional: false,
                },
            )?;
            push_flat(
                out,
                len,
                FlatOperand {
                    kind: OperandConstraintKind::Immediate,
                    value: i64::from(shift.amount),
                    optional: false,
                },
            )?;
        }
        Modifier::Extend(extend) => {
            push_flat(
                out,
                len,
                FlatOperand {
                    kind: OperandConstraintKind::ExtendKind,
                    value: extend_kind_value(extend.kind),
                    optional: false,
                },
            )?;
            push_flat(
                out,
                len,
                FlatOperand {
                    kind: OperandConstraintKind::Immediate,
                    value: i64::from(extend.amount.unwrap_or(0)),
                    optional: false,
                },
            )?;
        }
    }
    Ok(())
}

fn flatten_operand(
    operand: Operand,
    out: &mut [FlatOperand; 64],
    len: &mut usize,
) -> Result<(), EncodeError> {
    match operand {
        Operand::Register(reg) => flatten_register(reg, out, len),
        Operand::Immediate(value) => push_flat(
            out,
            len,
            FlatOperand {
                kind: OperandConstraintKind::Immediate,
                value,
                optional: false,
            },
        ),
        Operand::Condition(cond) => push_flat(
            out,
            len,
            FlatOperand {
                kind: OperandConstraintKind::Condition,
                value: condition_code_value(cond),
                optional: false,
            },
        ),
        Operand::Shift(shift) => flatten_modifier(Modifier::Shift(shift), out, len),
        Operand::Extend(extend) => flatten_modifier(Modifier::Extend(extend), out, len),
        Operand::Memory(mem) => {
            flatten_register(mem.base, out, len)?;
            match mem.offset {
                MemoryOffset::None => {
                    if mem.addressing == AddressingMode::Offset {
                        // Bare `[base]` can map either to no-offset forms (e.g. LDAR)
                        // or to encodings with an explicit zero immediate (e.g. LDR).
                        push_flat_optional(out, len, OperandConstraintKind::Immediate, 0)?;
                    }
                }
                MemoryOffset::Immediate(value) => {
                    push_flat(
                        out,
                        len,
                        FlatOperand {
                            kind: OperandConstraintKind::Immediate,
                            value,
                            optional: false,
                        },
                    )?;
                }
                MemoryOffset::Register { reg, modifier } => {
                    flatten_register(reg, out, len)?;
                    if let Some(modifier) = modifier {
                        flatten_modifier(modifier, out, len)?;
                    }
                }
            }

            if let AddressingMode::PostIndex(offset) = mem.addressing {
                match offset {
                    PostIndexOffset::Immediate(value) => {
                        push_flat(
                            out,
                            len,
                            FlatOperand {
                                kind: OperandConstraintKind::Immediate,
                                value,
                                optional: false,
                            },
                        )?;
                    }
                    PostIndexOffset::Register(reg) => {
                        flatten_register(reg, out, len)?;
                    }
                }
            }

            Ok(())
        }
        Operand::RegisterList(list) => {
            if list.count == 0 {
                return Err(EncodeError::OperandCountMismatch);
            }

            let mut idx = 0u8;
            while idx < list.count {
                let code = list.first.code.saturating_add(idx);
                if code > 31 {
                    return Err(EncodeError::OperandOutOfRange {
                        field: "reglist",
                        value: i64::from(code),
                        width: 5,
                        signed: false,
                    });
                }
                flatten_register(
                    RegisterOperand {
                        code,
                        class: list.first.class,
                        arrangement: list.first.arrangement,
                        lane: list.first.lane,
                    },
                    out,
                    len,
                )?;
                idx = idx.saturating_add(1);
            }
            Ok(())
        }
        Operand::SysReg(sys) => {
            let o0 = match sys.op0 {
                2 | 3 => sys.op0 - 2,
                _ => {
                    return Err(EncodeError::OperandOutOfRange {
                        field: "op0",
                        value: i64::from(sys.op0),
                        width: 2,
                        signed: false,
                    });
                }
            };

            for value in [sys.op1, sys.crn, sys.crm, sys.op2] {
                push_flat(
                    out,
                    len,
                    FlatOperand {
                        kind: OperandConstraintKind::SysRegPart,
                        value: i64::from(value),
                        optional: false,
                    },
                )?;
            }
            push_flat(
                out,
                len,
                FlatOperand {
                    kind: OperandConstraintKind::Immediate,
                    value: i64::from(o0),
                    optional: false,
                },
            )?;
            Ok(())
        }
    }
}

fn flatten_operands(
    operands: &[Operand],
    out: &mut [FlatOperand; 64],
) -> Result<usize, EncodeError> {
    let mut len = 0usize;
    for operand in operands.iter().copied() {
        flatten_operand(operand, out, &mut len)?;
    }
    Ok(len)
}

fn materialize_flat_for_expected_len(
    expected_len: usize,
    flat: &[FlatOperand],
    out: &mut [FlatOperand; 64],
) -> Result<usize, EncodeError> {
    // A single flattened stream may represent multiple user-visible forms when it
    // contains optional tokens. This materializer picks one concrete length to
    // compare against a specific variant signature.
    let mut required_len = 0usize;
    let mut optional_len = 0usize;
    for operand in flat {
        if operand.optional {
            optional_len += 1;
        } else {
            required_len += 1;
        }
    }

    if expected_len < required_len || expected_len > required_len + optional_len {
        return Err(EncodeError::OperandCountMismatch);
    }

    let mut include_optional = expected_len - required_len;
    let mut out_len = 0usize;
    for operand in flat {
        if !operand.optional || include_optional > 0 {
            if out_len >= out.len() {
                return Err(EncodeError::OperandCountMismatch);
            }
            out[out_len] = FlatOperand {
                kind: operand.kind,
                value: operand.value,
                optional: false,
            };
            out_len += 1;
            if operand.optional {
                include_optional = include_optional.saturating_sub(1);
            }
        }
    }

    if out_len != expected_len {
        return Err(EncodeError::OperandCountMismatch);
    }
    Ok(out_len)
}

#[inline]
fn kind_matches(expected: OperandConstraintKind, actual: OperandConstraintKind) -> bool {
    if expected == actual {
        return true;
    }

    matches!(
        (expected, actual),
        (
            OperandConstraintKind::GprRegister,
            OperandConstraintKind::Gpr32Register
        ) | (
            OperandConstraintKind::GprRegister,
            OperandConstraintKind::Gpr64Register
        ) | (
            OperandConstraintKind::SysRegPart,
            OperandConstraintKind::Immediate
        )
    )
}

#[inline]
fn kind_matches_for_slot(
    spec: &EncodingSpec,
    slot: usize,
    expected: OperandConstraintKind,
    actual: OperandConstraintKind,
) -> bool {
    if kind_matches(expected, actual) {
        return true;
    }

    if !(expected == OperandConstraintKind::Gpr64Register
        && actual == OperandConstraintKind::Gpr32Register)
    {
        return false;
    }
    if slot >= u64::BITS as usize {
        return false;
    }
    ((spec.gpr32_extend_compatibility >> slot) & 1) != 0
}

#[inline]
fn operand_shape_code(kind: OperandConstraintKind) -> u8 {
    match kind {
        OperandConstraintKind::GprRegister => 1,
        OperandConstraintKind::Gpr32Register => 2,
        OperandConstraintKind::Gpr64Register => 3,
        OperandConstraintKind::SimdRegister => 4,
        OperandConstraintKind::SveZRegister => 5,
        OperandConstraintKind::PredicateRegister => 6,
        OperandConstraintKind::Immediate => 7,
        OperandConstraintKind::Condition => 8,
        OperandConstraintKind::ShiftKind => 9,
        OperandConstraintKind::ExtendKind => 10,
        OperandConstraintKind::SysRegPart => 11,
        OperandConstraintKind::Arrangement => 12,
        OperandConstraintKind::Lane => 13,
    }
}

#[inline]
fn memory_shape_code_from_operands(operands: &[Operand]) -> Option<u8> {
    let mut code = None;
    for operand in operands {
        let Operand::Memory(memory) = operand else {
            continue;
        };
        let next = match memory.addressing {
            AddressingMode::Offset => 14,
            AddressingMode::PreIndex => 15,
            AddressingMode::PostIndex(_) => 0,
        };
        match code {
            None => code = Some(next),
            Some(existing) if existing == next => {}
            Some(_) => return None,
        }
    }
    code
}

#[inline]
fn expected_kind_name(kind: OperandConstraintKind) -> &'static str {
    match kind {
        OperandConstraintKind::GprRegister => "general-purpose register",
        OperandConstraintKind::Gpr32Register => "32-bit general-purpose register",
        OperandConstraintKind::Gpr64Register => "64-bit general-purpose register",
        OperandConstraintKind::SimdRegister => "SIMD register",
        OperandConstraintKind::SveZRegister => "SVE Z register",
        OperandConstraintKind::PredicateRegister => "predicate register",
        OperandConstraintKind::Immediate => "immediate",
        OperandConstraintKind::Condition => "condition code",
        OperandConstraintKind::ShiftKind => "shift kind",
        OperandConstraintKind::ExtendKind => "extend kind",
        OperandConstraintKind::SysRegPart => "system register field",
        OperandConstraintKind::Arrangement => "vector arrangement",
        OperandConstraintKind::Lane => "vector lane",
    }
}

fn field_name_matches(field: &BitFieldSpec, expected: &str) -> bool {
    field.name.eq_ignore_ascii_case(expected)
}

fn spec_has_field(spec: &EncodingSpec, expected: &str) -> bool {
    spec.fields
        .iter()
        .any(|field| field_name_matches(field, expected))
}

fn spec_matches_memory_addressing(spec: &EncodingSpec, operands: &[Operand]) -> bool {
    operands.iter().all(|operand| match operand {
        Operand::Memory(mem) => match (spec.memory_addressing, mem.addressing) {
            (MemoryAddressingConstraintSpec::None, _) => true,
            (MemoryAddressingConstraintSpec::NoOffset, AddressingMode::Offset) => {
                matches!(mem.offset, MemoryOffset::None)
            }
            (MemoryAddressingConstraintSpec::Offset, AddressingMode::Offset) => true,
            (MemoryAddressingConstraintSpec::PreIndex, AddressingMode::PreIndex) => true,
            (MemoryAddressingConstraintSpec::PostIndex, AddressingMode::PostIndex(_)) => true,
            (_, _) => false,
        },
        _ => true,
    })
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct SplitImmediatePlan {
    first_slot: usize,
    second_slot: usize,
    kind: SplitImmediateKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SplitImmediateKind {
    AdrLike {
        immlo_field_idx: usize,
        immhi_field_idx: usize,
        scale: i64,
    },
    BitIndex6 {
        b5_field_idx: usize,
        b40_field_idx: usize,
    },
}

#[inline]
fn spec_split_immediate_plan(spec: &EncodingSpec) -> Option<SplitImmediatePlan> {
    spec.split_immediate_plan.map(|plan| SplitImmediatePlan {
        first_slot: usize::from(plan.first_slot),
        second_slot: usize::from(plan.second_slot),
        kind: match plan.kind {
            SplitImmediateKindSpec::AdrLike {
                immlo_field_index,
                immhi_field_index,
                scale,
            } => SplitImmediateKind::AdrLike {
                immlo_field_idx: usize::from(immlo_field_index),
                immhi_field_idx: usize::from(immhi_field_index),
                scale,
            },
            SplitImmediateKindSpec::BitIndex6 {
                b5_field_index,
                b40_field_index,
            } => SplitImmediateKind::BitIndex6 {
                b5_field_idx: usize::from(b5_field_index),
                b40_field_idx: usize::from(b40_field_index),
            },
        },
    })
}

#[inline]
fn spec_maybe_split_immediate(spec: &EncodingSpec) -> bool {
    spec.split_immediate_plan.is_some()
}

fn scale_immediate(field: &'static str, value: i64, scale: i64) -> Result<i64, EncodeError> {
    if scale <= 1 {
        return Ok(value);
    }
    if value % scale != 0 {
        return Err(EncodeError::ImmediateNotAligned {
            field,
            value,
            scale,
        });
    }
    Ok(value / scale)
}

fn normalize_field_value(
    spec: &EncodingSpec,
    field: BitFieldSpec,
    field_idx: usize,
    value: i64,
) -> Result<i64, EncodeError> {
    let scale = spec.field_scales.get(field_idx).copied().unwrap_or(1);
    scale_immediate(field.name, value, i64::from(scale))
}

#[inline]
fn encode_field(field: BitFieldSpec, value: i64) -> Result<u32, EncodeError> {
    let raw = if field.signed {
        if field.width == 0 || field.width > 32 {
            return Err(EncodeError::OperandOutOfRange {
                field: field.name,
                value,
                width: field.width,
                signed: true,
            });
        }

        let min = -(1i64 << (field.width - 1));
        let max = (1i64 << (field.width - 1)) - 1;
        if value < min || value > max {
            return Err(EncodeError::OperandOutOfRange {
                field: field.name,
                value,
                width: field.width,
                signed: true,
            });
        }

        (value as i32 as u32) & field_mask(field.width)
    } else {
        if value < 0 {
            return Err(EncodeError::OperandOutOfRange {
                field: field.name,
                value,
                width: field.width,
                signed: false,
            });
        }

        let max = i64::from(field_mask(field.width));
        if value > max {
            return Err(EncodeError::OperandOutOfRange {
                field: field.name,
                value,
                width: field.width,
                signed: false,
            });
        }
        value as u32
    };

    Ok(raw << field.lsb)
}

/// Encodes one instruction variant using structured operands in asm-like order.
///
/// Field mapping and kind checks are driven by generated metadata stored in
/// [`EncodingSpec::operand_order`], [`EncodingSpec::operand_kinds`] and
/// [`EncodingSpec::implicit_defaults`].
///
/// # Errors
///
/// Returns [`EncodeError`] when operand count, kind, or ranges are invalid.
#[inline]
fn encode_flat_ordered(
    spec: &EncodingSpec,
    selected_flat: &[FlatOperand],
    selected_len: usize,
    split_plan: Option<SplitImmediatePlan>,
) -> Result<InstructionCode, EncodeError> {
    // Values are collected by field index first, then emitted in field order.
    // This keeps split-immediate and implicit-default handling centralized.
    let mut values = [0i64; 64];
    let mut filled_mask = 0u64;
    let mut input_idx = 0usize;
    let mut slot = 0usize;
    while slot < spec.operand_order.len() {
        if let Some(plan) = split_plan {
            if slot == plan.first_slot {
                if input_idx >= selected_len {
                    return Err(EncodeError::OperandCountMismatch);
                }

                let first_field_idx = spec.operand_order[slot] as usize;
                let first_field = spec.fields[first_field_idx];
                let expected = spec.operand_kinds[slot];
                let actual = selected_flat[input_idx];
                if !kind_matches_for_slot(spec, slot, expected, actual.kind) {
                    return Err(EncodeError::InvalidOperandKind {
                        field: first_field.name,
                        expected: expected_kind_name(expected),
                        got: expected_kind_name(actual.kind),
                    });
                }

                match plan.kind {
                    SplitImmediateKind::AdrLike {
                        immlo_field_idx,
                        immhi_field_idx,
                        scale,
                    } => {
                        let encoded = scale_immediate(first_field.name, actual.value, scale)?;
                        if immlo_field_idx >= values.len() || immhi_field_idx >= values.len() {
                            return Err(EncodeError::OperandCountMismatch);
                        }
                        values[immlo_field_idx] = encoded & 0b11;
                        values[immhi_field_idx] = encoded >> 2;
                        filled_mask |= 1u64 << immlo_field_idx;
                        filled_mask |= 1u64 << immhi_field_idx;
                    }
                    SplitImmediateKind::BitIndex6 {
                        b5_field_idx,
                        b40_field_idx,
                    } => {
                        if b5_field_idx >= values.len() || b40_field_idx >= values.len() {
                            return Err(EncodeError::OperandCountMismatch);
                        }
                        values[b5_field_idx] = actual.value >> 5;
                        values[b40_field_idx] = actual.value & 0x1f;
                        filled_mask |= 1u64 << b5_field_idx;
                        filled_mask |= 1u64 << b40_field_idx;
                    }
                }
                input_idx += 1;
                slot += 1;
                continue;
            }

            if slot == plan.second_slot {
                slot += 1;
                continue;
            }
        }

        if input_idx >= selected_len {
            return Err(EncodeError::OperandCountMismatch);
        }

        let field_idx = spec.operand_order[slot] as usize;
        if field_idx >= spec.fields.len() || field_idx >= values.len() {
            return Err(EncodeError::OperandCountMismatch);
        }

        let field = spec.fields[field_idx];
        let expected = spec.operand_kinds[slot];
        let actual = selected_flat[input_idx];
        if !kind_matches_for_slot(spec, slot, expected, actual.kind) {
            return Err(EncodeError::InvalidOperandKind {
                field: field.name,
                expected: expected_kind_name(expected),
                got: expected_kind_name(actual.kind),
            });
        }
        values[field_idx] = actual.value;
        filled_mask |= 1u64 << field_idx;
        input_idx += 1;
        slot += 1;
    }

    if input_idx != selected_len {
        return Err(EncodeError::OperandCountMismatch);
    }

    for implicit in spec.implicit_defaults {
        let idx = implicit.field_index as usize;
        if idx < spec.fields.len() && idx < values.len() && ((filled_mask >> idx) & 1) == 0 {
            values[idx] = implicit.value;
            filled_mask |= 1u64 << idx;
        }
    }

    let mut args = [0i64; 64];
    for idx in 0..spec.fields.len() {
        if idx >= values.len() || ((filled_mask >> idx) & 1) == 0 {
            return Err(EncodeError::OperandCountMismatch);
        }
        let raw = values[idx];
        args[idx] = normalize_field_value(spec, spec.fields[idx], idx, raw)?;
    }

    encode_by_spec(spec, &args[..spec.fields.len()])
}

fn spec_has_arrangement_lane(spec: &EncodingSpec) -> bool {
    spec.operand_kinds.iter().any(|kind| {
        matches!(
            kind,
            OperandConstraintKind::Arrangement | OperandConstraintKind::Lane
        )
    })
}

fn flat_has_arrangement_lane(flat: &[FlatOperand], len: usize) -> bool {
    flat[..len].iter().any(|operand| {
        matches!(
            operand.kind,
            OperandConstraintKind::Arrangement | OperandConstraintKind::Lane
        )
    })
}

fn reorder_flat_arrangement_lane_for_expected(
    spec: &EncodingSpec,
    flat: &[FlatOperand],
    len: usize,
    out: &mut [FlatOperand; 64],
) -> Option<usize> {
    let mut expected = [OperandConstraintKind::Immediate; 64];
    let expected_len = expected_user_operand_kinds(spec, &mut expected);
    if expected_len != len {
        return None;
    }

    let mut non_arr = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let mut arr = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let mut non_arr_len = 0usize;
    let mut arr_len = 0usize;

    for operand in &flat[..len] {
        if matches!(
            operand.kind,
            OperandConstraintKind::Arrangement | OperandConstraintKind::Lane
        ) {
            arr[arr_len] = *operand;
            arr_len += 1;
        } else {
            non_arr[non_arr_len] = *operand;
            non_arr_len += 1;
        }
    }

    let mut non_idx = 0usize;
    let mut arr_idx = 0usize;
    let mut out_len = 0usize;
    for expected_kind in expected[..expected_len].iter().copied() {
        if matches!(
            expected_kind,
            OperandConstraintKind::Arrangement | OperandConstraintKind::Lane
        ) {
            if arr_idx >= arr_len {
                return None;
            }
            if arr[arr_idx].kind != expected_kind {
                let Some(rel_idx) = arr[arr_idx..arr_len]
                    .iter()
                    .position(|operand| operand.kind == expected_kind)
                else {
                    return None;
                };
                arr.swap(arr_idx, arr_idx + rel_idx);
            }
            out[out_len] = arr[arr_idx];
            arr_idx += 1;
            out_len += 1;
        } else {
            if non_idx >= non_arr_len {
                return None;
            }
            out[out_len] = non_arr[non_idx];
            non_idx += 1;
            out_len += 1;
        }
    }

    if non_idx != non_arr_len || arr_idx != arr_len {
        return None;
    }
    Some(out_len)
}

#[inline]
fn encode_flat(
    spec: &EncodingSpec,
    selected_flat: &[FlatOperand],
    selected_len: usize,
    split_plan: Option<SplitImmediatePlan>,
) -> Result<InstructionCode, EncodeError> {
    // Fast path: direct ordered encoding. If that fails and both expected/actual
    // streams contain arrangement/lane tokens, retry with a stable reorder that
    // aligns trailing arrangement/lane decorations to the expected slots.
    let direct = encode_flat_ordered(spec, selected_flat, selected_len, split_plan);
    if direct.is_ok()
        || !spec_has_arrangement_lane(spec)
        || !flat_has_arrangement_lane(selected_flat, selected_len)
    {
        return direct;
    }

    let mut reordered = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let Some(reordered_len) = reorder_flat_arrangement_lane_for_expected(
        spec,
        selected_flat,
        selected_len,
        &mut reordered,
    ) else {
        return direct;
    };
    if reordered_len != selected_len || reordered[..selected_len] == selected_flat[..selected_len] {
        return direct;
    }

    match encode_flat_ordered(spec, &reordered, reordered_len, split_plan) {
        Ok(code) => Ok(code),
        Err(_) => direct,
    }
}

fn diagnostic_priority(err: &EncodeError) -> u8 {
    match err {
        EncodeError::OperandOutOfRange { .. } | EncodeError::ImmediateNotAligned { .. } => 0,
        EncodeError::InvalidOperandKind { .. } => 1,
        EncodeError::OperandCountMismatch | EncodeError::OperandCountRange { .. } => 2,
        EncodeError::NoMatchingVariant | EncodeError::NoMatchingVariantHint { .. } => 3,
        EncodeError::AmbiguousVariant | EncodeError::UnknownMnemonic => 4,
    }
}

fn prefer_diagnostic_error(lhs: EncodeError, rhs: EncodeError) -> EncodeError {
    if diagnostic_priority(&rhs) < diagnostic_priority(&lhs) {
        rhs
    } else {
        lhs
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CandidateScore {
    fixed_bits: u16,
    kind_specificity: u16,
    immediate_narrowness: u16,
    explicit_operands: u8,
    implicit_penalty: u8,
}

#[inline]
fn kind_specificity(kind: OperandConstraintKind) -> u16 {
    match kind {
        OperandConstraintKind::GprRegister => 3,
        OperandConstraintKind::Gpr32Register
        | OperandConstraintKind::Gpr64Register
        | OperandConstraintKind::SimdRegister
        | OperandConstraintKind::SveZRegister
        | OperandConstraintKind::PredicateRegister => 4,
        OperandConstraintKind::Immediate => 1,
        OperandConstraintKind::Condition
        | OperandConstraintKind::ShiftKind
        | OperandConstraintKind::ExtendKind
        | OperandConstraintKind::SysRegPart
        | OperandConstraintKind::Arrangement
        | OperandConstraintKind::Lane => 2,
    }
}

fn candidate_score(spec: &EncodingSpec) -> CandidateScore {
    // Tie-break candidates deterministically so "best" means:
    // - more fixed opcode bits,
    // - more specific operand kinds,
    // - narrower immediate fields,
    // - fewer implicit defaults.
    let fixed_bits = spec.opcode_mask.count_ones().min(u32::from(u16::MAX)) as u16;

    let mut kind_specificity_sum = 0u16;
    let mut immediate_narrowness = 0u16;
    for (slot, kind) in spec.operand_kinds.iter().copied().enumerate() {
        kind_specificity_sum = kind_specificity_sum.saturating_add(kind_specificity(kind));
        if kind == OperandConstraintKind::Immediate {
            let Some(field_idx) = spec.operand_order.get(slot).copied() else {
                continue;
            };
            let field_idx = field_idx as usize;
            if let Some(field) = spec.fields.get(field_idx) {
                immediate_narrowness =
                    immediate_narrowness.saturating_add((64u16).saturating_sub(field.width as u16));
            }
        }
    }

    let explicit_operands = spec
        .operand_order
        .len()
        .saturating_sub(spec.implicit_defaults.len())
        .min(usize::from(u8::MAX)) as u8;

    let implicit_penalty = (u8::MAX as usize)
        .saturating_sub(spec.implicit_defaults.len().min(usize::from(u8::MAX)))
        as u8;

    CandidateScore {
        fixed_bits,
        kind_specificity: kind_specificity_sum,
        immediate_narrowness,
        explicit_operands,
        implicit_penalty,
    }
}

#[inline]
fn should_replace_selected_candidate(
    selected_score: CandidateScore,
    selected_mask: u32,
    selected_opcode: u32,
    next_score: CandidateScore,
    next_mask: u32,
    next_opcode: u32,
) -> bool {
    match next_score.cmp(&selected_score) {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Less => false,
        // Deterministic tie-break for equivalent-shape duplicate variants.
        core::cmp::Ordering::Equal => (next_mask, next_opcode) > (selected_mask, selected_opcode),
    }
}

#[inline]
fn encode_by_spec_from_flattened(
    spec: &EncodingSpec,
    flat: &[FlatOperand],
    flat_len: usize,
) -> Result<InstructionCode, EncodeError> {
    if spec.fields.len() > 64 {
        return Err(EncodeError::OperandCountMismatch);
    }
    let split_before_materialize = if flat_len + 1 == spec.operand_order.len() {
        if spec_maybe_split_immediate(spec) {
            spec_split_immediate_plan(spec)
        } else {
            None
        }
    } else {
        None
    };
    let expected_flat_len = if split_before_materialize.is_some() {
        spec.operand_order
            .len()
            .checked_sub(1)
            .ok_or(EncodeError::OperandCountMismatch)?
    } else {
        spec.operand_order.len()
    };
    let mut selected_flat = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let selected_len = materialize_flat_for_expected_len(
        expected_flat_len,
        &flat[..flat_len],
        &mut selected_flat,
    )?;
    encode_flat(spec, &selected_flat, selected_len, split_before_materialize)
}

/// Returns [`EncodeError`] when operand count, kind, or ranges are invalid.
#[must_use]
pub fn encode_by_spec_operands(
    spec: &EncodingSpec,
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    if spec.fields.len() > 64 {
        return Err(EncodeError::OperandCountMismatch);
    }
    if spec.operand_order.len() != spec.operand_kinds.len() {
        return Err(EncodeError::OperandCountMismatch);
    }

    let mut flat = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let flat_len = flatten_operands(operands, &mut flat)?;
    encode_by_spec_from_flattened(spec, &flat, flat_len)
}

/// Encodes one mnemonic using a generated shortlist of candidate variant indices.
///
/// The shortlist is expected to contain only variants for the same mnemonic.
/// This function keeps strict validation and ambiguity handling while avoiding
/// a full linear scan over all mnemonic variants.
///
/// # Errors
///
/// Returns [`EncodeError`] when no candidate can be encoded unambiguously.
#[must_use]
pub fn encode_candidates(
    specs: &[EncodingSpec],
    candidate_indices: &[u16],
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    if candidate_indices.is_empty() {
        return Err(EncodeError::NoMatchingVariant);
    }

    let mut flat_operands = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let flat_len = flatten_operands(operands, &mut flat_operands)?;

    let mut best_detail_error: Option<EncodeError> = None;
    let mut selected: Option<(InstructionCode, CandidateScore, u32, u32)> = None;
    let mut saw_count_mismatch = false;
    let mut saw_shape_mismatch = false;

    for candidate_index in candidate_indices.iter().copied() {
        let Some(spec) = specs.get(usize::from(candidate_index)) else {
            continue;
        };
        if !spec_matches_memory_addressing(spec, operands) {
            saw_shape_mismatch = true;
            continue;
        }

        match encode_by_spec_from_flattened(spec, &flat_operands, flat_len) {
            Ok(code) => {
                let score = candidate_score(spec);
                if let Some((_, prev_score, prev_mask, prev_opcode)) = selected {
                    if should_replace_selected_candidate(
                        prev_score,
                        prev_mask,
                        prev_opcode,
                        score,
                        spec.opcode_mask,
                        spec.opcode,
                    ) {
                        selected = Some((code, score, spec.opcode_mask, spec.opcode));
                    }
                    continue;
                }
                selected = Some((code, score, spec.opcode_mask, spec.opcode));
            }
            Err(EncodeError::OperandCountMismatch) => {
                saw_count_mismatch = true;
            }
            Err(
                err @ (EncodeError::OperandOutOfRange { .. }
                | EncodeError::ImmediateNotAligned { .. }
                | EncodeError::InvalidOperandKind { .. }
                | EncodeError::NoMatchingVariant
                | EncodeError::NoMatchingVariantHint { .. }),
            ) => {
                saw_shape_mismatch = true;
                best_detail_error = Some(match best_detail_error {
                    Some(current) => prefer_diagnostic_error(current, err),
                    None => err,
                });
            }
            Err(err) => return Err(err),
        }
    }

    if let Some((code, _, _, _)) = selected {
        return Ok(code);
    }
    if let Some(detail) = best_detail_error {
        return Err(detail);
    }
    if saw_shape_mismatch || saw_count_mismatch {
        return Err(EncodeError::NoMatchingVariant);
    }
    Err(EncodeError::NoMatchingVariant)
}

/// Computes packed operand-shape keys for all materialized flattenings of `operands`.
///
/// Some operands can contribute optional flattened fields (for example bare `[base]`
/// memory offsets). This function returns one key per valid materialized shape.
///
/// # Errors
///
/// Returns [`EncodeError`] when flattening fails or `out` is too small.
#[must_use]
pub fn operand_shape_keys(
    operands: &[Operand],
    out: &mut [OperandShapeKey],
) -> Result<usize, EncodeError> {
    let memory_shape_code = memory_shape_code_from_operands(operands);
    let mut flat = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let flat_len = flatten_operands(operands, &mut flat)?;
    let mut required_len = 0usize;
    let mut optional_len = 0usize;
    for operand in &flat[..flat_len] {
        if operand.optional {
            optional_len += 1;
        } else {
            required_len += 1;
        }
    }

    let mut selected = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let mut out_len = 0usize;
    let mut kind_codes = [0u8; 64];
    for expected_len in required_len..=required_len + optional_len {
        let selected_len =
            materialize_flat_for_expected_len(expected_len, &flat[..flat_len], &mut selected)?;
        for (idx, operand) in selected[..selected_len].iter().enumerate() {
            kind_codes[idx] = operand_shape_code(operand.kind);
        }

        let mut key_len = selected_len;
        if memory_shape_code.is_some() {
            key_len = key_len.saturating_add(1);
        }
        if key_len > 30 {
            continue;
        }
        let mut key = key_len as OperandShapeKey;
        for (slot, code) in kind_codes[..selected_len].iter().copied().enumerate() {
            let shift = 8 + (slot * 4);
            key |= OperandShapeKey::from(code) << shift;
        }
        if let Some(memory_shape_code) = memory_shape_code {
            let shift = 8 + (selected_len * 4);
            key |= OperandShapeKey::from(memory_shape_code) << shift;
        }

        if out[..out_len].contains(&key) {
            continue;
        }
        if out_len >= out.len() {
            return Err(EncodeError::OperandCountMismatch);
        }
        out[out_len] = key;
        out_len += 1;
    }
    Ok(out_len)
}

/// Encodes one instruction variant by spec.
///
/// # Errors
///
/// Returns [`EncodeError`] when operand count or operand ranges are invalid.
#[must_use]
pub fn encode_by_spec(spec: &EncodingSpec, args: &[i64]) -> Result<InstructionCode, EncodeError> {
    if spec.fields.len() != args.len() {
        return Err(EncodeError::OperandCountMismatch);
    }

    let mut word = spec.opcode;
    for (field, arg) in spec.fields.iter().copied().zip(args.iter().copied()) {
        let encoded = encode_field(field, arg)?;
        word |= encoded;
    }

    Ok(InstructionCode::from_u32(word))
}

fn mnemonic_has_memory_immediate_variants(specs: &[EncodingSpec], mnemonic: &str) -> bool {
    specs.iter().any(|spec| {
        spec.mnemonic == mnemonic
            && spec.memory_addressing != MemoryAddressingConstraintSpec::None
            && (spec_has_field(spec, "imm12")
                || spec_has_field(spec, "imm9")
                || spec_has_field(spec, "imm7"))
            && spec_has_field(spec, "rn")
            && spec_has_field(spec, "rt")
    })
}

fn operands_have_bare_memory_offset(operands: &[Operand]) -> bool {
    operands.iter().any(|operand| {
        matches!(
            operand,
            Operand::Memory(MemoryOperand {
                offset: MemoryOffset::None,
                addressing: AddressingMode::Offset,
                ..
            })
        )
    })
}

fn operands_have_mixed_gpr_width(operands: &[Operand]) -> bool {
    let mut saw_w = false;
    let mut saw_x = false;
    for operand in operands {
        if let Operand::Register(reg) = operand {
            match reg.class {
                RegClass::W | RegClass::Wsp => saw_w = true,
                RegClass::X | RegClass::Xsp => saw_x = true,
                _ => {}
            }
        }
    }
    saw_w && saw_x
}

fn expected_user_operand_kinds(
    spec: &EncodingSpec,
    out: &mut [OperandConstraintKind; 64],
) -> usize {
    let split = spec_split_immediate_plan(spec);
    let mut out_len = 0usize;
    let mut slot = 0usize;
    while slot < spec.operand_kinds.len() {
        if let Some(plan) = split {
            if slot == plan.first_slot {
                out[out_len] = OperandConstraintKind::Immediate;
                out_len += 1;
                slot = slot.saturating_add(2);
                continue;
            }
        }

        out[out_len] = spec.operand_kinds[slot];
        out_len += 1;
        slot += 1;
    }
    out_len
}

fn infer_no_matching_hint(
    specs: &[EncodingSpec],
    mnemonic: &str,
    operands: &[Operand],
) -> Option<CoreNoMatchHint> {
    // Keep this heuristic-only and data-driven by operand forms/fields.
    // No opcode-name based acceptance logic is introduced here.
    if operands_have_bare_memory_offset(operands)
        && mnemonic_has_memory_immediate_variants(specs, mnemonic)
    {
        return Some(CoreNoMatchHint::MemoryMayRequireExplicitZeroOffset);
    }

    if operands_have_mixed_gpr_width(operands) {
        return Some(CoreNoMatchHint::RegisterWidthMismatch);
    }

    None
}

fn operand_shape_tag(operand: &Operand) -> OperandShapeTag {
    match operand {
        Operand::Register(reg) => match reg.class {
            RegClass::W | RegClass::Wsp => OperandShapeTag::Gpr32,
            RegClass::X | RegClass::Xsp => OperandShapeTag::Gpr64,
            RegClass::V | RegClass::B | RegClass::H | RegClass::S | RegClass::D | RegClass::Q => {
                OperandShapeTag::Simd
            }
            RegClass::Z => OperandShapeTag::SveZ,
            RegClass::P => OperandShapeTag::Predicate,
        },
        Operand::Immediate(_) => OperandShapeTag::Immediate,
        Operand::Memory(_) => OperandShapeTag::Memory,
        Operand::Shift(_) => OperandShapeTag::Shift,
        Operand::Extend(_) => OperandShapeTag::Extend,
        Operand::Condition(_) => OperandShapeTag::Condition,
        Operand::RegisterList(_) => OperandShapeTag::RegisterList,
        Operand::SysReg(_) => OperandShapeTag::SysReg,
    }
}

fn build_input_shape_signature(operands: &[Operand]) -> OperandShapeSignature {
    let mut slots = Vec::with_capacity(operands.len());
    for operand in operands {
        slots.push(operand_shape_tag(operand));
    }
    OperandShapeSignature {
        slots: slots.into_boxed_slice(),
    }
}

fn expected_signature_from_spec(spec: &EncodingSpec) -> OperandConstraintSignature {
    let mut kinds = [OperandConstraintKind::Immediate; 64];
    let len = expected_user_operand_kinds(spec, &mut kinds);
    OperandConstraintSignature {
        slots: kinds[..len].to_vec().into_boxed_slice(),
    }
}

fn build_shape_mismatch_hint(
    specs: &[EncodingSpec],
    mnemonic: &str,
    operands: &[Operand],
) -> Option<CoreNoMatchHint> {
    let mut expected = Vec::<OperandConstraintSignature>::new();
    for spec in specs {
        if spec.mnemonic != mnemonic {
            continue;
        }
        let signature = expected_signature_from_spec(spec);
        if expected.iter().any(|existing| existing == &signature) {
            continue;
        }
        expected.push(signature);
    }

    if expected.is_empty() {
        return None;
    }

    expected.sort_by(|lhs, rhs| lhs.slots.len().cmp(&rhs.slots.len()));
    let shown = expected.len().min(4);
    let expected_additional = expected
        .len()
        .saturating_sub(shown)
        .min(usize::from(u16::MAX)) as u16;

    Some(CoreNoMatchHint::ShapeMismatch {
        expected: expected
            .into_iter()
            .take(shown)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        expected_additional,
        got: build_input_shape_signature(operands),
    })
}

fn operand_hint_kind(operand: &Operand) -> Option<OperandConstraintKind> {
    match operand {
        Operand::Register(reg) => Some(reg_constraint_for_class(reg.class)),
        Operand::Immediate(_) => Some(OperandConstraintKind::Immediate),
        Operand::Condition(_) => Some(OperandConstraintKind::Condition),
        Operand::Shift(_) => Some(OperandConstraintKind::ShiftKind),
        Operand::Extend(_) => Some(OperandConstraintKind::ExtendKind),
        Operand::Memory(_) | Operand::RegisterList(_) | Operand::SysReg(_) => None,
    }
}

fn push_unique_expected_kind(
    out: &mut Vec<OperandConstraintKind>,
    expected: OperandConstraintKind,
) {
    if !out.contains(&expected) {
        out.push(expected);
    }
}

fn build_operand_delta_hint(
    specs: &[EncodingSpec],
    mnemonic: &str,
    operands: &[Operand],
) -> Option<CoreNoMatchHint> {
    // Produce the earliest actionable delta:
    // - missing operand at a given slot, or
    // - single-slot kind mismatch.
    let actual = operands
        .iter()
        .map(operand_hint_kind)
        .collect::<Vec<Option<OperandConstraintKind>>>();

    let mut missing_slot: Option<usize> = None;
    let mut missing_expected = Vec::<OperandConstraintKind>::new();
    let mut mismatch_slot: Option<usize> = None;
    let mut mismatch_expected = Vec::<OperandConstraintKind>::new();

    for spec in specs {
        if spec.mnemonic != mnemonic {
            continue;
        }

        let mut expected = [OperandConstraintKind::Immediate; 64];
        let expected_len = expected_user_operand_kinds(spec, &mut expected);

        if operands.len() + 1 == expected_len {
            let mut compatible_prefix = true;
            for idx in 0..operands.len() {
                let Some(actual_kind) = actual[idx] else {
                    compatible_prefix = false;
                    break;
                };
                if !kind_matches(expected[idx], actual_kind) {
                    compatible_prefix = false;
                    break;
                }
            }

            if compatible_prefix {
                let idx = expected_len - 1;
                if match missing_slot {
                    None => true,
                    Some(current) => idx < current,
                } {
                    missing_slot = Some(idx);
                    missing_expected.clear();
                }
                if missing_slot == Some(idx) {
                    push_unique_expected_kind(&mut missing_expected, expected[idx]);
                }
            }
        }

        if operands.len() == expected_len {
            let mut mismatch_idx: Option<usize> = None;
            let mut valid = true;
            for idx in 0..operands.len() {
                let Some(actual_kind) = actual[idx] else {
                    valid = false;
                    break;
                };
                if !kind_matches(expected[idx], actual_kind) {
                    if mismatch_idx.is_some() {
                        valid = false;
                        break;
                    }
                    mismatch_idx = Some(idx);
                }
            }

            if valid && let Some(idx) = mismatch_idx {
                if match mismatch_slot {
                    None => true,
                    Some(current) => idx < current,
                } {
                    mismatch_slot = Some(idx);
                    mismatch_expected.clear();
                }
                if mismatch_slot == Some(idx) {
                    push_unique_expected_kind(&mut mismatch_expected, expected[idx]);
                }
            }
        }
    }

    if let Some(idx) = missing_slot
        && !missing_expected.is_empty()
    {
        return Some(CoreNoMatchHint::OperandMissing {
            index: (idx + 1).min(usize::from(u8::MAX)) as u8,
            expected: missing_expected.into_boxed_slice(),
        });
    }

    if let Some(idx) = mismatch_slot
        && !mismatch_expected.is_empty()
        && idx < operands.len()
    {
        return Some(CoreNoMatchHint::OperandKindMismatch {
            index: (idx + 1).min(usize::from(u8::MAX)) as u8,
            expected: mismatch_expected.into_boxed_slice(),
            got: operand_shape_tag(&operands[idx]),
        });
    }

    None
}

fn spec_expected_user_operand_count(spec: &EncodingSpec) -> usize {
    if spec_split_immediate_plan(spec).is_some() {
        spec.operand_order.len().saturating_sub(1)
    } else {
        spec.operand_order.len()
    }
}

/// Selects a variant by mnemonic and typed operands and encodes it.
///
/// # Errors
///
/// Returns [`EncodeError`] if no variant can be selected and encoded unambiguously.
#[must_use]
pub fn encode(
    specs: &[EncodingSpec],
    mnemonic: &str,
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    let mut saw_mnemonic = false;
    let mut saw_count_mismatch = false;
    let mut saw_shape_mismatch = false;
    let mut best_detail_error: Option<EncodeError> = None;
    let mut selected: Option<(InstructionCode, CandidateScore, u32, u32)> = None;
    let mut flat_operands = [FlatOperand {
        kind: OperandConstraintKind::Immediate,
        value: 0,
        optional: false,
    }; 64];
    let mut flat_len = 0usize;
    let mut flat_attempted = false;
    let mut flat_error: Option<EncodeError> = None;
    let mut expected_min_count = usize::MAX;
    let mut expected_max_count = 0usize;

    for spec in specs {
        if spec.mnemonic != mnemonic {
            continue;
        }
        saw_mnemonic = true;
        let expected_count = spec_expected_user_operand_count(spec);
        if expected_count < expected_min_count {
            expected_min_count = expected_count;
        }
        if expected_count > expected_max_count {
            expected_max_count = expected_count;
        }

        if !spec_matches_memory_addressing(spec, operands) {
            saw_shape_mismatch = true;
            continue;
        }

        if !flat_attempted {
            flat_attempted = true;
            match flatten_operands(operands, &mut flat_operands) {
                Ok(len) => flat_len = len,
                Err(err) => flat_error = Some(err),
            }
        }

        let encode_result = if let Some(err) = flat_error.clone() {
            Err(err)
        } else {
            encode_by_spec_from_flattened(spec, &flat_operands, flat_len)
        };

        match encode_result {
            Ok(code) => {
                let score = candidate_score(spec);
                if let Some((_, prev_score, prev_mask, prev_opcode)) = selected {
                    if should_replace_selected_candidate(
                        prev_score,
                        prev_mask,
                        prev_opcode,
                        score,
                        spec.opcode_mask,
                        spec.opcode,
                    ) {
                        selected = Some((code, score, spec.opcode_mask, spec.opcode));
                    }
                    continue;
                }
                selected = Some((code, score, spec.opcode_mask, spec.opcode));
            }
            Err(EncodeError::OperandCountMismatch) => {
                saw_count_mismatch = true;
            }
            Err(
                err @ (EncodeError::OperandOutOfRange { .. }
                | EncodeError::ImmediateNotAligned { .. }
                | EncodeError::InvalidOperandKind { .. }
                | EncodeError::NoMatchingVariant
                | EncodeError::NoMatchingVariantHint { .. }),
            ) => {
                saw_shape_mismatch = true;
                best_detail_error = Some(match best_detail_error {
                    Some(current) => prefer_diagnostic_error(current, err),
                    None => err,
                });
            }
            Err(err) => return Err(err),
        }
    }

    if let Some((code, _, _, _)) = selected {
        return Ok(code);
    }
    if !saw_mnemonic {
        return Err(EncodeError::UnknownMnemonic);
    }
    if let Some(detail) = best_detail_error {
        return Err(detail);
    }
    if saw_shape_mismatch {
        if let Some(hint) = infer_no_matching_hint(specs, mnemonic, operands) {
            return Err(EncodeError::NoMatchingVariantHint {
                hint: NoMatchingHint::Core(hint),
            });
        }
    }
    if saw_shape_mismatch {
        if let Some(hint) = build_operand_delta_hint(specs, mnemonic, operands) {
            return Err(EncodeError::NoMatchingVariantHint {
                hint: NoMatchingHint::Core(hint),
            });
        }
        if let Some(hint) = build_shape_mismatch_hint(specs, mnemonic, operands) {
            return Err(EncodeError::NoMatchingVariantHint {
                hint: NoMatchingHint::Core(hint),
            });
        }
        return Err(EncodeError::NoMatchingVariant);
    }
    if saw_count_mismatch {
        if expected_min_count != usize::MAX {
            let min = expected_min_count.min(usize::from(u8::MAX)) as u8;
            let max = expected_max_count.min(usize::from(u8::MAX)) as u8;
            let got = operands.len().min(usize::from(u8::MAX)) as u8;
            return Err(EncodeError::OperandCountRange { min, max, got });
        }
        return Err(EncodeError::OperandCountMismatch);
    }

    Err(EncodeError::NoMatchingVariant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const ADD_FIELDS: &[BitFieldSpec] = &[
        BitFieldSpec {
            name: "sh",
            lsb: 22,
            width: 1,
            signed: false,
        },
        BitFieldSpec {
            name: "imm12",
            lsb: 10,
            width: 12,
            signed: false,
        },
        BitFieldSpec {
            name: "Rn",
            lsb: 5,
            width: 5,
            signed: false,
        },
        BitFieldSpec {
            name: "Rd",
            lsb: 0,
            width: 5,
            signed: false,
        },
    ];

    const ADD_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "add",
        variant: "ADD_64_addsub_imm",
        opcode: 0b100100010u32 << 23,
        opcode_mask: 0b111111111u32 << 23,
        fields: ADD_FIELDS,
        operand_order: &[3, 2, 1],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[ImplicitField {
            field_index: 0,
            value: 0,
        }],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const SEL_P_FIELDS: &[BitFieldSpec] = &[
        BitFieldSpec {
            name: "Pm",
            lsb: 16,
            width: 4,
            signed: false,
        },
        BitFieldSpec {
            name: "Pg",
            lsb: 10,
            width: 4,
            signed: false,
        },
        BitFieldSpec {
            name: "Pn",
            lsb: 5,
            width: 4,
            signed: false,
        },
        BitFieldSpec {
            name: "Pd",
            lsb: 0,
            width: 4,
            signed: false,
        },
    ];

    const SEL_P_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "sel",
        variant: "sel_p_p_pp_",
        opcode: 0x2500_4210,
        opcode_mask: 0xfff0_c210,
        fields: SEL_P_FIELDS,
        operand_order: &[3, 1, 2, 0],
        operand_kinds: &[
            OperandConstraintKind::PredicateRegister,
            OperandConstraintKind::PredicateRegister,
            OperandConstraintKind::PredicateRegister,
            OperandConstraintKind::PredicateRegister,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const STR_64_POS_FIELDS: &[BitFieldSpec] = &[
        BitFieldSpec {
            name: "imm12",
            lsb: 10,
            width: 12,
            signed: false,
        },
        BitFieldSpec {
            name: "Rn",
            lsb: 5,
            width: 5,
            signed: false,
        },
        BitFieldSpec {
            name: "Rt",
            lsb: 0,
            width: 5,
            signed: false,
        },
    ];

    const STR_64_POS_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "str",
        variant: "STR_64_ldst_pos",
        opcode: 0xf900_0000,
        opcode_mask: 0xffc0_0000,
        fields: STR_64_POS_FIELDS,
        operand_order: &[2, 1, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::Offset,
        field_scales: &[8, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const STLR_NO_OFFSET_FIELDS: &[BitFieldSpec] = &[
        BitFieldSpec {
            name: "Rn",
            lsb: 5,
            width: 5,
            signed: false,
        },
        BitFieldSpec {
            name: "Rt",
            lsb: 0,
            width: 5,
            signed: false,
        },
    ];

    const STLR_NO_OFFSET_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "stlr",
        variant: "STLR_SL64_ldstord",
        opcode: 0xc89f_fc00,
        opcode_mask: 0xffff_fc00,
        fields: STLR_NO_OFFSET_FIELDS,
        operand_order: &[1, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::NoOffset,
        field_scales: &[1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const STP_64_PRE_FIELDS: &[BitFieldSpec] = &[
        BitFieldSpec {
            name: "imm7",
            lsb: 15,
            width: 7,
            signed: true,
        },
        BitFieldSpec {
            name: "Rt2",
            lsb: 10,
            width: 5,
            signed: false,
        },
        BitFieldSpec {
            name: "Rn",
            lsb: 5,
            width: 5,
            signed: false,
        },
        BitFieldSpec {
            name: "Rt",
            lsb: 0,
            width: 5,
            signed: false,
        },
    ];

    const STP_64_PRE_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "stp",
        variant: "STP_64_ldstpair_pre",
        opcode: 0xa980_0000,
        opcode_mask: 0xffc0_0000,
        fields: STP_64_PRE_FIELDS,
        operand_order: &[3, 1, 2, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::PreIndex,
        field_scales: &[8, 1, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const B_IMM_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "b",
        variant: "B_only_branch_imm",
        opcode: 0x1400_0000,
        opcode_mask: 0xfc00_0000,
        fields: &[BitFieldSpec {
            name: "imm26",
            lsb: 0,
            width: 26,
            signed: true,
        }],
        operand_order: &[0],
        operand_kinds: &[OperandConstraintKind::Immediate],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[4],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const ADR_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "adr",
        variant: "ADR_only_pcreladdr",
        opcode: 0x1000_0000,
        opcode_mask: 0x9f00_0000,
        fields: &[
            BitFieldSpec {
                name: "immlo",
                lsb: 29,
                width: 2,
                signed: false,
            },
            BitFieldSpec {
                name: "immhi",
                lsb: 5,
                width: 19,
                signed: true,
            },
            BitFieldSpec {
                name: "Rd",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[2, 0, 1],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1],
        split_immediate_plan: Some(SplitImmediatePlanSpec {
            first_slot: 1,
            second_slot: 2,
            kind: SplitImmediateKindSpec::AdrLike {
                immlo_field_index: 0,
                immhi_field_index: 1,
                scale: 1,
            },
        }),
        gpr32_extend_compatibility: 0,
    };

    const ADRP_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "adrp",
        variant: "ADRP_only_pcreladdr",
        opcode: 0x9000_0000,
        opcode_mask: 0x9f00_0000,
        fields: &[
            BitFieldSpec {
                name: "immlo",
                lsb: 29,
                width: 2,
                signed: false,
            },
            BitFieldSpec {
                name: "immhi",
                lsb: 5,
                width: 19,
                signed: true,
            },
            BitFieldSpec {
                name: "Rd",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[2, 0, 1],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1],
        split_immediate_plan: Some(SplitImmediatePlanSpec {
            first_slot: 1,
            second_slot: 2,
            kind: SplitImmediateKindSpec::AdrLike {
                immlo_field_index: 0,
                immhi_field_index: 1,
                scale: 4096,
            },
        }),
        gpr32_extend_compatibility: 0,
    };

    const ADD_EXT_64_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "add",
        variant: "ADD_64_addsub_ext",
        opcode: 0x8b20_0000,
        opcode_mask: 0xffe0_0000,
        fields: &[
            BitFieldSpec {
                name: "Rm",
                lsb: 16,
                width: 5,
                signed: false,
            },
            BitFieldSpec {
                name: "option",
                lsb: 13,
                width: 3,
                signed: false,
            },
            BitFieldSpec {
                name: "imm3",
                lsb: 10,
                width: 3,
                signed: false,
            },
            BitFieldSpec {
                name: "Rn",
                lsb: 5,
                width: 5,
                signed: false,
            },
            BitFieldSpec {
                name: "Rd",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[4, 3, 0, 1, 2],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::ExtendKind,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0b100,
    };

    const LDR_64_UNSIGNED_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "ldr",
        variant: "LDR_64_ldst_pos",
        opcode: 0xf940_0000,
        opcode_mask: 0xffc0_0000,
        fields: &[
            BitFieldSpec {
                name: "imm12",
                lsb: 10,
                width: 12,
                signed: false,
            },
            BitFieldSpec {
                name: "Rn",
                lsb: 5,
                width: 5,
                signed: false,
            },
            BitFieldSpec {
                name: "Rt",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[2, 1, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::Offset,
        field_scales: &[8, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const LDAR_64_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "ldar",
        variant: "LDAR_LR64_ldstord",
        opcode: 0xc8df_fc00,
        opcode_mask: 0xffff_fc00,
        fields: &[
            BitFieldSpec {
                name: "Rn",
                lsb: 5,
                width: 5,
                signed: false,
            },
            BitFieldSpec {
                name: "Rt",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[1, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Gpr64Register,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    const TBNZ_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "tbnz",
        variant: "TBNZ_only_testbranch",
        opcode: 0x3700_0000,
        opcode_mask: 0x7f00_0000,
        fields: &[
            BitFieldSpec {
                name: "b5",
                lsb: 31,
                width: 1,
                signed: false,
            },
            BitFieldSpec {
                name: "b40",
                lsb: 19,
                width: 5,
                signed: false,
            },
            BitFieldSpec {
                name: "imm14",
                lsb: 5,
                width: 14,
                signed: true,
            },
            BitFieldSpec {
                name: "Rt",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[3, 0, 1, 2],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::Immediate,
            OperandConstraintKind::Immediate,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 4, 1],
        split_immediate_plan: Some(SplitImmediatePlanSpec {
            first_slot: 1,
            second_slot: 2,
            kind: SplitImmediateKindSpec::BitIndex6 {
                b5_field_index: 0,
                b40_field_index: 1,
            },
        }),
        gpr32_extend_compatibility: 0,
    };

    const MRS_SPEC: EncodingSpec = EncodingSpec {
        mnemonic: "mrs",
        variant: "MRS_RS_systemmove",
        opcode: 0xd530_0000,
        opcode_mask: 0xfff0_0000,
        fields: &[
            BitFieldSpec {
                name: "o0",
                lsb: 19,
                width: 1,
                signed: false,
            },
            BitFieldSpec {
                name: "op1",
                lsb: 16,
                width: 3,
                signed: false,
            },
            BitFieldSpec {
                name: "CRn",
                lsb: 12,
                width: 4,
                signed: false,
            },
            BitFieldSpec {
                name: "CRm",
                lsb: 8,
                width: 4,
                signed: false,
            },
            BitFieldSpec {
                name: "op2",
                lsb: 5,
                width: 3,
                signed: false,
            },
            BitFieldSpec {
                name: "Rt",
                lsb: 0,
                width: 5,
                signed: false,
            },
        ],
        operand_order: &[5, 1, 2, 3, 4, 0],
        operand_kinds: &[
            OperandConstraintKind::Gpr64Register,
            OperandConstraintKind::SysRegPart,
            OperandConstraintKind::SysRegPart,
            OperandConstraintKind::SysRegPart,
            OperandConstraintKind::SysRegPart,
            OperandConstraintKind::Immediate,
        ],
        implicit_defaults: &[],
        memory_addressing: MemoryAddressingConstraintSpec::None,
        field_scales: &[1, 1, 1, 1, 1, 1],
        split_immediate_plan: None,
        gpr32_extend_compatibility: 0,
    };

    #[test]
    fn encode_add_imm() {
        let code = encode_by_spec(&ADD_SPEC, &[0, 1, 2, 1]).expect("encode should succeed");
        assert_eq!(code.unpack(), 0x91000441);
    }

    #[test]
    fn rejects_imm_overflow() {
        let err =
            encode_by_spec(&ADD_SPEC, &[0, 0x2000, 2, 1]).expect_err("should reject overflow");
        assert_eq!(
            err,
            EncodeError::OperandOutOfRange {
                field: "imm12",
                value: 0x2000,
                width: 12,
                signed: false,
            }
        );
    }

    #[test]
    fn unknown_mnemonic() {
        let err = encode(
            &[ADD_SPEC],
            "sub",
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(1),
            ],
        )
        .expect_err("must fail");
        assert_eq!(err, EncodeError::UnknownMnemonic);
    }

    #[test]
    fn encode_add_imm_asm_operands() {
        let code = encode_by_spec_operands(
            &ADD_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(1),
            ],
        )
        .expect("encode should succeed");
        assert_eq!(code.unpack(), 0x91000441);
    }

    #[test]
    fn invalid_kind_rejected() {
        let err = encode_by_spec_operands(
            &ADD_SPEC,
            &[
                Operand::Immediate(1),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(1),
            ],
        )
        .expect_err("must fail");
        assert_eq!(
            err,
            EncodeError::InvalidOperandKind {
                field: "Rd",
                expected: "64-bit general-purpose register",
                got: "immediate",
            }
        );
    }

    #[test]
    fn encode_predicate_operands() {
        let code = encode_by_spec_operands(
            &SEL_P_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 3,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 4,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
            ],
        )
        .expect("predicate encoding should succeed");
        assert_eq!(
            code.unpack(),
            0x2500_4210u32 | (4u32 << 16) | (2u32 << 10) | (3u32 << 5) | 1u32
        );
    }

    #[test]
    fn reject_wrong_register_family() {
        let err = encode_by_spec_operands(
            &SEL_P_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 3,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 4,
                    class: RegClass::P,
                    arrangement: None,
                    lane: None,
                }),
            ],
        )
        .expect_err("mismatched family should fail");
        assert_eq!(
            err,
            EncodeError::InvalidOperandKind {
                field: "Pd",
                expected: "predicate register",
                got: "64-bit general-purpose register",
            }
        );
    }

    #[test]
    fn memory_imm12_is_scaled_from_bytes() {
        let code = encode_by_spec_operands(
            &STR_64_POS_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 30,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 31,
                        class: RegClass::Xsp,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::Immediate(16),
                    addressing: AddressingMode::Offset,
                }),
            ],
        )
        .expect("str should encode");
        assert_eq!(code.unpack(), 0xf900_0bfe);
    }

    #[test]
    fn memory_imm7_pair_is_scaled_from_bytes() {
        let code = encode_by_spec_operands(
            &STP_64_PRE_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 27,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 28,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 31,
                        class: RegClass::Xsp,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::Immediate(-32),
                    addressing: AddressingMode::PreIndex,
                }),
            ],
        )
        .expect("stp should encode");
        assert_eq!(code.unpack(), 0xa9be_73fb);
    }

    #[test]
    fn misaligned_memory_immediate_is_rejected() {
        let err = encode_by_spec_operands(
            &STR_64_POS_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 30,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 31,
                        class: RegClass::Xsp,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::Immediate(10),
                    addressing: AddressingMode::Offset,
                }),
            ],
        )
        .expect_err("misalignment must fail");
        assert_eq!(
            err,
            EncodeError::ImmediateNotAligned {
                field: "imm12",
                value: 10,
                scale: 8,
            }
        );
    }

    #[test]
    fn branch_immediate_is_scaled_from_bytes() {
        let code =
            encode_by_spec_operands(&B_IMM_SPEC, &[Operand::Immediate(8)]).expect("b encode");
        assert_eq!(code.unpack(), 0x1400_0002);
    }

    #[test]
    fn branch_immediate_alignment_is_enforced() {
        let err = encode_by_spec_operands(&B_IMM_SPEC, &[Operand::Immediate(6)])
            .expect_err("misaligned branch immediate must fail");
        assert_eq!(
            err,
            EncodeError::ImmediateNotAligned {
                field: "imm26",
                value: 6,
                scale: 4,
            }
        );
    }

    #[test]
    fn adr_accepts_single_pc_relative_immediate() {
        let code = encode_by_spec_operands(
            &ADR_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(4),
            ],
        )
        .expect("adr must encode with single immediate");
        assert_eq!(code.unpack(), 0x1000_0020);
    }

    #[test]
    fn adrp_immediate_uses_page_scale() {
        let code = encode_by_spec_operands(
            &ADRP_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(4096),
            ],
        )
        .expect("adrp must encode with page immediate");
        assert_eq!(code.unpack(), 0xb000_0000);
    }

    #[test]
    fn adrp_immediate_alignment_is_enforced() {
        let err = encode_by_spec_operands(
            &ADRP_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(2048),
            ],
        )
        .expect_err("misaligned adrp immediate must fail");
        assert_eq!(
            err,
            EncodeError::ImmediateNotAligned {
                field: "immlo",
                value: 2048,
                scale: 4096,
            }
        );
    }

    #[test]
    fn add_ext_64_accepts_w_register_source() {
        let w_source = encode_by_spec_operands(
            &ADD_EXT_64_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 3,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 4,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 5,
                    class: RegClass::W,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Extend(ExtendOperand {
                    kind: ExtendKind::Uxtw,
                    amount: Some(2),
                }),
            ],
        )
        .expect("64-bit add ext form should accept Wm");

        let x_source = encode_by_spec_operands(
            &ADD_EXT_64_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 3,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 4,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 5,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Extend(ExtendOperand {
                    kind: ExtendKind::Uxtw,
                    amount: Some(2),
                }),
            ],
        )
        .expect("64-bit add ext form should accept Xm as encoded source register");

        assert_eq!(w_source.unpack(), x_source.unpack());
    }

    #[test]
    fn tbnz_accepts_single_bit_index_operand() {
        let code = encode_by_spec_operands(
            &TBNZ_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(3),
                Operand::Immediate(8),
            ],
        )
        .expect("tbnz with split bit-index should encode");

        let expected = encode_by_spec(&TBNZ_SPEC, &[0, 3, 2, 1]).expect("expected split encoding");
        assert_eq!(code.unpack(), expected.unpack());
    }

    #[test]
    fn bare_memory_operand_maps_to_zero_or_no_offset_forms() {
        let ldr = encode_by_spec_operands(
            &LDR_64_UNSIGNED_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 6,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 7,
                        class: RegClass::X,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::None,
                    addressing: AddressingMode::Offset,
                }),
            ],
        )
        .expect("ldr [xn] should map to imm12=#0 form");
        let expected_ldr = encode_by_spec(&LDR_64_UNSIGNED_SPEC, &[0, 7, 6]).expect("ldr #0");
        assert_eq!(ldr.unpack(), expected_ldr.unpack());

        let ldar = encode_by_spec_operands(
            &LDAR_64_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 8,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 9,
                        class: RegClass::X,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::None,
                    addressing: AddressingMode::Offset,
                }),
            ],
        )
        .expect("ldar [xn] should match no-offset form");
        let expected_ldar = encode_by_spec(&LDAR_64_SPEC, &[9, 8]).expect("ldar");
        assert_eq!(ldar.unpack(), expected_ldar.unpack());
    }

    #[test]
    fn sysreg_operand_converts_arch_op0_to_encoded_o0() {
        let code = encode_by_spec_operands(
            &MRS_SPEC,
            &[
                Operand::Register(RegisterOperand {
                    code: 8,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::SysReg(SysRegOperand {
                    op0: 3,
                    op1: 0,
                    crn: 15,
                    crm: 2,
                    op2: 0,
                }),
            ],
        )
        .expect("mrs sysreg form should encode");

        let expected = encode_by_spec(&MRS_SPEC, &[1, 0, 15, 2, 0, 8]).expect("mrs expected");
        assert_eq!(code.unpack(), expected.unpack());
    }

    #[test]
    fn operand_count_range_reports_expected_bounds() {
        let err = encode(
            &[ADD_SPEC],
            "add",
            &[Operand::Register(RegisterOperand {
                code: 1,
                class: RegClass::X,
                arrangement: None,
                lane: None,
            })],
        )
        .expect_err("must reject operand count");
        assert_eq!(
            err,
            EncodeError::OperandCountRange {
                min: 3,
                max: 3,
                got: 1,
            }
        );
    }

    #[test]
    fn operand_count_range_uses_user_visible_split_immediate_count() {
        let err = encode(
            &[TBNZ_SPEC],
            "tbnz",
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(3),
            ],
        )
        .expect_err("must reject missing branch immediate");
        assert_eq!(
            err,
            EncodeError::OperandCountRange {
                min: 3,
                max: 3,
                got: 2,
            }
        );
    }

    #[test]
    fn no_matching_variant_hint_includes_expected_shapes() {
        let err = encode(
            &[STR_64_POS_SPEC],
            "str",
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 1,
                        class: RegClass::X,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::Immediate(16),
                    addressing: AddressingMode::PreIndex,
                }),
            ],
        )
        .expect_err("shape mismatch should return hint");

        match err {
            EncodeError::NoMatchingVariantHint {
                hint:
                    NoMatchingHint::Core(CoreNoMatchHint::ShapeMismatch {
                        expected,
                        expected_additional,
                        got,
                    }),
            } => {
                assert_eq!(expected_additional, 0);
                assert_eq!(expected.len(), 1);
                assert_eq!(
                    expected[0].slots.as_ref(),
                    &[
                        OperandConstraintKind::Gpr64Register,
                        OperandConstraintKind::Gpr64Register,
                        OperandConstraintKind::Immediate,
                    ]
                );
                assert_eq!(
                    got.slots.as_ref(),
                    &[OperandShapeTag::Gpr64, OperandShapeTag::Memory]
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn no_offset_memory_constraint_rejects_explicit_zero_offset_without_count_range() {
        let err = encode(
            &[STLR_NO_OFFSET_SPEC],
            "stlr",
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Memory(MemoryOperand {
                    base: RegisterOperand {
                        code: 1,
                        class: RegClass::X,
                        arrangement: None,
                        lane: None,
                    },
                    offset: MemoryOffset::Immediate(0),
                    addressing: AddressingMode::Offset,
                }),
            ],
        )
        .expect_err("stlr no-offset form must reject explicit #0");

        match err {
            EncodeError::NoMatchingVariant | EncodeError::NoMatchingVariantHint { .. } => {}
            EncodeError::OperandCountRange { .. } => {
                panic!("explicit #0 on no-offset memory form must not map to operand count range")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn operand_delta_hint_reports_missing_trailing_operand() {
        let hint = build_operand_delta_hint(
            &[ADD_SPEC],
            "add",
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
            ],
        )
        .expect("missing operand hint should be available");
        assert_eq!(
            hint,
            CoreNoMatchHint::OperandMissing {
                index: 3,
                expected: vec![OperandConstraintKind::Immediate].into_boxed_slice(),
            }
        );
    }

    #[test]
    fn operand_delta_hint_reports_single_slot_kind_mismatch() {
        let hint = build_operand_delta_hint(
            &[ADD_SPEC],
            "add",
            &[
                Operand::Register(RegisterOperand {
                    code: 0,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(7),
                Operand::Immediate(9),
            ],
        )
        .expect("kind mismatch hint should be available");
        assert_eq!(
            hint,
            CoreNoMatchHint::OperandKindMismatch {
                index: 2,
                expected: vec![OperandConstraintKind::Gpr64Register].into_boxed_slice(),
                got: OperandShapeTag::Immediate,
            }
        );
    }

    #[test]
    fn mnemonic_selection_prefers_canonical_variant_for_conflicting_fixed_bits() {
        const WITH_DEFAULT: EncodingSpec = EncodingSpec {
            mnemonic: "pick",
            variant: "PICK_64_defaulted",
            opcode: 0x1000_0000,
            opcode_mask: 0xff00_0000,
            fields: &[
                BitFieldSpec {
                    name: "sh",
                    lsb: 22,
                    width: 1,
                    signed: false,
                },
                BitFieldSpec {
                    name: "imm12",
                    lsb: 10,
                    width: 12,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rn",
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rd",
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
            operand_order: &[3, 2, 1],
            operand_kinds: &[
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Immediate,
            ],
            implicit_defaults: &[ImplicitField {
                field_index: 0,
                value: 0,
            }],
            memory_addressing: MemoryAddressingConstraintSpec::None,
            field_scales: &[1, 1, 1, 1],
            split_immediate_plan: None,
            gpr32_extend_compatibility: 0,
        };

        const NO_DEFAULT: EncodingSpec = EncodingSpec {
            mnemonic: "pick",
            variant: "PICK_64_direct",
            opcode: 0x2000_0000,
            opcode_mask: 0xff00_0000,
            fields: &[
                BitFieldSpec {
                    name: "imm12",
                    lsb: 10,
                    width: 12,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rn",
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rd",
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
            operand_order: &[2, 1, 0],
            operand_kinds: &[
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Immediate,
            ],
            implicit_defaults: &[],
            memory_addressing: MemoryAddressingConstraintSpec::None,
            field_scales: &[1, 1, 1],
            split_immediate_plan: None,
            gpr32_extend_compatibility: 0,
        };

        let code = encode(
            &[WITH_DEFAULT, NO_DEFAULT],
            "pick",
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(7),
            ],
        )
        .expect("conflicting fixed bits should resolve to one canonical variant");
        let expected = encode_by_spec_operands(
            &NO_DEFAULT,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Immediate(7),
            ],
        )
        .expect("no-default form should encode");
        assert_eq!(code.unpack(), expected.unpack());
    }

    #[test]
    fn mnemonic_selection_prefers_specific_score_when_fixed_bits_are_compatible() {
        const PICK_NARROW: EncodingSpec = EncodingSpec {
            mnemonic: "pickcompat",
            variant: "PICKCOMPAT_narrow",
            opcode: 0x1000_0000,
            opcode_mask: 0xf000_0000,
            fields: &[
                BitFieldSpec {
                    name: "imm1",
                    lsb: 0,
                    width: 1,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rn",
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rd",
                    lsb: 10,
                    width: 5,
                    signed: false,
                },
            ],
            operand_order: &[2, 1, 0],
            operand_kinds: &[
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Immediate,
            ],
            implicit_defaults: &[],
            memory_addressing: MemoryAddressingConstraintSpec::None,
            field_scales: &[1, 1, 1],
            split_immediate_plan: None,
            gpr32_extend_compatibility: 0,
        };

        const PICK_WIDE_DEFAULTED: EncodingSpec = EncodingSpec {
            mnemonic: "pickcompat",
            variant: "PICKCOMPAT_wide_defaulted",
            opcode: 0x1000_0000,
            opcode_mask: 0xf000_0000,
            fields: &[
                BitFieldSpec {
                    name: "sh",
                    lsb: 22,
                    width: 1,
                    signed: false,
                },
                BitFieldSpec {
                    name: "imm12",
                    lsb: 10,
                    width: 12,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rn",
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rd",
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
            operand_order: &[3, 2, 1],
            operand_kinds: &[
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Immediate,
            ],
            implicit_defaults: &[ImplicitField {
                field_index: 0,
                value: 0,
            }],
            memory_addressing: MemoryAddressingConstraintSpec::None,
            field_scales: &[1, 1, 1, 1],
            split_immediate_plan: None,
            gpr32_extend_compatibility: 0,
        };

        let operands = [
            Operand::Register(RegisterOperand {
                code: 1,
                class: RegClass::X,
                arrangement: None,
                lane: None,
            }),
            Operand::Register(RegisterOperand {
                code: 2,
                class: RegClass::X,
                arrangement: None,
                lane: None,
            }),
            Operand::Immediate(1),
        ];

        let selected = encode(&[PICK_WIDE_DEFAULTED, PICK_NARROW], "pickcompat", &operands)
            .expect("compatible fixed bits should allow score-based selection");
        let expected = encode_by_spec_operands(&PICK_NARROW, &operands)
            .expect("narrow immediate form should encode");
        assert_eq!(selected.unpack(), expected.unpack());
    }

    #[test]
    fn mnemonic_selection_uses_stable_order_on_score_tie() {
        const PICK_A: EncodingSpec = EncodingSpec {
            mnemonic: "picktie",
            variant: "PICKTIE_64_a",
            opcode: 0x4000_0000,
            opcode_mask: 0xf000_0000,
            fields: &[
                BitFieldSpec {
                    name: "Rn",
                    lsb: 5,
                    width: 5,
                    signed: false,
                },
                BitFieldSpec {
                    name: "Rd",
                    lsb: 0,
                    width: 5,
                    signed: false,
                },
            ],
            operand_order: &[1, 0],
            operand_kinds: &[
                OperandConstraintKind::Gpr64Register,
                OperandConstraintKind::Gpr64Register,
            ],
            implicit_defaults: &[],
            memory_addressing: MemoryAddressingConstraintSpec::None,
            field_scales: &[1, 1],
            split_immediate_plan: None,
            gpr32_extend_compatibility: 0,
        };

        const PICK_B: EncodingSpec = EncodingSpec {
            mnemonic: "picktie",
            variant: "PICKTIE_64_b",
            opcode: 0x4000_0000,
            opcode_mask: 0xf000_0000,
            fields: PICK_A.fields,
            operand_order: &[0, 1],
            operand_kinds: PICK_A.operand_kinds,
            implicit_defaults: PICK_A.implicit_defaults,
            memory_addressing: PICK_A.memory_addressing,
            field_scales: PICK_A.field_scales,
            split_immediate_plan: PICK_A.split_immediate_plan,
            gpr32_extend_compatibility: PICK_A.gpr32_extend_compatibility,
        };

        let code = encode(
            &[PICK_A, PICK_B],
            "picktie",
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
            ],
        )
        .expect("equal-score candidates should resolve deterministically");
        let expected = encode_by_spec_operands(
            &PICK_A,
            &[
                Operand::Register(RegisterOperand {
                    code: 1,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
                Operand::Register(RegisterOperand {
                    code: 2,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                }),
            ],
        )
        .expect("first tie candidate should encode");
        assert_eq!(code.unpack(), expected.unpack());
    }
}
