use crate::generated::MnemonicId;
use jit_core::{
    AddressingMode, AliasNoMatchHint, ConditionCode, EncodeError, MemoryOffset, NoMatchingHint,
    Operand, RegClass, RegisterOperand, ShiftKind,
};

pub(crate) const ALIAS_OPERAND_CAP: usize = 16;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum AliasTransform {
    PureRename,
    RetDefault,
    CmpLike,
    CmnLike,
    TstLike,
    MovLike,
    MvnLike,
    CincLike,
    CsetLike,
    CnegLike,
    BitfieldBfi,
    BitfieldBfxil,
    BitfieldBfc,
    BitfieldUbfx,
    BitfieldSbfx,
    BitfieldSbfiz,
    ExtendLongZero,
    StsetlLike,
    DcLike,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct AliasRule {
    alias: &'static str,
    canonical: &'static str,
    canonical_id: u16,
    transform: AliasTransform,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ConditionalBranchAliasRule {
    alias: &'static str,
    base_mnemonic: &'static str,
    base_mnemonic_id: u16,
    condition_code: u8,
}

include!("generated_alias_rules.rs");

#[inline]
fn is_gpr_class(class: RegClass) -> bool {
    matches!(
        class,
        RegClass::W | RegClass::X | RegClass::Wsp | RegClass::Xsp
    )
}

#[inline]
fn zero_register_for_class(class: RegClass) -> Option<RegisterOperand> {
    let class = match class {
        RegClass::W | RegClass::Wsp => RegClass::W,
        RegClass::X | RegClass::Xsp => RegClass::X,
        _ => return None,
    };

    Some(RegisterOperand {
        code: 31,
        class,
        arrangement: None,
        lane: None,
    })
}

#[inline]
fn invert_condition(cond: ConditionCode) -> ConditionCode {
    match cond {
        ConditionCode::Eq => ConditionCode::Ne,
        ConditionCode::Ne => ConditionCode::Eq,
        ConditionCode::Cs => ConditionCode::Cc,
        ConditionCode::Cc => ConditionCode::Cs,
        ConditionCode::Mi => ConditionCode::Pl,
        ConditionCode::Pl => ConditionCode::Mi,
        ConditionCode::Vs => ConditionCode::Vc,
        ConditionCode::Vc => ConditionCode::Vs,
        ConditionCode::Hi => ConditionCode::Ls,
        ConditionCode::Ls => ConditionCode::Hi,
        ConditionCode::Ge => ConditionCode::Lt,
        ConditionCode::Lt => ConditionCode::Ge,
        ConditionCode::Gt => ConditionCode::Le,
        ConditionCode::Le => ConditionCode::Gt,
        ConditionCode::Al => ConditionCode::Nv,
        ConditionCode::Nv => ConditionCode::Al,
    }
}

#[inline]
fn condition_code_from_u8(code: u8) -> Option<ConditionCode> {
    match code {
        0 => Some(ConditionCode::Eq),
        1 => Some(ConditionCode::Ne),
        2 => Some(ConditionCode::Cs),
        3 => Some(ConditionCode::Cc),
        4 => Some(ConditionCode::Mi),
        5 => Some(ConditionCode::Pl),
        6 => Some(ConditionCode::Vs),
        7 => Some(ConditionCode::Vc),
        8 => Some(ConditionCode::Hi),
        9 => Some(ConditionCode::Ls),
        10 => Some(ConditionCode::Ge),
        11 => Some(ConditionCode::Lt),
        12 => Some(ConditionCode::Gt),
        13 => Some(ConditionCode::Le),
        14 => Some(ConditionCode::Al),
        15 => Some(ConditionCode::Nv),
        _ => None,
    }
}

#[inline]
fn alias_hint(hint: AliasNoMatchHint) -> EncodeError {
    EncodeError::NoMatchingVariantHint {
        hint: NoMatchingHint::Alias(hint),
    }
}

#[inline]
fn bits_to_u8(bits: i64) -> u8 {
    bits.clamp(0, i64::from(u8::MAX)) as u8
}

#[inline]
fn gpr_data_class(class: RegClass) -> Option<RegClass> {
    match class {
        RegClass::W | RegClass::Wsp => Some(RegClass::W),
        RegClass::X | RegClass::Xsp => Some(RegClass::X),
        _ => None,
    }
}

#[inline]
fn gpr_data_bits(class: RegClass) -> Option<i64> {
    match class {
        RegClass::W | RegClass::Wsp => Some(32),
        RegClass::X | RegClass::Xsp => Some(64),
        _ => None,
    }
}

#[inline]
fn require_gpr_register(
    operand: Operand,
    hint: AliasNoMatchHint,
) -> Result<RegisterOperand, EncodeError> {
    let Operand::Register(reg) = operand else {
        return Err(alias_hint(hint));
    };
    if !is_gpr_class(reg.class) {
        return Err(alias_hint(hint));
    }
    Ok(reg)
}

#[inline]
fn require_immediate(operand: Operand, hint: AliasNoMatchHint) -> Result<i64, EncodeError> {
    match operand {
        Operand::Immediate(value) => Ok(value),
        _ => Err(alias_hint(hint)),
    }
}

fn validate_lsb_width(
    bits: i64,
    lsb: i64,
    width: i64,
    out_of_range_hint: AliasNoMatchHint,
) -> Result<(), EncodeError> {
    if lsb < 0 || width <= 0 || lsb >= bits || width > bits - lsb {
        return Err(alias_hint(out_of_range_hint));
    }
    Ok(())
}

fn extract_stsetl_base(operand: Operand) -> Result<RegisterOperand, EncodeError> {
    match operand {
        Operand::Register(rn) if is_gpr_class(rn.class) => Ok(rn),
        Operand::Memory(mem) => {
            if mem.addressing != AddressingMode::Offset {
                return Err(alias_hint(AliasNoMatchHint::StsetlMemoryOffsetOnly));
            }
            match mem.offset {
                MemoryOffset::None | MemoryOffset::Immediate(0) => {}
                _ => {
                    return Err(alias_hint(AliasNoMatchHint::StsetlMemoryOffsetOnly));
                }
            }
            if !is_gpr_class(mem.base.class) {
                return Err(alias_hint(AliasNoMatchHint::StsetlMemoryBaseMustBeGpr));
            }
            Ok(mem.base)
        }
        _ => Err(alias_hint(AliasNoMatchHint::StsetlSecondOperandInvalid)),
    }
}

#[inline]
fn copy_operands(
    scratch: &mut [Operand; ALIAS_OPERAND_CAP],
    dst_start: usize,
    operands: &[Operand],
) -> Result<(), EncodeError> {
    if dst_start + operands.len() > ALIAS_OPERAND_CAP {
        return Err(EncodeError::OperandCountMismatch);
    }
    for (idx, operand) in operands.iter().copied().enumerate() {
        scratch[dst_start + idx] = operand;
    }
    Ok(())
}

pub(crate) fn canonicalize_alias<'a>(
    mnemonic: &'a str,
    operands: &'a [Operand],
    scratch: &'a mut [Operand; ALIAS_OPERAND_CAP],
) -> Result<(Option<MnemonicId>, &'a str, &'a [Operand]), EncodeError> {
    if let Some(rule) = lookup_conditional_branch_alias(mnemonic) {
        let Some(cond) = condition_code_from_u8(rule.condition_code) else {
            return Err(alias_hint(AliasNoMatchHint::InvalidGeneratedConditionCode));
        };
        scratch[0] = Operand::Condition(cond);
        copy_operands(scratch, 1, operands)?;
        return Ok((
            Some(MnemonicId(rule.base_mnemonic_id)),
            rule.base_mnemonic,
            &scratch[..operands.len() + 1],
        ));
    }
    let Some(rule) = lookup_alias_rule(mnemonic) else {
        return Ok((None, mnemonic, operands));
    };

    match rule.transform {
        AliasTransform::PureRename => Ok((
            Some(MnemonicId(rule.canonical_id)),
            rule.canonical,
            operands,
        )),
        AliasTransform::RetDefault => {
            if operands.is_empty() {
                scratch[0] = Operand::Register(RegisterOperand {
                    code: 30,
                    class: RegClass::X,
                    arrangement: None,
                    lane: None,
                });
                Ok((
                    Some(MnemonicId(rule.canonical_id)),
                    rule.canonical,
                    &scratch[..1],
                ))
            } else {
                Ok((
                    Some(MnemonicId(rule.canonical_id)),
                    rule.canonical,
                    operands,
                ))
            }
        }
        AliasTransform::CmpLike | AliasTransform::CmnLike | AliasTransform::TstLike => {
            if operands.len() < 2 {
                return Err(alias_hint(
                    AliasNoMatchHint::CompareTestNeedsAtLeastTwoOperands,
                ));
            }
            let lhs = require_gpr_register(
                operands[0],
                AliasNoMatchHint::CompareTestFirstOperandMustBeGpr,
            )?;
            let Some(zero) = zero_register_for_class(lhs.class) else {
                return Err(alias_hint(
                    AliasNoMatchHint::CompareTestFirstOperandMustBeGpr,
                ));
            };
            scratch[0] = Operand::Register(zero);
            scratch[1] = Operand::Register(lhs);
            copy_operands(scratch, 2, &operands[1..])?;
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..operands.len() + 1],
            ))
        }
        AliasTransform::MovLike => {
            if operands.len() != 2 {
                return Err(alias_hint(AliasNoMatchHint::MovNeedsExactlyTwoOperands));
            }

            let dst = require_gpr_register(operands[0], AliasNoMatchHint::MovDestinationMustBeGpr)?;
            if gpr_data_class(dst.class).is_none() {
                return Err(alias_hint(AliasNoMatchHint::MovDestinationMustBeGpr));
            }

            let src = match operands[1] {
                Operand::Register(reg) if is_gpr_class(reg.class) => reg,
                Operand::Immediate(0) => {
                    let Some(zero) = zero_register_for_class(dst.class) else {
                        return Err(alias_hint(AliasNoMatchHint::MovDestinationMustBeGpr));
                    };
                    zero
                }
                _ => {
                    return Err(alias_hint(AliasNoMatchHint::MovSourceMustBeGprOrZero));
                }
            };

            scratch[0] = Operand::Register(dst);
            scratch[1] = Operand::Register(src);
            scratch[2] = Operand::Immediate(0);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..3],
            ))
        }
        AliasTransform::MvnLike => {
            if operands.len() < 2 || operands.len() > 3 {
                return Err(alias_hint(AliasNoMatchHint::MvnOperandFormInvalid));
            }
            let dst = require_gpr_register(operands[0], AliasNoMatchHint::MvnDestinationMustBeGpr)?;
            let src = require_gpr_register(operands[1], AliasNoMatchHint::MvnSourceMustBeGpr)?;
            let Some(zero) = zero_register_for_class(dst.class) else {
                return Err(alias_hint(AliasNoMatchHint::MvnDestinationMustBeGpr));
            };
            let shift_amount = if operands.len() == 2 {
                0
            } else {
                match operands[2] {
                    Operand::Immediate(value) => value,
                    Operand::Shift(shift) if shift.kind == ShiftKind::Lsl => {
                        i64::from(shift.amount)
                    }
                    _ => {
                        return Err(alias_hint(
                            AliasNoMatchHint::MvnOptionalShiftMustBeImmediateOrLsl,
                        ));
                    }
                }
            };

            scratch[0] = Operand::Register(dst);
            scratch[1] = Operand::Register(zero);
            scratch[2] = Operand::Register(src);
            scratch[3] = Operand::Immediate(shift_amount);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::CincLike => {
            if operands.len() != 3 {
                return Err(alias_hint(AliasNoMatchHint::CincNeedsExactlyThreeOperands));
            }
            let rd = require_gpr_register(operands[0], AliasNoMatchHint::CincOperandsMustBeGpr)?;
            let rn = require_gpr_register(operands[1], AliasNoMatchHint::CincOperandsMustBeGpr)?;
            if !is_gpr_class(rd.class) || !is_gpr_class(rn.class) {
                return Err(alias_hint(AliasNoMatchHint::CincOperandsMustBeGpr));
            }
            let Operand::Condition(cond) = operands[2] else {
                return Err(alias_hint(AliasNoMatchHint::CincThirdMustBeCondition));
            };

            scratch[0] = Operand::Register(rd);
            scratch[1] = Operand::Register(rn);
            scratch[2] = Operand::Register(rn);
            scratch[3] = Operand::Condition(invert_condition(cond));
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::CsetLike => {
            if operands.len() != 2 {
                return Err(alias_hint(AliasNoMatchHint::CsetNeedsExactlyTwoOperands));
            }
            let rd = require_gpr_register(operands[0], AliasNoMatchHint::CsetDestinationMustBeGpr)?;
            let Some(zero) = zero_register_for_class(rd.class) else {
                return Err(alias_hint(AliasNoMatchHint::CsetDestinationMustBeGpr));
            };
            let Operand::Condition(cond) = operands[1] else {
                return Err(alias_hint(AliasNoMatchHint::CsetSecondMustBeCondition));
            };

            scratch[0] = Operand::Register(rd);
            scratch[1] = Operand::Register(zero);
            scratch[2] = Operand::Register(zero);
            scratch[3] = Operand::Condition(invert_condition(cond));
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::CnegLike => {
            if operands.len() != 3 {
                return Err(alias_hint(AliasNoMatchHint::CnegNeedsExactlyThreeOperands));
            }
            let rd = require_gpr_register(operands[0], AliasNoMatchHint::CnegDestinationMustBeGpr)?;
            let rn = require_gpr_register(operands[1], AliasNoMatchHint::CnegSourceMustBeGpr)?;
            let Operand::Condition(cond) = operands[2] else {
                return Err(alias_hint(AliasNoMatchHint::CnegThirdMustBeCondition));
            };

            scratch[0] = Operand::Register(rd);
            scratch[1] = Operand::Register(rn);
            scratch[2] = Operand::Register(rn);
            scratch[3] = Operand::Condition(invert_condition(cond));
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::BitfieldBfi
        | AliasTransform::BitfieldBfxil
        | AliasTransform::BitfieldUbfx
        | AliasTransform::BitfieldSbfx
        | AliasTransform::BitfieldSbfiz => {
            if operands.len() != 4 {
                return Err(alias_hint(
                    AliasNoMatchHint::BitfieldNeedsExactlyFourOperands,
                ));
            }
            let rd =
                require_gpr_register(operands[0], AliasNoMatchHint::BitfieldDestinationMustBeGpr)?;
            let rn = require_gpr_register(operands[1], AliasNoMatchHint::BitfieldSourceMustBeGpr)?;
            let Some(bits) = gpr_data_bits(rd.class) else {
                return Err(alias_hint(AliasNoMatchHint::BitfieldDestinationMustBeGpr));
            };
            if gpr_data_bits(rn.class) != Some(bits) {
                return Err(alias_hint(
                    AliasNoMatchHint::BitfieldSourceWidthMustMatchDestination,
                ));
            }

            let lsb = require_immediate(operands[2], AliasNoMatchHint::BitfieldLsbMustBeImmediate)?;
            let width =
                require_immediate(operands[3], AliasNoMatchHint::BitfieldWidthMustBeImmediate)?;
            validate_lsb_width(
                bits,
                lsb,
                width,
                AliasNoMatchHint::BitfieldRangeInvalid {
                    bits: bits_to_u8(bits),
                },
            )?;

            let (immr, imms) = match rule.transform {
                AliasTransform::BitfieldBfi => ((-lsb).rem_euclid(bits), width - 1),
                AliasTransform::BitfieldBfxil => (lsb, lsb + width - 1),
                AliasTransform::BitfieldUbfx => (lsb, lsb + width - 1),
                AliasTransform::BitfieldSbfx => (lsb, lsb + width - 1),
                AliasTransform::BitfieldSbfiz => ((-lsb).rem_euclid(bits), width - 1),
                _ => unreachable!("matched outer bitfield transform set"),
            };

            scratch[0] = Operand::Register(rd);
            scratch[1] = Operand::Register(rn);
            scratch[2] = Operand::Immediate(immr);
            scratch[3] = Operand::Immediate(imms);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::BitfieldBfc => {
            if operands.len() != 3 {
                return Err(alias_hint(AliasNoMatchHint::BfcNeedsExactlyThreeOperands));
            }
            let rd = require_gpr_register(operands[0], AliasNoMatchHint::BfcDestinationMustBeGpr)?;
            let Some(bits) = gpr_data_bits(rd.class) else {
                return Err(alias_hint(AliasNoMatchHint::BfcDestinationMustBeGpr));
            };
            let Some(zero) = zero_register_for_class(rd.class) else {
                return Err(alias_hint(AliasNoMatchHint::BfcDestinationMustBeGpr));
            };
            let lsb = require_immediate(operands[1], AliasNoMatchHint::BfcLsbMustBeImmediate)?;
            let width = require_immediate(operands[2], AliasNoMatchHint::BfcWidthMustBeImmediate)?;
            validate_lsb_width(
                bits,
                lsb,
                width,
                AliasNoMatchHint::BfcRangeInvalid {
                    bits: bits_to_u8(bits),
                },
            )?;

            scratch[0] = Operand::Register(rd);
            scratch[1] = Operand::Register(zero);
            scratch[2] = Operand::Immediate((-lsb).rem_euclid(bits));
            scratch[3] = Operand::Immediate(width - 1);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..4],
            ))
        }
        AliasTransform::ExtendLongZero => {
            if operands.len() < 2 {
                return Err(alias_hint(
                    AliasNoMatchHint::ExtendLongNeedsAtLeastTwoOperands,
                ));
            }
            copy_operands(scratch, 0, operands)?;
            if operands.len() == 2 {
                scratch[2] = Operand::Immediate(0);
                Ok((
                    Some(MnemonicId(rule.canonical_id)),
                    rule.canonical,
                    &scratch[..3],
                ))
            } else {
                Ok((
                    Some(MnemonicId(rule.canonical_id)),
                    rule.canonical,
                    &scratch[..operands.len()],
                ))
            }
        }
        AliasTransform::StsetlLike => {
            if operands.len() != 2 {
                return Err(alias_hint(AliasNoMatchHint::StsetlNeedsExactlyTwoOperands));
            }
            let rs =
                require_gpr_register(operands[0], AliasNoMatchHint::StsetlFirstOperandMustBeGpr)?;
            let Some(data_class) = gpr_data_class(rs.class) else {
                return Err(alias_hint(AliasNoMatchHint::StsetlFirstOperandMustBeGpr));
            };
            let Some(zero) = zero_register_for_class(data_class) else {
                return Err(alias_hint(AliasNoMatchHint::StsetlFirstOperandMustBeGpr));
            };
            let mut rn = extract_stsetl_base(operands[1])?;
            rn.class = data_class;

            scratch[0] = Operand::Register(zero);
            scratch[1] = Operand::Register(rn);
            scratch[2] = Operand::Register(rs);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..3],
            ))
        }
        AliasTransform::DcLike => {
            if operands.len() != 2 {
                return Err(alias_hint(AliasNoMatchHint::DcNeedsExactlyTwoOperands));
            }
            let subop = require_immediate(
                operands[0],
                AliasNoMatchHint::DcFirstOperandMustBeImmediateSubop,
            )?;
            let rt =
                require_gpr_register(operands[1], AliasNoMatchHint::DcSecondOperandMustBeRegister)?;
            if subop < 0 || subop > u32::MAX as i64 {
                return Err(alias_hint(AliasNoMatchHint::DcSubopMustBeNonNegativeU32));
            }
            let subop = subop as u32;
            const DC_SUBOP_MASK: u32 = (0x7 << 16) | (0xf << 8) | (0x7 << 5);
            if subop & !DC_SUBOP_MASK != 0 {
                return Err(alias_hint(AliasNoMatchHint::DcSubopUnsupportedBits));
            }
            let op1 = i64::from((subop >> 16) & 0x7);
            let crm = i64::from((subop >> 8) & 0xf);
            let op2 = i64::from((subop >> 5) & 0x7);

            scratch[0] = Operand::Register(rt);
            scratch[1] = Operand::Immediate(op1);
            scratch[2] = Operand::Immediate(7);
            scratch[3] = Operand::Immediate(crm);
            scratch[4] = Operand::Immediate(op2);
            Ok((
                Some(MnemonicId(rule.canonical_id)),
                rule.canonical,
                &scratch[..5],
            ))
        }
    }
}

/// Returns the canonical mnemonic for an asm alias, if one exists.
#[must_use]
pub fn alias_canonical_mnemonic(mnemonic: &str) -> Option<&'static str> {
    if let Some(rule) = lookup_conditional_branch_alias(mnemonic) {
        return Some(rule.base_mnemonic);
    }
    lookup_alias_rule(mnemonic).map(|rule| rule.canonical)
}

/// Returns whether this mnemonic is accepted as an asm alias.
#[must_use]
pub fn supports_alias_mnemonic(mnemonic: &str) -> bool {
    alias_canonical_mnemonic(mnemonic).is_some()
}

#[inline]
pub(crate) fn has_alias(mnemonic: &str) -> bool {
    supports_alias_mnemonic(mnemonic)
}
