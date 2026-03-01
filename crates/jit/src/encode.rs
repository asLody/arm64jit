use crate::alias::{ALIAS_OPERAND_CAP, canonicalize_alias, has_alias};
use crate::generated::{self, MnemonicId, VariantId};
use jit_core::{EncodeError, EncodingSpec, InstructionCode, Operand};

// Encoding dispatch strategy:
// 1) Try shape-key dispatch (O(1)-style shortlist) when it yields a unique variant.
// 2) Fall back to full core matcher for complete diagnostics and edge cases.
// 3) Apply alias canonicalization only when required by the mnemonic.

/// Returns the generated spec slice for one canonical mnemonic.
#[must_use]
pub fn specs_for_mnemonic(mnemonic: &str) -> Option<&'static [EncodingSpec]> {
    generated::specs_for_mnemonic(mnemonic)
}

/// Resolves a canonical mnemonic to a compact dispatcher ID.
#[must_use]
pub fn mnemonic_id(mnemonic: &str) -> Option<MnemonicId> {
    generated::mnemonic_id_from_str(mnemonic)
}

/// Encodes one instruction by compile-time mnemonic ID and typed operands,
/// without applying alias normalization.
///
/// # Errors
///
/// Returns [`EncodeError`] if no variant can be selected and encoded.
#[must_use]
pub fn encode_mnemonic_id_const_no_alias<const MNEMONIC: u16>(
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    let mnemonic_name =
        generated::mnemonic_name_const::<MNEMONIC>().ok_or(EncodeError::UnknownMnemonic)?;
    let specs =
        generated::specs_for_mnemonic_id_const::<MNEMONIC>().ok_or(EncodeError::UnknownMnemonic)?;
    jit_core::encode(specs, mnemonic_name, operands)
}

/// Returns the generated spec slice for one mnemonic ID.
#[must_use]
pub fn specs_for_mnemonic_id(id: MnemonicId) -> Option<&'static [EncodingSpec]> {
    generated::specs_for_mnemonic_id(id)
}

/// Encodes one instruction by pre-resolved variant ID.
///
/// # Errors
///
/// Returns [`EncodeError`] if operands do not satisfy the selected variant.
#[must_use]
pub fn encode_variant(
    variant: VariantId,
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    generated::encode_variant(variant, operands)
}

/// Encodes one instruction by compile-time variant ID.
///
/// This enables link-time dead-code elimination of unrelated variant metadata
/// when call sites use fixed variant IDs (for example, `jit!` fast paths).
///
/// # Errors
///
/// Returns [`EncodeError`] if operands do not satisfy the selected variant.
#[must_use]
pub fn encode_variant_const<const VARIANT: u16>(
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    generated::encode_variant_const::<VARIANT>(operands)
}

/// Returns the canonical encoding spec referenced by a variant ID.
#[must_use]
pub fn spec_for_variant(variant: VariantId) -> Option<&'static EncodingSpec> {
    generated::spec_for_variant(variant)
}

fn encode_mnemonic_id_no_alias(
    mnemonic_id: MnemonicId,
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    // Decision-tree fast path:
    // 1) derive shape keys (including optional materializations),
    // 2) resolve each key into generated candidates,
    // 3) if all resolved keys agree on one unique candidate, encode directly.
    let mut shape_keys = [0u128; 32];
    if let Ok(shape_len) = jit_core::operand_shape_keys(operands, &mut shape_keys) {
        let mut selected_variant: Option<VariantId> = None;
        let mut saw_shape_match = false;
        for shape_key in shape_keys[..shape_len].iter().copied() {
            let Some(shape_match) = generated::variant_match_for_shape(mnemonic_id, shape_key)
            else {
                continue;
            };
            saw_shape_match = true;
            match shape_match {
                generated::ShapeVariantMatch::Ambiguous => {
                    selected_variant = None;
                    break;
                }
                generated::ShapeVariantMatch::Unique(candidate) => match selected_variant {
                    None => selected_variant = Some(candidate),
                    Some(previous) if previous == candidate => {}
                    Some(_) => {
                        selected_variant = None;
                        break;
                    }
                },
            }
        }
        if saw_shape_match && let Some(variant) = selected_variant {
            match generated::encode_variant(variant, operands) {
                Ok(code) => return Ok(code),
                Err(
                    EncodeError::NoMatchingVariant
                    | EncodeError::AmbiguousVariant
                    | EncodeError::OperandCountMismatch
                    | EncodeError::OperandCountRange { .. }
                    | EncodeError::NoMatchingVariantHint { .. },
                ) => {}
                Err(err) => return Err(err),
            }
        }
    }

    let mnemonic_name =
        generated::mnemonic_name(mnemonic_id).ok_or(EncodeError::UnknownMnemonic)?;
    let specs =
        generated::specs_for_mnemonic_id(mnemonic_id).ok_or(EncodeError::UnknownMnemonic)?;
    jit_core::encode(specs, mnemonic_name, operands)
}

pub(crate) fn encode_mnemonic_id(
    mnemonic_id: MnemonicId,
    operands: &[Operand],
) -> Result<InstructionCode, EncodeError> {
    // Alias expansion is intentionally a second pass to keep the canonical path fast.
    match encode_mnemonic_id_no_alias(mnemonic_id, operands) {
        Ok(code) => Ok(code),
        Err(direct_err) => {
            let mnemonic_name =
                generated::mnemonic_name(mnemonic_id).ok_or(EncodeError::UnknownMnemonic)?;
            if !has_alias(mnemonic_name) {
                return Err(direct_err);
            }

            let mut scratch = [Operand::Immediate(0); ALIAS_OPERAND_CAP];
            let (canonical_id, canonical_mnemonic, canonical_operands) =
                canonicalize_alias(mnemonic_name, operands, &mut scratch)?;
            if canonical_mnemonic == mnemonic_name && canonical_operands == operands {
                return Err(direct_err);
            }
            let canonical_id = canonical_id
                .or_else(|| generated::mnemonic_id_from_str(canonical_mnemonic))
                .ok_or(EncodeError::UnknownMnemonic)?;
            encode_mnemonic_id_no_alias(canonical_id, canonical_operands)
        }
    }
}

/// Encodes one instruction by mnemonic and typed operands in asm-like order.
///
/// # Errors
///
/// Returns [`EncodeError`] if no variant can be selected and encoded.
#[must_use]
pub fn encode(mnemonic: &str, operands: &[Operand]) -> Result<InstructionCode, EncodeError> {
    // If canonical lookup succeeds, try it first. Alias normalization is only attempted
    // when canonical resolution fails or the mnemonic itself is an alias.
    if let Some(id) = generated::mnemonic_id_from_str(mnemonic) {
        match encode_mnemonic_id(id, operands) {
            Ok(code) => return Ok(code),
            Err(direct_err) => {
                if !has_alias(mnemonic) {
                    return Err(direct_err);
                }
            }
        }
    } else if !has_alias(mnemonic) {
        return Err(EncodeError::UnknownMnemonic);
    }

    let mut scratch = [Operand::Immediate(0); ALIAS_OPERAND_CAP];
    let (canonical_id, canonical_mnemonic, canonical_operands) =
        canonicalize_alias(mnemonic, operands, &mut scratch)?;
    let Some(canonical_id) =
        canonical_id.or_else(|| generated::mnemonic_id_from_str(canonical_mnemonic))
    else {
        return Err(EncodeError::UnknownMnemonic);
    };
    encode_mnemonic_id(canonical_id, canonical_operands)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use jit_core::{
        AddressingMode, ConditionCode, MemoryOffset, MemoryOperand, RegClass, RegisterOperand,
    };

    fn x(code: u8) -> Operand {
        Operand::Register(RegisterOperand {
            code,
            class: RegClass::X,
            arrangement: None,
            lane: None,
        })
    }

    fn w(code: u8) -> Operand {
        Operand::Register(RegisterOperand {
            code,
            class: RegClass::W,
            arrangement: None,
            lane: None,
        })
    }

    fn imm(value: i64) -> Operand {
        Operand::Immediate(value)
    }

    fn mem(base: u8, offset: MemoryOffset, addressing: AddressingMode) -> Operand {
        Operand::Memory(MemoryOperand {
            base: RegisterOperand {
                code: base,
                class: RegClass::X,
                arrangement: None,
                lane: None,
            },
            offset,
            addressing,
        })
    }

    fn shortlist_ids(mnemonic: &str, operands: &[Operand]) -> Vec<u16> {
        let Some(id) = generated::mnemonic_id_from_str(mnemonic) else {
            return Vec::new();
        };
        let mut shape_keys = [0u128; 32];
        let Ok(shape_len) = jit_core::operand_shape_keys(operands, &mut shape_keys) else {
            return Vec::new();
        };

        let mut shortlist = Vec::new();
        for shape_key in shape_keys[..shape_len].iter().copied() {
            let Some(candidates) = generated::variants_for_shape(id, shape_key) else {
                continue;
            };
            for candidate in candidates.iter().copied() {
                if shortlist.contains(&candidate.0) {
                    continue;
                }
                shortlist.push(candidate.0);
            }
        }
        shortlist
    }

    #[test]
    fn shortlist_candidates_match_full_encoder() {
        let cases: [(&str, Vec<Operand>); 4] = [
            ("add", vec![x(1), x(2), imm(7)]),
            ("sub", vec![x(3), x(4), imm(9)]),
            (
                "ldr",
                vec![
                    x(5),
                    mem(6, MemoryOffset::Immediate(16), AddressingMode::Offset),
                ],
            ),
            ("b", vec![Operand::Condition(ConditionCode::Le), imm(12)]),
        ];

        for (mnemonic, operands) in cases {
            let ids = shortlist_ids(mnemonic, &operands);
            assert!(
                !ids.is_empty(),
                "empty shortlist for mnemonic={mnemonic} operands={operands:?}"
            );

            let full = encode(mnemonic, &operands)
                .unwrap_or_else(|err| panic!("full encode failed for {mnemonic}: {err}"));
            let shortlist = jit_core::encode_candidates(generated::SPECS, &ids, &operands);
            if let Ok(shortlisted_code) = shortlist {
                assert_eq!(
                    full.unpack(),
                    shortlisted_code.unpack(),
                    "shortlist mismatch for {mnemonic}"
                );
            }
        }
    }

    #[test]
    fn encode_supports_scalar_sxt_uxt_alias_family() {
        let cases = [
            ("sxtb", "sbfm", 7),
            ("sxth", "sbfm", 15),
            ("sxtw", "sbfm", 31),
            ("uxtb", "ubfm", 7),
            ("uxth", "ubfm", 15),
            ("uxtw", "ubfm", 31),
        ];

        for (alias, canonical, imms) in cases {
            let got = encode(alias, &[x(0), w(1)]).unwrap_or_else(|err| {
                panic!("{alias} alias should encode, got error: {err:?}");
            });
            let expected =
                encode(canonical, &[x(0), x(1), imm(0), imm(imms)]).expect("canonical bitfield");
            assert_eq!(got.unpack(), expected.unpack(), "alias {alias} mismatch");
        }
    }

    #[test]
    fn memory_addressing_contributes_to_shape_keys() {
        let offset = [
            x(0),
            mem(1, MemoryOffset::Immediate(16), AddressingMode::Offset),
        ];
        let pre = [
            x(0),
            mem(1, MemoryOffset::Immediate(16), AddressingMode::PreIndex),
        ];
        let post = [
            x(0),
            mem(
                1,
                MemoryOffset::None,
                AddressingMode::PostIndex(jit_core::PostIndexOffset::Immediate(16)),
            ),
        ];

        let mut keys_offset = [0u128; 16];
        let mut keys_pre = [0u128; 16];
        let mut keys_post = [0u128; 16];
        let len_offset =
            jit_core::operand_shape_keys(&offset, &mut keys_offset).expect("offset keys");
        let len_pre = jit_core::operand_shape_keys(&pre, &mut keys_pre).expect("pre keys");
        let len_post = jit_core::operand_shape_keys(&post, &mut keys_post).expect("post keys");

        assert!(len_offset > 0 && len_pre > 0 && len_post > 0);
        assert_ne!(
            keys_offset[0], keys_pre[0],
            "offset/preindex keys should differ"
        );
        assert_ne!(
            keys_offset[0], keys_post[0],
            "offset/postindex keys should differ"
        );
        assert_ne!(
            keys_pre[0], keys_post[0],
            "preindex/postindex keys should differ"
        );
    }
}
