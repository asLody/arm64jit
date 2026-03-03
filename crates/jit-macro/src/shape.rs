use crate::ast::{
    JitArg, OperandAst, ParsedMemory, ParsedMemoryOffset, ParsedModifier, ParsedPostIndex,
    ParsedRegister,
};
use std::vec::Vec;

#[derive(Copy, Clone)]
struct FlatKindToken {
    kind_code: u8,
    optional: bool,
    generic_kind_code: Option<u8>,
}

#[inline]
fn push_shape_token(
    out: &mut Vec<FlatKindToken>,
    kind_code: u8,
    optional: bool,
    generic_kind_code: Option<u8>,
) {
    out.push(FlatKindToken {
        kind_code,
        optional,
        generic_kind_code,
    });
}

fn register_kind_codes(class: &'static str) -> Option<(u8, Option<u8>)> {
    match class {
        "W" | "Wsp" => Some((2, Some(1))),
        "X" | "Xsp" => Some((3, Some(1))),
        "V" | "B" | "H" | "S" | "D" | "Q" => Some((4, None)),
        "Z" => Some((5, None)),
        "P" => Some((6, None)),
        _ => None,
    }
}

#[inline]
fn memory_addressing_shape_code(memory: &ParsedMemory) -> u8 {
    if memory.post_index.is_some() {
        // `0` is valid because the packed key also includes total length.
        0
    } else if memory.pre_index {
        15
    } else {
        14
    }
}

fn push_modifier_shape(modifier: &ParsedModifier, out: &mut Vec<FlatKindToken>) {
    match modifier {
        ParsedModifier::Shift { .. } => {
            push_shape_token(out, 9, false, None);
            push_shape_token(out, 7, false, None);
        }
        ParsedModifier::Extend { .. } => {
            push_shape_token(out, 10, false, None);
            push_shape_token(out, 7, false, None);
        }
    }
}

fn push_register_shape(reg: &ParsedRegister, out: &mut Vec<FlatKindToken>) -> Option<()> {
    let (kind_code, generic_kind_code) = register_kind_codes(reg.class)?;
    push_shape_token(out, kind_code, false, generic_kind_code);
    if reg.arrangement.is_some() {
        push_shape_token(out, 12, false, None);
    }
    if reg.lane.is_some() {
        push_shape_token(out, 13, false, None);
    }
    Some(())
}

fn shape_tokens_for_arg(arg: &JitArg, out: &mut Vec<FlatKindToken>) -> Option<()> {
    match arg {
        JitArg::DirectionalLabelRef { .. } => {
            push_shape_token(out, 7, false, None);
            Some(())
        }
        JitArg::Operand(operand) => match operand {
            OperandAst::Immediate(_) => {
                push_shape_token(out, 7, false, None);
                Some(())
            }
            OperandAst::Register(reg) => push_register_shape(reg, out),
            OperandAst::Memory(memory) => {
                push_register_shape(&memory.base, out)?;
                match &memory.offset {
                    ParsedMemoryOffset::None => {
                        if !memory.pre_index && memory.post_index.is_none() {
                            push_shape_token(out, 7, true, None);
                        }
                    }
                    ParsedMemoryOffset::Immediate(_) => {
                        push_shape_token(out, 7, false, None);
                    }
                    ParsedMemoryOffset::Register { reg, modifier } => {
                        push_register_shape(reg, out)?;
                        if let Some(modifier) = modifier {
                            push_modifier_shape(modifier, out);
                        }
                    }
                }
                if let Some(post_index) = &memory.post_index {
                    match post_index {
                        ParsedPostIndex::Immediate(_) => push_shape_token(out, 7, false, None),
                        ParsedPostIndex::Register(reg) => push_register_shape(reg, out)?,
                    }
                }
                push_shape_token(out, memory_addressing_shape_code(memory), false, None);
                Some(())
            }
            OperandAst::Shift { .. } => {
                push_shape_token(out, 9, false, None);
                push_shape_token(out, 7, false, None);
                Some(())
            }
            OperandAst::Extend { .. } => {
                push_shape_token(out, 10, false, None);
                push_shape_token(out, 7, false, None);
                Some(())
            }
            OperandAst::Condition(_) => {
                push_shape_token(out, 8, false, None);
                Some(())
            }
            OperandAst::RegisterList(_) => None,
            OperandAst::SysReg(_) => {
                out.extend_from_slice(&[
                    FlatKindToken {
                        kind_code: 11,
                        optional: false,
                        generic_kind_code: None,
                    },
                    FlatKindToken {
                        kind_code: 11,
                        optional: false,
                        generic_kind_code: None,
                    },
                    FlatKindToken {
                        kind_code: 11,
                        optional: false,
                        generic_kind_code: None,
                    },
                    FlatKindToken {
                        kind_code: 11,
                        optional: false,
                        generic_kind_code: None,
                    },
                    FlatKindToken {
                        kind_code: 7,
                        optional: false,
                        generic_kind_code: None,
                    },
                ]);
                Some(())
            }
        },
    }
}

fn encode_shape_key(kind_codes: &[u8]) -> Option<u128> {
    if kind_codes.len() > 30 {
        return None;
    }

    let mut key = kind_codes.len() as u128;
    for (idx, kind_code) in kind_codes.iter().copied().enumerate() {
        let shift = 8 + (idx * 4);
        key |= u128::from(kind_code) << shift;
    }
    Some(key)
}

fn shape_keys_for_args(args: &[JitArg]) -> Option<Vec<u128>> {
    let mut tokens = Vec::<FlatKindToken>::with_capacity(32);
    for arg in args {
        shape_tokens_for_arg(arg, &mut tokens)?;
    }

    let optional_len = tokens.iter().filter(|token| token.optional).count();
    let mut keys = Vec::<u128>::with_capacity(16);
    let mut materialized = Vec::<FlatKindToken>::with_capacity(tokens.len());
    let mut base_codes = Vec::<u8>::with_capacity(tokens.len());
    let mut codes = Vec::<u8>::with_capacity(tokens.len());
    let mut generic_slots = Vec::<(usize, u8)>::with_capacity(12);
    for include_optional in 0..=optional_len {
        materialized.clear();
        let mut remaining_optional = include_optional;
        for token in tokens.iter().copied() {
            if token.optional {
                if remaining_optional == 0 {
                    continue;
                }
                remaining_optional -= 1;
            }
            materialized.push(token);
        }

        base_codes.clear();
        generic_slots.clear();
        for (idx, token) in materialized.iter().enumerate() {
            base_codes.push(token.kind_code);
            if let Some(code) = token.generic_kind_code {
                generic_slots.push((idx, code));
            }
        }
        if generic_slots.len() > 12 {
            return None;
        }

        let combinations = 1usize << generic_slots.len();
        codes.clear();
        codes.extend_from_slice(&base_codes);
        for mask in 0..combinations {
            codes.copy_from_slice(&base_codes);
            for (bit, (slot, generic_code)) in generic_slots.iter().copied().enumerate() {
                if ((mask >> bit) & 1) != 0 {
                    codes[slot] = generic_code;
                }
            }
            let key = encode_shape_key(&codes)?;
            if keys.contains(&key) {
                continue;
            }
            keys.push(key);
        }
    }
    Some(keys)
}

fn lookup_canonical_mnemonic_id(mnemonic: &str) -> Option<u16> {
    let idx = crate::rules::generated::MNEMONIC_ID_RULES
        .binary_search_by(|rule| rule.mnemonic.cmp(mnemonic))
        .ok()?;
    Some(crate::rules::generated::MNEMONIC_ID_RULES[idx].id)
}

pub(crate) fn lookup_mnemonic_id(op_name: &str) -> Option<u16> {
    if let Some(id) = lookup_canonical_mnemonic_id(op_name) {
        return Some(id);
    }
    if let Some(rule) = crate::rules::generated::lookup_alias_rule(op_name) {
        return lookup_canonical_mnemonic_id(rule.canonical);
    }
    if let Ok(idx) = crate::rules::generated::CONDITIONAL_BRANCH_ALIAS_RULES
        .binary_search_by(|rule| rule.alias.cmp(op_name))
    {
        return lookup_canonical_mnemonic_id(
            crate::rules::generated::CONDITIONAL_BRANCH_ALIAS_RULES[idx].base_mnemonic,
        );
    }
    None
}

pub(crate) fn lookup_direct_variant_id(mnemonic_id: u16, args: &[JitArg]) -> Option<u16> {
    let shape_keys = shape_keys_for_args(args)?;
    if shape_keys.is_empty() {
        return None;
    }

    let mut selected: Option<u16> = None;
    for shape_key in shape_keys {
        let Ok(idx) = crate::rules::generated::DIRECT_VARIANT_RULES.binary_search_by(|rule| {
            rule.mnemonic_id
                .cmp(&mnemonic_id)
                .then(rule.shape_key.cmp(&shape_key))
        }) else {
            continue;
        };
        let candidate = crate::rules::generated::DIRECT_VARIANT_RULES[idx].variant_id;
        match selected {
            None => selected = Some(candidate),
            Some(previous) if previous == candidate => {}
            Some(_) => return None,
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::lookup_mnemonic_id;
    use crate::rules::generated::lookup_alias_rule;

    #[test]
    fn lookup_mnemonic_id_resolves_aliases_to_canonical_ids() {
        assert_eq!(lookup_mnemonic_id("bfi"), lookup_mnemonic_id("bfm"));
        assert_eq!(lookup_mnemonic_id("bfxil"), lookup_mnemonic_id("bfm"));
        assert_eq!(lookup_mnemonic_id("sbfiz"), lookup_mnemonic_id("sbfm"));
        assert_eq!(lookup_mnemonic_id("stsetl"), lookup_mnemonic_id("ldsetl"));
    }

    #[test]
    fn bitfield_extract_fixed_alias_rules_exist_in_macro_table() {
        let sxtb = lookup_alias_rule("sxtb").expect("missing sxtb alias rule");
        let sxth = lookup_alias_rule("sxth").expect("missing sxth alias rule");
        let sxtw = lookup_alias_rule("sxtw").expect("missing sxtw alias rule");
        let uxtb = lookup_alias_rule("uxtb").expect("missing uxtb alias rule");
        let uxth = lookup_alias_rule("uxth").expect("missing uxth alias rule");
        let uxtw = lookup_alias_rule("uxtw").expect("missing uxtw alias rule");

        assert_eq!(sxtb.canonical, "sbfm");
        assert_eq!(sxth.canonical, "sbfm");
        assert_eq!(sxtw.canonical, "sbfm");
        assert_eq!(uxtb.canonical, "ubfm");
        assert_eq!(uxth.canonical, "ubfm");
        assert_eq!(uxtw.canonical, "ubfm");

        assert_eq!(sxtb.fixed_imms, 7);
        assert_eq!(sxth.fixed_imms, 15);
        assert_eq!(sxtw.fixed_imms, 31);
        assert_eq!(uxtb.fixed_imms, 7);
        assert_eq!(uxth.fixed_imms, 15);
        assert_eq!(uxtw.fixed_imms, 31);
    }
}
