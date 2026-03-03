use crate::ast::{InstructionStmt, JitArg, OperandAst, ShiftKindAst};
use syn::{Result, parse_quote};

fn is_gpr_class(class: &'static str) -> bool {
    matches!(class, "W" | "X" | "Wsp" | "Xsp")
}

fn is_gpr_register_arg(arg: &JitArg) -> bool {
    matches!(
        arg,
        JitArg::Operand(OperandAst::Register(reg)) if is_gpr_class(reg.class)
    )
}

fn is_sysreg_arg(arg: &JitArg) -> bool {
    matches!(arg, JitArg::Operand(OperandAst::SysReg(_)))
}

fn arg_lsl_shift_zero() -> JitArg {
    JitArg::Operand(OperandAst::Shift {
        kind: ShiftKindAst::Lsl,
        amount: Some(parse_quote!(0)),
    })
}

fn is_conditional_branch_alias_mnemonic(mnemonic: &str) -> bool {
    crate::rules::generated::CONDITIONAL_BRANCH_ALIAS_RULES
        .binary_search_by(|rule| rule.alias.cmp(mnemonic))
        .is_ok()
}

fn lookup_mnemonic_normalization_rule(
    mnemonic_id: u16,
) -> Option<&'static crate::rules::generated::MnemonicNormalizationRule> {
    let idx = crate::rules::generated::MNEMONIC_NORMALIZATION_RULES
        .binary_search_by(|rule| rule.mnemonic_id.cmp(&mnemonic_id))
        .ok()?;
    Some(&crate::rules::generated::MNEMONIC_NORMALIZATION_RULES[idx])
}

fn mnemonic_has_norm_flag(mnemonic_id: u16, flag: u8) -> bool {
    lookup_mnemonic_normalization_rule(mnemonic_id)
        .map(|rule| (rule.flags & flag) != 0)
        .unwrap_or(false)
}

pub(crate) fn mnemonic_reloc_mask(mnemonic_id: u16) -> u8 {
    lookup_mnemonic_normalization_rule(mnemonic_id)
        .map(|rule| rule.reloc_mask)
        .unwrap_or(0)
}

pub(crate) fn normalize_instruction_stmt(mut inst: InstructionStmt) -> Result<InstructionStmt> {
    if is_conditional_branch_alias_mnemonic(inst.op_name.as_str())
        && matches!(
            inst.args.first(),
            Some(JitArg::Operand(OperandAst::Condition(_)))
        )
    {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "conditional branch alias cannot take an explicit condition operand",
        ));
    }

    let mnemonic_id = crate::shape::lookup_mnemonic_id(inst.op_name.as_str());

    if mnemonic_id
        .map(|id| mnemonic_has_norm_flag(id, crate::rules::generated::NORM_FLAG_SYSREG_GPR_SWAP))
        .unwrap_or(false)
        && inst.args.len() == 2
        && is_sysreg_arg(&inst.args[0])
        && is_gpr_register_arg(&inst.args[1])
    {
        inst.args.swap(0, 1);
    }

    if mnemonic_id
        .map(|id| mnemonic_has_norm_flag(id, crate::rules::generated::NORM_FLAG_SHIFT_TO_IMMEDIATE))
        .unwrap_or(false)
        && inst.args.len() >= 3
        && inst.args[..3].iter().all(is_gpr_register_arg)
    {
        if inst.args.len() == 3 {
            inst.args.push(arg_lsl_shift_zero());
        } else if inst.args.len() == 4
            && let JitArg::Operand(OperandAst::Shift { kind, amount }) = &inst.args[3]
        {
            let amount = amount.clone().unwrap_or_else(|| parse_quote!(0));
            inst.args[3] = JitArg::Operand(OperandAst::Shift {
                kind: *kind,
                amount: Some(amount),
            });
        } else if inst.args.len() == 4
            && let JitArg::Operand(OperandAst::Immediate(amount)) = &inst.args[3]
        {
            inst.args[3] = JitArg::Operand(OperandAst::Shift {
                kind: ShiftKindAst::Lsl,
                amount: Some(amount.clone()),
            });
        }
    }

    Ok(inst)
}
