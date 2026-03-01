use crate::ast::{ConditionAst, InstructionStmt, JitArg, OperandAst, ParsedRegister, ShiftKindAst};
use quote::quote;
use syn::ExprUnary;
use syn::{Expr, ExprLit, Lit, Result, UnOp, parse_quote};

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

fn arg_immediate_zero() -> JitArg {
    JitArg::Operand(OperandAst::Immediate(parse_quote!(0)))
}

fn gpr_data_class(class: &'static str) -> Option<&'static str> {
    match class {
        "W" | "Wsp" => Some("W"),
        "X" | "Xsp" => Some("X"),
        _ => None,
    }
}

fn zero_register_for_class(class: &'static str) -> Option<ParsedRegister> {
    let class = gpr_data_class(class)?;
    Some(ParsedRegister {
        code: quote! { 31u8 },
        class,
        arrangement: None,
        lane: None,
    })
}

fn invert_condition(condition: ConditionAst) -> ConditionAst {
    match condition {
        ConditionAst::Eq => ConditionAst::Ne,
        ConditionAst::Ne => ConditionAst::Eq,
        ConditionAst::Cs => ConditionAst::Cc,
        ConditionAst::Cc => ConditionAst::Cs,
        ConditionAst::Mi => ConditionAst::Pl,
        ConditionAst::Pl => ConditionAst::Mi,
        ConditionAst::Vs => ConditionAst::Vc,
        ConditionAst::Vc => ConditionAst::Vs,
        ConditionAst::Hi => ConditionAst::Ls,
        ConditionAst::Ls => ConditionAst::Hi,
        ConditionAst::Ge => ConditionAst::Lt,
        ConditionAst::Lt => ConditionAst::Ge,
        ConditionAst::Gt => ConditionAst::Le,
        ConditionAst::Le => ConditionAst::Gt,
        ConditionAst::Al => ConditionAst::Nv,
        ConditionAst::Nv => ConditionAst::Al,
    }
}

fn immediate_expr_i64(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(literal),
            ..
        }) => literal.base10_parse::<i64>().ok(),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => immediate_expr_i64(expr)?.checked_neg(),
        _ => None,
    }
}

fn is_immediate_zero_arg(arg: &JitArg) -> bool {
    matches!(
        arg,
        JitArg::Operand(OperandAst::Immediate(expr)) if immediate_expr_i64(expr) == Some(0)
    )
}

fn operand_as_gpr_register(arg: &JitArg) -> Option<ParsedRegister> {
    match arg {
        JitArg::Operand(OperandAst::Register(reg)) if is_gpr_class(reg.class) => Some(reg.clone()),
        _ => None,
    }
}

fn operand_as_condition(arg: &JitArg) -> Option<ConditionAst> {
    match arg {
        JitArg::Operand(OperandAst::Condition(condition)) => Some(*condition),
        _ => None,
    }
}

fn arg_register(reg: ParsedRegister) -> JitArg {
    JitArg::Operand(OperandAst::Register(reg))
}

fn arg_condition(condition: ConditionAst) -> JitArg {
    JitArg::Operand(OperandAst::Condition(condition))
}

fn lookup_alias_rule(op_name: &str) -> Option<&'static crate::rules::generated::AliasRule> {
    crate::rules::generated::lookup_alias_rule(op_name)
}

fn lookup_conditional_branch_alias(
    op_name: &str,
) -> Option<&'static crate::rules::generated::ConditionalBranchAliasRule> {
    let idx = crate::rules::generated::CONDITIONAL_BRANCH_ALIAS_RULES
        .binary_search_by(|rule| rule.alias.cmp(op_name))
        .ok()?;
    Some(&crate::rules::generated::CONDITIONAL_BRANCH_ALIAS_RULES[idx])
}

fn condition_ast_from_code(code: u8) -> Option<ConditionAst> {
    match code {
        0 => Some(ConditionAst::Eq),
        1 => Some(ConditionAst::Ne),
        2 => Some(ConditionAst::Cs),
        3 => Some(ConditionAst::Cc),
        4 => Some(ConditionAst::Mi),
        5 => Some(ConditionAst::Pl),
        6 => Some(ConditionAst::Vs),
        7 => Some(ConditionAst::Vc),
        8 => Some(ConditionAst::Hi),
        9 => Some(ConditionAst::Ls),
        10 => Some(ConditionAst::Ge),
        11 => Some(ConditionAst::Lt),
        12 => Some(ConditionAst::Gt),
        13 => Some(ConditionAst::Le),
        14 => Some(ConditionAst::Al),
        15 => Some(ConditionAst::Nv),
        _ => None,
    }
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

fn normalize_alias_instruction(
    mut inst: InstructionStmt,
    rule: &crate::rules::generated::AliasRule,
) -> Option<InstructionStmt> {
    use crate::rules::generated::AliasTransform;

    match rule.transform {
        AliasTransform::PureRename => {
            inst.op_name = rule.canonical.to_owned();
            Some(inst)
        }
        AliasTransform::RetDefault => {
            inst.op_name = rule.canonical.to_owned();
            if inst.args.is_empty() {
                let ret = ParsedRegister {
                    code: quote! { 30u8 },
                    class: "X",
                    arrangement: None,
                    lane: None,
                };
                inst.args.push(arg_register(ret));
            }
            Some(inst)
        }
        AliasTransform::CmpLike | AliasTransform::CmnLike | AliasTransform::TstLike => {
            if inst.args.len() < 2 {
                return None;
            }
            let lhs = operand_as_gpr_register(inst.args.first()?)?;
            let zero = zero_register_for_class(lhs.class)?;
            let mut args = Vec::with_capacity(inst.args.len() + 1);
            args.push(arg_register(zero));
            args.push(arg_register(lhs));
            args.extend(inst.args.into_iter().skip(1));
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::MovLike => {
            if inst.args.len() != 2 {
                return None;
            }
            let dst = operand_as_gpr_register(inst.args.first()?)?;
            let src = if let Some(src_reg) = operand_as_gpr_register(inst.args.get(1)?) {
                src_reg
            } else if is_immediate_zero_arg(inst.args.get(1)?) {
                zero_register_for_class(dst.class)?
            } else {
                return None;
            };
            let args = vec![arg_register(dst), arg_register(src), arg_immediate_zero()];
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::MvnLike => {
            if inst.args.len() < 2 || inst.args.len() > 3 {
                return None;
            }
            let dst = operand_as_gpr_register(inst.args.first()?)?;
            let src = operand_as_gpr_register(inst.args.get(1)?)?;
            let zero = zero_register_for_class(dst.class)?;
            let shift_amount = if inst.args.len() == 2 {
                parse_quote!(0)
            } else {
                match inst.args.get(2)? {
                    JitArg::Operand(OperandAst::Immediate(expr)) => expr.clone(),
                    JitArg::Operand(OperandAst::Shift {
                        kind: ShiftKindAst::Lsl,
                        amount,
                    }) => amount.clone().unwrap_or_else(|| parse_quote!(0)),
                    _ => return None,
                }
            };
            let args = vec![
                arg_register(dst),
                arg_register(zero),
                arg_register(src),
                JitArg::Operand(OperandAst::Immediate(shift_amount)),
            ];
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::CincLike => {
            if inst.args.len() != 3 {
                return None;
            }
            let rd = operand_as_gpr_register(inst.args.first()?)?;
            let rn = operand_as_gpr_register(inst.args.get(1)?)?;
            let cond = operand_as_condition(inst.args.get(2)?)?;
            let args = vec![
                arg_register(rd),
                arg_register(rn.clone()),
                arg_register(rn),
                arg_condition(invert_condition(cond)),
            ];
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::CsetLike => {
            if inst.args.len() != 2 {
                return None;
            }
            let rd = operand_as_gpr_register(inst.args.first()?)?;
            let cond = operand_as_condition(inst.args.get(1)?)?;
            let zero = zero_register_for_class(rd.class)?;
            let args = vec![
                arg_register(rd),
                arg_register(zero.clone()),
                arg_register(zero),
                arg_condition(invert_condition(cond)),
            ];
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::CnegLike => {
            if inst.args.len() != 3 {
                return None;
            }
            let rd = operand_as_gpr_register(inst.args.first()?)?;
            let rn = operand_as_gpr_register(inst.args.get(1)?)?;
            let cond = operand_as_condition(inst.args.get(2)?)?;
            let args = vec![
                arg_register(rd),
                arg_register(rn.clone()),
                arg_register(rn),
                arg_condition(invert_condition(cond)),
            ];
            inst.op_name = rule.canonical.to_owned();
            inst.args = args;
            Some(inst)
        }
        AliasTransform::ExtendLongZero => {
            if inst.args.len() < 2 {
                return None;
            }
            inst.op_name = rule.canonical.to_owned();
            if inst.args.len() == 2 {
                inst.args.push(arg_immediate_zero());
            }
            Some(inst)
        }
        _ => None,
    }
}

pub(crate) fn normalize_instruction_stmt(mut inst: InstructionStmt) -> Result<InstructionStmt> {
    if let Some(rule) = lookup_conditional_branch_alias(inst.op_name.as_str()) {
        if matches!(
            inst.args.first(),
            Some(JitArg::Operand(OperandAst::Condition(_)))
        ) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "conditional branch alias cannot take an explicit condition operand",
            ));
        }
        let Some(condition) = condition_ast_from_code(rule.condition_code) else {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "internal conditional branch alias rule is invalid",
            ));
        };
        inst.op_name = rule.base_mnemonic.to_owned();
        inst.args
            .insert(0, JitArg::Operand(OperandAst::Condition(condition)));
    }

    if let Some(rule) = lookup_alias_rule(inst.op_name.as_str())
        && let Some(normalized) = normalize_alias_instruction(inst.clone(), rule)
    {
        inst = normalized;
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
            inst.args.push(arg_immediate_zero());
        } else if inst.args.len() == 4
            && let JitArg::Operand(OperandAst::Shift { kind, amount }) = &inst.args[3]
        {
            if *kind != ShiftKindAst::Lsl {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "register-shift form only supports lsl shift in this syntax",
                ));
            }
            let amount = amount.clone().unwrap_or_else(|| parse_quote!(0));
            inst.args[3] = JitArg::Operand(OperandAst::Immediate(amount));
        }
    }
    Ok(inst)
}
