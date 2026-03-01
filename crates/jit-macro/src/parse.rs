use crate::ast::{
    ArrangementAst, ConditionAst, ExtendKindAst, InstructionStmt, JitArg, JitBlockInput, JitStmt,
    LabelDirection, OperandAst, ParsedMemory, ParsedMemoryOffset, ParsedModifier, ParsedPostIndex,
    ParsedRegister, ParsedRegisterList, ParsedSysReg, ShiftKindAst,
};
use crate::normalize::normalize_instruction_stmt;
use quote::quote;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, ExprCall, ExprField, ExprIndex, ExprPath, Ident, LitInt, Member, Result, Token};

fn parse_label_name(input: ParseStream<'_>) -> Result<String> {
    if input.peek(Ident) {
        return Ok(input.parse::<Ident>()?.to_string());
    }
    if input.peek(LitInt) {
        let literal = input.parse::<LitInt>()?;
        if !literal.suffix().is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "numeric label definitions cannot use suffixes",
            ));
        }
        return Ok(literal.base10_digits().to_owned());
    }
    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "expected label name",
    ))
}

fn try_parse_numeric_directional_label_ref(
    input: ParseStream<'_>,
) -> Result<Option<(String, LabelDirection)>> {
    if !input.peek(LitInt) {
        return Ok(None);
    }

    let fork = input.fork();
    let literal = fork.parse::<LitInt>()?;
    let direction = match literal.suffix().to_ascii_lowercase().as_str() {
        "f" => LabelDirection::Forward,
        "b" => LabelDirection::Backward,
        _ => return Ok(None),
    };

    let name = literal.base10_digits().to_owned();
    input.advance_to(&fork);
    Ok(Some((name, direction)))
}

impl Parse for JitArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        if input.peek(Token![<]) {
            let _: Token![<] = input.parse()?;
            let name = parse_label_name(input)?;
            return Ok(Self::DirectionalLabelRef {
                name,
                direction: LabelDirection::Backward,
            });
        }

        if input.peek(Token![>]) {
            let _: Token![>] = input.parse()?;
            let name = parse_label_name(input)?;
            return Ok(Self::DirectionalLabelRef {
                name,
                direction: LabelDirection::Forward,
            });
        }

        if input.peek(Token![=]) && input.peek2(Token![>]) {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "dynamic labels (`=>...`) were removed; use local labels or external patching",
            ));
        }

        if let Some((name, direction)) = try_parse_numeric_directional_label_ref(input)? {
            return Ok(Self::DirectionalLabelRef { name, direction });
        }

        if input.peek(syn::token::Brace) {
            let reglist = parse_register_list_operand(input)?;
            return Ok(Self::Operand(OperandAst::RegisterList(reglist)));
        }

        if input.peek(syn::token::Bracket) {
            let memory = parse_memory_operand(input)?;
            return Ok(Self::Operand(OperandAst::Memory(memory)));
        }

        if let Some(modifier) = try_parse_modifier(input)? {
            return match modifier {
                ParsedModifier::Shift { kind, amount } => {
                    Ok(Self::Operand(OperandAst::Shift { kind, amount }))
                }
                ParsedModifier::Extend { kind, amount } => {
                    Ok(Self::Operand(OperandAst::Extend { kind, amount }))
                }
            };
        }

        let has_hash = if input.peek(Token![#]) {
            let _: Token![#] = input.parse()?;
            true
        } else {
            false
        };

        let expr = input.parse::<Expr>()?;
        if has_hash {
            return Ok(Self::Operand(OperandAst::Immediate(expr)));
        }

        if let Some(cond) = parse_condition_expr(&expr) {
            return Ok(Self::Operand(OperandAst::Condition(cond)));
        }

        if let Some(sysreg) = parse_sysreg_expr(&expr) {
            return Ok(Self::Operand(OperandAst::SysReg(sysreg)));
        }

        if let Some(reg) = parse_register_expr(&expr) {
            return Ok(Self::Operand(OperandAst::Register(reg)));
        }

        Ok(Self::Operand(OperandAst::Immediate(expr)))
    }
}

impl Parse for JitBlockInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let target = input.parse::<Expr>()?;
        let mut statements = Vec::new();

        while !input.is_empty() {
            let _: Token![;] = input.parse()?;
            if input.is_empty() {
                break;
            }

            if input.peek(Token![.]) {
                let _: Token![.] = input.parse()?;
                let directive = input.parse::<Ident>()?.to_string().to_ascii_lowercase();
                match directive.as_str() {
                    "arch" => {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "aarch64-only macro; remove .arch directive",
                        ));
                    }
                    "bytes" => {
                        let expr = input.parse::<Expr>()?;
                        statements.push(JitStmt::Bytes(expr));
                    }
                    _ => {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "unsupported jit directive",
                        ));
                    }
                }
                continue;
            }

            if input.peek(Token![=]) && input.peek2(Token![>]) {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "dynamic labels (`=>...`) were removed; use local labels or external patching",
                ));
            }

            if input.peek(LitInt) && input.peek2(Token![:]) {
                let name = parse_label_name(input)?;
                let _: Token![:] = input.parse()?;
                statements.push(JitStmt::StaticLabelDef(name));
                continue;
            }

            if input.peek(Ident) && input.peek2(Token![:]) {
                let name = input.parse::<Ident>()?.to_string();
                let _: Token![:] = input.parse()?;
                statements.push(JitStmt::StaticLabelDef(name));
                continue;
            }

            let head = input.parse::<Ident>()?;
            let mut op_name = head.to_string();
            while input.peek(Token![.]) {
                let _: Token![.] = input.parse()?;
                let seg = input.parse::<Ident>()?;
                op_name.push('.');
                op_name.push_str(&seg.to_string());
            }
            if op_name == "MOV" {
                // Upper-case `MOV` is a macro-level pseudo instruction:
                // auto materialize arbitrary immediates into GPRs.
                op_name = String::from("__mov_auto");
            } else {
                op_name.make_ascii_lowercase();
            }

            let mut args = Vec::new();
            if !input.is_empty() && !input.peek(Token![;]) {
                args.push(input.parse::<JitArg>()?);
                while input.peek(Token![,]) {
                    let _: Token![,] = input.parse()?;
                    if input.is_empty() || input.peek(Token![;]) {
                        break;
                    }
                    args.push(input.parse::<JitArg>()?);
                }
            }

            let inst = normalize_instruction_stmt(InstructionStmt { op_name, args })?;
            statements.push(JitStmt::Instruction(inst));
        }

        Ok(Self { target, statements })
    }
}

fn parse_prefixed_expr(input: ParseStream<'_>) -> Result<(bool, Expr)> {
    let has_hash = if input.peek(Token![#]) {
        let _: Token![#] = input.parse()?;
        true
    } else {
        false
    };
    let expr = input.parse::<Expr>()?;
    Ok((has_hash, expr))
}

fn try_parse_modifier(input: ParseStream<'_>) -> Result<Option<ParsedModifier>> {
    let fork = input.fork();
    let Some(modifier) = parse_modifier(&fork)? else {
        return Ok(None);
    };
    input.advance_to(&fork);
    Ok(Some(modifier))
}

fn parse_modifier(input: ParseStream<'_>) -> Result<Option<ParsedModifier>> {
    if !input.peek(Ident) {
        return Ok(None);
    }

    let ident = input.parse::<Ident>()?;
    let name = ident.to_string().to_ascii_lowercase();
    let amount = parse_optional_modifier_amount(input)?;

    let modifier = match name.as_str() {
        "lsl" => ParsedModifier::Shift {
            kind: ShiftKindAst::Lsl,
            amount,
        },
        "lsr" => ParsedModifier::Shift {
            kind: ShiftKindAst::Lsr,
            amount,
        },
        "asr" => ParsedModifier::Shift {
            kind: ShiftKindAst::Asr,
            amount,
        },
        "ror" => ParsedModifier::Shift {
            kind: ShiftKindAst::Ror,
            amount,
        },
        "msl" => ParsedModifier::Shift {
            kind: ShiftKindAst::Msl,
            amount,
        },
        "uxtb" => ParsedModifier::Extend {
            kind: ExtendKindAst::Uxtb,
            amount,
        },
        "uxth" => ParsedModifier::Extend {
            kind: ExtendKindAst::Uxth,
            amount,
        },
        "uxtw" => ParsedModifier::Extend {
            kind: ExtendKindAst::Uxtw,
            amount,
        },
        "uxtx" => ParsedModifier::Extend {
            kind: ExtendKindAst::Uxtx,
            amount,
        },
        "sxtb" => ParsedModifier::Extend {
            kind: ExtendKindAst::Sxtb,
            amount,
        },
        "sxth" => ParsedModifier::Extend {
            kind: ExtendKindAst::Sxth,
            amount,
        },
        "sxtw" => ParsedModifier::Extend {
            kind: ExtendKindAst::Sxtw,
            amount,
        },
        "sxtx" => ParsedModifier::Extend {
            kind: ExtendKindAst::Sxtx,
            amount,
        },
        _ => return Ok(None),
    };

    Ok(Some(modifier))
}

fn parse_optional_modifier_amount(input: ParseStream<'_>) -> Result<Option<Expr>> {
    if input.is_empty() || input.peek(Token![,]) || input.peek(Token![;]) || input.peek(Token![!]) {
        return Ok(None);
    }

    if input.peek(Token![#]) {
        let _: Token![#] = input.parse()?;
    }

    let expr = input.parse::<Expr>()?;
    Ok(Some(expr))
}

fn parse_required_register(input: ParseStream<'_>) -> Result<ParsedRegister> {
    let expr = input.parse::<Expr>()?;
    parse_register_expr(&expr).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "expected register expression",
        )
    })
}

fn parse_memory_operand(input: ParseStream<'_>) -> Result<ParsedMemory> {
    let inner;
    let _ = syn::bracketed!(inner in input);

    let base = parse_required_register(&inner)?;
    let mut offset = ParsedMemoryOffset::None;

    if !inner.is_empty() {
        let _: Token![,] = inner.parse()?;
        let (second_hash, second_expr) = parse_prefixed_expr(&inner)?;

        if !second_hash {
            if let Some(reg) = parse_register_expr(&second_expr) {
                let mut modifier = None;
                if !inner.is_empty() {
                    let _: Token![,] = inner.parse()?;
                    modifier = parse_modifier(&inner)?;
                    if modifier.is_none() {
                        return Err(syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "expected shift/extend modifier after memory index register",
                        ));
                    }
                }
                offset = ParsedMemoryOffset::Register { reg, modifier };
            } else {
                offset = ParsedMemoryOffset::Immediate(second_expr);
            }
        } else {
            offset = ParsedMemoryOffset::Immediate(second_expr);
        }
    }

    if !inner.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "unexpected extra tokens in memory reference",
        ));
    }

    let mut pre_index = false;
    if input.peek(Token![!]) {
        let _: Token![!] = input.parse()?;
        pre_index = true;
    }

    let mut post_index = None;
    if input.peek(Token![,]) {
        let _: Token![,] = input.parse()?;
        let (is_hash, expr) = parse_prefixed_expr(input)?;
        if !is_hash {
            if let Some(reg) = parse_register_expr(&expr) {
                post_index = Some(ParsedPostIndex::Register(reg));
            } else {
                post_index = Some(ParsedPostIndex::Immediate(expr));
            }
        } else {
            post_index = Some(ParsedPostIndex::Immediate(expr));
        }
    }

    Ok(ParsedMemory {
        base,
        offset,
        pre_index,
        post_index,
    })
}

fn parse_register_list_operand(input: ParseStream<'_>) -> Result<ParsedRegisterList> {
    let inner;
    let _ = syn::braced!(inner in input);

    let first_expr = inner.parse::<Expr>()?;
    let (mut first, count) = if let Expr::Binary(binary) = &first_expr {
        if matches!(binary.op, syn::BinOp::Mul(_)) {
            let first = parse_register_expr(&binary.left).ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "expected register expression before '*' in register list",
                )
            })?;
            let amount = (*binary.right).clone();
            (first, quote! { (#amount) })
        } else if matches!(binary.op, syn::BinOp::Sub(_)) {
            let first = parse_register_expr(&binary.left).ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "expected register expression before '-' in register list",
                )
            })?;
            let last = parse_register_expr(&binary.right).ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "expected register expression after '-' in register list",
                )
            })?;
            let start = first.code.clone();
            let end = last.code;
            (
                first,
                quote! {{
                    let __jit_start = (#start) as i16;
                    let __jit_end = (#end) as i16;
                    (__jit_end - __jit_start + 1)
                }},
            )
        } else {
            let first = parse_register_expr(&first_expr).ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "expected register expression in register list",
                )
            })?;
            let mut count: usize = 1;
            while inner.peek(Token![,]) {
                let _: Token![,] = inner.parse()?;
                let _ = parse_required_register(&inner)?;
                count += 1;
            }
            (first, quote! { #count })
        }
    } else {
        let first = parse_register_expr(&first_expr).ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "expected register expression in register list",
            )
        })?;
        let mut count: usize = 1;
        while inner.peek(Token![,]) {
            let _: Token![,] = inner.parse()?;
            let _ = parse_required_register(&inner)?;
            count += 1;
        }
        (first, quote! { #count })
    };

    if !inner.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "unexpected trailing tokens in register list",
        ));
    }

    if input.peek(syn::token::Bracket) {
        let lane_inner;
        let _ = syn::bracketed!(lane_inner in input);
        let lane = lane_inner.parse::<Expr>()?;
        first.lane = Some(lane);
    }

    Ok(ParsedRegisterList { first, count })
}

fn parse_condition_expr(expr: &Expr) -> Option<ConditionAst> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    if path.segments.len() != 1 {
        return None;
    }

    condition_from_name(&path.segments[0].ident.to_string().to_ascii_lowercase())
}

fn condition_from_name(name: &str) -> Option<ConditionAst> {
    match name {
        "eq" => Some(ConditionAst::Eq),
        "ne" => Some(ConditionAst::Ne),
        "cs" | "hs" => Some(ConditionAst::Cs),
        "cc" | "lo" => Some(ConditionAst::Cc),
        "mi" => Some(ConditionAst::Mi),
        "pl" => Some(ConditionAst::Pl),
        "vs" => Some(ConditionAst::Vs),
        "vc" => Some(ConditionAst::Vc),
        "hi" => Some(ConditionAst::Hi),
        "ls" => Some(ConditionAst::Ls),
        "ge" => Some(ConditionAst::Ge),
        "lt" => Some(ConditionAst::Lt),
        "gt" => Some(ConditionAst::Gt),
        "le" => Some(ConditionAst::Le),
        "al" => Some(ConditionAst::Al),
        "nv" => Some(ConditionAst::Nv),
        _ => None,
    }
}

fn parse_sysreg_expr(expr: &Expr) -> Option<ParsedSysReg> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => {
            if path.segments.len() != 1 {
                return None;
            }
            let ident = path.segments[0].ident.to_string();
            let [op0, op1, crn, crm, op2] = parse_sysreg_literal(&ident)?;
            Some(ParsedSysReg {
                op0: quote! { #op0 },
                op1: quote! { #op1 },
                crn: quote! { #crn },
                crm: quote! { #crm },
                op2: quote! { #op2 },
            })
        }
        Expr::Call(ExprCall { func, args, .. }) => {
            let Expr::Path(ExprPath { path, .. }) = &**func else {
                return None;
            };
            if path.segments.len() != 1 {
                return None;
            }
            let func_name = path.segments[0].ident.to_string().to_ascii_lowercase();
            if func_name != "sys" || args.len() != 5 {
                return None;
            }

            let mut it = args.iter();
            let op0 = it.next()?.clone();
            let op1 = it.next()?.clone();
            let crn = it.next()?.clone();
            let crm = it.next()?.clone();
            let op2 = it.next()?.clone();

            Some(ParsedSysReg {
                op0: quote! { ((#op0) as u8) },
                op1: quote! { ((#op1) as u8) },
                crn: quote! { ((#crn) as u8) },
                crm: quote! { ((#crm) as u8) },
                op2: quote! { ((#op2) as u8) },
            })
        }
        _ => None,
    }
}

fn parse_sysreg_literal(text: &str) -> Option<[u8; 5]> {
    let lower = text.to_ascii_lowercase();
    let mut parts = lower.split('_');

    let op0 = parts.next()?;
    let op1 = parts.next()?;
    let crn = parts.next()?;
    let crm = parts.next()?;
    let op2 = parts.next()?;
    if parts.next().is_some() {
        return None;
    }

    let op0 = op0.strip_prefix('s')?.parse::<u8>().ok()?;
    let op1 = op1.parse::<u8>().ok()?;
    let crn = crn.strip_prefix('c')?.parse::<u8>().ok()?;
    let crm = crm.strip_prefix('c')?.parse::<u8>().ok()?;
    let op2 = op2.parse::<u8>().ok()?;
    Some([op0, op1, crn, crm, op2])
}

fn parse_register_expr(expr: &Expr) -> Option<ParsedRegister> {
    match expr {
        Expr::Index(ExprIndex { expr, index, .. }) => {
            let mut reg = parse_register_expr(expr)?;
            reg.lane = Some((**index).clone());
            Some(reg)
        }
        Expr::Field(ExprField { base, member, .. }) => {
            let mut reg = parse_register_expr(base)?;
            let Member::Named(member_ident) = member else {
                return None;
            };

            let member_name = member_ident.to_string();
            let lower = member_name.to_ascii_lowercase();
            if let Some(spec) = parse_arrangement_spec(&lower) {
                reg.arrangement = Some(spec);
                return Some(reg);
            }

            if let Some(class) = parse_plain_vector_class(&lower) {
                reg.class = class;
                return Some(reg);
            }

            None
        }
        Expr::Path(path) => parse_register_literal_expr(path),
        Expr::Call(call) => parse_dynamic_register_expr(call),
        _ => None,
    }
}

fn parse_arrangement_spec(text: &str) -> Option<ArrangementAst> {
    match text {
        "b8" => Some(ArrangementAst::B8),
        "b16" => Some(ArrangementAst::B16),
        "h4" => Some(ArrangementAst::H4),
        "h8" => Some(ArrangementAst::H8),
        "s2" => Some(ArrangementAst::S2),
        "s4" => Some(ArrangementAst::S4),
        "d1" => Some(ArrangementAst::D1),
        "d2" => Some(ArrangementAst::D2),
        "q1" => Some(ArrangementAst::Q1),
        _ => None,
    }
}

fn parse_plain_vector_class(text: &str) -> Option<&'static str> {
    match text {
        "b" => Some("B"),
        "h" => Some("H"),
        "s" => Some("S"),
        "d" => Some("D"),
        "q" => Some("Q"),
        _ => None,
    }
}

fn parse_register_literal_expr(expr: &ExprPath) -> Option<ParsedRegister> {
    if expr.path.segments.len() != 1 {
        return None;
    }

    let ident = expr.path.segments[0].ident.to_string();
    let parsed = parse_register_literal(&ident)?;
    let code = parsed.code;
    let class = parsed.class;
    Some(ParsedRegister {
        code: quote! { #code },
        class,
        arrangement: None,
        lane: None,
    })
}

struct ParsedRegisterLiteral {
    code: u8,
    class: &'static str,
}

fn parse_dynamic_register_expr(call: &ExprCall) -> Option<ParsedRegister> {
    let Expr::Path(ExprPath { path, .. }) = &*call.func else {
        return None;
    };
    if path.segments.len() != 1 || call.args.len() != 1 {
        return None;
    }

    let class = match path.segments[0]
        .ident
        .to_string()
        .to_ascii_uppercase()
        .as_str()
    {
        "X" => "X",
        "W" => "W",
        "XSP" => "Xsp",
        "WSP" => "Wsp",
        "V" => "V",
        "B" => "B",
        "H" => "H",
        "S" => "S",
        "D" => "D",
        "Q" => "Q",
        "Z" => "Z",
        "P" | "PN" => "P",
        _ => return None,
    };

    let arg = call.args.first()?.clone();
    Some(ParsedRegister {
        code: quote! { ((#arg) as u8) & 31 },
        class,
        arrangement: None,
        lane: None,
    })
}

fn parse_register_literal(ident: &str) -> Option<ParsedRegisterLiteral> {
    let lower = ident.to_ascii_lowercase();
    match lower.as_str() {
        "sp" => {
            return Some(ParsedRegisterLiteral {
                code: 31,
                class: "Xsp",
            });
        }
        "wsp" => {
            return Some(ParsedRegisterLiteral {
                code: 31,
                class: "Wsp",
            });
        }
        "xzr" => {
            return Some(ParsedRegisterLiteral {
                code: 31,
                class: "X",
            });
        }
        "wzr" => {
            return Some(ParsedRegisterLiteral {
                code: 31,
                class: "W",
            });
        }
        _ => {}
    }

    parse_prefixed_register(&lower, "x")
        .map(|code| ParsedRegisterLiteral { code, class: "X" })
        .or_else(|| {
            parse_prefixed_register(&lower, "w")
                .map(|code| ParsedRegisterLiteral { code, class: "W" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "v")
                .map(|code| ParsedRegisterLiteral { code, class: "V" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "b")
                .map(|code| ParsedRegisterLiteral { code, class: "B" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "h")
                .map(|code| ParsedRegisterLiteral { code, class: "H" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "s")
                .map(|code| ParsedRegisterLiteral { code, class: "S" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "d")
                .map(|code| ParsedRegisterLiteral { code, class: "D" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "q")
                .map(|code| ParsedRegisterLiteral { code, class: "Q" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "z")
                .map(|code| ParsedRegisterLiteral { code, class: "Z" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "pn")
                .map(|code| ParsedRegisterLiteral { code, class: "P" })
        })
        .or_else(|| {
            parse_prefixed_register(&lower, "p")
                .map(|code| ParsedRegisterLiteral { code, class: "P" })
        })
}

fn parse_prefixed_register(ident: &str, prefix: &str) -> Option<u8> {
    let digits = ident.strip_prefix(prefix)?;
    if digits.is_empty() {
        return None;
    }

    let value: u8 = digits.parse().ok()?;
    if value <= 31 { Some(value) } else { None }
}
