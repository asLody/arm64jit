use crate::ast::{
    ArrangementAst, ConditionAst, ExtendKindAst, InstructionStmt, JitArg, JitBlockInput, JitStmt,
    LabelDirection, OperandAst, ParsedMemoryOffset, ParsedModifier, ParsedPostIndex,
    ParsedRegister, ShiftKindAst,
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::HashMap;

fn class_ident_tokens(class: &'static str) -> TokenStream2 {
    match class {
        "W" => quote! { ::arm64jit::__private::RegClass::W },
        "X" => quote! { ::arm64jit::__private::RegClass::X },
        "Wsp" => quote! { ::arm64jit::__private::RegClass::Wsp },
        "Xsp" => quote! { ::arm64jit::__private::RegClass::Xsp },
        "V" => quote! { ::arm64jit::__private::RegClass::V },
        "B" => quote! { ::arm64jit::__private::RegClass::B },
        "H" => quote! { ::arm64jit::__private::RegClass::H },
        "S" => quote! { ::arm64jit::__private::RegClass::S },
        "D" => quote! { ::arm64jit::__private::RegClass::D },
        "Q" => quote! { ::arm64jit::__private::RegClass::Q },
        "Z" => quote! { ::arm64jit::__private::RegClass::Z },
        "P" => quote! { ::arm64jit::__private::RegClass::P },
        _ => quote! { ::arm64jit::__private::RegClass::X },
    }
}

fn arrangement_tokens(arrangement: Option<ArrangementAst>) -> TokenStream2 {
    match arrangement {
        Some(ArrangementAst::B8) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::B8) }
        }
        Some(ArrangementAst::B16) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::B16) }
        }
        Some(ArrangementAst::H4) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::H4) }
        }
        Some(ArrangementAst::H8) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::H8) }
        }
        Some(ArrangementAst::S2) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::S2) }
        }
        Some(ArrangementAst::S4) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::S4) }
        }
        Some(ArrangementAst::D1) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::D1) }
        }
        Some(ArrangementAst::D2) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::D2) }
        }
        Some(ArrangementAst::Q1) => {
            quote! { ::core::option::Option::Some(::arm64jit::__private::VectorArrangement::Q1) }
        }
        None => quote! { ::core::option::Option::None },
    }
}

fn condition_tokens(condition: ConditionAst) -> TokenStream2 {
    match condition {
        ConditionAst::Eq => quote! { ::arm64jit::__private::ConditionCode::Eq },
        ConditionAst::Ne => quote! { ::arm64jit::__private::ConditionCode::Ne },
        ConditionAst::Cs => quote! { ::arm64jit::__private::ConditionCode::Cs },
        ConditionAst::Cc => quote! { ::arm64jit::__private::ConditionCode::Cc },
        ConditionAst::Mi => quote! { ::arm64jit::__private::ConditionCode::Mi },
        ConditionAst::Pl => quote! { ::arm64jit::__private::ConditionCode::Pl },
        ConditionAst::Vs => quote! { ::arm64jit::__private::ConditionCode::Vs },
        ConditionAst::Vc => quote! { ::arm64jit::__private::ConditionCode::Vc },
        ConditionAst::Hi => quote! { ::arm64jit::__private::ConditionCode::Hi },
        ConditionAst::Ls => quote! { ::arm64jit::__private::ConditionCode::Ls },
        ConditionAst::Ge => quote! { ::arm64jit::__private::ConditionCode::Ge },
        ConditionAst::Lt => quote! { ::arm64jit::__private::ConditionCode::Lt },
        ConditionAst::Gt => quote! { ::arm64jit::__private::ConditionCode::Gt },
        ConditionAst::Le => quote! { ::arm64jit::__private::ConditionCode::Le },
        ConditionAst::Al => quote! { ::arm64jit::__private::ConditionCode::Al },
        ConditionAst::Nv => quote! { ::arm64jit::__private::ConditionCode::Nv },
    }
}

fn shift_kind_tokens(kind: ShiftKindAst) -> TokenStream2 {
    match kind {
        ShiftKindAst::Lsl => quote! { ::arm64jit::__private::ShiftKind::Lsl },
        ShiftKindAst::Lsr => quote! { ::arm64jit::__private::ShiftKind::Lsr },
        ShiftKindAst::Asr => quote! { ::arm64jit::__private::ShiftKind::Asr },
        ShiftKindAst::Ror => quote! { ::arm64jit::__private::ShiftKind::Ror },
        ShiftKindAst::Msl => quote! { ::arm64jit::__private::ShiftKind::Msl },
    }
}

fn extend_kind_tokens(kind: ExtendKindAst) -> TokenStream2 {
    match kind {
        ExtendKindAst::Uxtb => quote! { ::arm64jit::__private::ExtendKind::Uxtb },
        ExtendKindAst::Uxth => quote! { ::arm64jit::__private::ExtendKind::Uxth },
        ExtendKindAst::Uxtw => quote! { ::arm64jit::__private::ExtendKind::Uxtw },
        ExtendKindAst::Uxtx => quote! { ::arm64jit::__private::ExtendKind::Uxtx },
        ExtendKindAst::Sxtb => quote! { ::arm64jit::__private::ExtendKind::Sxtb },
        ExtendKindAst::Sxth => quote! { ::arm64jit::__private::ExtendKind::Sxth },
        ExtendKindAst::Sxtw => quote! { ::arm64jit::__private::ExtendKind::Sxtw },
        ExtendKindAst::Sxtx => quote! { ::arm64jit::__private::ExtendKind::Sxtx },
    }
}

fn register_tokens(reg: &ParsedRegister) -> TokenStream2 {
    let code = reg.code.clone();
    let class = class_ident_tokens(reg.class);
    let arrangement = arrangement_tokens(reg.arrangement);
    let lane = if let Some(lane_expr) = &reg.lane {
        quote! { ::core::option::Option::Some((#lane_expr) as u8) }
    } else {
        quote! { ::core::option::Option::None }
    };

    quote! {
        ::arm64jit::__private::RegisterOperand {
            code: ((#code) as u8) & 31,
            class: #class,
            arrangement: #arrangement,
            lane: #lane,
        }
    }
}

fn modifier_tokens(modifier: &ParsedModifier) -> TokenStream2 {
    match modifier {
        ParsedModifier::Shift { kind, amount } => {
            let kind = shift_kind_tokens(*kind);
            let amount = if let Some(expr) = amount {
                quote! { (#expr) as u8 }
            } else {
                quote! { 0u8 }
            };
            quote! {
                ::arm64jit::__private::Modifier::Shift(
                    ::arm64jit::__private::ShiftOperand {
                        kind: #kind,
                        amount: #amount,
                    }
                )
            }
        }
        ParsedModifier::Extend { kind, amount } => {
            let kind = extend_kind_tokens(*kind);
            let amount = if let Some(expr) = amount {
                quote! { ::core::option::Option::Some((#expr) as u8) }
            } else {
                quote! { ::core::option::Option::None }
            };
            quote! {
                ::arm64jit::__private::Modifier::Extend(
                    ::arm64jit::__private::ExtendOperand {
                        kind: #kind,
                        amount: #amount,
                    }
                )
            }
        }
    }
}

fn operand_tokens(operand: OperandAst) -> TokenStream2 {
    match operand {
        OperandAst::Immediate(expr) => {
            quote! { ::arm64jit::__private::Operand::Immediate((#expr) as i64) }
        }
        OperandAst::Register(reg) => {
            let reg = register_tokens(&reg);
            quote! { ::arm64jit::__private::Operand::Register(#reg) }
        }
        OperandAst::Memory(memory) => {
            let base = register_tokens(&memory.base);
            let offset = match memory.offset {
                ParsedMemoryOffset::None => {
                    quote! { ::arm64jit::__private::MemoryOffset::None }
                }
                ParsedMemoryOffset::Immediate(expr) => {
                    quote! { ::arm64jit::__private::MemoryOffset::Immediate((#expr) as i64) }
                }
                ParsedMemoryOffset::Register { reg, modifier } => {
                    let reg = register_tokens(&reg);
                    let modifier = if let Some(modifier) = modifier {
                        let modifier = modifier_tokens(&modifier);
                        quote! { ::core::option::Option::Some(#modifier) }
                    } else {
                        quote! { ::core::option::Option::None }
                    };
                    quote! {
                        ::arm64jit::__private::MemoryOffset::Register {
                            reg: #reg,
                            modifier: #modifier,
                        }
                    }
                }
            };

            let addressing = if let Some(post) = memory.post_index {
                match post {
                    ParsedPostIndex::Immediate(expr) => {
                        quote! {
                            ::arm64jit::__private::AddressingMode::PostIndex(
                                ::arm64jit::__private::PostIndexOffset::Immediate((#expr) as i64)
                            )
                        }
                    }
                    ParsedPostIndex::Register(reg) => {
                        let reg = register_tokens(&reg);
                        quote! {
                            ::arm64jit::__private::AddressingMode::PostIndex(
                                ::arm64jit::__private::PostIndexOffset::Register(#reg)
                            )
                        }
                    }
                }
            } else if memory.pre_index {
                quote! { ::arm64jit::__private::AddressingMode::PreIndex }
            } else {
                quote! { ::arm64jit::__private::AddressingMode::Offset }
            };

            quote! {
                ::arm64jit::__private::Operand::Memory(
                    ::arm64jit::__private::MemoryOperand {
                        base: #base,
                        offset: #offset,
                        addressing: #addressing,
                    }
                )
            }
        }
        OperandAst::Shift { kind, amount } => {
            let kind = shift_kind_tokens(kind);
            let amount = if let Some(expr) = amount {
                quote! { (#expr) as u8 }
            } else {
                quote! { 0u8 }
            };

            quote! {
                ::arm64jit::__private::Operand::Shift(
                    ::arm64jit::__private::ShiftOperand {
                        kind: #kind,
                        amount: #amount,
                    }
                )
            }
        }
        OperandAst::Extend { kind, amount } => {
            let kind = extend_kind_tokens(kind);
            let amount = if let Some(expr) = amount {
                quote! { ::core::option::Option::Some((#expr) as u8) }
            } else {
                quote! { ::core::option::Option::None }
            };

            quote! {
                ::arm64jit::__private::Operand::Extend(
                    ::arm64jit::__private::ExtendOperand {
                        kind: #kind,
                        amount: #amount,
                    }
                )
            }
        }
        OperandAst::Condition(condition) => {
            let condition = condition_tokens(condition);
            quote! { ::arm64jit::__private::Operand::Condition(#condition) }
        }
        OperandAst::RegisterList(list) => {
            let first = register_tokens(&list.first);
            let count = list.count;
            quote! {
                ::arm64jit::__private::Operand::RegisterList(
                    ::arm64jit::__private::RegisterListOperand {
                        first: #first,
                        count: ((#count) as u8),
                    }
                )
            }
        }
        OperandAst::SysReg(sys) => {
            let op0 = sys.op0;
            let op1 = sys.op1;
            let crn = sys.crn;
            let crm = sys.crm;
            let op2 = sys.op2;
            quote! {
                ::arm64jit::__private::Operand::SysReg(
                    ::arm64jit::__private::SysRegOperand {
                        op0: (#op0) as u8,
                        op1: (#op1) as u8,
                        crn: (#crn) as u8,
                        crm: (#crm) as u8,
                        op2: (#op2) as u8,
                    }
                )
            }
        }
    }
}

fn reloc_kind_tokens(mnemonic_id: Option<u16>, args: &[JitArg]) -> Option<TokenStream2> {
    let mnemonic_id = mnemonic_id?;
    let reloc_mask = crate::normalize::mnemonic_reloc_mask(mnemonic_id);

    if (reloc_mask & crate::rules::generated::RELOC_MASK_BCOND19) != 0
        && matches!(
            args.first(),
            Some(JitArg::Operand(OperandAst::Condition(_)))
        )
    {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::BCond19 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_B26) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::B26 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_CBZ19) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::Cbz19 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_IMM19) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::Imm19 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_TBZ14) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::Tbz14 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_ADR21) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::Adr21 });
    }
    if (reloc_mask & crate::rules::generated::RELOC_MASK_ADRP21) != 0 {
        return Some(quote! { ::arm64jit::__private::BranchRelocKind::Adrp21 });
    }
    None
}

fn resolve_directional_label_target(
    defs_by_name: &HashMap<String, Vec<(usize, usize)>>,
    name: &str,
    stmt_idx: usize,
    direction: LabelDirection,
) -> Option<usize> {
    let defs = defs_by_name.get(name)?;
    match direction {
        LabelDirection::Backward => defs
            .iter()
            .rev()
            .find(|(def_stmt, _)| *def_stmt < stmt_idx)
            .map(|(_, id)| *id),
        LabelDirection::Forward => defs
            .iter()
            .find(|(def_stmt, _)| *def_stmt > stmt_idx)
            .map(|(_, id)| *id),
    }
}

pub(crate) fn expand(input: TokenStream) -> TokenStream {
    let JitBlockInput { target, statements } = syn::parse_macro_input!(input as JitBlockInput);

    let mut defs_by_name: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    let mut def_id_by_stmt: Vec<Option<usize>> = vec![None; statements.len()];
    let mut def_count = 0usize;

    for (stmt_idx, stmt) in statements.iter().enumerate() {
        if let JitStmt::StaticLabelDef(name) = stmt {
            let def_id = def_count;
            def_count += 1;
            def_id_by_stmt[stmt_idx] = Some(def_id);
            defs_by_name
                .entry(name.clone())
                .or_default()
                .push((stmt_idx, def_id));
        }
    }

    let mut lowered = Vec::new();
    let mut mnemonic_id_cache = HashMap::<String, u16>::new();
    let mut fixup_slot_count = 0usize;

    for (stmt_idx, stmt) in statements.into_iter().enumerate() {
        match stmt {
            JitStmt::Bytes(expr) => lowered.push(quote! {
                let __jit_bytes_ref = #expr;
                let __jit_bytes = ::core::convert::AsRef::<[u8]>::as_ref(&__jit_bytes_ref);
                if (__jit_bytes.len() & 3usize) != 0 {
                    return ::core::result::Result::Err(
                        ::arm64jit::__private::AssembleError::RawBytesNotWordAligned {
                            len: __jit_bytes.len(),
                        }
                    );
                }
                for __jit_chunk in __jit_bytes.chunks_exact(4) {
                    let __jit_word = u32::from_le_bytes([
                        __jit_chunk[0],
                        __jit_chunk[1],
                        __jit_chunk[2],
                        __jit_chunk[3],
                    ]);
                    __jit_asm.emit_word(__jit_word)?;
                }
            }),
            JitStmt::StaticLabelDef(_) => {
                let Some(def_id) = def_id_by_stmt[stmt_idx] else {
                    let err = syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "internal static label resolution failure",
                    );
                    return err.to_compile_error().into();
                };
                lowered.push(quote! {
                    __jit_label_pos[#def_id] = __jit_asm.pos();
                    __jit_label_bound[#def_id] = true;
                });
            }
            JitStmt::Instruction(inst) => {
                let InstructionStmt {
                    op_name,
                    args: inst_args,
                } = inst;

                if op_name == "__mov_auto" {
                    let [
                        JitArg::Operand(OperandAst::Register(dst)),
                        JitArg::Operand(OperandAst::Immediate(immediate)),
                    ] = inst_args.as_slice()
                    else {
                        let err = syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "MOV pseudo expects exactly: MOV <Wn|Xn>, <immediate>",
                        );
                        return err.to_compile_error().into();
                    };

                    let dst = register_tokens(dst);
                    let immediate = immediate.clone();
                    lowered.push(quote! {
                        ::arm64jit::__private::emit_mov_imm_auto(__jit_asm, #dst, (#immediate) as i64)?;
                    });
                    continue;
                }

                let resolved_mnemonic_id = if let Some(existing) = mnemonic_id_cache.get(&op_name) {
                    *existing
                } else {
                    let Some(resolved) = crate::shape::lookup_mnemonic_id(&op_name) else {
                        let err = syn::Error::new(
                            proc_macro2::Span::call_site(),
                            format!("unknown mnemonic in jit! block: {op_name}"),
                        );
                        return err.to_compile_error().into();
                    };
                    mnemonic_id_cache.insert(op_name.clone(), resolved);
                    resolved
                };
                let reloc_kind_hint = reloc_kind_tokens(Some(resolved_mnemonic_id), &inst_args);
                let direct_variant_id =
                    crate::shape::lookup_direct_variant_id(resolved_mnemonic_id, &inst_args);
                let mut label_fixup_def: Option<usize> = None;
                let mut args = Vec::new();

                for arg in inst_args {
                    match arg {
                        JitArg::DirectionalLabelRef { name, direction } => {
                            if label_fixup_def.is_some() {
                                let err = syn::Error::new(
                                    proc_macro2::Span::call_site(),
                                    "only one label reference is supported per instruction",
                                );
                                return err.to_compile_error().into();
                            }

                            let Some(target_def_id) = resolve_directional_label_target(
                                &defs_by_name,
                                &name,
                                stmt_idx,
                                direction,
                            ) else {
                                let err = syn::Error::new(
                                    proc_macro2::Span::call_site(),
                                    "cannot resolve directional label reference",
                                );
                                return err.to_compile_error().into();
                            };

                            label_fixup_def = Some(target_def_id);
                            args.push(quote! { ::arm64jit::__private::Operand::Immediate(0) });
                        }
                        JitArg::Operand(operand) => args.push(operand_tokens(operand)),
                    }
                }

                let len = args.len();

                if let Some(target_def_id) = label_fixup_def {
                    let Some(reloc_kind) = reloc_kind_hint else {
                        let err = syn::Error::new(
                            proc_macro2::Span::call_site(),
                            "label references are unsupported for this mnemonic",
                        );
                        return err.to_compile_error().into();
                    };

                    let emit_stmt = if let Some(variant_id) = direct_variant_id {
                        quote! {
                            ::arm64jit::__private::emit_variant_const_into::<#variant_id>(
                                __jit_asm,
                                &__jit_args,
                            )?;
                        }
                    } else {
                        quote! {
                            ::arm64jit::__private::emit_mnemonic_id_const_no_alias_into::<#resolved_mnemonic_id>(
                                __jit_asm,
                                &__jit_args,
                            )?;
                        }
                    };

                    let fixup_slot = fixup_slot_count;
                    fixup_slot_count += 1;

                    lowered.push(quote! {
                        let __jit_fixup_from = __jit_asm.pos();
                        let __jit_args: [::arm64jit::__private::Operand; #len] = [#(#args),*];
                        #emit_stmt
                        __jit_fixup_from_pos[#fixup_slot] = __jit_fixup_from;
                        __jit_fixup_target_label[#fixup_slot] = #target_def_id;
                        __jit_fixup_kind[#fixup_slot] = #reloc_kind;
                    });
                } else {
                    let emit_stmt = if let Some(variant_id) = direct_variant_id {
                        quote! {
                            ::arm64jit::__private::emit_variant_const_into::<#variant_id>(
                                __jit_asm,
                                &__jit_args,
                            )?;
                        }
                    } else {
                        quote! {
                            ::arm64jit::__private::emit_mnemonic_id_const_no_alias_into::<#resolved_mnemonic_id>(
                                __jit_asm,
                                &__jit_args,
                            )?;
                        }
                    };

                    lowered.push(quote! {
                        let __jit_args: [::arm64jit::__private::Operand; #len] = [#(#args),*];
                        #emit_stmt
                    });
                }
            }
        }
    }

    let expanded = quote! {{
        (|| -> ::core::result::Result<(), ::arm64jit::__private::AssembleError> {
            let __jit_asm = &mut #target;
            let mut __jit_label_pos: [usize; #def_count] = [0usize; #def_count];
            let mut __jit_label_bound: [bool; #def_count] = [false; #def_count];
            let mut __jit_fixup_from_pos: [usize; #fixup_slot_count] = [0usize; #fixup_slot_count];
            let mut __jit_fixup_target_label: [usize; #fixup_slot_count] = [0usize; #fixup_slot_count];
            let mut __jit_fixup_kind: [::arm64jit::__private::BranchRelocKind; #fixup_slot_count] =
                [::arm64jit::__private::BranchRelocKind::B26; #fixup_slot_count];

            #(#lowered)*

            let mut __jit_idx = 0usize;
            while __jit_idx < #fixup_slot_count {
                let __jit_target_label = __jit_fixup_target_label[__jit_idx];
                if !__jit_label_bound[__jit_target_label] {
                    return ::core::result::Result::Err(
                        ::arm64jit::__private::AssembleError::UnboundLocalLabel,
                    );
                }
                ::arm64jit::__private::patch_relocation(
                    __jit_asm,
                    __jit_fixup_from_pos[__jit_idx],
                    __jit_label_pos[__jit_target_label],
                    __jit_fixup_kind[__jit_idx],
                )?;
                __jit_idx += 1;
            }

            ::core::result::Result::<(), ::arm64jit::__private::AssembleError>::Ok(())
        })()
    }};

    expanded.into()
}
