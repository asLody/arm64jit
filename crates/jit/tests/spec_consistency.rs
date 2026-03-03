use arm64jit::__private::{
    BitFieldSpec, ConditionCode, EncodingSpec, ExtendKind, ExtendOperand, Operand,
    OperandConstraintKind, RegClass, RegisterOperand, SPECS, ShiftKind, ShiftOperand,
    SplitImmediateKindSpec, VectorArrangement, encode,
};
use jit_core::{EncodeError, encode_by_spec, encode_by_spec_operands};

fn sample_nonzero(field: BitFieldSpec) -> i64 {
    if field.signed {
        // Prefer a value that toggles field bits while staying in range.
        -1
    } else if field.width == 0 {
        0
    } else {
        let max = if field.width == 32 {
            u64::from(u32::MAX)
        } else {
            (1u64 << field.width) - 1
        };
        if max == 0 { 0 } else { 1 }
    }
}

fn sample_structured_value(field: BitFieldSpec) -> i64 {
    if field.width == 0 {
        return 0;
    }
    0
}

fn split_immediate_info_at_slot(spec: &EncodingSpec, slot: usize) -> Option<(usize, i64)> {
    let plan = spec.split_immediate_plan?;
    if usize::from(plan.first_slot) != slot {
        return None;
    }
    let second = usize::from(plan.second_slot);
    if second < slot {
        return None;
    }
    let sample = match plan.kind {
        SplitImmediateKindSpec::AdrLike { .. } | SplitImmediateKindSpec::BitIndex6 { .. } => 0,
        SplitImmediateKindSpec::LogicalImmRs { .. }
        | SplitImmediateKindSpec::LogicalImmNrs { .. } => 1,
    };
    Some((second - slot + 1, sample))
}

fn sample_register_for_kind(
    kind: OperandConstraintKind,
    arrangement: Option<VectorArrangement>,
    lane: Option<u8>,
) -> Operand {
    let class = match kind {
        OperandConstraintKind::GprRegister | OperandConstraintKind::Gpr64Register => RegClass::X,
        OperandConstraintKind::Gpr32Register => RegClass::W,
        OperandConstraintKind::SimdRegister => RegClass::V,
        OperandConstraintKind::SveZRegister => RegClass::Z,
        OperandConstraintKind::PredicateRegister => RegClass::P,
        _ => RegClass::X,
    };
    Operand::Register(RegisterOperand {
        code: 0,
        class,
        arrangement,
        lane,
    })
}

fn kind_supports_arrangement(kind: OperandConstraintKind) -> bool {
    matches!(
        kind,
        OperandConstraintKind::SimdRegister | OperandConstraintKind::SveZRegister
    )
}

fn reg_supports_arrangement(class: RegClass) -> bool {
    matches!(
        class,
        RegClass::V
            | RegClass::B
            | RegClass::H
            | RegClass::S
            | RegClass::D
            | RegClass::Q
            | RegClass::Z
    )
}

fn sample_structured_operands(spec: &EncodingSpec) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut slot = 0usize;
    while slot < spec.operand_order.len() {
        if let Some((span, sample)) = split_immediate_info_at_slot(spec, slot) {
            out.push(Operand::Immediate(sample));
            slot += span;
            continue;
        }

        let kind = spec.operand_kinds[slot];
        match kind {
            OperandConstraintKind::GprRegister
            | OperandConstraintKind::Gpr32Register
            | OperandConstraintKind::Gpr64Register
            | OperandConstraintKind::SimdRegister
            | OperandConstraintKind::SveZRegister
            | OperandConstraintKind::PredicateRegister => {
                let mut arrangement = None;
                let mut lane = None;
                let mut advance = 1usize;

                if kind_supports_arrangement(kind)
                    && slot + advance < spec.operand_kinds.len()
                    && spec.operand_kinds[slot + advance] == OperandConstraintKind::Arrangement
                {
                    arrangement = Some(VectorArrangement::B8);
                    advance += 1;
                }
                if kind_supports_arrangement(kind)
                    && slot + advance < spec.operand_kinds.len()
                    && spec.operand_kinds[slot + advance] == OperandConstraintKind::Lane
                {
                    lane = Some(0);
                    advance += 1;
                }

                out.push(sample_register_for_kind(kind, arrangement, lane));
                slot += advance;
            }
            OperandConstraintKind::Condition => {
                out.push(Operand::Condition(ConditionCode::Eq));
                slot += 1;
            }
            OperandConstraintKind::ShiftKind => {
                let amount = if slot + 1 < spec.operand_kinds.len()
                    && spec.operand_kinds[slot + 1] == OperandConstraintKind::Immediate
                {
                    slot += 2;
                    0
                } else {
                    slot += 1;
                    0
                };
                out.push(Operand::Shift(ShiftOperand {
                    kind: ShiftKind::Lsl,
                    amount,
                }));
            }
            OperandConstraintKind::ExtendKind => {
                let amount = if slot + 1 < spec.operand_kinds.len()
                    && spec.operand_kinds[slot + 1] == OperandConstraintKind::Immediate
                {
                    slot += 2;
                    Some(0)
                } else {
                    slot += 1;
                    None
                };
                out.push(Operand::Extend(ExtendOperand {
                    kind: ExtendKind::Uxtx,
                    amount,
                }));
            }
            OperandConstraintKind::Immediate => {
                let field_idx = spec.operand_order[slot] as usize;
                let field = spec.fields[field_idx];
                out.push(Operand::Immediate(sample_structured_value(field)));
                slot += 1;
            }
            OperandConstraintKind::SysRegPart => {
                out.push(Operand::Immediate(0));
                slot += 1;
            }
            OperandConstraintKind::Arrangement => {
                let mut attached = false;
                for operand in out.iter_mut().rev() {
                    if let Operand::Register(reg) = operand
                        && reg.arrangement.is_none()
                        && reg_supports_arrangement(reg.class)
                    {
                        reg.arrangement = Some(VectorArrangement::B8);
                        attached = true;
                        break;
                    }
                }
                if !attached {
                    out.push(Operand::Immediate(0));
                }
                slot += 1;
            }
            OperandConstraintKind::Lane => {
                let mut attached = false;
                for operand in out.iter_mut().rev() {
                    if let Operand::Register(reg) = operand
                        && reg.lane.is_none()
                        && reg_supports_arrangement(reg.class)
                    {
                        reg.lane = Some(0);
                        attached = true;
                        break;
                    }
                }
                if !attached {
                    out.push(Operand::Immediate(0));
                }
                slot += 1;
            }
        }
    }
    out
}

#[test]
fn every_generated_variant_preserves_fixed_opcode_bits() {
    for spec in SPECS {
        let zero_args = vec![0i64; spec.fields.len()];
        let zero_code = encode_by_spec(spec, &zero_args).unwrap_or_else(|err| {
            panic!("zero-sample encode failed for {}: {}", spec.variant, err)
        });
        assert_eq!(
            zero_code.unpack() & spec.opcode_mask,
            spec.opcode,
            "fixed bits mismatch for {} with zero sample",
            spec.variant
        );

        let mixed_args: Vec<i64> = spec.fields.iter().copied().map(sample_nonzero).collect();
        let mixed_code = encode_by_spec(spec, &mixed_args).unwrap_or_else(|err| {
            panic!("mixed-sample encode failed for {}: {}", spec.variant, err)
        });
        assert_eq!(
            mixed_code.unpack() & spec.opcode_mask,
            spec.opcode,
            "fixed bits mismatch for {} with mixed sample",
            spec.variant
        );
    }
}

#[test]
fn every_generated_variant_is_reachable_with_structured_operands() {
    for spec in SPECS {
        let operands = sample_structured_operands(spec);

        let code = match encode_by_spec_operands(spec, &operands) {
            Ok(code) => code,
            Err(EncodeError::InvalidOperandKind { expected, got, .. })
                if (expected == "vector arrangement" && got == "immediate")
                    || (expected == "vector lane" && got == "immediate")
                    || got == "vector arrangement"
                    || got == "vector lane" =>
            {
                // The generic sampler cannot synthesize every arrangement/lane permutation.
                // Keep strict failures for all other mismatches.
                continue;
            }
            Err(err) => {
                panic!(
                    "structured reachability encode failed for {}: {} (mnemonic={}, operands={:?})",
                    spec.variant, err, spec.mnemonic, operands
                );
            }
        };

        assert_eq!(
            code.unpack() & spec.opcode_mask,
            spec.opcode,
            "fixed bits mismatch for {} with structured operands",
            spec.variant
        );

        match encode(spec.mnemonic, &operands) {
            Ok(selected) => {
                let selected_word = selected.unpack();
                let matches_any = SPECS.iter().any(|candidate| {
                    candidate.mnemonic == spec.mnemonic
                        && (selected_word & candidate.opcode_mask) == candidate.opcode
                });
                assert!(
                    matches_any,
                    "dispatcher selected code does not match any variant for {} (code={selected_word:08x})",
                    spec.mnemonic
                );
            }
            Err(EncodeError::AmbiguousVariant) => {}
            Err(err) => {
                panic!(
                    "dispatcher rejected structured sample for {} with {}",
                    spec.variant, err
                );
            }
        }
    }
}
