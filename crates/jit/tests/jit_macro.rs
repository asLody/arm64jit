use arm64jit::__private::{
    AddressingMode, ConditionCode, EncodeError, MemoryOffset, MemoryOperand, Operand, RegClass,
    RegisterOperand, emit_mov_imm_auto, encode,
};
use arm64jit::{AssembleError, CodeWriter, jit};

fn emitted(writer: &CodeWriter<'_>) -> Vec<u32> {
    let mut out = Vec::with_capacity(writer.pos());
    for idx in 0..writer.pos() {
        out.push(writer.read_u32_at(idx).expect("read emitted word"));
    }
    out
}

fn word_at(code: &[u32], index: usize) -> u32 {
    code[index]
}

fn w(code: u8) -> Operand {
    Operand::Register(RegisterOperand {
        code,
        class: RegClass::W,
        arrangement: None,
        lane: None,
    })
}

fn x(code: u8) -> Operand {
    Operand::Register(RegisterOperand {
        code,
        class: RegClass::X,
        arrangement: None,
        lane: None,
    })
}

fn imm(value: i64) -> Operand {
    Operand::Immediate(value)
}

fn mem_x(base: u8) -> Operand {
    Operand::Memory(MemoryOperand {
        base: RegisterOperand {
            code: base,
            class: RegClass::X,
            arrangement: None,
            lane: None,
        },
        offset: MemoryOffset::None,
        addressing: AddressingMode::Offset,
    })
}

#[test]
fn block_emits_multiple_instructions() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; add x1, x2, #1
        ; b #8
    )
    .expect("block should encode");

    let code = emitted(&ops);
    assert_eq!(ops.pos(), 2);
    assert_eq!(word_at(&code, 0), 0x9100_0441);
    assert_eq!(word_at(&code, 1), 0x1400_0002);
}

#[test]
fn block_supports_dynamic_and_sve_predicates() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    let pd = 1u8;
    let pg = 2u8;
    let pn = 3u8;
    let pm = 4u8;

    jit!(ops
        ; sel P(pd), P(pg), P(pn), P(pm)
    )
    .expect("block should encode");

    assert_eq!(ops.pos(), 1);
    let code = emitted(&ops);
    assert_eq!(
        word_at(&code, 0),
        0x2500_4210u32 | (4u32 << 16) | (2u32 << 10) | (3u32 << 5) | 1u32
    );
}

#[test]
fn block_supports_raw_bytes_directive() {
    let mut storage = [0u32; 2];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; .bytes [0xaa_u8, 0xbb_u8, 0xcc_u8, 0xdd_u8]
        ; b #0
    )
    .expect("bytes directive should work");

    let code = emitted(&ops);
    assert_eq!(code, &[0xddcc_bbaa, 0x1400_0000]);
}

#[test]
fn block_reports_buffer_overflow() {
    let mut storage = [0u32; 1];
    let mut ops = CodeWriter::new(&mut storage);

    let err = jit!(ops
        ; b #0
        ; b #4
    )
    .expect_err("second instruction must overflow");

    assert_eq!(
        err,
        AssembleError::BufferOverflow {
            needed: 1,
            remaining: 0,
        }
    );
}

#[test]
fn block_resolves_directional_labels() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; top:
        ; b >exit
        ; b <top
        ; exit:
        ; b <top
    )
    .expect("block should encode");

    let code = emitted(&ops);
    assert_eq!(ops.pos(), 3);
    assert_eq!(word_at(&code, 0), 0x1400_0002);
    assert_eq!(word_at(&code, 1), 0x17ff_ffff);
    assert_eq!(word_at(&code, 2), 0x17ff_fffe);
}

#[test]
fn block_patches_cbnz_with_label() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; top:
        ; b #0
        ; cbnz w1, <top
    )
    .expect("cbnz should patch");

    let expected = encode("cbnz", &[w(1), Operand::Immediate(-4)]).expect("encode expected");
    let code = emitted(&ops);

    assert_eq!(word_at(&code, 1), expected.unpack());
}

#[test]
fn block_supports_condition_operands() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; csel x0, x1, x2, eq
    )
    .expect("condition operands should parse and encode");

    assert_eq!(ops.pos(), 1);
}

#[test]
fn block_supports_asm_aliases_and_numeric_labels() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; 2:
        ; cmp x0, 1
        ; ble 1f
        ; mov x28, 0
        ; mov x0, x27
        ; cinc x0, x28, eq
        ; bgt 2b
        ; 1:
        ; ret
    )
    .expect("alias syntax should encode");

    let code = emitted(&ops);
    assert_eq!(ops.pos(), 7);
    assert_eq!(
        word_at(&code, 0),
        encode("subs", &[x(31), x(0), imm(1)])
            .expect("subs")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 1),
        encode("b", &[Operand::Condition(ConditionCode::Le), imm(20)])
            .expect("b.le")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 2),
        encode("add", &[x(28), x(31), imm(0)])
            .expect("mov->add")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 3),
        encode("add", &[x(0), x(27), imm(0)])
            .expect("mov->add")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 4),
        encode(
            "csinc",
            &[x(0), x(28), x(28), Operand::Condition(ConditionCode::Ne),],
        )
        .expect("cinc->csinc")
        .unpack()
    );
    assert_eq!(
        word_at(&code, 5),
        encode("b", &[Operand::Condition(ConditionCode::Gt), imm(-20)])
            .expect("b.gt")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 6),
        encode("ret", &[x(30)]).expect("ret").unpack()
    );
}

#[test]
fn block_supports_uppercase_mov_pseudo_immediate() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; MOV x0, 0x1234_5678_9abc_def0_i64
        ; MOV w1, -1
    )
    .expect("uppercase MOV pseudo should encode");

    let mut expected_storage = [0u32; 16];
    let mut expected = CodeWriter::new(&mut expected_storage);
    emit_mov_imm_auto(
        &mut expected,
        RegisterOperand {
            code: 0,
            class: RegClass::X,
            arrangement: None,
            lane: None,
        },
        0x1234_5678_9abc_def0_i64,
    )
    .expect("expected x MOV materialization");
    emit_mov_imm_auto(
        &mut expected,
        RegisterOperand {
            code: 1,
            class: RegClass::W,
            arrangement: None,
            lane: None,
        },
        -1,
    )
    .expect("expected w MOV materialization");

    assert_eq!(emitted(&ops), emitted(&expected));
}

#[test]
fn block_supports_mov_alias_with_w_dynamic_registers() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; mov W(12), W(13)
    )
    .expect("mov W(d), W(s) should encode without recursion");

    assert_eq!(ops.pos(), 1);
    let code = emitted(&ops);
    assert_eq!(
        word_at(&code, 0),
        encode("add", &[w(12), w(13), imm(0)])
            .expect("mov alias canonical add")
            .unpack()
    );
}

#[test]
fn block_normalizes_addsub_register_shift_forms() {
    let mut storage = [0u32; 4];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; add x28, x28, x0
        ; sub x3, x4, x5, lsl #7
    )
    .expect("gpr add/sub forms should normalize to imm6");

    let code = emitted(&ops);
    assert_eq!(ops.pos(), 2);
    assert_eq!(
        word_at(&code, 0),
        encode("add", &[x(28), x(28), x(0), imm(0)])
            .expect("add shifted register")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 1),
        encode("sub", &[x(3), x(4), x(5), imm(7)])
            .expect("sub shifted register")
            .unpack()
    );
}

#[test]
fn block_supports_broader_asm_forms() {
    macro_rules! assert_ok_block {
        ($($tt:tt)*) => {{
            let mut storage = [0u32; 16];
            let mut ops = CodeWriter::new(&mut storage);
            let result = jit!(ops ; $($tt)*);
            assert!(
                result.is_ok(),
                "failed block `{}` with {:?}",
                stringify!($($tt)*),
                result.err()
            );
        }};
    }

    assert_ok_block!(add x0, x1, x2, lsl #1);
    assert_ok_block!(add x3, x4, w5, uxtw #2);
    assert_ok_block!(ldr x6, [x7]);
    assert_ok_block!(ldr x8, [x9, #16]);
    assert_ok_block!(ldr x10, [x11], #8);
    assert_ok_block!(str x12, [x13, #24]);
    assert_ok_block!(stp x14, x15, [sp, -16]!);
    assert_ok_block!(ldp x14, x15, [sp], 16);
    assert_ok_block!(csel x2, x3, x4, lt);
    assert_ok_block!(csinc x5, x6, x7, ne);
    assert_ok_block!(mrs x8, S3_0_C15_C2_0);
    assert_ok_block!(msr S3_0_C15_C2_0, x8);

    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);
    jit!(ops
        ; top:
        ; cbnz w0, <top
        ; tbnz x1, #3, <top
    )
    .expect("label-based cbnz/tbnz should parse and encode");
}

#[test]
fn block_supports_add_extended_w_operand() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; add x3, x4, w5, uxtw #2
    )
    .expect("w-extended source form should encode");
}

#[test]
fn block_supports_implicit_zero_offset_memory() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; ldr x6, [x7]
    )
    .expect("implicit #0 memory offset form should encode");
}

#[test]
fn block_supports_msr_asm_order() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; msr S3_0_C15_C2_0, x8
    )
    .expect("msr asm operand order should encode");
}

#[test]
fn block_supports_tbnz_with_bit_index() {
    let mut storage = [0u32; 16];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; top:
        ; tbnz x1, #3, <top
    )
    .expect("tbnz operand mapping should encode");
}

#[test]
fn block_supports_ldar_with_dynamic_registers() {
    let mut storage = [0u32; 8];
    let mut ops = CodeWriter::new(&mut storage);
    let rt = 0u8;
    let rn = 1u8;

    jit!(ops
        ; ldar X(rt), [X(rn)]
    )
    .expect("ldar x dynamic regs should encode");

    let code = emitted(&ops);
    assert_eq!(
        word_at(&code, 0),
        encode("ldar", &[x(rt), mem_x(rn)])
            .expect("expected ldar x")
            .unpack()
    );
}

#[test]
fn block_supports_ldxr_with_dynamic_register_widths() {
    let mut storage = [0u32; 8];
    let mut ops = CodeWriter::new(&mut storage);
    let rt_w = 0u8;
    let rt_x = 2u8;
    let rn = 1u8;

    jit!(ops
        ; ldxr W(rt_w), [X(rn)]
        ; ldxr X(rt_x), [X(rn)]
    )
    .expect("ldxr dynamic regs should encode");

    let code = emitted(&ops);
    assert_eq!(
        word_at(&code, 0),
        encode("ldxr", &[w(rt_w), mem_x(rn)])
            .expect("expected ldxr w")
            .unpack()
    );
    assert_eq!(
        word_at(&code, 1),
        encode("ldxr", &[x(rt_x), mem_x(rn)])
            .expect("expected ldxr x")
            .unpack()
    );
}

#[test]
fn block_supports_stlr_with_dynamic_and_static_registers() {
    let mut storage = [0u32; 8];
    let mut ops = CodeWriter::new(&mut storage);
    let src = 0u8;
    let addr = 1u8;

    jit!(ops
        ; stlr X(src), [X(addr)]
        ; stlr x0, [x1]
    )
    .expect("stlr forms should encode");

    let code = emitted(&ops);
    assert_eq!(word_at(&code, 0), 0xc89f_fc20);
    assert_eq!(word_at(&code, 1), 0xc89f_fc20);
}

#[test]
fn block_rejects_stlr_with_explicit_zero_offset() {
    let mut storage = [0u32; 8];
    let mut ops = CodeWriter::new(&mut storage);

    let err = jit!(ops
        ; stlr x0, [x1, #0]
    )
    .expect_err("stlr no-offset form must reject explicit #0");

    match err {
        AssembleError::Encode(EncodeError::OperandCountRange { .. }) => {
            panic!("stlr [x1,#0] must not be reported as operand-count-range mismatch");
        }
        AssembleError::Encode(EncodeError::NoMatchingVariant)
        | AssembleError::Encode(EncodeError::NoMatchingVariantHint { .. }) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}
