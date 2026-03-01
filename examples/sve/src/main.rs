use arm64jit::{AssembleError, CodeWriter, jit};
use capstone::prelude::*;

fn emit_sve_kernel(mut ops: &mut CodeWriter<'_>) -> Result<(), AssembleError> {
    jit!(ops
        ; eor x10, x10, x10
        ; eor x11, x11, x11
        ; add x11, x11, 8
        ; loop_head:
        ; sel P(2), P(0), P(1), P(0)
        ; eor Z(0), Z(1), Z(2)
        ; and Z(3), Z(0), Z(1)
        ; orr Z(4), Z(3), Z(2)
        ; bic Z(5), Z(4), Z(1)
        ; zip1 Z(6).b16, Z(4), Z(5)
        ; uzp1 Z(7).b16, Z(6), Z(3)
        ; sub x11, x11, 1
        ; cbnz x11, <loop_head
        ; ret
    )?;

    Ok(())
}

fn dump_words(code: &[u32]) {
    println!("machine code words:");
    for (idx, word) in code.iter().copied().enumerate() {
        println!("  {idx:02}: 0x{word:08x}");
    }
}

fn dump_disasm(code_words: &[u32], base: u64) -> Result<(), Box<dyn std::error::Error>> {
    let cs = Capstone::new()
        .arm64()
        .mode(arch::arm64::ArchMode::Arm)
        .build()?;
    let mut code_bytes = Vec::with_capacity(code_words.len() * 4);
    for word in code_words.iter().copied() {
        code_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let insns = cs.disasm_all(&code_bytes, base)?;

    println!("capstone disassembly:");
    for insn in insns.iter() {
        let mnemonic = insn.mnemonic().unwrap_or("<?>");
        let operands = insn.op_str().unwrap_or("");
        println!("  0x{:04x}: {:8} {}", insn.address(), mnemonic, operands);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut storage = [0u32; 128];
    let mut ops = CodeWriter::new(&mut storage);
    let base = 0x4000_u64;

    emit_sve_kernel(&mut ops).map_err(|err| std::io::Error::other(err.to_string()))?;

    let code_len = ops.pos();
    drop(ops);
    let code = &storage[..code_len];
    println!("encoded {} instructions", code.len());
    dump_words(code);
    dump_disasm(code, base)?;

    Ok(())
}
