use arm64jit::{AssembleError, CodeWriter, jit};

fn dump_machine_code(code: &[u32]) {
    for (idx, word) in code.iter().copied().enumerate() {
        println!("{idx:02}: 0x{word:08x}");
    }
}

fn main() -> Result<(), AssembleError> {
    let mut storage = [0u32; 128];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; fib:
        ; cmp x0, 1
        ; ble 1f
        ; stp x27, x28, [sp, -32]!
        ; str x30, [sp, 16]
        ; mov x28, 0
        ; sub x27, x0, 1
        ; 2:
        ; mov x0, x27
        ; bl <fib
        ; add x28, x28, x0
        ; subs x27, x27, 2
        ; bgt 2b
        ; cinc x0, x28, eq
        ; ldr x30, [sp, 16]
        ; ldp x27, x28, [sp], 32
        ; 1:
        ; ret
    )?;

    let code_len = ops.pos();
    drop(ops);
    let code = &storage[..code_len];
    println!("encoded {} instructions", code.len());
    dump_machine_code(code);
    Ok(())
}
