use arm64jit::{AssembleError, CodeWriter, jit};

fn main() -> Result<(), AssembleError> {
    let mut storage = [0u32; 1];
    let mut ops = CodeWriter::new(&mut storage);

    jit!(ops
        ; mov W(12), W(13)
    )?;

    let code_len = ops.pos();
    drop(ops);
    println!("{:08x?}", &storage[..code_len]);
    Ok(())
}
