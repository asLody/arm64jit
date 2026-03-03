# arm64jit

Spec-driven AArch64 JIT assembler for Rust.

`arm64jit` gives you assembly-style authoring with Rust ergonomics, while encoder correctness comes from Arm official machine-readable data.

## AArch64 ISA Coverage

Generated from vendored Arm AARCHMRS data (`AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12`).

| Instruction set area | Status |
| --- | --- |
| A64 core (integer, control flow, load/store) | Full |
| Advanced SIMD / FP (NEON) | Full |
| SVE | Full |
| SME | Full |

## Install

```toml
[dependencies]
arm64jit = "0.3.8"
```

## Usage

```rust
use arm64jit::{AssembleError, CodeWriter, jit};

fn emit(ops: &mut CodeWriter<'_>) -> Result<(), AssembleError> {
    jit!(ops
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
        ; fib:
        ; ret
    )
}

let mut buf = [0u32; 128];
let mut ops = CodeWriter::new(&mut buf);
emit(&mut ops)?;
let code_len = ops.pos();
drop(ops);
let code_words = &buf[..code_len];
# Ok::<(), arm64jit::AssembleError>(())
```

## Syntax Highlights (`jit!`)

- GPR/SIMD/SVE registers: `x0`, `w1`, `v0.b16`, `z0`, `p0`
- Memory forms: `[x0]`, `[x0, #16]`, `[x0, #16]!`, `[x0], #16`, `[x0, x1, lsl #3]`
- Modifiers: `lsl #n`, `lsr #n`, `asr #n`, `uxtw #n`, `sxtw #n`
- Conditions: `eq`, `ne`, `lt`, `ge`, `gt`, `le`, ...
- Labels: symbolic (`loop:` / `<loop`) and numeric (`1:` / `1f` / `1b`)
- Raw bytes: `.bytes [0xaa_u8, 0xbb_u8]`
- Pseudo immediate materialization: `MOV x0, 0x1234_5678_9abc_def0`

## External Dynamic Linker

`CodeWriter` does not embed label/fixup metadata. Dynamic labels are handled by a standalone `Linker`.

```rust
use arm64jit::{BranchRelocKind, CodeWriter, Linker, jit};

let mut storage = [0u32; 4];
let mut code = CodeWriter::new(&mut storage);
let mut linker = Linker::new();

let text = linker.new_block();
let loop_label = linker.new_label();

// Bind label position first.
linker.bind(text, loop_label, code.pos())?;

// Emit placeholder immediate=0, then register a fixup.
let br_at = code.pos();
jit!(code ; b #0 ; ret)?;
linker.add_fixup(text, br_at, loop_label, BranchRelocKind::B26)?;

// Patch in a later stage.
linker.patch_writer(text, &mut code)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

MIT.
