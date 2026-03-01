use arm64jit::__private::{
    AddressingMode, MemoryOffset, MemoryOperand, Operand, RegClass, RegisterOperand, encode,
};
use std::hint::black_box;

#[inline]
fn x(code: u8) -> Operand {
    Operand::Register(RegisterOperand {
        code,
        class: RegClass::X,
        arrangement: None,
        lane: None,
    })
}

#[inline]
fn imm(value: i64) -> Operand {
    Operand::Immediate(value)
}

#[inline]
fn mem_x_imm(base: u8, offset: i64) -> Operand {
    Operand::Memory(MemoryOperand {
        base: RegisterOperand {
            code: base,
            class: RegClass::X,
            arrangement: None,
            lane: None,
        },
        offset: MemoryOffset::Immediate(offset),
        addressing: AddressingMode::Offset,
    })
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<u64>().ok())
        .unwrap_or(3_000_000);

    let mut checksum = 0u64;
    for i in 0..iters {
        let imm12 = (i as i64) & 0xfff;
        let ldst_scaled = ((i as i64) & 0x1ff) * 8;
        let bit_idx = (i & 63) as i64;
        let branch = (((i as i64) & 0x7ff) - 0x400) * 4;

        let add = encode("add", &[x(1), x(2), imm(imm12)]).expect("add");
        checksum ^= u64::from(add.unpack());

        let sub = encode("sub", &[x(3), x(4), imm(imm12)]).expect("sub");
        checksum = checksum.rotate_left(5) ^ u64::from(sub.unpack());

        let ldr = encode("ldr", &[x(5), mem_x_imm(6, ldst_scaled)]).expect("ldr");
        checksum = checksum.rotate_left(7) ^ u64::from(ldr.unpack());

        let str_ = encode("str", &[x(7), mem_x_imm(8, ldst_scaled)]).expect("str");
        checksum = checksum.rotate_left(11) ^ u64::from(str_.unpack());

        let tbnz = encode("tbnz", &[x(9), imm(bit_idx), imm(branch)]).expect("tbnz");
        checksum = checksum.rotate_left(13) ^ u64::from(tbnz.unpack());

        let ble = encode("ble", &[imm(branch)]).expect("ble");
        checksum = checksum.rotate_left(17) ^ u64::from(ble.unpack());
    }

    black_box(checksum);
    println!("encode_hotpath checksum={checksum} iters={iters}");
}
