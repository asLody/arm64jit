use core::fmt;

use crate::encode::{encode, encode_mnemonic_id_const_no_alias, encode_variant_const};
use jit_core::{EncodeError, Operand, RegClass, RegisterOperand};

/// Emission errors for writing encoded instructions into a caller-provided word buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AssembleError {
    /// Encoding failed.
    Encode(EncodeError),
    /// Output buffer does not have enough free words.
    BufferOverflow {
        /// Required words for this write.
        needed: usize,
        /// Remaining free words in the buffer.
        remaining: usize,
    },
    /// Position update would move outside current emitted range.
    PositionOutOfBounds {
        /// Requested word position.
        position: usize,
        /// Current emitted length in words.
        len: usize,
    },
    /// In-place write would exceed current emitted range.
    WriteOutOfBounds {
        /// Word index for the write.
        at: usize,
        /// Current emitted length in words.
        len: usize,
    },
    /// In-place read would exceed current emitted range.
    ReadOutOfBounds {
        /// Word index for the read.
        at: usize,
        /// Current emitted length in words.
        len: usize,
    },
    /// Raw byte directive payload is not aligned to full instruction words.
    RawBytesNotWordAligned {
        /// Input payload length in bytes.
        len: usize,
    },
    /// JIT block contains an unresolved local label.
    UnboundLocalLabel,
    /// Relocation value exceeds the supported signed field width.
    RelocationOutOfRange {
        /// Relocation kind.
        kind: BranchRelocKind,
        /// Source instruction word index.
        from: usize,
        /// Target instruction word index.
        to: usize,
        /// Signed relocation value after kind-specific scaling.
        value: i64,
        /// Signed field width in bits.
        bits: u8,
    },
    /// `MOV` pseudo destination is not a general-purpose data register (`Wn`/`Xn`).
    MovPseudoInvalidDestination {
        /// Destination register class provided by the caller.
        class: RegClass,
    },
    /// `MOV Wn, #imm` immediate does not fit 32-bit materialization range.
    MovPseudoImmediateOutOfRange {
        /// Destination register class (`W` for this error path).
        class: RegClass,
        /// Immediate provided by the caller.
        value: i64,
    },
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "{err}"),
            Self::BufferOverflow { needed, remaining } => {
                write!(
                    f,
                    "buffer overflow while emitting code (needed={}, remaining={})",
                    needed, remaining
                )
            }
            Self::PositionOutOfBounds { position, len } => {
                write!(
                    f,
                    "position {} is outside emitted range (len={})",
                    position, len
                )
            }
            Self::WriteOutOfBounds { at, len } => {
                write!(
                    f,
                    "write index {} is outside emitted range (len={})",
                    at, len
                )
            }
            Self::ReadOutOfBounds { at, len } => {
                write!(
                    f,
                    "read index {} is outside emitted range (len={})",
                    at, len
                )
            }
            Self::RawBytesNotWordAligned { len } => {
                write!(f, "raw bytes length {} is not a multiple of 4", len)
            }
            Self::UnboundLocalLabel => write!(f, "jit block contains an unbound local label"),
            Self::RelocationOutOfRange {
                kind,
                from,
                to,
                value,
                bits,
            } => write!(
                f,
                "relocation {:?} out of range (from={}, to={}, value={}, bits={})",
                kind, from, to, value, bits
            ),
            Self::MovPseudoInvalidDestination { class } => write!(
                f,
                "MOV pseudo destination must be Wn/Xn (got class={:?})",
                class
            ),
            Self::MovPseudoImmediateOutOfRange { class, value } => write!(
                f,
                "MOV pseudo immediate out of range (class={:?}, value={})",
                class, value
            ),
        }
    }
}

impl From<EncodeError> for AssembleError {
    fn from(value: EncodeError) -> Self {
        Self::Encode(value)
    }
}

/// Relocation kinds supported by `jit!` local-label patching.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BranchRelocKind {
    /// `b` / `bl` immediate branch (`imm26`).
    B26,
    /// `b.<cond>` immediate branch (`imm19`).
    BCond19,
    /// `cbz` / `cbnz` branch (`imm19`).
    Cbz19,
    /// Generic `imm19` PC-relative relocation.
    Imm19,
    /// `tbz` / `tbnz` branch (`imm14`).
    Tbz14,
    /// `adr` split immediate relocation.
    Adr21,
    /// `adrp` split immediate relocation.
    Adrp21,
}

/// Thin instruction-word writer for JIT emission.
///
/// This type intentionally owns no label/fixup metadata.
pub struct CodeWriter<'a> {
    buf: &'a mut [u32],
    pos: usize,
}

impl<'a> CodeWriter<'a> {
    /// Creates a new writer over a caller-provided word buffer.
    #[must_use]
    pub fn new(buf: &'a mut [u32]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Returns the current write cursor in words.
    #[must_use]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Returns remaining writable words.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Moves the write cursor to an earlier or equal emitted position.
    ///
    /// # Errors
    ///
    /// Returns [`AssembleError::PositionOutOfBounds`] if `position` is outside emitted range.
    pub fn set_position(&mut self, position: usize) -> Result<(), AssembleError> {
        if position > self.pos {
            return Err(AssembleError::PositionOutOfBounds {
                position,
                len: self.pos,
            });
        }
        self.pos = position;
        Ok(())
    }

    /// Emits one already-encoded instruction word.
    ///
    /// # Errors
    ///
    /// Returns [`AssembleError::BufferOverflow`] if the buffer is full.
    pub fn emit_word(&mut self, word: u32) -> Result<(), AssembleError> {
        if self.remaining() == 0 {
            return Err(AssembleError::BufferOverflow {
                needed: 1,
                remaining: 0,
            });
        }
        self.buf[self.pos] = word;
        self.pos += 1;
        Ok(())
    }

    /// Reads one emitted instruction word at `at`.
    ///
    /// # Errors
    ///
    /// Returns [`AssembleError::ReadOutOfBounds`] if `at` is outside emitted range.
    pub fn read_u32_at(&self, at: usize) -> Result<u32, AssembleError> {
        if at >= self.pos {
            return Err(AssembleError::ReadOutOfBounds { at, len: self.pos });
        }
        Ok(self.buf[at])
    }

    /// Overwrites one emitted instruction word at `at`.
    ///
    /// # Errors
    ///
    /// Returns [`AssembleError::WriteOutOfBounds`] if `at` is outside emitted range.
    pub fn write_u32_at(&mut self, at: usize, word: u32) -> Result<(), AssembleError> {
        if at >= self.pos {
            return Err(AssembleError::WriteOutOfBounds { at, len: self.pos });
        }
        self.buf[at] = word;
        Ok(())
    }
}

#[inline]
fn field_mask(width: u8) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

#[inline]
fn fits_signed(value: i64, bits: u8) -> bool {
    if bits == 0 {
        return value == 0;
    }
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    value >= min && value <= max
}

#[inline]
fn patch_signed_field(
    word: u32,
    value: i64,
    bits: u8,
    lsb: u8,
    kind: BranchRelocKind,
    from: usize,
    to: usize,
) -> Result<u32, AssembleError> {
    if !fits_signed(value, bits) {
        return Err(AssembleError::RelocationOutOfRange {
            kind,
            from,
            to,
            value,
            bits,
        });
    }
    let width_mask = field_mask(bits);
    let encoded = (value as i32 as u32) & width_mask;
    let mask = width_mask << lsb;
    Ok((word & !mask) | (encoded << lsb))
}

/// Applies one relocation patch in-place for local-label fixups.
///
/// `from` and `to` are instruction-word indices.
pub fn patch_relocation(
    writer: &mut CodeWriter<'_>,
    from: usize,
    to: usize,
    kind: BranchRelocKind,
) -> Result<(), AssembleError> {
    let old = writer.read_u32_at(from)?;
    let from_words = from as i64;
    let to_words = to as i64;

    let patched = match kind {
        BranchRelocKind::B26 => {
            patch_signed_field(old, to_words - from_words, 26, 0, kind, from, to)?
        }
        BranchRelocKind::BCond19 | BranchRelocKind::Cbz19 | BranchRelocKind::Imm19 => {
            patch_signed_field(old, to_words - from_words, 19, 5, kind, from, to)?
        }
        BranchRelocKind::Tbz14 => {
            patch_signed_field(old, to_words - from_words, 14, 5, kind, from, to)?
        }
        BranchRelocKind::Adr21 => {
            let value = (to_words - from_words) << 2;
            if !fits_signed(value, 21) {
                return Err(AssembleError::RelocationOutOfRange {
                    kind,
                    from,
                    to,
                    value,
                    bits: 21,
                });
            }
            let encoded = (value as i32 as u32) & field_mask(21);
            let immlo = encoded & 0b11;
            let immhi = (encoded >> 2) & field_mask(19);
            let cleared = old & !((0b11 << 29) | (field_mask(19) << 5));
            cleared | (immlo << 29) | (immhi << 5)
        }
        BranchRelocKind::Adrp21 => {
            let from_page = ((from_words << 2) >> 12) as i64;
            let to_page = ((to_words << 2) >> 12) as i64;
            let value = to_page - from_page;
            if !fits_signed(value, 21) {
                return Err(AssembleError::RelocationOutOfRange {
                    kind,
                    from,
                    to,
                    value,
                    bits: 21,
                });
            }
            let encoded = (value as i32 as u32) & field_mask(21);
            let immlo = encoded & 0b11;
            let immhi = (encoded >> 2) & field_mask(19);
            let cleared = old & !((0b11 << 29) | (field_mask(19) << 5));
            cleared | (immlo << 29) | (immhi << 5)
        }
    };

    writer.write_u32_at(from, patched)
}

/// Encodes by compile-time mnemonic ID (no alias) and emits into a [`CodeWriter`].
pub fn emit_mnemonic_id_const_no_alias_into<const MNEMONIC: u16>(
    writer: &mut CodeWriter<'_>,
    operands: &[Operand],
) -> Result<(), AssembleError> {
    let code = encode_mnemonic_id_const_no_alias::<MNEMONIC>(operands)?;
    writer.emit_word(code.unpack())
}

/// Encodes by compile-time variant ID and emits into a [`CodeWriter`].
pub fn emit_variant_const_into<const VARIANT: u16>(
    writer: &mut CodeWriter<'_>,
    operands: &[Operand],
) -> Result<(), AssembleError> {
    let code = encode_variant_const::<VARIANT>(operands)?;
    writer.emit_word(code.unpack())
}

#[inline]
fn normalize_mov_imm_value(class: RegClass, immediate: i64) -> Result<(u64, usize), AssembleError> {
    match class {
        RegClass::X => Ok((immediate as u64, 4)),
        RegClass::W => {
            if immediate >= 0 {
                if immediate > i64::from(u32::MAX) {
                    return Err(AssembleError::MovPseudoImmediateOutOfRange {
                        class,
                        value: immediate,
                    });
                }
                Ok((immediate as u32 as u64, 2))
            } else {
                if immediate < i64::from(i32::MIN) {
                    return Err(AssembleError::MovPseudoImmediateOutOfRange {
                        class,
                        value: immediate,
                    });
                }
                Ok((immediate as i32 as u32 as u64, 2))
            }
        }
        _ => Err(AssembleError::MovPseudoInvalidDestination { class }),
    }
}

fn emit_encoded(
    writer: &mut CodeWriter<'_>,
    mnemonic: &str,
    operands: &[Operand],
) -> Result<(), AssembleError> {
    let code = encode(mnemonic, operands)?;
    writer.emit_word(code.unpack())
}

/// Emits pseudo `MOV` immediate materialization for one GPR destination.
///
/// - `Xn` destinations accept full 64-bit two's-complement values.
/// - `Wn` destinations accept 32-bit values (`-2147483648..4294967295`).
///
/// # Errors
///
/// Returns [`AssembleError`] when destination class is unsupported,
/// immediate width is invalid for `Wn`, or emission fails.
pub fn emit_mov_imm_auto(
    writer: &mut CodeWriter<'_>,
    dst: RegisterOperand,
    immediate: i64,
) -> Result<(), AssembleError> {
    let (value, chunk_count) = normalize_mov_imm_value(dst.class, immediate)?;

    let mut chunks = [0u16; 4];
    for (idx, chunk) in chunks.iter_mut().enumerate().take(chunk_count) {
        *chunk = ((value >> (idx * 16)) & 0xffff) as u16;
    }

    let mut non_zero = [0usize; 4];
    let mut non_zero_count = 0usize;
    let mut non_ones = [0usize; 4];
    let mut non_ones_count = 0usize;

    for (idx, chunk) in chunks.iter().copied().enumerate().take(chunk_count) {
        if chunk != 0 {
            non_zero[non_zero_count] = idx;
            non_zero_count += 1;
        }
        if chunk != 0xffff {
            non_ones[non_ones_count] = idx;
            non_ones_count += 1;
        }
    }

    let count_z = if non_zero_count == 0 {
        1
    } else {
        non_zero_count
    };
    let count_n = if non_ones_count == 0 {
        1
    } else {
        non_ones_count
    };
    let use_movn = count_n < count_z;

    let (base_idx, base_imm, base_mnemonic) = if use_movn {
        let idx = if non_ones_count == 0 { 0 } else { non_ones[0] };
        let imm16 = !chunks[idx];
        (idx, imm16, "movn")
    } else {
        let idx = if non_zero_count == 0 { 0 } else { non_zero[0] };
        let imm16 = chunks[idx];
        (idx, imm16, "movz")
    };

    let base_ops = [
        Operand::Register(dst),
        Operand::Immediate(base_idx as i64),
        Operand::Immediate(i64::from(base_imm)),
    ];
    emit_encoded(writer, base_mnemonic, &base_ops)?;

    for (idx, chunk) in chunks.iter().copied().enumerate().take(chunk_count) {
        if idx == base_idx {
            continue;
        }

        let target = if use_movn { !chunk } else { chunk };
        if target == 0 {
            continue;
        }

        let ops = [
            Operand::Register(dst),
            Operand::Immediate(idx as i64),
            Operand::Immediate(i64::from(chunk)),
        ];
        emit_encoded(writer, "movk", &ops)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jit_core::{ConditionCode, Operand, RegClass, RegisterOperand};

    fn reg(class: RegClass, code: u8) -> Operand {
        Operand::Register(RegisterOperand {
            code,
            class,
            arrangement: None,
            lane: None,
        })
    }

    fn x(code: u8) -> Operand {
        reg(RegClass::X, code)
    }

    fn w(code: u8) -> Operand {
        reg(RegClass::W, code)
    }

    fn imm(value: i64) -> Operand {
        Operand::Immediate(value)
    }

    #[test]
    fn code_writer_emits_and_reads_words() {
        let mut storage = [0u32; 2];
        let mut w = CodeWriter::new(&mut storage);

        w.emit_word(0x1111_1111).expect("emit");
        w.emit_word(0x2222_2222).expect("emit");

        assert_eq!(w.pos(), 2);
        assert_eq!(w.remaining(), 0);
        assert_eq!(w.read_u32_at(0).expect("read"), 0x1111_1111);
        assert_eq!(w.read_u32_at(1).expect("read"), 0x2222_2222);
    }

    #[test]
    fn code_writer_reports_bounds() {
        let mut storage = [0u32; 1];
        let mut w = CodeWriter::new(&mut storage);
        assert!(matches!(
            w.read_u32_at(0),
            Err(AssembleError::ReadOutOfBounds { at: 0, len: 0 })
        ));

        w.emit_word(0).expect("emit");
        assert!(matches!(
            w.emit_word(1),
            Err(AssembleError::BufferOverflow {
                needed: 1,
                remaining: 0
            })
        ));
        assert!(matches!(
            w.write_u32_at(1, 0),
            Err(AssembleError::WriteOutOfBounds { at: 1, len: 1 })
        ));
    }

    #[test]
    fn patch_relocation_b26_local() {
        let mut storage = [0u32; 2];
        let mut w = CodeWriter::new(&mut storage);
        w.emit_word(0x1400_0000).expect("placeholder b");
        w.emit_word(0xd65f_03c0).expect("ret");

        patch_relocation(&mut w, 0, 1, BranchRelocKind::B26).expect("patch");
        assert_eq!(w.read_u32_at(0).expect("read"), 0x1400_0001);
    }

    #[test]
    fn patch_relocation_range_checks() {
        let mut storage = [0u32; 1];
        let mut w = CodeWriter::new(&mut storage);
        w.emit_word(0x1400_0000).expect("placeholder b");

        let err = patch_relocation(&mut w, 0, (1usize << 25) + 8, BranchRelocKind::B26)
            .expect_err("must fail");
        assert!(matches!(err, AssembleError::RelocationOutOfRange { .. }));
    }

    #[test]
    fn emit_mov_imm_auto_materializes_words() {
        let mut storage = [0u32; 8];
        let mut w = CodeWriter::new(&mut storage);

        emit_mov_imm_auto(
            &mut w,
            RegisterOperand {
                code: 0,
                class: RegClass::X,
                arrangement: None,
                lane: None,
            },
            0x1234_5678,
        )
        .expect("mov pseudo");

        assert_eq!(w.pos(), 2);
        let expected0 = encode("movz", &[x(0), imm(0), imm(0x5678)])
            .expect("movz")
            .unpack();
        let expected1 = encode("movk", &[x(0), imm(1), imm(0x1234)])
            .expect("movk")
            .unpack();
        assert_eq!(w.read_u32_at(0).expect("read0"), expected0);
        assert_eq!(w.read_u32_at(1).expect("read1"), expected1);
    }

    #[test]
    fn emit_mov_imm_auto_rejects_invalid_dest() {
        let mut storage = [0u32; 4];
        let mut w = CodeWriter::new(&mut storage);
        let err = emit_mov_imm_auto(
            &mut w,
            RegisterOperand {
                code: 31,
                class: RegClass::Xsp,
                arrangement: None,
                lane: None,
            },
            1,
        )
        .expect_err("must fail");
        assert_eq!(
            err,
            AssembleError::MovPseudoInvalidDestination {
                class: RegClass::Xsp
            }
        );
    }

    #[test]
    fn patch_relocation_condition_branch() {
        let mut storage = [0u32; 2];
        let mut w = CodeWriter::new(&mut storage);
        // b.eq #0 placeholder
        w.emit_word(0x5400_0000 | ConditionCode::Eq as u32)
            .expect("emit");
        w.emit_word(0xd65f_03c0).expect("emit");

        patch_relocation(&mut w, 0, 1, BranchRelocKind::BCond19).expect("patch");
        let patched = w.read_u32_at(0).expect("read");
        assert_eq!(patched & 0x00ff_ffe0, 0x0000_0020);
    }

    #[test]
    fn w_register_constructor_still_usable_in_tests() {
        let _ = w(0);
    }
}
