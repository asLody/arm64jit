use alloc::boxed::Box;
use core::fmt;

/// AArch64 instruction with little-endian byte order.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(align(4))]
pub struct InstructionCode(pub [u8; 4]);

impl InstructionCode {
    /// Creates an instruction code from a native-endian `u32` word.
    #[inline]
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        Self(value.to_le_bytes())
    }

    /// Returns the encoded instruction as a native-endian `u32` word.
    #[inline]
    #[must_use]
    pub const fn unpack(self) -> u32 {
        u32::from_le_bytes(self.0)
    }
}

impl fmt::Debug for InstructionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InstructionCode({:08x})", self.unpack())
    }
}

impl fmt::Display for InstructionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08x}", self.unpack())
    }
}

/// Field metadata for an instruction operand bitfield.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitFieldSpec {
    /// Field name from the architecture dataset.
    pub name: &'static str,
    /// Least significant bit index.
    pub lsb: u8,
    /// Number of bits in this field.
    pub width: u8,
    /// Whether this field is interpreted as signed.
    pub signed: bool,
}

/// Encoding metadata for a single canonical instruction variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EncodingSpec {
    /// Lower-case mnemonic (e.g. `"add"`).
    pub mnemonic: &'static str,
    /// Canonical variant ID from the source dataset.
    pub variant: &'static str,
    /// Base opcode with fixed bits already applied.
    pub opcode: u32,
    /// Mask of fixed bits in the opcode.
    pub opcode_mask: u32,
    /// Operand field list in deterministic argument order.
    pub fields: &'static [BitFieldSpec],
    /// Field indices, in user-facing operand order.
    pub operand_order: &'static [u8],
    /// Operand class constraints, aligned with [`EncodingSpec::operand_order`].
    pub operand_kinds: &'static [OperandConstraintKind],
    /// Implicit field defaults automatically filled during encoding.
    pub implicit_defaults: &'static [ImplicitField],
    /// Precomputed memory addressing-mode constraint for this variant.
    pub memory_addressing: MemoryAddressingConstraintSpec,
    /// Per-field immediate scaling factors (1 means no scaling).
    ///
    /// The slice is aligned with [`EncodingSpec::fields`] indices.
    pub field_scales: &'static [u16],
    /// Precomputed split-immediate packing plan, if this variant uses one logical
    /// immediate split across two bitfields (e.g. `immlo/immhi`, `b5/b40`).
    pub split_immediate_plan: Option<SplitImmediatePlanSpec>,
    /// Bitset of operand slots where `Gpr64Register` accepts a `Gpr32Register`
    /// when followed by an extend operand in the same variant.
    ///
    /// Bit `n` corresponds to operand slot `n`.
    pub gpr32_extend_compatibility: u64,
}

/// Per-slot operand constraint kind generated from AARCHMRS metadata.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OperandConstraintKind {
    /// General-purpose register with no width constraint (`W/X/SP` families).
    GprRegister,
    /// 32-bit general-purpose register (`W/WSP`).
    Gpr32Register,
    /// 64-bit general-purpose register (`X/SP`).
    Gpr64Register,
    /// SIMD register (`B/H/S/D/Q/V` families).
    SimdRegister,
    /// SVE/SME vector register (`Z`).
    SveZRegister,
    /// SVE/SME predicate register (`P/PN`).
    PredicateRegister,
    /// Plain immediate field.
    Immediate,
    /// Condition-code field.
    Condition,
    /// Shift-kind selector field.
    ShiftKind,
    /// Extend-kind selector field.
    ExtendKind,
    /// System register fragment (`op0/op1/crn/crm/op2`).
    SysRegPart,
    /// Vector arrangement selector.
    Arrangement,
    /// Vector lane selector.
    Lane,
}

/// Packed key for one flattened operand-kind sequence.
///
/// This key is used by generated dispatch tables to shortlist candidate
/// instruction variants before full validation.
pub type OperandShapeKey = u128;

/// Precomputed memory addressing constraint for one encoding variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryAddressingConstraintSpec {
    /// No addressing-mode filtering is required.
    None,
    /// Bare base form only (`[base]`).
    NoOffset,
    /// Offset addressing only.
    Offset,
    /// Pre-indexed addressing only.
    PreIndex,
    /// Post-indexed addressing only.
    PostIndex,
}

/// Precomputed split-immediate descriptor for one encoding variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SplitImmediatePlanSpec {
    /// First user-facing operand slot participating in the split.
    pub first_slot: u8,
    /// Second user-facing operand slot participating in the split.
    pub second_slot: u8,
    /// Split packing kind.
    pub kind: SplitImmediateKindSpec,
}

/// Split-immediate packing kind for one encoding variant.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SplitImmediateKindSpec {
    /// ADR/ADRP-style split: low 2 bits in `immlo`, remaining bits in `immhi`.
    AdrLike {
        /// Field index of `immlo`.
        immlo_field_index: u8,
        /// Field index of `immhi`.
        immhi_field_index: u8,
        /// Required immediate scaling in bytes before split packing.
        scale: i64,
    },
    /// TBZ/TBNZ-style split: top bit in `b5`, low 5 bits in `b40`.
    BitIndex6 {
        /// Field index of `b5`.
        b5_field_index: u8,
        /// Field index of `b40`.
        b40_field_index: u8,
    },
    /// Logical-immediate split: one user bitmask immediate into `immr`/`imms`.
    LogicalImmRs {
        /// Field index of `immr`.
        immr_field_index: u8,
        /// Field index of `imms`.
        imms_field_index: u8,
        /// Register width used by this variant (`32` or `64`).
        reg_size: u8,
    },
    /// Logical-immediate split: one user bitmask immediate into `N`/`immr`/`imms`.
    LogicalImmNrs {
        /// Field index of `N`.
        n_field_index: u8,
        /// Field index of `immr`.
        immr_field_index: u8,
        /// Field index of `imms`.
        imms_field_index: u8,
        /// Register width used by this variant (`32` or `64`).
        reg_size: u8,
    },
}

/// Implicit default for one encoding field.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ImplicitField {
    /// Index into [`EncodingSpec::fields`].
    pub field_index: u8,
    /// Default value to apply.
    pub value: i64,
}

/// Structured operand used by `jit!` and direct runtime encoding.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Operand {
    /// Register operand.
    Register(RegisterOperand),
    /// Immediate operand.
    Immediate(i64),
    /// Memory operand.
    Memory(MemoryOperand),
    /// Shift modifier operand.
    Shift(ShiftOperand),
    /// Extend modifier operand.
    Extend(ExtendOperand),
    /// Condition-code operand.
    Condition(ConditionCode),
    /// Register list operand (expanded as consecutive registers).
    RegisterList(RegisterListOperand),
    /// System register operand.
    SysReg(SysRegOperand),
}

/// Register operand with optional vector arrangement/lane decorations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RegisterOperand {
    /// Encoded register code (`0..=31`).
    pub code: u8,
    /// Register family class.
    pub class: RegClass,
    /// Optional vector arrangement suffix.
    pub arrangement: Option<VectorArrangement>,
    /// Optional lane index.
    pub lane: Option<u8>,
}

/// Memory operand.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MemoryOperand {
    /// Base register.
    pub base: RegisterOperand,
    /// Addressing offset.
    pub offset: MemoryOffset,
    /// Addressing mode.
    pub addressing: AddressingMode,
}

/// Addressing mode for memory operands.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AddressingMode {
    /// Standard offset addressing (`[base, ...]`).
    Offset,
    /// Pre-indexed addressing (`[base, ...]!`).
    PreIndex,
    /// Post-indexed addressing (`[base], #imm` / `[base], Xm`).
    PostIndex(PostIndexOffset),
}

/// Post-index update component.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PostIndexOffset {
    /// Immediate post-index increment.
    Immediate(i64),
    /// Register post-index increment.
    Register(RegisterOperand),
}

/// Offset for memory operands.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryOffset {
    /// No explicit offset.
    None,
    /// Immediate offset.
    Immediate(i64),
    /// Register offset with optional modifier.
    Register {
        /// Index register.
        reg: RegisterOperand,
        /// Optional shift/extend modifier.
        modifier: Option<Modifier>,
    },
}

/// Shift modifier operand.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ShiftOperand {
    /// Shift kind.
    pub kind: ShiftKind,
    /// Shift amount.
    pub amount: u8,
}

/// Extend modifier operand.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ExtendOperand {
    /// Extend kind.
    pub kind: ExtendKind,
    /// Optional shift amount.
    pub amount: Option<u8>,
}

/// Modifier used in memory index expressions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Modifier {
    /// Shift modifier.
    Shift(ShiftOperand),
    /// Extend modifier.
    Extend(ExtendOperand),
}

/// Shift kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ShiftKind {
    /// Logical shift left.
    Lsl,
    /// Logical shift right.
    Lsr,
    /// Arithmetic shift right.
    Asr,
    /// Rotate right.
    Ror,
    /// MSL shift kind.
    Msl,
}

/// Extend kind.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExtendKind {
    /// Unsigned extend byte.
    Uxtb,
    /// Unsigned extend half-word.
    Uxth,
    /// Unsigned extend word.
    Uxtw,
    /// Unsigned extend xword.
    Uxtx,
    /// Signed extend byte.
    Sxtb,
    /// Signed extend half-word.
    Sxth,
    /// Signed extend word.
    Sxtw,
    /// Signed extend xword.
    Sxtx,
}

/// AArch64 condition code.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ConditionCode {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Carry set.
    Cs,
    /// Carry clear.
    Cc,
    /// Minus / negative.
    Mi,
    /// Plus / positive.
    Pl,
    /// Overflow set.
    Vs,
    /// Overflow clear.
    Vc,
    /// Unsigned higher.
    Hi,
    /// Unsigned lower-or-same.
    Ls,
    /// Signed greater-or-equal.
    Ge,
    /// Signed less-than.
    Lt,
    /// Signed greater-than.
    Gt,
    /// Signed less-or-equal.
    Le,
    /// Always.
    Al,
    /// Never.
    Nv,
}

/// Vector arrangement suffix.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VectorArrangement {
    /// `8b`
    B8,
    /// `16b`
    B16,
    /// `4h`
    H4,
    /// `8h`
    H8,
    /// `2s`
    S2,
    /// `4s`
    S4,
    /// `1d`
    D1,
    /// `2d`
    D2,
    /// `1q`
    Q1,
}

/// Register-list operand.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RegisterListOperand {
    /// First register in the list.
    pub first: RegisterOperand,
    /// Register count.
    pub count: u8,
}

/// System-register operand encoded as architectural parts.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SysRegOperand {
    /// `op0` field.
    pub op0: u8,
    /// `op1` field.
    pub op1: u8,
    /// `CRn` field.
    pub crn: u8,
    /// `CRm` field.
    pub crm: u8,
    /// `op2` field.
    pub op2: u8,
}

/// Register family class used for mnemonic disambiguation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RegClass {
    /// `Wn` / `wzr`.
    W,
    /// `Xn` / `xzr`.
    X,
    /// `WSP`.
    Wsp,
    /// `SP`.
    Xsp,
    /// Vector `Vn`.
    V,
    /// SIMD scalar/vector `Bn`.
    B,
    /// SIMD scalar/vector `Hn`.
    H,
    /// SIMD scalar/vector `Sn`.
    S,
    /// SIMD scalar/vector `Dn`.
    D,
    /// SIMD scalar/vector `Qn`.
    Q,
    /// SVE/SME vector `Zn`.
    Z,
    /// SVE/SME predicate `Pn`.
    P,
}

/// Operand-shape atom used in structured mismatch diagnostics.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OperandShapeTag {
    /// 32-bit general-purpose register.
    Gpr32,
    /// 64-bit general-purpose register.
    Gpr64,
    /// SIMD register family.
    Simd,
    /// SVE vector register.
    SveZ,
    /// Predicate register.
    Predicate,
    /// Immediate.
    Immediate,
    /// Memory operand.
    Memory,
    /// Shift modifier.
    Shift,
    /// Extend modifier.
    Extend,
    /// Condition operand.
    Condition,
    /// Register-list operand.
    RegisterList,
    /// System-register operand.
    SysReg,
}

/// One expected operand-form signature expressed as constraint kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandConstraintSignature {
    /// Ordered operand slots.
    pub slots: Box<[OperandConstraintKind]>,
}

/// One provided operand-form signature expressed as concrete shape tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandShapeSignature {
    /// Ordered operand slots.
    pub slots: Box<[OperandShapeTag]>,
}

/// Structured core mismatch hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreNoMatchHint {
    /// Memory addressing might need explicit `#0`.
    MemoryMayRequireExplicitZeroOffset,
    /// Register width mix likely mismatched (`W` vs `X`).
    RegisterWidthMismatch,
    /// One trailing operand is missing.
    OperandMissing {
        /// 1-based user-visible operand index.
        index: u8,
        /// Expected kind candidates for this slot.
        expected: Box<[OperandConstraintKind]>,
    },
    /// One operand has an incompatible kind.
    OperandKindMismatch {
        /// 1-based user-visible operand index.
        index: u8,
        /// Expected kind candidates for this slot.
        expected: Box<[OperandConstraintKind]>,
        /// Actual shape category for the slot.
        got: OperandShapeTag,
    },
    /// Generic shape mismatch with expected/got signatures.
    ShapeMismatch {
        /// Up to the first few expected signatures.
        expected: Box<[OperandConstraintSignature]>,
        /// Count of remaining expected signatures not listed in `expected`.
        expected_additional: u16,
        /// Actual user-provided operand shape.
        got: OperandShapeSignature,
    },
}

/// Structured alias mismatch hint.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AliasNoMatchHint {
    /// Generated conditional-branch alias metadata is invalid.
    InvalidGeneratedConditionCode,
    /// Compare/test alias requires at least two operands.
    CompareTestNeedsAtLeastTwoOperands,
    /// Compare/test first operand must be a GPR.
    CompareTestFirstOperandMustBeGpr,
    /// `mov` alias expects exactly two operands.
    MovNeedsExactlyTwoOperands,
    /// `mov` destination must be a GPR.
    MovDestinationMustBeGpr,
    /// `mov` source must be GPR or zero immediate.
    MovSourceMustBeGprOrZero,
    /// `mul` alias expects exactly three operands.
    MulNeedsExactlyThreeOperands,
    /// `mul` destination must be a data GPR (`Wn`/`Xn`).
    MulDestinationMustBeDataGpr,
    /// `mul` sources must be data GPRs matching destination width.
    MulSourcesMustMatchDestinationWidth,
    /// `ror` alias expects exactly three operands.
    RorNeedsExactlyThreeOperands,
    /// `ror` destination must be a data GPR (`Wn`/`Xn`).
    RorDestinationMustBeDataGpr,
    /// `ror` source must match destination width in data GPR class.
    RorSourceMustMatchDestinationWidth,
    /// `ror` shift must be immediate or same-width data GPR register.
    RorShiftMustBeImmediateOrMatchingRegister,
    /// `lsl`/`lsr`/`asr` immediate alias expects exactly three operands.
    ShiftImmediateNeedsExactlyThreeOperands,
    /// `lsl`/`lsr`/`asr` immediate destination must be a data GPR (`Wn`/`Xn`).
    ShiftImmediateDestinationMustBeDataGpr,
    /// `lsl`/`lsr`/`asr` immediate source must match destination width.
    ShiftImmediateSourceMustMatchDestinationWidth,
    /// `lsl`/`lsr`/`asr` immediate shift must be an immediate.
    ShiftImmediateAmountMustBeImmediate,
    /// `lsl`/`lsr`/`asr` immediate shift out of range.
    ShiftImmediateAmountOutOfRange {
        /// Destination bit width (`32` or `64`).
        bits: u8,
    },
    /// `mvn` operand form is invalid.
    MvnOperandFormInvalid,
    /// `mvn` destination must be a GPR.
    MvnDestinationMustBeGpr,
    /// `mvn` source must be a GPR.
    MvnSourceMustBeGpr,
    /// `mvn` optional third operand must be immediate/`lsl`.
    MvnOptionalShiftMustBeImmediateOrLsl,
    /// `cinc` alias expects exactly three operands.
    CincNeedsExactlyThreeOperands,
    /// `cinc` operands must be GPRs.
    CincOperandsMustBeGpr,
    /// `cinc` third operand must be condition.
    CincThirdMustBeCondition,
    /// `cset`/`csetm` alias expects exactly two operands.
    CsetNeedsExactlyTwoOperands,
    /// `cset`/`csetm` destination must be GPR.
    CsetDestinationMustBeGpr,
    /// `cset`/`csetm` second operand must be condition.
    CsetSecondMustBeCondition,
    /// `cneg` alias expects exactly three operands.
    CnegNeedsExactlyThreeOperands,
    /// `cneg` destination must be GPR.
    CnegDestinationMustBeGpr,
    /// `cneg` source must be GPR.
    CnegSourceMustBeGpr,
    /// `cneg` third operand must be condition.
    CnegThirdMustBeCondition,
    /// Bitfield alias expects exactly four operands.
    BitfieldNeedsExactlyFourOperands,
    /// Bitfield destination must be GPR.
    BitfieldDestinationMustBeGpr,
    /// Bitfield source must be GPR.
    BitfieldSourceMustBeGpr,
    /// Bitfield source width must match destination width.
    BitfieldSourceWidthMustMatchDestination,
    /// Bitfield `lsb` must be immediate.
    BitfieldLsbMustBeImmediate,
    /// Bitfield `width` must be immediate.
    BitfieldWidthMustBeImmediate,
    /// Bitfield lsb/width range is invalid.
    BitfieldRangeInvalid {
        /// Destination bit width (`32` or `64`).
        bits: u8,
    },
    /// `bfc` alias expects exactly three operands.
    BfcNeedsExactlyThreeOperands,
    /// `bfc` destination must be GPR.
    BfcDestinationMustBeGpr,
    /// `bfc` `lsb` must be immediate.
    BfcLsbMustBeImmediate,
    /// `bfc` `width` must be immediate.
    BfcWidthMustBeImmediate,
    /// `bfc` lsb/width range is invalid.
    BfcRangeInvalid {
        /// Destination bit width (`32` or `64`).
        bits: u8,
    },
    /// Extend-long alias requires at least two operands.
    ExtendLongNeedsAtLeastTwoOperands,
    /// `stsetl` expects exactly two operands.
    StsetlNeedsExactlyTwoOperands,
    /// `stsetl` first operand must be GPR.
    StsetlFirstOperandMustBeGpr,
    /// `stsetl` memory form only supports `[rn]` or `[rn, #0]`.
    StsetlMemoryOffsetOnly,
    /// `stsetl` memory base must be GPR.
    StsetlMemoryBaseMustBeGpr,
    /// `stsetl` second operand form is invalid.
    StsetlSecondOperandInvalid,
    /// `smull`/`umull` alias expects exactly three operands.
    MAddLongNeedsExactlyThreeOperands,
    /// `smull`/`umull` destination must be 64-bit data GPR (`Xd`).
    MAddLongDestinationMustBeX,
    /// `smull`/`umull` sources must be 32-bit data GPRs (`Wn`, `Wm`).
    MAddLongSourcesMustBeW,
    /// `dc` expects exactly two operands.
    DcNeedsExactlyTwoOperands,
    /// `dc` first operand must be immediate subop.
    DcFirstOperandMustBeImmediateSubop,
    /// `dc` second operand must be register.
    DcSecondOperandMustBeRegister,
    /// `dc` subop must be non-negative 32-bit.
    DcSubopMustBeNonNegativeU32,
    /// `dc` subop has unsupported bits.
    DcSubopUnsupportedBits,
}

/// Structured no-match hint for [`EncodeError::NoMatchingVariantHint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoMatchingHint {
    /// Core matcher produced this hint.
    Core(CoreNoMatchHint),
    /// Alias canonicalization produced this hint.
    Alias(AliasNoMatchHint),
}

/// Encoder errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The mnemonic is unknown in the spec set.
    UnknownMnemonic,
    /// The provided operand count does not match any variant for the mnemonic.
    OperandCountMismatch,
    /// The provided operand count is outside the valid range for the mnemonic.
    OperandCountRange {
        /// Minimum accepted operand count across variants.
        min: u8,
        /// Maximum accepted operand count across variants.
        max: u8,
        /// Provided operand count.
        got: u8,
    },
    /// An operand value does not fit its target field.
    OperandOutOfRange {
        /// Field name.
        field: &'static str,
        /// Provided value.
        value: i64,
        /// Field width.
        width: u8,
        /// Whether the field expects signed encoding.
        signed: bool,
    },
    /// Immediate must satisfy a scale/alignment constraint before encoding.
    ImmediateNotAligned {
        /// Field name.
        field: &'static str,
        /// Provided value.
        value: i64,
        /// Required positive scale.
        scale: i64,
    },
    /// Operand kind does not match expected field kind.
    InvalidOperandKind {
        /// Field name.
        field: &'static str,
        /// Expected operand class.
        expected: &'static str,
        /// Actual operand class from input.
        got: &'static str,
    },
    /// Multiple variants matched equally.
    AmbiguousVariant,
    /// No variant matched the provided operands.
    NoMatchingVariant,
    /// No variant matched, with a structured hint.
    NoMatchingVariantHint {
        /// Structured guidance for adjusting operand shape.
        hint: NoMatchingHint,
    },
}

fn expected_kind_name(expected: OperandConstraintKind) -> &'static str {
    match expected {
        OperandConstraintKind::GprRegister => "general-purpose register",
        OperandConstraintKind::Gpr32Register => "32-bit general-purpose register",
        OperandConstraintKind::Gpr64Register => "64-bit general-purpose register",
        OperandConstraintKind::SimdRegister => "SIMD register",
        OperandConstraintKind::SveZRegister => "SVE Z register",
        OperandConstraintKind::PredicateRegister => "predicate register",
        OperandConstraintKind::Immediate => "immediate",
        OperandConstraintKind::Condition => "condition",
        OperandConstraintKind::ShiftKind => "shift",
        OperandConstraintKind::ExtendKind => "extend",
        OperandConstraintKind::SysRegPart => "system register part",
        OperandConstraintKind::Arrangement => "arrangement",
        OperandConstraintKind::Lane => "lane",
    }
}

fn shape_tag_name(tag: OperandShapeTag) -> &'static str {
    match tag {
        OperandShapeTag::Gpr32 => "gpr32",
        OperandShapeTag::Gpr64 => "gpr64",
        OperandShapeTag::Simd => "simd",
        OperandShapeTag::SveZ => "z",
        OperandShapeTag::Predicate => "p",
        OperandShapeTag::Immediate => "imm",
        OperandShapeTag::Memory => "mem",
        OperandShapeTag::Shift => "shift",
        OperandShapeTag::Extend => "extend",
        OperandShapeTag::Condition => "cond",
        OperandShapeTag::RegisterList => "reglist",
        OperandShapeTag::SysReg => "sysreg",
    }
}

fn write_expected_list(
    f: &mut fmt::Formatter<'_>,
    expected: &[OperandConstraintKind],
) -> fmt::Result {
    for (idx, kind) in expected.iter().copied().enumerate() {
        if idx > 0 {
            write!(f, " / ")?;
        }
        write!(f, "{}", expected_kind_name(kind))?;
    }
    Ok(())
}

fn write_shape_signature(f: &mut fmt::Formatter<'_>, got: &OperandShapeSignature) -> fmt::Result {
    for (idx, tag) in got.slots.iter().copied().enumerate() {
        if idx > 0 {
            write!(f, ",")?;
        }
        write!(f, "{}", shape_tag_name(tag))?;
    }
    Ok(())
}

fn write_expected_signature(
    f: &mut fmt::Formatter<'_>,
    expected: &OperandConstraintSignature,
) -> fmt::Result {
    write!(f, "(")?;
    for (idx, kind) in expected.slots.iter().copied().enumerate() {
        if idx > 0 {
            write!(f, ", ")?;
        }
        write!(
            f,
            "{}",
            shape_tag_name(match kind {
                OperandConstraintKind::GprRegister => OperandShapeTag::Gpr64,
                OperandConstraintKind::Gpr32Register => OperandShapeTag::Gpr32,
                OperandConstraintKind::Gpr64Register => OperandShapeTag::Gpr64,
                OperandConstraintKind::SimdRegister => OperandShapeTag::Simd,
                OperandConstraintKind::SveZRegister => OperandShapeTag::SveZ,
                OperandConstraintKind::PredicateRegister => OperandShapeTag::Predicate,
                OperandConstraintKind::Immediate => OperandShapeTag::Immediate,
                OperandConstraintKind::Condition => OperandShapeTag::Condition,
                OperandConstraintKind::ShiftKind => OperandShapeTag::Shift,
                OperandConstraintKind::ExtendKind => OperandShapeTag::Extend,
                OperandConstraintKind::SysRegPart => OperandShapeTag::SysReg,
                OperandConstraintKind::Arrangement => OperandShapeTag::Immediate,
                OperandConstraintKind::Lane => OperandShapeTag::Immediate,
            })
        )?;
    }
    write!(f, ")")
}

fn write_no_matching_hint(hint: &NoMatchingHint, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match hint {
        NoMatchingHint::Core(CoreNoMatchHint::MemoryMayRequireExplicitZeroOffset) => {
            write!(f, "memory addressing may require explicit #0 offset")
        }
        NoMatchingHint::Core(CoreNoMatchHint::RegisterWidthMismatch) => {
            write!(f, "register width mismatch: check W/X operand sizes")
        }
        NoMatchingHint::Core(CoreNoMatchHint::OperandMissing { index, expected }) => {
            write!(f, "operand #{} is missing; expected ", index)?;
            write_expected_list(f, expected)
        }
        NoMatchingHint::Core(CoreNoMatchHint::OperandKindMismatch {
            index,
            expected,
            got,
        }) => {
            write!(f, "operand #{} expects ", index)?;
            write_expected_list(f, expected)?;
            write!(f, "; got {}", shape_tag_name(*got))
        }
        NoMatchingHint::Core(CoreNoMatchHint::ShapeMismatch {
            expected,
            expected_additional,
            got,
        }) => {
            write!(f, "expected operand forms: ")?;
            for (idx, form) in expected.iter().enumerate() {
                if idx > 0 {
                    write!(f, "; ")?;
                }
                write_expected_signature(f, form)?;
            }
            if *expected_additional > 0 {
                write!(f, "; ... (+{} more)", expected_additional)?;
            }
            write!(f, "; got operand form: ")?;
            write_shape_signature(f, got)
        }
        NoMatchingHint::Alias(AliasNoMatchHint::InvalidGeneratedConditionCode) => {
            write!(
                f,
                "internal error: invalid generated condition code in branch alias"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CompareTestNeedsAtLeastTwoOperands) => {
            write!(f, "compare/test alias requires at least two operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CompareTestFirstOperandMustBeGpr) => {
            write!(
                f,
                "compare/test first operand must be a general-purpose register"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MovNeedsExactlyTwoOperands) => {
            write!(f, "mov alias expects exactly two operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MovDestinationMustBeGpr) => {
            write!(f, "mov destination must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MovSourceMustBeGprOrZero) => {
            write!(f, "mov source must be a general-purpose register or #0")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MulNeedsExactlyThreeOperands) => {
            write!(f, "mul alias expects exactly three operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MulDestinationMustBeDataGpr) => {
            write!(
                f,
                "mul destination must be a data general-purpose register (Wn/Xn)"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MulSourcesMustMatchDestinationWidth) => {
            write!(
                f,
                "mul source registers must be data general-purpose registers matching destination width"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::RorNeedsExactlyThreeOperands) => {
            write!(f, "ror alias expects exactly three operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::RorDestinationMustBeDataGpr) => {
            write!(
                f,
                "ror destination must be a data general-purpose register (Wn/Xn)"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::RorSourceMustMatchDestinationWidth) => {
            write!(
                f,
                "ror source register must be a data general-purpose register matching destination width"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::RorShiftMustBeImmediateOrMatchingRegister) => {
            write!(
                f,
                "ror third operand must be an immediate shift or same-width data general-purpose register"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::ShiftImmediateNeedsExactlyThreeOperands) => {
            write!(
                f,
                "lsl/lsr/asr immediate alias expects exactly three operands"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::ShiftImmediateDestinationMustBeDataGpr) => {
            write!(
                f,
                "lsl/lsr/asr immediate destination must be a data general-purpose register (Wn/Xn)"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::ShiftImmediateSourceMustMatchDestinationWidth) => {
            write!(
                f,
                "lsl/lsr/asr immediate source register must be a data general-purpose register matching destination width"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::ShiftImmediateAmountMustBeImmediate) => {
            write!(f, "lsl/lsr/asr immediate shift amount must be an immediate")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::ShiftImmediateAmountOutOfRange { bits }) => {
            write!(
                f,
                "lsl/lsr/asr immediate shift out of range (expected 0 <= shift < {bits})"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MvnOperandFormInvalid) => write!(
            f,
            "mvn expects two operands or three operands with an optional lsl shift amount"
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::MvnDestinationMustBeGpr) => {
            write!(f, "mvn destination must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MvnSourceMustBeGpr) => {
            write!(f, "mvn source must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MvnOptionalShiftMustBeImmediateOrLsl) => write!(
            f,
            "mvn optional third operand must be an immediate or lsl #<amount>"
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::CincNeedsExactlyThreeOperands) => {
            write!(f, "cinc alias expects exactly three operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CincOperandsMustBeGpr) => {
            write!(f, "cinc operands must be general-purpose registers")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CincThirdMustBeCondition) => {
            write!(f, "cinc third operand must be a condition code")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CsetNeedsExactlyTwoOperands) => {
            write!(f, "cset/csetm alias expects exactly two operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CsetDestinationMustBeGpr) => {
            write!(
                f,
                "cset/csetm destination must be a general-purpose register"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CsetSecondMustBeCondition) => {
            write!(f, "cset/csetm second operand must be a condition code")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CnegNeedsExactlyThreeOperands) => {
            write!(f, "cneg alias expects exactly three operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CnegDestinationMustBeGpr) => {
            write!(f, "cneg destination must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CnegSourceMustBeGpr) => {
            write!(f, "cneg source must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::CnegThirdMustBeCondition) => {
            write!(f, "cneg third operand must be a condition code")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldNeedsExactlyFourOperands) => write!(
            f,
            "bitfield alias expects exactly four operands: <rd>, <rn>, #<lsb>, #<width>"
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldDestinationMustBeGpr) => {
            write!(f, "bitfield destination must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldSourceMustBeGpr) => {
            write!(f, "bitfield source must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldSourceWidthMustMatchDestination) => {
            write!(
                f,
                "bitfield source register width must match destination register width"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldLsbMustBeImmediate) => {
            write!(f, "bitfield lsb must be an immediate")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldWidthMustBeImmediate) => {
            write!(f, "bitfield width must be an immediate")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BitfieldRangeInvalid { bits }) => write!(
            f,
            "bitfield immediates out of range (expected 0 <= lsb < {} and 1 <= width <= {} - lsb)",
            bits, bits
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::BfcNeedsExactlyThreeOperands) => {
            write!(
                f,
                "bfc alias expects exactly three operands: <rd>, #<lsb>, #<width>"
            )
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BfcDestinationMustBeGpr) => {
            write!(f, "bfc destination must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BfcLsbMustBeImmediate) => {
            write!(f, "bfc lsb must be an immediate")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BfcWidthMustBeImmediate) => {
            write!(f, "bfc width must be an immediate")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::BfcRangeInvalid { bits }) => write!(
            f,
            "bfc immediates out of range (expected 0 <= lsb < {} and 1 <= width <= {} - lsb)",
            bits, bits
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::ExtendLongNeedsAtLeastTwoOperands) => {
            write!(f, "extend-long alias requires at least two operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::StsetlNeedsExactlyTwoOperands) => write!(
            f,
            "stsetl expects exactly two operands: <rs>, <rn> or <rs>, [<rn>]"
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::StsetlFirstOperandMustBeGpr) => {
            write!(f, "stsetl first operand must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::StsetlMemoryOffsetOnly) => write!(
            f,
            "stsetl memory form only supports offset addressing: [rn] or [rn, #0]"
        ),
        NoMatchingHint::Alias(AliasNoMatchHint::StsetlMemoryBaseMustBeGpr) => {
            write!(f, "stsetl memory base must be a general-purpose register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::StsetlSecondOperandInvalid) => {
            write!(f, "stsetl expects stsetl <rs>, <rn> or stsetl <rs>, [<rn>]")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MAddLongNeedsExactlyThreeOperands) => {
            write!(f, "smull/umull alias expects exactly three operands")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MAddLongDestinationMustBeX) => {
            write!(f, "smull/umull destination must be a 64-bit register (Xd)")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::MAddLongSourcesMustBeW) => {
            write!(f, "smull/umull sources must be 32-bit registers (Wn, Wm)")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::DcNeedsExactlyTwoOperands) => {
            write!(f, "dc expects exactly two operands: #<subop>, <rt>")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::DcFirstOperandMustBeImmediateSubop) => {
            write!(f, "dc first operand must be an immediate subop")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::DcSecondOperandMustBeRegister) => {
            write!(f, "dc second operand must be a register")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::DcSubopMustBeNonNegativeU32) => {
            write!(f, "dc subop must be a non-negative 32-bit value")
        }
        NoMatchingHint::Alias(AliasNoMatchHint::DcSubopUnsupportedBits) => write!(
            f,
            "dc subop contains unsupported bits; expected encoding is (op1<<16)|(crm<<8)|(op2<<5)"
        ),
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMnemonic => write!(f, "unknown mnemonic"),
            Self::OperandCountMismatch => write!(f, "operand count mismatch"),
            Self::OperandCountRange { min, max, got } => write!(
                f,
                "operand count mismatch (expected {}..={}, got={})",
                min, max, got
            ),
            Self::OperandOutOfRange {
                field,
                value,
                width,
                signed,
            } => write!(
                f,
                "operand out of range for field {} (value={}, width={}, signed={})",
                field, value, width, signed
            ),
            Self::ImmediateNotAligned {
                field,
                value,
                scale,
            } => write!(
                f,
                "immediate for field {} is not aligned (value={}, scale={})",
                field, value, scale
            ),
            Self::InvalidOperandKind {
                field,
                expected,
                got,
            } => {
                write!(
                    f,
                    "invalid operand kind for field {} (expected {}, got {})",
                    field, expected, got
                )
            }
            Self::AmbiguousVariant => write!(f, "ambiguous variant"),
            Self::NoMatchingVariant => write!(f, "no matching variant"),
            Self::NoMatchingVariantHint { hint } => {
                write!(f, "no matching variant (")?;
                write_no_matching_hint(hint, f)?;
                write!(f, ")")
            }
        }
    }
}
