#![no_std]
#![forbid(unsafe_code)]
#![deny(
    missing_docs,
    dead_code,
    nonstandard_style,
    unused_imports,
    unused_mut,
    unused_variables,
    unused_unsafe,
    unreachable_patterns
)]

//! Macro-first AArch64 JIT assembler.
//!
//! Public user-facing API is intentionally centered on:
//! - [`jit!`](macro@jit) for emission,
//! - [`CodeWriter`] as caller-owned output buffer.
//! - [`Linker`] for external dynamic-label/fixup management.
//!
//! Low-level encoder pieces remain available under [`__private`] for macro expansion
//! and internal validation, but are not part of the primary public API.

extern crate alloc;

pub(crate) mod generated {
    include!("generated_specs.rs");
}

mod alias;
mod asm;
mod encode;
mod linker;

pub use asm::{AssembleError, BranchRelocKind, CodeWriter};
pub use jit_macro::jit;
pub use linker::{
    BlockId, DynamicLabel, LinkError, LinkPatchError, Linker, ResolvedFixup, ResolvedPatch,
};

/// Macro/runtime plumbing exposed for generated expansion and project-internal checks.
#[doc(hidden)]
pub mod __private {
    pub use crate::alias::{alias_canonical_mnemonic, supports_alias_mnemonic};
    pub use crate::asm::{
        AssembleError, BranchRelocKind, CodeWriter, emit_mnemonic_id_const_no_alias_into,
        emit_mov_imm_auto, emit_variant_const_into, patch_relocation,
    };
    pub use crate::encode::{
        encode, encode_mnemonic_id_const_no_alias, encode_variant, mnemonic_id, spec_for_variant,
        specs_for_mnemonic, specs_for_mnemonic_id,
    };
    pub use crate::generated::{MnemonicId, SPECS, VARIANT_COUNT, VariantId};
    pub use jit_core::{
        AddressingMode, BitFieldSpec, ConditionCode, EncodeError, EncodingSpec, ExtendKind,
        ExtendOperand, ImplicitField, InstructionCode, MemoryAddressingConstraintSpec,
        MemoryOffset, MemoryOperand, Modifier, Operand, OperandConstraintKind, PostIndexOffset,
        RegClass, RegisterListOperand, RegisterOperand, ShiftKind, ShiftOperand,
        SplitImmediateKindSpec, SplitImmediatePlanSpec, SysRegOperand, VectorArrangement,
    };
}
