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

//! Core `no_std` types and encoding logic for AArch64 instruction synthesis.
//!
//! This crate is the runtime heart of the project:
//! - [`types`] defines the structured operand model and generated spec metadata.
//! - [`engine`] implements variant selection, validation, and bitfield encoding.
//!
//! Design goals:
//! - `no_std` compatibility (with `alloc` only where needed for diagnostics).
//! - deterministic, metadata-driven encoding (no handwritten opcode tables).
//! - strict error reporting for ambiguous or invalid operand forms.
//!
//! Typical call paths:
//! - high-level: [`encode`] with canonical mnemonic + structured operands.
//! - low-level: [`encode_by_spec_operands`] when variant/spec is already known.
//! - dispatch assist: [`operand_shape_keys`] for precomputed shape-based routing.

extern crate alloc;

mod engine;
mod types;

pub use engine::{
    encode, encode_by_spec, encode_by_spec_operands, encode_candidates, operand_shape_keys,
};
pub use types::*;
pub use types::{AliasNoMatchHint, NoMatchingHint};
