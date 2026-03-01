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

//! Code generation utilities that turn normalized AARCHMRS data into compact Rust tables.
//!
//! This crate does not participate in runtime encoding directly. Instead it generates:
//! - deduplicated [`EncodingSpec`](jit_core::EncodingSpec) pools,
//! - mnemonic/variant dispatch tables,
//! - macro-side normalization and fast-dispatch metadata.
//!
//! The output is consumed by `jit` at build time.

mod core;

pub use core::*;
