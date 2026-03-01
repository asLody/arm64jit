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

//! Proc-macro front-end for AArch64-only `jit!` block syntax.
//!
//! The macro lowers assembly-like statements into structured operand calls to
//! `jit`, including:
//! - labels (`name:`, `1:`, `<name`, `1f`),
//! - memory operands, modifiers, register lists, and system-register syntax,
//! - alias normalization and relocation-aware label patching.
//!
//! The expansion stays strict: unsupported syntax or unresolved/invalid variants are
//! surfaced as compile-time or runtime errors instead of hidden fallback behavior.

use proc_macro::TokenStream;

mod ast;
mod emit;
mod normalize;
mod parse;
mod rules;
mod shape;

/// JIT block emitter for AArch64.
///
/// Example:
/// - `jit!(ops ; top: ; add x1, x2, #1 ; cbnz w1, <top)`
#[proc_macro]
pub fn jit(input: TokenStream) -> TokenStream {
    emit::expand(input)
}
