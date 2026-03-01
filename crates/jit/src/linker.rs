use alloc::vec::Vec;
use core::fmt;

use crate::asm::{AssembleError, BranchRelocKind, CodeWriter, patch_relocation};

/// Logical code block identifier used by [`Linker`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(usize);

/// Dynamic label identifier allocated by [`Linker`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DynamicLabel(usize);

/// Fully resolved relocation patch for a specific block.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ResolvedPatch {
    /// Block that owns the instruction to patch.
    pub block: BlockId,
    /// Instruction word index to patch.
    pub at: usize,
    /// Target instruction word index inside `block`.
    pub to: usize,
    /// Relocation kind.
    pub kind: BranchRelocKind,
}

/// Resolved relocation patch without block metadata.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ResolvedFixup {
    /// Instruction word index to patch.
    pub at: usize,
    /// Target instruction word index.
    pub to: usize,
    /// Relocation kind.
    pub kind: BranchRelocKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct PendingFixup {
    block: BlockId,
    at: usize,
    target: DynamicLabel,
    kind: BranchRelocKind,
}

/// Errors produced by [`Linker`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinkError {
    /// Label was not created by this linker instance.
    UnknownLabel {
        /// Label id.
        label: DynamicLabel,
    },
    /// Label was already bound once.
    LabelAlreadyBound {
        /// Label id.
        label: DynamicLabel,
    },
    /// Label is referenced but not bound.
    LabelUnbound {
        /// Label id.
        label: DynamicLabel,
    },
    /// Cross-block relocations are unsupported by `patch_relocation`.
    CrossBlockRelocationUnsupported {
        /// Block containing the source instruction.
        from_block: BlockId,
        /// Block containing the target label.
        to_block: BlockId,
        /// Target label.
        label: DynamicLabel,
    },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownLabel { label } => write!(f, "unknown linker label {:?}", label),
            Self::LabelAlreadyBound { label } => {
                write!(f, "label {:?} is already bound", label)
            }
            Self::LabelUnbound { label } => write!(f, "label {:?} is not bound", label),
            Self::CrossBlockRelocationUnsupported {
                from_block,
                to_block,
                label,
            } => write!(
                f,
                "cross-block relocation is unsupported (from={:?}, to={:?}, label={:?})",
                from_block, to_block, label
            ),
        }
    }
}

/// Errors produced by [`Linker::patch_writer`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinkPatchError {
    /// Linker resolution failed.
    Link(LinkError),
    /// Code patch failed while writing instruction words.
    Assemble(AssembleError),
}

impl fmt::Display for LinkPatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Link(err) => write!(f, "{err}"),
            Self::Assemble(err) => write!(f, "{err}"),
        }
    }
}

impl From<LinkError> for LinkPatchError {
    fn from(value: LinkError) -> Self {
        Self::Link(value)
    }
}

impl From<AssembleError> for LinkPatchError {
    fn from(value: AssembleError) -> Self {
        Self::Assemble(value)
    }
}

/// Independent dynamic-label/fixup manager.
///
/// This type intentionally stores only label/fixup metadata and can manage
/// multiple code writers via [`BlockId`].
pub struct Linker {
    next_block: usize,
    bindings: Vec<Option<(BlockId, usize)>>,
    fixups: Vec<PendingFixup>,
}

impl Default for Linker {
    fn default() -> Self {
        Self::new()
    }
}

impl Linker {
    /// Creates an empty linker state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_block: 0,
            bindings: Vec::new(),
            fixups: Vec::new(),
        }
    }

    /// Allocates a new logical code block id.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    /// Allocates a new dynamic label.
    pub fn new_label(&mut self) -> DynamicLabel {
        let id = DynamicLabel(self.bindings.len());
        self.bindings.push(None);
        id
    }

    fn label_index(&self, label: DynamicLabel) -> Result<usize, LinkError> {
        if label.0 >= self.bindings.len() {
            return Err(LinkError::UnknownLabel { label });
        }
        Ok(label.0)
    }

    /// Binds a dynamic label to a concrete word position in one block.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::UnknownLabel`] if `label` was not allocated by this
    /// linker, or [`LinkError::LabelAlreadyBound`] if the label is already bound.
    pub fn bind(
        &mut self,
        block: BlockId,
        label: DynamicLabel,
        position: usize,
    ) -> Result<(), LinkError> {
        let index = self.label_index(label)?;
        if self.bindings[index].is_some() {
            return Err(LinkError::LabelAlreadyBound { label });
        }
        self.bindings[index] = Some((block, position));
        Ok(())
    }

    /// Adds one pending relocation fixup for a dynamic label.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::UnknownLabel`] if `target` was not allocated by this
    /// linker.
    pub fn add_fixup(
        &mut self,
        block: BlockId,
        at: usize,
        target: DynamicLabel,
        kind: BranchRelocKind,
    ) -> Result<(), LinkError> {
        let _ = self.label_index(target)?;
        self.fixups.push(PendingFixup {
            block,
            at,
            target,
            kind,
        });
        Ok(())
    }

    /// Resolves all pending fixups across every block.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError::LabelUnbound`] for unresolved labels or
    /// [`LinkError::CrossBlockRelocationUnsupported`] for cross-block fixups.
    pub fn resolve(&self) -> Result<Vec<ResolvedPatch>, LinkError> {
        let mut out = Vec::with_capacity(self.fixups.len());
        for fixup in self.fixups.iter().copied() {
            let index = self.label_index(fixup.target)?;
            let Some((target_block, target_pos)) = self.bindings[index] else {
                return Err(LinkError::LabelUnbound {
                    label: fixup.target,
                });
            };
            if target_block != fixup.block {
                return Err(LinkError::CrossBlockRelocationUnsupported {
                    from_block: fixup.block,
                    to_block: target_block,
                    label: fixup.target,
                });
            }
            out.push(ResolvedPatch {
                block: fixup.block,
                at: fixup.at,
                to: target_pos,
                kind: fixup.kind,
            });
        }
        Ok(out)
    }

    /// Resolves pending fixups for one block.
    ///
    /// # Errors
    ///
    /// Returns [`LinkError`] when a label is missing, unbound, or cross-block.
    pub fn resolve_for_block(&self, block: BlockId) -> Result<Vec<ResolvedFixup>, LinkError> {
        let mut out = Vec::new();
        for patch in self.resolve()? {
            if patch.block != block {
                continue;
            }
            out.push(ResolvedFixup {
                at: patch.at,
                to: patch.to,
                kind: patch.kind,
            });
        }
        Ok(out)
    }

    /// Resolves and applies all fixups for one block onto `writer`.
    ///
    /// # Errors
    ///
    /// Returns [`LinkPatchError::Link`] when metadata resolution fails, or
    /// [`LinkPatchError::Assemble`] when in-place patching fails.
    pub fn patch_writer(
        &self,
        block: BlockId,
        writer: &mut CodeWriter<'_>,
    ) -> Result<(), LinkPatchError> {
        let patches = self.resolve_for_block(block)?;
        for patch in patches {
            patch_relocation(writer, patch.at, patch.to, patch.kind)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linker_patches_single_block() {
        let mut storage = [0u32; 2];
        let mut writer = CodeWriter::new(&mut storage);
        writer.emit_word(0x1400_0000).expect("emit placeholder");
        writer.emit_word(0xd65f_03c0).expect("emit ret");

        let mut linker = Linker::new();
        let text = linker.new_block();
        let top = linker.new_label();

        linker.bind(text, top, 1).expect("bind");
        linker
            .add_fixup(text, 0, top, BranchRelocKind::B26)
            .expect("add_fixup");

        linker.patch_writer(text, &mut writer).expect("patch");
        assert_eq!(writer.read_u32_at(0).expect("read"), 0x1400_0001);
    }

    #[test]
    fn linker_rejects_cross_block_fixup() {
        let mut linker = Linker::new();
        let a = linker.new_block();
        let b = linker.new_block();
        let label = linker.new_label();

        linker.bind(b, label, 0).expect("bind");
        linker
            .add_fixup(a, 0, label, BranchRelocKind::B26)
            .expect("fixup");

        let err = linker.resolve_for_block(a).expect_err("must fail");
        assert!(matches!(
            err,
            LinkError::CrossBlockRelocationUnsupported {
                from_block,
                to_block,
                label: _
            } if from_block == a && to_block == b
        ));
    }

    #[test]
    fn linker_reports_unbound_label() {
        let mut linker = Linker::new();
        let text = linker.new_block();
        let label = linker.new_label();
        linker
            .add_fixup(text, 0, label, BranchRelocKind::B26)
            .expect("fixup");

        let err = linker.resolve().expect_err("must fail");
        assert_eq!(err, LinkError::LabelUnbound { label });
    }
}
