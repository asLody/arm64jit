use proc_macro2::TokenStream as TokenStream2;
use syn::Expr;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LabelDirection {
    Backward,
    Forward,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShiftKindAst {
    Lsl,
    Lsr,
    Asr,
    Ror,
    Msl,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExtendKindAst {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConditionAst {
    Eq,
    Ne,
    Cs,
    Cc,
    Mi,
    Pl,
    Vs,
    Vc,
    Hi,
    Ls,
    Ge,
    Lt,
    Gt,
    Le,
    Al,
    Nv,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ArrangementAst {
    B8,
    B16,
    H4,
    H8,
    S2,
    S4,
    D1,
    D2,
    Q1,
}

#[derive(Clone)]
pub(crate) struct ParsedRegister {
    pub(crate) code: TokenStream2,
    pub(crate) class: &'static str,
    pub(crate) arrangement: Option<ArrangementAst>,
    pub(crate) lane: Option<Expr>,
}

#[derive(Clone)]
pub(crate) enum ParsedModifier {
    Shift {
        kind: ShiftKindAst,
        amount: Option<Expr>,
    },
    Extend {
        kind: ExtendKindAst,
        amount: Option<Expr>,
    },
}

#[derive(Clone)]
pub(crate) enum ParsedMemoryOffset {
    None,
    Immediate(Expr),
    Register {
        reg: ParsedRegister,
        modifier: Option<ParsedModifier>,
    },
}

#[derive(Clone)]
pub(crate) enum ParsedPostIndex {
    Immediate(Expr),
    Register(ParsedRegister),
}

#[derive(Clone)]
pub(crate) struct ParsedMemory {
    pub(crate) base: ParsedRegister,
    pub(crate) offset: ParsedMemoryOffset,
    pub(crate) pre_index: bool,
    pub(crate) post_index: Option<ParsedPostIndex>,
}

#[derive(Clone)]
pub(crate) struct ParsedRegisterList {
    pub(crate) first: ParsedRegister,
    pub(crate) count: TokenStream2,
}

#[derive(Clone)]
pub(crate) struct ParsedSysReg {
    pub(crate) op0: TokenStream2,
    pub(crate) op1: TokenStream2,
    pub(crate) crn: TokenStream2,
    pub(crate) crm: TokenStream2,
    pub(crate) op2: TokenStream2,
}

#[derive(Clone)]
pub(crate) enum OperandAst {
    Immediate(Expr),
    Register(ParsedRegister),
    Memory(ParsedMemory),
    Shift {
        kind: ShiftKindAst,
        amount: Option<Expr>,
    },
    Extend {
        kind: ExtendKindAst,
        amount: Option<Expr>,
    },
    Condition(ConditionAst),
    DynamicCondition(TokenStream2),
    RegisterList(ParsedRegisterList),
    SysReg(ParsedSysReg),
}

#[derive(Clone)]
pub(crate) enum JitArg {
    Operand(OperandAst),
    DirectionalLabelRef {
        name: String,
        direction: LabelDirection,
    },
}

#[derive(Clone)]
pub(crate) struct InstructionStmt {
    pub(crate) op_name: String,
    pub(crate) args: Vec<JitArg>,
}

pub(crate) enum JitStmt {
    Bytes(Expr),
    StaticLabelDef(String),
    Instruction(InstructionStmt),
}

pub(crate) struct JitBlockInput {
    pub(crate) target: Expr,
    pub(crate) statements: Vec<JitStmt>,
}
