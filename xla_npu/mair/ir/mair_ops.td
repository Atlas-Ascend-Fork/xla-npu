#ifndef MAIR_OPS
#define MAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mair_ops_base.td"

//===--------------------------------------------------------------------------===//
// ConstantOp
//===--------------------------------------------------------------------------===//

def Air_ConstantOp : Air_Op<"Const", [
    ConstantLike,
    Pure
    ]>{
        let summary = "Constant operation";
        let description = [{
            Represents a constant value in the MAIR dialect.
        }];

        let arguments = (ins
            AnyAttr:$value
        );

        let results = (outs
           AnyType:$output
     );
}

//===--------------------------------------------------------------------------===//
// AddOp
//===--------------------------------------------------------------------------===//

def Air_AddOp : Air_Op<"Add",[
    Pure, 
    SameOperandsAndResultType
    ]>{
        let summary = "element-wise addition operation";
        let description = [{
            The "add" operation performs element-wise addition between two tensors.
        }];

        let arguments = (ins
            Air_Tensor:$lhs,
            Air_Tensor:$rhs
        );

        let results = (outs Air_Tensor);
    }


//===--------------------------------------------------------------------------===//
// BroadcastOp
//===--------------------------------------------------------------------------===//

def Air_BroadcastOp : Air_Op<"BroadcastTo", [Pure]>{
    let summary = "broadcast to a tensor type.";
    let description = "broadcast to a tensor type.";

    let arguments = (ins
        Air_Tensor:$input,
        TensorOf<[I32, I64]>:$shape
    );

    let results = (outs Air_Tensor);
}


//===--------------------------------------------------------------------------===//
// MultiplyOp
//===--------------------------------------------------------------------------===//

def Air_MulOp : Air_Op<"Mul", [Pure]>{
    let summary = "Returns x * y element-wise.";
    let description = [{
        Returns x * y element-wise. 
    }];

    let arguments = (ins
        Air_Tensor:$x,
        Air_Tensor:$y
    );

    let results = (outs Air_Tensor);
}


//===--------------------------------------------------------------------------===//
// TanhOp
//===--------------------------------------------------------------------------===//

def Air_TanhOp : Air_Op<"Tanh", [Pure]>{
    let summary = "Tanh operator.";
    let description = "Tanh operator.";

    let arguments = (ins
        Air_Tensor:$x
    );

    let results = (outs Air_Tensor);
}

//===----------------------------------------------------------------------===//
// ascend graph op definitions
//===----------------------------------------------------------------------===//
def Air_GraphOp : Air_Op<"graph"> {
  let summary = [{The `ascend.graph` operation contains a region with a
    single block that lists the operations in a Ascend graph.}];


  let description = [{
    The operations are topologically sorted in-order (no cycles are allowed in
    the values).
  }];

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  }];

  let hasCanonicalizer = 1;
}


//===--------------------------------------------------------------------------===//
// HcomAllReduceOp
//===--------------------------------------------------------------------------===//

def Air_HcomAllReduceOp : Air_Op<"HcomAllReduce", [Pure]>{
    let summary = "HcomAllReduce operator.";
    let description = "HcomAllReduce operator.";

    let arguments = (ins
        Air_Tensor:$x, // 当前仅支持对一个张量做allreduce
        I32:$allreduce_type // 0: SUM, 1: MAX, 2: MIN, 3: PROD
    );

    let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// HcomAllGatherOp
//===--------------------------------------------------------------------------===//

def Air_HcomAllGatherOp : Air_Op<"HcomAllGather", [Pure]>{
    let summary = "HcomAllGather operator.";
    let description = "HcomAllGather operator.";

    let arguments = (ins
        Air_Tensor:$x,
        I64:$rank_size
    );

    let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// ConcatV2Op
//===--------------------------------------------------------------------------===//
def Air_ConcatV2Op : Air_Op<"ConcatV2", [Pure]> {
    let summary = "Concatenates tensors along one dimension.";

    let arguments = (ins
    Arg<Variadic<TensorOf<[Air_BasicType]>>, [{List of `N` Tensors to concatenate. Their ranks and types must match,
    and their sizes must match in all dimensions except `concat_dim`.}]>:$x,
    Arg<Air_I32OrI64, [{0-D.  The dimension along which to concatenate.}]>:$concat_dim,
    DefaultValuedAttr<I64Attr, "1">:$N
    );

    let results = (outs
    Res<TensorOf<[Air_BasicType]>, [{A `Tensor` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.}]>:$y
    );
}

//===--------------------------------------------------------------------------===//
// DivOp
//===--------------------------------------------------------------------------===//
def Air_DivOp : Air_Op<"Div", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise division operation.";
  let description = [{
    Performs element-wise division of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// MatMulOp
//===--------------------------------------------------------------------------===//
def Air_MatMulOp : Air_Op<"MatMul", [Pure]> {
  let summary = "Standard matrix multiplication operation";
  let description = [{
    Performs standard matrix multiplication on two input tensors, summing over contracting dimensions.
    For example: tensor<AxB> * tensor<BxC> -> tensor<AxC>
  }];

  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2,
    Optional<Air_Tensor>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_x1,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_x2
  );

  let results = (outs
    Air_Tensor:$output
  );
}

//===--------------------------------------------------------------------------===//
// ExpOp
//===--------------------------------------------------------------------------===//
def Air_ExpOp : Air_Op<"Exp", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise exponential operation.";
  let description = [{
    Computes the exponential value of each element in the input tensor.
  }];
  let arguments = (ins Air_Tensor:$operand);
  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// MaxOp
//===--------------------------------------------------------------------------===//
def Air_MaximumOp : Air_Op<"Maximum", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise maximum operation.";
  let description = [{
    Computes the element-wise maximum of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// NegOp
//===--------------------------------------------------------------------------===//
def Air_NegNegOp : Air_Op<"Neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise negation operation.";
  let description = [{
    Computes the element-wise negation of the input tensor.
  }];
  let arguments = (ins Air_Tensor:$operand);
  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// ReduceSumOp
//===--------------------------------------------------------------------------===//
def Air_ReduceSumOp : Air_Op<"ReduceSum", [Pure]> {
  let summary = "Reduction sum operation.";
  let description = [{
    Applies sum reduction to elements along specified dimensions of the input tensor.
  }];
  let arguments = (ins
    Air_Tensor:$inputs,
    AnyType:$init_values,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReduceMaxOp
//===--------------------------------------------------------------------------===//
def Air_ReduceMaxOp : Air_Op<"ReduceMax", [Pure]> {
  let summary = "Reduction max operation.";
  let description = [{
    Applies max reduction to elements along specified dimensions of the input tensor.
  }];
  let arguments = (ins
    Air_Tensor:$inputs,
    AnyType:$init_values,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReshapeOp
//===--------------------------------------------------------------------------===//
def Air_ReshapeOp : Air_Op<"Reshape", [Pure]> {
  let summary = "Reshape operation.";
  let description = [{
    Reshapes a tensor to the specified shape while preserving total element count.
  }];
  let arguments = (ins
    Air_Tensor:$operand
  );
    let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$shape,
    DefaultValuedAttr<I64Attr, "0">:$axis,
    DefaultValuedAttr<I64Attr, "-1">:$num_axes
  );

  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// SubOp
//===--------------------------------------------------------------------------===//
def Air_SubOp : Air_Op<"Sub", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise subtraction operation.";
  let description = [{
    Performs element-wise subtraction of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

// --- func.call (for calling other functions like @relu) ---
// This would typically reuse existing func.call or define a similar call op
// def Air_CallOp : Air_Op<"call", [NoMemoryEffect]> { ... }

// --- func.func (for defining functions like @relu) ---
// This would typically reuse existing func.func or define a similar func op
// def Air_FuncOp : Air_Op<"func", [IsolatedFromAbove, Symbol]> { ... }

#endif // MAIR_OPS