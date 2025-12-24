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

def Air_ConstantOp : Air_Op<"constant", [
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

def Air_AddOp : Air_Op<"add",[
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

def Air_BroadcastOp : Air_Op<"broadcast", [Pure]>{
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

def Air_MulOp : Air_Op<"mul", [Pure]>{
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

def Air_TanhOp : Air_Op<"tanh", [Pure]>{
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

#endif // MAIR_OPS