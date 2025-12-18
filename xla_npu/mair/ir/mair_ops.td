#ifndef MAIR_OPS
#define MAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mair_ops_base.td"

def Mair_AddOp : Mair_Op<"Add", [Commutative]> {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    AC_Tensor:$x,
    AC_Tensor:$y
  );

  let results = (outs
    AC_Tensor:$z
  );
}

#endif // MAIR_OPS