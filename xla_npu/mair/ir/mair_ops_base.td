#ifndef MAIR_OP_BASE
#define MAIR_OP_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AirDialect : Dialect {
  let name = "mair";

  let description = [{
    The mair dialect.
  }];

  let cppNamespace = "::mlir::air";
}

class Air_Op<string mnemonic, list<Trait> traits = []> :
    Op<AirDialect, mnemonic, traits>;

def Air_I1 : AnyTypeOf<[I<1>], "1-bit integer">;
def Air_I32OrI64 : AnyTypeOf<[I32, I64], "32/64-bit signed integer">;
def Air_UI32OrUI64 : AnyTypeOf<[UI<32>, UI<64>], "32/64-bit unsigned integer">;
def Air_Tensor : TensorOf<[AnyFloat, Air_I32OrI64, Air_UI32OrUI64, Air_I1]>;

#endif // MAIR_OP_BASE
