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

class Mair_Op<string mnemonic, list<Trait> traits = []> :
    Op<AirDialect, mnemonic, traits>;

def AC_I1 : AnyTypeOf<[I<1>], "1-bit integer">;
def AC_I32OrI64 : AnyTypeOf<[I32, I64], "32/64-bit signed integer">;
def AC_UI32OrUI64 : AnyTypeOf<[UI<32>, UI<64>], "32/64-bit unsigned integer">;
def AC_Tensor : TensorOf<[AnyFloat, AC_I32OrI64, AC_UI32OrUI64, AC_I1]>;

#endif // MAIR_OP_BASE
