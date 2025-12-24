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
def Air_Int8 : AnyTypeOf<[I8], "8-bit integer">;
def Air_Int16 : AnyTypeOf<[I16], "16-bit integer">;
def Air_Int32 : AnyTypeOf<[I32], "32-bit integer">;
def Air_Int64 : AnyTypeOf<[I64], "64-bit integer">;
def Air_Uint8 : AnyTypeOf<[UI<8>], "8-bit unsigned integer">;
def Air_Uint16 : AnyTypeOf<[UI<16>], "16-bit unsigned integer">;
def Air_Uint32 : AnyTypeOf<[UI<32>], "32-bit unsigned integer">;
def Air_Uint64 : AnyTypeOf<[UI<64>], "64-bit unsigned integer">;
def Air_I32OrI64 : AnyTypeOf<[I32, I64], "32/64-bit signed integer">;
def Air_I32OrI64Tensor : TensorOf<[Air_I32OrI64]>;
def Air_UI32OrUI64 : AnyTypeOf<[UI<32>, UI<64>], "32/64-bit unsigned integer">;
def Air_Tensor : TensorOf<[AnyFloat, Air_I32OrI64, Air_UI32OrUI64, Air_I1]>;
def Air_BasicType : AnyTypeOf<[F64, F32, F16, BF16, Air_Int8, Air_Int16, Air_Int32, Air_Int64,
                              Air_Uint8, Air_Uint16, Air_Uint32, Air_Uint64], "int uint qint float complex">;

#endif // MAIR_OP_BASE
