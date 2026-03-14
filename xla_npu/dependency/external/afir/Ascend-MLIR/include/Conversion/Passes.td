/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Passes.td - Conversion passes -----------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StableHLO to AFIR Conversion
//===----------------------------------------------------------------------===//

//def ConvertStableHLOToAFIR : Pass<"convert-stablehlo-to-afir", "mlir::ModuleOp"> {
//  let summary = "Convert StableHLO dialect to AFIR dialect";
//  let description = [{
//    This pass converts supported StableHLO operations to AFIR operations.
//    Currently supports: add, subtract, multiply, divide.
//  }];
//  let dependentDialects = [
//    "mlir::afir::AFIRDialect"
//  ];
//}

//===----------------------------------------------------------------------===//
// AFIR to ASC-IR (PyAsc) Conversion
//===----------------------------------------------------------------------===//

def ConvertAFIRToASCIRPass : Pass<"convert-afir-to-ascir", "mlir::ModuleOp"> {
  let summary = "Convert AFIR dialect to ASC-IR dialect";
  let description = [{
    This pass converts AFIR operations to ASC-IR operations from PyAsc.
    This is the final lowering step for Ascend hardware.
    It outputs the converted graph to stderr or to a file if --ascir-path is specified.
  }];
  let constructor = "mlir::afir::createConvertAFIRToASCIRPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect"
  ];
  let options = [
    Option<"ascirPath", "ascir-path", "std::string", /*default=*/"\"\"",
           "Path to output ASCIR text file">,
  ];
}

//===----------------------------------------------------------------------===//
// AFIR to ASCIR Text Conversion
//===----------------------------------------------------------------------===//

def ConvertAFIRToASCIRTextPass : Pass<"convert-afir-to-ascir-text", "mlir::ModuleOp"> {
  let summary = "Convert AFIR dialect to ASCIR text format";
  let description = [{
    This pass converts AFIR operations to ASCIR protobuf text format.
    It outputs the converted graph to stderr or to a file if --ascir-path is specified.
  }];
  let constructor = "mlir::afir::createConvertAFIRToASCIRTextPass()";
  let dependentDialects = [
    "mlir::afir::AFIRDialect"
  ];
  let options = [
    Option<"ascirPath", "ascir-path", "std::string", /*default=*/"\"\"",
           "Path to output ASCIR text file">,
  ];
}

#endif // AFIR_CONVERSION_PASSES
