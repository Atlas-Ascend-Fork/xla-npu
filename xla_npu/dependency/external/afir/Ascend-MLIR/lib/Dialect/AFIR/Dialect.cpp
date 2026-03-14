/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- AFIRDialect.cpp - AFIR dialect implementation ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// AFIR dialect initialization
//===----------------------------------------------------------------------===//

// Include generated dialect implementation
#include "Dialect/AFIR/Dialect.cpp.inc"

// Include generated enum definitions
#include "Dialect/AFIR/Enums.cpp.inc"

// Include generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.cpp.inc"

// Include generated operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.cpp.inc"

void AFIRDialect::initialize() {
  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/AFIR/Attrs.cpp.inc"
      >();

  // Register operations
  addOperations<
#define GET_OP_LIST
#include "Dialect/AFIR/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type Parsing and Printing
//===----------------------------------------------------------------------===//

Type AFIRDialect::parseType(DialectAsmParser &parser) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated parser:
  // return generatedTypeParser(parser);
  return Type();
}

void AFIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // Currently AFIR uses standard MLIR types, no custom types yet
  // When custom types are added, use generated printer:
  // (void)generatedTypePrinter(type, printer);
}

//===----------------------------------------------------------------------===//
// Attribute Parsing and Printing
//===----------------------------------------------------------------------===//

Attribute AFIRDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrType;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrType, type, attr);
  if (parseResult.has_value() && succeeded(parseResult.value())) {
    return attr;
  }
  return Attribute();
}

void AFIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  if (failed(generatedAttributePrinter(attr, printer))) {
    return;
  }
}
