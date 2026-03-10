/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Copyright 2025 The OpenXLA Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Dialect/AFIR/AFIR.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

ABSL_FLAG(std::string, input, "", "Input AFIR file path");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);

    std::string input_file = absl::GetFlag(FLAGS_input);
    if (input_file.empty()) {
        std::cerr << "Error: Input file must be specified with --input flag" << std::endl;
        return 1;
    }

    std::cout << "Loading AFIR from: " << input_file << std::endl;

    // Read AFIR file
    std::ifstream ifs(input_file);
    if (!ifs) {
        std::cerr << "Error: Failed to open input file: " << input_file << std::endl;
        return 1;
    }

    std::string afir_string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    ifs.close();

    std::cout << "  AFIR size: " << afir_string.size() << " bytes" << std::endl;

    // Create MLIR context and register dialects
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::afir::AFIRDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    context->getOrLoadDialect<mlir::arith::ArithDialect>();

    // Parse AFIR string to MLIR module
    std::cout << "Parsing AFIR..." << std::endl;
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(afir_string), context.get());

    if (!module) {
        std::cerr << "Error: Failed to parse AFIR module" << std::endl;
        return 1;
    }

    std::cout << "  Parsed successfully" << std::endl;

    // Create PassManager and add AFIR to ASCIR conversion pass
    std::cout << "Converting AFIR to ASCIR..." << std::endl;
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::afir::createConvertAFIRToASCIRTextPass());

    // Run the pass
    mlir::LogicalResult result = pm.run(*module);

    if (mlir::failed(result)) {
        std::cerr << "Error: Failed to run AFIR to ASCIR conversion pass" << std::endl;
        return 1;
    }

    std::cout << "  Conversion successful" << std::endl;

    return 0;
}
