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

#include "xla/xla_npu/service/convert_to_afir/ascir_backend.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Python C API
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/xla_npu/dependency/tool/log.h"
#include "xla/xla_npu/service/convert_to_afir/convert_to_afir.h"

// Ascend-MLIR AFIR Dialect and Conversion
#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Dialect/AFIR/AFIR.h"

namespace xla {

namespace {

// Helper function to execute a shell command and get output
absl::StatusOr<std::string> ExecuteCommand(const std::string& command) {
    char buffer[128];
    std::string result;

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return absl::InternalError(absl::StrCat("Failed to execute command: ", command));
    }

    try {
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        return absl::InternalError("Error reading command output");
    }

    int return_code = pclose(pipe);
    if (return_code != 0) {
        return absl::InternalError(absl::StrCat("Command failed with code ", return_code, ": ", command));
    }

    return result;
}

// Helper function to write string to file
absl::Status WriteFile(absl::string_view content, const std::string& filepath) {
    std::ofstream ofs(filepath);
    if (!ofs) {
        return absl::InternalError(absl::StrCat("Failed to open file: ", filepath));
    }
    ofs << content;
    ofs.close();
    return absl::OkStatus();
}

// Helper function to read file to string
absl::StatusOr<std::string> ReadFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs) {
        return absl::InternalError(absl::StrCat("Failed to open file: ", filepath));
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

// Helper function to extract graph_name from ascir text
absl::StatusOr<std::string> ExtractGraphName(const std::string& ascir_text) {
    // Find the last occurrence of "graph_name:"
    size_t pos = ascir_text.rfind("graph_name:");
    if (pos == std::string::npos) {
        return absl::InternalError("graph_name not found in ascir text");
    }

    // Extract the line containing graph_name
    size_t line_start = ascir_text.rfind('\n', pos);
    if (line_start == std::string::npos) {
        line_start = 0;
    } else {
        line_start += 1;  // Skip the newline
    }

    size_t line_end = ascir_text.find('\n', pos);
    if (line_end == std::string::npos) {
        line_end = ascir_text.length();
    }

    std::string graph_line = ascir_text.substr(line_start, line_end - line_start);

    // Parse: graph_name: "name"
    size_t quote_start = graph_line.find('"', pos - line_start);
    if (quote_start == std::string::npos) {
        return absl::InternalError("Invalid graph_name format: missing opening quote");
    }

    size_t quote_end = graph_line.find('"', quote_start + 1);
    if (quote_end == std::string::npos) {
        return absl::InternalError("Invalid graph_name format: missing closing quote");
    }

    std::string graph_name = graph_line.substr(quote_start + 1, quote_end - quote_start - 1);
    return graph_name;
}

}  // namespace

AscirBackend::AscirBackend() = default;

AscirBackend::~AscirBackend() = default;

absl::StatusOr<std::string> AscirBackend::ConvertHloToAfirModule(const HloModule& hlo_module,
                                                                 const AscirBackendOptions& options) {
    // Create MLIR context
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::afir::AFIRDialect>();

    // Convert HLO to AFIR
    HloToAfirOptions afir_options;
    afir_options.preserve_names = true;
    afir_options.verify_module = true;
    afir_options.emit_afir_dialect = true;

    TF_ASSIGN_OR_RETURN(auto mlir_module, ConvertHloToAfir(hlo_module, context.get(), afir_options));

    // Serialize MLIR module to string immediately
    std::string afir_string;
    llvm::raw_string_ostream stream(afir_string);
    mlir_module->print(stream);

    return afir_string;
}

absl::StatusOr<std::string> AscirBackend::ConvertAfirToAscir(const std::string& afir_string,
                                                             const AscirBackendOptions& options) {
    // Create MLIR context for parsing AFIR
    auto context = std::make_unique<mlir::MLIRContext>();
    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::afir::AFIRDialect>();

    // Parse AFIR string to MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(afir_string), context.get());
    if (!module) {
        return absl::InternalError("Failed to parse AFIR string");
    }

    // Create temporary file for ASCIR output
    std::string temp_ascir_file = absl::StrCat(options.output_dir, "/temp_output.ascir");

    // Create PassManager to run the conversion pass
    mlir::PassManager pm(context.get());
    pm.addPass(mlir::afir::createConvertAFIRToASCIRTextPass());

    // Capture stderr to get the output (since pass outputs to stderr by default)
    // Save the original stderr
    FILE* original_stderr = stderr;
    fflush(stderr);

    // Redirect stderr to a file
    FILE* temp_stderr = freopen(temp_ascir_file.c_str(), "w", stderr);
    if (!temp_stderr) {
        return absl::InternalError(absl::StrCat("Failed to open temp file: ", temp_ascir_file));
    }

    // Run the pass
    mlir::LogicalResult result = pm.run(*module);

    // Restore stderr
    fflush(stderr);
    stderr = original_stderr;
    fclose(temp_stderr);

    if (mlir::failed(result)) {
        return absl::InternalError("Failed to run AFIR to ASCIR conversion pass");
    }

    // Read the generated ASCIR file
    TF_ASSIGN_OR_RETURN(auto ascir_text, ReadFile(temp_ascir_file));

    // Clean up temporary file if not preserving intermediates
    if (!options.preserve_intermediates) {
        std::remove(temp_ascir_file.c_str());
    }

    return ascir_text;
}

absl::StatusOr<AscirBackendResult> AscirBackend::GenerateKernel(const std::string& ascir_text,
                                                                const AscirBackendOptions& options) {
    AscirBackendResult result;

    // Extract graph_name from ascir text
    TF_ASSIGN_OR_RETURN(auto graph_name, ExtractGraphName(ascir_text));
    result.graph_name = graph_name;
    std::cerr << "[DEBUG] Extracted graph_name: " << graph_name << std::endl;

    // Save ASCIR to file for reference
    std::string ascir_file = absl::StrCat(options.output_dir, "/graph.ascir");
    TF_RETURN_IF_ERROR(WriteFile(ascir_text, ascir_file));
    result.ascir_file = ascir_file;

    // Initialize Python interpreter
    NPU_LOG_FORMAT_PRINT(DEBUG, "About to initialize Python interpreter...");

    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            return absl::InternalError("Failed to initialize Python interpreter");
        }
    }
    NPU_LOG_FORMAT_PRINT(DEBUG, "Python initialized successfully");

    // Set environment variable for SOC version
    std::string set_soc_cmd = absl::StrFormat("import os; os.environ['SOC_VERSION'] = '%s'", options.soc_version);
    PyRun_SimpleString(set_soc_cmd.c_str());

    // Add runtime path to Python path
    const char* ascend_mlir_path = std::getenv("ASCEND_MLIR_PYTHON_PATH");
    std::string runtime_path = ascend_mlir_path ? ascend_mlir_path : "";
    if (runtime_path.empty()) {
        return absl::InternalError(
            "ASCEND_MLIR_PYTHON_PATH environment variable not set. "
            "Please set it to xla_npu/third_party/Ascend-MLIR/python");
    }
    std::cerr << "[DEBUG] Adding runtime path: " << runtime_path << std::endl;
    std::string import_cmd = absl::StrFormat("import sys; sys.path.insert(0, '%s')", runtime_path);
    PyRun_SimpleString(import_cmd.c_str());
    std::cerr << "[DEBUG] Runtime path added" << std::endl;

    // Import runtime module
    std::cerr << "[DEBUG] Importing runtime module..." << std::endl;
    PyObject* runtime_module = PyImport_ImportModule("runtime");
    std::cerr << "[DEBUG] Runtime module imported" << std::endl;
    if (!runtime_module) {
        PyErr_Print();
        return absl::InternalError("Failed to import runtime module");
    }

    // Get AscGen class
    PyObject* ascgen_class = PyObject_GetAttrString(runtime_module, "AscGen");
    Py_DECREF(runtime_module);
    if (!ascgen_class) {
        PyErr_Print();
        return absl::InternalError("Failed to get AscGen class");
    }

    // Create AscGen instance
    PyObject* ascgen_instance = PyObject_CallObject(ascgen_class, nullptr);
    Py_DECREF(ascgen_class);
    if (!ascgen_instance) {
        PyErr_Print();
        return absl::InternalError("Failed to create AscGen instance");
    }

    // Prepare compile_graph arguments
    // Create Path object for output_dir
    PyObject* sys_module = PyImport_ImportModule("pathlib");
    if (!sys_module) {
        Py_DECREF(ascgen_instance);
        PyErr_Print();
        return absl::InternalError("Failed to import pathlib");
    }
    PyObject* path_class = PyObject_GetAttrString(sys_module, "Path");
    Py_DECREF(sys_module);
    if (!path_class) {
        Py_DECREF(ascgen_instance);
        PyErr_Print();
        return absl::InternalError("Failed to get Path class");
    }

    PyObject* output_dir_py = PyObject_CallFunction(path_class, "s", options.output_dir.c_str());
    if (!output_dir_py) {
        Py_DECREF(path_class);
        Py_DECREF(ascgen_instance);
        PyErr_Print();
        return absl::InternalError("Failed to create Path object");
    }

    // Get compile_graph method
    PyObject* compile_graph_method = PyObject_GetAttrString(ascgen_instance, "compile_graph");
    if (!compile_graph_method) {
        Py_DECREF(ascgen_instance);
        Py_DECREF(output_dir_py);
        PyErr_Print();
        return absl::InternalError("Failed to get compile_graph method");
    }

    // Call compile_graph(graph_text, output_dir, graph_name)
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(ascir_text.c_str()));
    PyTuple_SetItem(args, 1, output_dir_py);  // Steals reference
    PyTuple_SetItem(args, 2, PyUnicode_FromString(graph_name.c_str()));

    PyObject* compile_result = PyObject_CallObject(compile_graph_method, args);
    Py_DECREF(args);
    Py_DECREF(compile_graph_method);

    if (!compile_result) {
        Py_DECREF(ascgen_instance);
        PyErr_Print();
        return absl::InternalError("Failed to call compile_graph");
    }

    // Parse result: (host_file, device_file, host_tiling, tiling_def)
    if (!PyTuple_Check(compile_result) || PyTuple_Size(compile_result) != 4) {
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        return absl::InternalError("Invalid compile_graph result format");
    }

    // Extract host_file path (returns Path object, need to convert to string)
    PyObject* host_file_obj = PyTuple_GetItem(compile_result, 0);  // Borrowed
    PyObject* host_file_str_obj = PyObject_Str(host_file_obj);     // Convert Path to str
    if (host_file_str_obj) {
        const char* host_file_str = PyUnicode_AsUTF8(host_file_str_obj);
        if (host_file_str) {
            result.host_file = host_file_str;
        }
        Py_DECREF(host_file_str_obj);
    }

    // Extract device_file path (returns Path object, need to convert to string)
    PyObject* device_file_obj = PyTuple_GetItem(compile_result, 1);  // Borrowed
    PyObject* device_file_str_obj = PyObject_Str(device_file_obj);   // Convert Path to str
    if (device_file_str_obj) {
        const char* device_file_str = PyUnicode_AsUTF8(device_file_str_obj);
        if (device_file_str) {
            result.device_file = device_file_str;
        }
        Py_DECREF(device_file_str_obj);
    }

    // Extract host_tiling and tiling_def for calc_tiling_data
    PyObject* host_tiling_obj = PyTuple_GetItem(compile_result, 2);  // Borrowed
    PyObject* tiling_def_obj = PyTuple_GetItem(compile_result, 3);   // Borrowed

    // Both should be strings (unicode)
    if (!PyUnicode_Check(host_tiling_obj) || !PyUnicode_Check(tiling_def_obj)) {
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        return absl::InternalError("host_tiling and tiling_def must be strings");
    }

    // Increment references since we're using them in PyTuple_SetItem
    Py_INCREF(tiling_def_obj);
    Py_INCREF(host_tiling_obj);

    // Get calc_tiling_data method
    PyObject* calc_tiling_method = PyObject_GetAttrString(ascgen_instance, "calc_tiling_data");
    if (!calc_tiling_method) {
        Py_DECREF(tiling_def_obj);
        Py_DECREF(host_tiling_obj);
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        PyErr_Print();
        return absl::InternalError("Failed to get calc_tiling_data method");
    }

    // Call calc_tiling_data(tiling_def, host_tiling, output_dir)
    PyObject* tiling_args = PyTuple_New(3);
    PyTuple_SetItem(tiling_args, 0, tiling_def_obj);   // Steals reference
    PyTuple_SetItem(tiling_args, 1, host_tiling_obj);  // Steals reference

    // Create Path object for output_dir
    PyObject* output_dir_py2 = PyObject_CallFunction(path_class, "s", options.output_dir.c_str());
    if (!output_dir_py2) {
        Py_DECREF(tiling_args);
        Py_DECREF(calc_tiling_method);
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        PyErr_Print();
        return absl::InternalError("Failed to create Path object for output_dir");
    }
    PyTuple_SetItem(tiling_args, 2, output_dir_py2);  // Steals reference

    // Call the method
    PyObject* tiling_result = PyObject_CallObject(calc_tiling_method, tiling_args);
    Py_DECREF(tiling_args);
    Py_DECREF(calc_tiling_method);

    if (!tiling_result) {
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        PyErr_Print();
        return absl::InternalError("Failed to call calc_tiling_data");
    }

    // Extract tiling data bytes
    if (PyBytes_Check(tiling_result)) {
        char* tiling_buffer;
        Py_ssize_t tiling_size;
        if (PyBytes_AsStringAndSize(tiling_result, &tiling_buffer, &tiling_size) == 0) {
            result.tiling_data.assign(tiling_buffer, tiling_buffer + tiling_size);
            result.tiling_data_file = absl::StrCat(options.output_dir, "/tiling_data.bin");

            // Also save tiling data to file
            std::ofstream tiling_file(result.tiling_data_file, std::ios::binary);
            if (tiling_file) {
                tiling_file.write(tiling_buffer, tiling_size);
                tiling_file.close();
            }
        }
    }

    Py_DECREF(tiling_result);
    Py_DECREF(compile_result);
    Py_DECREF(ascgen_instance);
    Py_DECREF(path_class);

    // Step 4: Compile device kernel using runtime.compile_kernel (Bisheng compiler)
    // Re-import runtime module to get compile_kernel function
    PyObject* runtime_module_compile = PyImport_ImportModule("runtime");
    if (!runtime_module_compile) {
        PyErr_Print();
        return absl::InternalError("Failed to import runtime module for compile_kernel");
    }

    // Get compile_kernel function from runtime module
    PyObject* compile_kernel_func = PyObject_GetAttrString(runtime_module_compile, "compile_kernel");
    Py_DECREF(runtime_module_compile);
    if (!compile_kernel_func) {
        PyErr_Print();
        return absl::InternalError("Failed to get compile_kernel function from runtime");
    }

    // Check if it's callable
    if (!PyCallable_Check(compile_kernel_func)) {
        Py_DECREF(compile_kernel_func);
        return absl::InternalError("compile_kernel is not callable");
    }

    // Prepare positional arguments: src_file, output_dir, kernel_name
    PyObject* kernel_args = PyTuple_New(3);
    if (!kernel_args) {
        Py_DECREF(compile_kernel_func);
        return absl::InternalError("Failed to create kernel_args tuple");
    }

    // Set tuple items (steals references)
    PyObject* src_file_obj = PyUnicode_FromString(result.device_file.c_str());
    if (!src_file_obj) {
        Py_DECREF(kernel_args);
        Py_DECREF(compile_kernel_func);
        return absl::InternalError("Failed to create src_file string");
    }

    PyObject* output_dir_obj = PyUnicode_FromString(options.output_dir.c_str());
    if (!output_dir_obj) {
        Py_DECREF(src_file_obj);
        Py_DECREF(kernel_args);
        Py_DECREF(compile_kernel_func);
        return absl::InternalError("Failed to create output_dir string");
    }

    PyObject* kernel_name_obj = PyUnicode_FromString(graph_name.c_str());
    if (!kernel_name_obj) {
        Py_DECREF(src_file_obj);
        Py_DECREF(output_dir_obj);
        Py_DECREF(kernel_args);
        Py_DECREF(compile_kernel_func);
        return absl::InternalError("Failed to create kernel_name string");
    }

    // PyTuple_SetItem steals the reference, so we don't decref separately
    PyTuple_SetItem(kernel_args, 0, src_file_obj);
    PyTuple_SetItem(kernel_args, 1, output_dir_obj);
    PyTuple_SetItem(kernel_args, 2, kernel_name_obj);

    // Call compile_kernel(src_file, output_dir, kernel_name)
    PyObject* kernel_result = PyObject_CallObject(compile_kernel_func, kernel_args);
    Py_DECREF(kernel_args);
    Py_DECREF(compile_kernel_func);

    if (!kernel_result) {
        PyErr_Print();
        return absl::InternalError("Failed to call compile_kernel");
    }

    // Extract result: returns path to compiled binary
    if (!PyUnicode_Check(kernel_result)) {
        Py_DECREF(kernel_result);
        return absl::InternalError("compile_kernel should return a string");
    }

    const char* device_binary_str = PyUnicode_AsUTF8(kernel_result);
    if (!device_binary_str) {
        Py_DECREF(kernel_result);
        return absl::InternalError("Failed to convert result to string");
    }

    result.device_binary_file = device_binary_str;
    Py_DECREF(kernel_result);

    return result;
}

absl::StatusOr<AscirBackendResult> AscirBackend::Compile(const HloModule& hlo_module,
                                                         const AscirBackendOptions& options) {
    // Step 1: Convert HLO to AFIR (in-memory)
    TF_ASSIGN_OR_RETURN(auto afir_string, ConvertHloToAfirModule(hlo_module, options));

    // Create output directory if it doesn't exist
    std::string mkdir_cmd = absl::StrCat("mkdir -p ", options.output_dir);
    TF_ASSIGN_OR_RETURN(auto _, ExecuteCommand(mkdir_cmd));

    // Step 2: Convert AFIR to AscIR (using Python script)
    TF_ASSIGN_OR_RETURN(auto ascir_text, ConvertAfirToAscir(afir_string, options));

    // Step 3: Generate kernel and tiling files using Ascend runtime
    TF_ASSIGN_OR_RETURN(auto result, GenerateKernel(ascir_text, options));

    return result;
}

}  // namespace xla
