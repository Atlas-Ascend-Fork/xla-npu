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

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <map>
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

// Helper function to add axis information to ASCIR text
// This is needed when converting individual computations that lack axis context
std::string AddAxisInfoToAscir(const std::string& ascir_text, const std::vector<int64_t>& shape) {
    std::string result = ascir_text;

    // Find the asc_graph_attr section
    size_t attr_pos = result.find("asc_graph_attr {");
    if (attr_pos == std::string::npos) {
        return result;  // No asc_graph_attr found, return as-is
    }

    // Find the closing brace of asc_graph_attr
    size_t attr_end = result.find("}", attr_pos);
    if (attr_end == std::string::npos) {
        return result;
    }

    // Build axis information with proper id fields
    std::string axis_info;
    for (size_t i = 0; i < shape.size(); ++i) {
        // All axes need id field
        // 注意，align参数的类型在新老版本cann包的proto文件中的定义不同，新版本cann包定义为string类型（需要加双引号）
        // 老版本cann包定义为int32类型（不需要加双引号）
        axis_info += absl::StrFormat(
            "  axis {\n"
            "    id: %zu\n"
            "    name: \"z%zu\"\n"
            "    size: \"%lld\"\n"
            "    align: 1\n"
            "    allow_unaligned_tail: true\n"
            "  }\n",
            i, i, shape[i]);
    }

    // Insert axis info before the closing brace
    result.insert(attr_end, axis_info);

    return result;
}

// Helper function to fix sched.axis for nodes that have axis_ids but missing sched.axis
std::string FixNodeSchedAxis(const std::string& ascir_text) {
    std::string result = ascir_text;

    // Strategy: Count how many axis definitions are in asc_graph_attr
    // Then for any node with sched that's missing axis, add all axes
    size_t graph_attr_pos = result.find("asc_graph_attr {");
    size_t num_axes = 0;

    NPU_LOG_PRINT(DEBUG, "FixNodeSchedAxis: Starting to process ASCIR text");

    if (graph_attr_pos != std::string::npos) {
        // Find the end of asc_graph_attr (matching closing brace at start of line)
        size_t graph_attr_end = result.find("\n}\n", graph_attr_pos);
        if (graph_attr_end == std::string::npos) {
            graph_attr_end = result.find("}\nasc_node", graph_attr_pos);
        }

        if (graph_attr_end != std::string::npos) {
            std::string graph_attr_section = result.substr(graph_attr_pos, graph_attr_end - graph_attr_pos);

            // Count axis definitions
            size_t pos = 0;
            while ((pos = graph_attr_section.find("axis {", pos)) != std::string::npos) {
                num_axes++;
                pos += 7;
            }
        }
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Found %d axes in asc_graph_attr", num_axes);

    if (num_axes == 0) {
        // Couldn't determine axis count, return as-is
        NPU_LOG_PRINT(DEBUG, "No axes found, returning without modifications");
        return result;
    }

    // Now find all sched sections that are missing axis and add it
    size_t pos = 0;
    size_t fixed_count = 0;
    size_t total_sched = 0;

    while ((pos = result.find("sched {\n", pos)) != std::string::npos) {
        total_sched++;
        size_t sched_end = result.find("}", pos);
        NPU_LOG_FORMAT_PRINT(DEBUG, "Found sched at position %d, sched_end at %d", pos, sched_end);

        if (sched_end == std::string::npos || sched_end > pos + 200) {
            NPU_LOG_PRINT(DEBUG, "Skipping sched (invalid end)");
            pos += 8;
            continue;
        }

        // Extract the sched section content
        std::string sched_content = result.substr(pos + 8, sched_end - (pos + 8));
        NPU_LOG_FORMAT_PRINT(DEBUG, "Sched content: [%s]", sched_content.c_str());

        // Check if this sched already has axis fields (but not loop_axis)
        // We need to check for "axis:" at the start of a line (with whitespace)
        bool has_axis_field = false;
        size_t search_pos = 0;
        while ((search_pos = sched_content.find("axis:", search_pos)) != std::string::npos) {
            // Check if it's at start of line or preceded only by whitespace
            if (search_pos == 0 || sched_content[search_pos - 1] == ' ' || sched_content[search_pos - 1] == '\t' ||
                sched_content[search_pos - 1] == '\n') {
                // Make sure it's not part of "loop_axis"
                if (search_pos < 4 || sched_content.substr(search_pos - 4, 4) != "loop") {
                    has_axis_field = true;
                    break;
                }
            }
            search_pos += 5;
        }

        if (!has_axis_field) {
            NPU_LOG_FORMAT_PRINT(DEBUG, "This sched is missing axis, adding %d axes", num_axes);
            // This sched is missing axis, add all axes
            std::string axis_fields;
            for (size_t i = 0; i < num_axes; ++i) {
                axis_fields += absl::StrFormat("      axis: %zu\n", i);
            }

            // Insert before the closing brace
            result.insert(sched_end, axis_fields);
            sched_end += axis_fields.length();
            fixed_count++;

            NPU_LOG_FORMAT_PRINT(DEBUG, "Fixed sched at position %d, added %d axes", pos, num_axes);
        } else {
            NPU_LOG_PRINT(DEBUG, "This sched already has axis fields");
        }

        pos = sched_end + 1;
    }

    NPU_LOG_FORMAT_PRINT(DEBUG, "Processed %d sched sections, fixed %d nodes", total_sched, fixed_count);

    return result;
}

}  // namespace

AscirBackend::AscirBackend() = default;

AscirBackend::~AscirBackend() = default;

absl::StatusOr<std::vector<std::string>> AscirBackend::ConvertHloToAfirModule(const HloModule& hlo_module,
                                                                              const AscirBackendOptions& options) {
    std::vector<std::string> afir_strings;

    const HloComputation* entry = hlo_module.entry_computation();

    // Only convert non-entry computations
    for (const HloComputation* computation : hlo_module.computations()) {
        if (computation == entry) {
            NPU_LOG_FORMAT_PRINT(INFO, "Skipping entry computation: %s", std::string(computation->name()).c_str());
            continue;
        }

        NPU_LOG_FORMAT_PRINT(INFO, "Converting non-entry computation: %s", std::string(computation->name()).c_str());

        // Create MLIR context for this computation
        auto context = std::make_unique<mlir::MLIRContext>();
        context->loadDialect<mlir::func::FuncDialect>();
        context->loadDialect<mlir::arith::ArithDialect>();
        context->loadDialect<mlir::afir::AFIRDialect>();

        // Convert this single computation (with all its dependencies) to AFIR
        HloToAfirOptions afir_options;
        afir_options.preserve_names = true;
        afir_options.verify_module = true;
        afir_options.emit_afir_dialect = true;

        TF_ASSIGN_OR_RETURN(auto mlir_module,
                            ConvertHloComputationToAfir(*computation, hlo_module, context.get(), afir_options));

        // Serialize MLIR module to string immediately
        std::string afir_string;
        llvm::raw_string_ostream stream(afir_string);
        mlir_module->print(stream);

        afir_strings.push_back(afir_string);
    }

    NPU_LOG_FORMAT_PRINT(INFO, "Converted %d non-entry computations to AFIR", afir_strings.size());

    return afir_strings;
}

absl::StatusOr<std::string> AscirBackend::ConvertAfirToAscir(const std::string& afir_string,
                                                             const AscirBackendOptions& options) {
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Starting conversion");

    // Create MLIR context for parsing AFIR
    auto context = std::make_unique<mlir::MLIRContext>();
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Created MLIR context");

    context->loadDialect<mlir::func::FuncDialect>();
    context->loadDialect<mlir::arith::ArithDialect>();
    context->loadDialect<mlir::afir::AFIRDialect>();
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Loaded dialects");

    // Parse AFIR string to MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(afir_string), context.get());
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Parsed AFIR string");

    if (!module) {
        return absl::InternalError("Failed to parse AFIR string");
    }

    // Create temporary file for ASCIR output
    std::string temp_ascir_file = absl::StrCat(options.output_dir, "/temp_output.ascir");
    NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertAfirToAscir: Created temp file path: %s", temp_ascir_file.c_str());

    // Create PassManager to run the conversion pass
    mlir::PassManager pm(context.get());
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Created PassManager");

    pm.addPass(mlir::afir::createConvertAFIRToASCIRTextPass());
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Added conversion pass");

    // Capture stderr to get the output (since pass outputs to stderr by default)
    // Use dup2 to properly save and restore stderr
    int stderr_fd = dup(fileno(stderr));
    if (stderr_fd == -1) {
        return absl::InternalError("Failed to duplicate stderr file descriptor");
    }

    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: About to redirect stderr");

    // Redirect stderr to a file
    FILE* temp_file = fopen(temp_ascir_file.c_str(), "w");
    if (!temp_file) {
        close(stderr_fd);
        return absl::InternalError(absl::StrCat("Failed to open temp file: ", temp_ascir_file));
    }

    dup2(fileno(temp_file), fileno(stderr));

    // Run the pass
    mlir::LogicalResult result = pm.run(*module);

    // Flush and close the temp file
    fflush(stderr);
    fclose(temp_file);

    // Restore stderr using dup2
    dup2(stderr_fd, fileno(stderr));
    close(stderr_fd);

    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Stderr restored");

    if (mlir::failed(result)) {
        return absl::InternalError("Failed to run AFIR to ASCIR conversion pass");
    }
    NPU_LOG_PRINT(DEBUG, "ConvertAfirToAscir: Pass succeeded");

    // Read the generated ASCIR file
    TF_ASSIGN_OR_RETURN(auto ascir_text, ReadFile(temp_ascir_file));

    // Clean up temporary file if not preserving intermediates
    if (!options.preserve_intermediates) {
        std::remove(temp_ascir_file.c_str());
    }

    return ascir_text;
}

absl::StatusOr<SingleComputationResult> AscirBackend::GenerateKernel(const std::string& ascir_text,
                                                                     const std::string& computation_name,
                                                                     const AscirBackendOptions& options) {
    SingleComputationResult result;
    result.computation_name = computation_name;

    // Extract graph_name from ascir text
    TF_ASSIGN_OR_RETURN(auto graph_name, ExtractGraphName(ascir_text));
    result.graph_name = graph_name;
    NPU_LOG_FORMAT_PRINT(DEBUG, "Computation: %s, Extracted graph_name: %s", computation_name.c_str(),
                         graph_name.c_str());

    // Create computation-specific output directory
    std::string computation_output_dir = absl::StrCat(options.output_dir, "/", computation_name);
    std::string mkdir_cmd = absl::StrCat("mkdir -p ", computation_output_dir);
    TF_ASSIGN_OR_RETURN(auto _, ExecuteCommand(mkdir_cmd));

    // Save ASCIR to file for reference (use graph_name as filename)
    std::string ascir_file = absl::StrCat(computation_output_dir, "/", graph_name, ".ascir");
    TF_RETURN_IF_ERROR(WriteFile(ascir_text, ascir_file));
    result.ascir_file = ascir_file;
    NPU_LOG_FORMAT_PRINT(DEBUG, "ASCIR saved to: %s", ascir_file.c_str());

    // Initialize Python interpreter
    NPU_LOG_FORMAT_PRINT(DEBUG, "About to initialize Python interpreter for %s...", computation_name.c_str());
    fflush(stderr);
    if (!Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            return absl::InternalError("Failed to initialize Python interpreter");
        }
    }
    NPU_LOG_PRINT(DEBUG, "Python initialized successfully");

    PyGILState_STATE gstate = PyGILState_Ensure();
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
    NPU_LOG_FORMAT_PRINT(DEBUG, "Adding runtime path: %s", runtime_path.c_str());
    std::string import_cmd = absl::StrFormat("import sys; sys.path.insert(0, '%s')", runtime_path);
    PyRun_SimpleString(import_cmd.c_str());
    NPU_LOG_PRINT(DEBUG, "Runtime path added");

    // Import runtime module
    NPU_LOG_PRINT(DEBUG, "Importing runtime module...");
    PyObject* runtime_module = PyImport_ImportModule("runtime");
    NPU_LOG_PRINT(DEBUG, "Runtime module imported");
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

    PyObject* output_dir_py = PyObject_CallFunction(path_class, "s", computation_output_dir.c_str());
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

    // Extract host_tiling and tiling_def for jit_compile
    PyObject* host_tiling_obj = PyTuple_GetItem(compile_result, 2);  // Borrowed
    PyObject* tiling_def_obj = PyTuple_GetItem(compile_result, 3);   // Borrowed

    // Both should be strings (unicode)
    if (!PyUnicode_Check(host_tiling_obj) || !PyUnicode_Check(tiling_def_obj)) {
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        return absl::InternalError("host_tiling and tiling_def must be strings");
    }

    // Increment references since we need them after compile_result is decref'd
    Py_INCREF(host_tiling_obj);
    Py_INCREF(tiling_def_obj);

    // Read device_file content for jit_compile (device_impl is not in compile_result)
    std::string device_impl;
    std::ifstream device_file_stream(result.device_file);
    if (!device_file_stream.is_open()) {
        Py_DECREF(compile_result);
        Py_DECREF(ascgen_instance);
        Py_DECREF(path_class);
        return absl::InternalError(absl::StrCat("Failed to open device_file: ", result.device_file));
    }
    std::stringstream device_buffer;
    device_buffer << device_file_stream.rdbuf();
    device_impl = device_buffer.str();
    device_file_stream.close();

    NPU_LOG_FORMAT_PRINT(DEBUG, "Read device_file (%s) size: %d", result.device_file.c_str(), device_impl.size());
    NPU_LOG_PRINT(DEBUG, "Using host_tiling_obj and tiling_def_obj from memory");

    // Clean up compile_result and related objects
    Py_DECREF(compile_result);
    Py_DECREF(ascgen_instance);
    Py_DECREF(path_class);

    // Step 4: Compile device kernel using autofuse.compile_adapter.jit_compile
    // Import autofuse.compile_adapter module
    PyObject* autofuse_module = PyImport_ImportModule("autofuse.compile_adapter");
    if (!autofuse_module) {
        PyErr_Print();
        return absl::InternalError("Failed to import autofuse.compile_adapter module");
    }

    // Get jit_compile function from autofuse.compile_adapter module
    PyObject* jit_compile_func = PyObject_GetAttrString(autofuse_module, "jit_compile");
    Py_DECREF(autofuse_module);
    if (!jit_compile_func) {
        PyErr_Print();
        return absl::InternalError("Failed to get jit_compile function from autofuse.compile_adapter");
    }

    // Check if it's callable
    if (!PyCallable_Check(jit_compile_func)) {
        Py_DECREF(jit_compile_func);
        return absl::InternalError("jit_compile is not callable");
    }

    // Prepare output_file parameter
    std::string output_file = absl::StrCat(options.output_dir, "/", graph_name, ".so");

    // Prepare parameters list: [graph_name, output_file, output_path, "--force_unknown=True"]
    // Based on test_ascir.py test_mode_0: jit_compile(tiling_def, host_impl, device_impl, [graph_name, output_file,
    // output_path, "--force_unknown=True"])

    // Prepare arguments list (4th parameter is a list)
    PyObject* args_list = PyList_New(4);
    if (!args_list) {
        Py_DECREF(jit_compile_func);
        return absl::InternalError("Failed to create args_list");
    }

    // Set list items
    std::string graph_name_arg = absl::StrCat("--graph_name=", graph_name);
    std::string output_file_arg = absl::StrCat("--output_file=", output_file);
    std::string output_path_arg = absl::StrCat("--output_path=", options.output_dir);
    const char* force_unknown_arg = "--force_unknown=True";

    PyList_SetItem(args_list, 0, PyUnicode_FromString(graph_name_arg.c_str()));
    PyList_SetItem(args_list, 1, PyUnicode_FromString(output_file_arg.c_str()));
    PyList_SetItem(args_list, 2, PyUnicode_FromString(output_path_arg.c_str()));
    PyList_SetItem(args_list, 3, PyUnicode_FromString(force_unknown_arg));

    // Prepare positional arguments: tiling_def, host_impl, device_impl, args_list
    PyObject* kernel_args = PyTuple_New(4);
    if (!kernel_args) {
        Py_DECREF(args_list);
        Py_DECREF(jit_compile_func);
        return absl::InternalError("Failed to create kernel_args tuple");
    }

    // Set tuple items (steals references)
    // Note: tiling_def_obj and host_tiling_obj already have their references incremented
    // at line 383-384, so PyTuple_SetItem will steal those references correctly

    PyObject* device_impl_py = PyUnicode_FromString(device_impl.c_str());
    if (!device_impl_py) {
        Py_DECREF(tiling_def_obj);
        Py_DECREF(host_tiling_obj);
        Py_DECREF(args_list);
        Py_DECREF(kernel_args);
        Py_DECREF(jit_compile_func);
        return absl::InternalError("Failed to create device_impl string");
    }

    // PyTuple_SetItem steals the reference, so we don't decref separately
    PyTuple_SetItem(kernel_args, 0, tiling_def_obj);
    PyTuple_SetItem(kernel_args, 1, host_tiling_obj);
    PyTuple_SetItem(kernel_args, 2, device_impl_py);
    PyTuple_SetItem(kernel_args, 3, args_list);

    NPU_LOG_FORMAT_PRINT(DEBUG, "Calling jit_compile with graph_name: %s", graph_name.c_str());
    NPU_LOG_FORMAT_PRINT(DEBUG, "output_file: %s", output_file.c_str());
    NPU_LOG_FORMAT_PRINT(DEBUG, "output_path: %s", options.output_dir.c_str());

    // Call jit_compile(tiling_def, host_impl, device_impl, args_list)
    PyObject* kernel_result = PyObject_CallObject(jit_compile_func, kernel_args);
    Py_DECREF(kernel_args);
    Py_DECREF(jit_compile_func);

    if (!kernel_result) {
        PyErr_Print();
        return absl::InternalError("Failed to call jit_compile");
    }

    // jit_compile returns None, the compiled .so file is at output_file path
    Py_DECREF(kernel_result);

    // Set device_binary_file to the output .so file
    result.device_binary_file = output_file;

    NPU_LOG_FORMAT_PRINT(DEBUG, "jit_compile completed, output at: %s", result.device_binary_file.c_str());

    PyGILState_Release(gstate);
    return result;
}

absl::StatusOr<NpuCompileResult> AscirBackend::Compile(const HloModule& hlo_module,
                                                         const AscirBackendOptions& options) {
    NpuCompileResult final_result;

    // Step 1: Convert HLO to AFIR for each non-entry computation
    TF_ASSIGN_OR_RETURN(auto afir_strings, ConvertHloToAfirModule(hlo_module, options));

    // Create output directory if it doesn't exist
    std::string mkdir_cmd = absl::StrCat("mkdir -p ", options.output_dir);
    TF_ASSIGN_OR_RETURN(auto _, ExecuteCommand(mkdir_cmd));

    // Get entry computation to skip
    const HloComputation* entry = hlo_module.entry_computation();

    // Process each non-entry computation
    int idx = 0;
    for (const HloComputation* computation : hlo_module.computations()) {
        // Skip entry computation
        if (computation == entry) {
            continue;
        }

        const std::string& afir_string = afir_strings[idx];
        const std::string computation_name = std::string(computation->name());

        NPU_LOG_FORMAT_PRINT(INFO, "Processing computation %d/%d: %s", idx + 1, afir_strings.size(),
                             computation_name.c_str());
        NPU_LOG_FORMAT_PRINT(DEBUG, "About to call ConvertAfirToAscir for %s", computation_name.c_str());
        NPU_LOG_FORMAT_PRINT(DEBUG, "AFIR string size: %d bytes", afir_string.size());

        // Step 2: Convert AFIR to AscIR
        TF_ASSIGN_OR_RETURN(auto ascir_text, ConvertAfirToAscir(afir_string, options));

        NPU_LOG_FORMAT_PRINT(DEBUG, "ConvertAfirToAscir completed, ASCIR size: %d bytes", ascir_text.size());

        // Extract shape information from computation's root instruction
        // to add axis information to ASCIR
        const HloInstruction* root = computation->root_instruction();
        const Shape& root_shape = root->shape();
        std::vector<int64_t> shape_dims;
        if (root_shape.IsArray()) {
            auto dims_span = root_shape.dimensions();
            shape_dims.assign(dims_span.begin(), dims_span.end());
        }

        // Add axis information if shape is available
        if (!shape_dims.empty()) {
            ascir_text = AddAxisInfoToAscir(ascir_text, shape_dims);
            NPU_LOG_FORMAT_PRINT(DEBUG, "Added axis info for shape with %d dimensions", shape_dims.size());
        }

        // Fix sched.axis for nodes that have axis_ids but missing sched.axis
        ascir_text = FixNodeSchedAxis(ascir_text);
        NPU_LOG_PRINT(DEBUG, "Fixed sched.axis for nodes");

        // Step 3: Generate kernel for this computation
        TF_ASSIGN_OR_RETURN(auto comp_result, GenerateKernel(ascir_text, computation_name, options));

        NpuArtifactInfo npu_artifact_info;
        npu_artifact_info.name = computation_name;
        npu_artifact_info.dir = options.output_dir;
        final_result.artifact_infos[computation_name] = npu_artifact_info;   
        idx++;
    }

    NPU_LOG_FORMAT_PRINT(INFO, "Successfully compiled %d computations", final_result.artifact_infos.size());

    return final_result;
}

}  // namespace xla
