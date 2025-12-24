#include "./export_graphdef.h"
#include "./status.h"

#include <iostream>
#include <fstream>

#include "../ir/mair_ops.h"
// #include "utils/utils.h"
// #include "translations/export_utils.h"
// #include "utils/convert_type.h"
// #include "utils/convert_tensor.h"
// #include "graph/operator_factory.h"
// #include "graph/operator.h"

#include <utility>
#include <map>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/IR/Location.h"                // from @llvm-project
#include "mlir/IR/Operation.h"               // from @llvm-project
#include "mlir/IR/Types.h"                   // from @llvm-project
#include "mlir/Pass/Pass.h"                  // from @llvm-project
#include "mlir/Pass/PassManager.h"           // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"      // from @llvm-project

namespace mlir::air {
using llvm::dyn_cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::Dialect;
using mlir::Operation;
using mlir::Value;

constexpr char kEntryFuncAttr[] = "tf.entry_function";

Status Exporter::buildMlirAirGraphOp(mlir::ModuleOp module) {
    mlir::StringAttr entry_func_id = mlir::StringAttr::get(module.getContext(), "main");
    std::optional<mlir::func::FuncOp> entry_func;

    for (auto function : module.getOps<mlir::func::FuncOp>()) {
      if (function.isExternal()) {
        //LOG(DEBUG) << "External functions not supported" << std::endl;
        return Status::OK();
      }
      if (function.getName() == entry_func_id) {
        entry_func.emplace(function);
      }
    }
  
    if (!entry_func.has_value()) {
    //    LOG(DEBUG) << "entry function `main` must be present" << std::endl;
      return Status::OK();
    }

    mlir::func::FuncOp function = entry_func.value();

    // ==========================================
    // mlir part
    // construct graph
    auto loc = function.getLoc();
    mlir::Block &body = function.front();
    // Find region of interest and ReturnOp.
    auto copy_range = body.without_terminator();
    if (copy_range.begin() != copy_range.end() && std::next(copy_range.begin()) == copy_range.end() &&
        isa<GraphOp>(*copy_range.begin())) {
      // Already a graph.
      return Status::OK();
    }
  
    // check return op
    auto return_op = dyn_cast<mlir::func::ReturnOp>(body.getTerminator());
    if (!return_op) {
      //LOG(DEBUG) << "Expect function to end with return\n" << std::endl;
      return Status::OK();
    }
  
    // Build GraphOp.
    mlir::OpBuilder builder(&body, body.begin());
    auto graph_op = GraphOp::create(builder, loc, function.getFunctionType().getResult(0));
    graph_op.getBody().push_back(new mlir::Block);
    builder.setInsertionPointToEnd(&graph_op.getBody().back());
  
    // copy body
    graph_op.getBody().front().getOperations().splice(graph_op.getBody().front().begin(), body.getOperations(),
                                                   copy_range.begin(), copy_range.end());
  
    // renumber op
    for (auto item : llvm::enumerate(graph_op.getResults())) {
      return_op.setOperand(item.index(), item.value());
    }
  
    //LOG(DEBUG) << "mlir graph building finished" << std::endl;

    // Extract input & output names if set.
    llvm::SmallVector<llvm::StringRef, 2> input_names;
    llvm::SmallVector<llvm::StringRef, 2> output_names;
    auto dict_attr = function->getAttrOfType<mlir::DictionaryAttr>(kEntryFuncAttr);
    if (dict_attr) {
    //   if (!dict_attr.get("inputs").isa<mlir::StringAttr>()) ;//LOG(ERROR) << "inputs missing in entry function attribute\n";
    //   if (!dict_attr.get("outputs").isa<mlir::StringAttr>())
    //     ;
        //LOG(ERROR) << "outputs missing in entry function attribute\n";
      mlir::dyn_cast<mlir::StringAttr>(dict_attr.get("inputs")).getValue().split(input_names, ',', /*MaxSplit=*/-1,
                                                                        /*KeepEmpty=*/false);
      mlir::dyn_cast<mlir::StringAttr>(dict_attr.get("outputs")).getValue().split(output_names, ',', /*MaxSplit=*/-1,
                                                                         /*KeepEmpty=*/false);
    }

    // Set input and output names and increment the use counter for them to help
    // generate unique names.
    if (!output_names.empty()) {
      const int num_data_results = graph_op.getNumResults();
      const int output_names_size = output_names.size();
      if (!(output_names_size == num_data_results)) {
        //LOG(ERROR) << "output_name err\n";
        return Status::OK();
        // << "output names (" << output_names.size()
        // << ") != terminator operands (" << num_data_results << ")";
      }
      llvm::DenseMap<Operation *, llvm::StringRef> output_op_to_name;
      llvm::StringMap<Operation *> name_to_op;
      for (const auto &it : llvm::enumerate(graph_op.getBody().back())) {
        // Skip control rets.
        const int index = it.index();
        if (index >= num_data_results) break;
        // TODO(jpienaar): If there is a result index specified, ensure only one
        // and that it matches the result index of the op.
        std::string name(output_names[index]);
        assert(!name.empty() && "expected non-empty name");
      }
    }
  
    mlir::Block &block = function.front();
  
    if (!input_names.empty()) {
      if (!(input_names.size() == block.getNumArguments())) {
        //LOG(ERROR) << "input_name err\n";
      }
      for (const auto &it : llvm::enumerate(function.getArguments())) {
        // TODO(lyandy): Update when changing feed/fetch import.
        std::string name(input_names[it.index()]);
        assert(!name.empty() && "expected non-empty name");
      }
    }

    return Status::OK();
}



} // namespace mlir::air