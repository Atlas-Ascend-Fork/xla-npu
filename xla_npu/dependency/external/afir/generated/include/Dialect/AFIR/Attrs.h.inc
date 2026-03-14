/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::afir {

/// Position configuration including location, depth and buffer mode
class PositionConfigAttr;
/// Ascend tensor attribute groups
/// Comprehensive tensor attributes including data type, axis information,
///     memory layout (repeats, strides), vectorization info, and memory management.
class AscTensorGroupsAttr;
/// Axis definition for computational loops
/// Defines an axis for loop-based computation including axis type, size,
///     alignment, binding information, and optimization flags.
class AxisAttr;
/// Temporary buffer description
/// Describes a temporary buffer including its size expression and lifetime axis.
class TmpBufDescAttr;
/// Ascend graph attribute groups
/// Graph-level attributes including tiling key, axis definitions,
///     graph type, and size variables.
class AscGraphAttrGroupsAttr;
/// representing the value of scalar
class ScalarValueAttr;
namespace detail {

struct PositionConfigAttrStorage;

} // namespace detail
class PositionConfigAttr : public ::mlir::Attribute::AttrBase<PositionConfigAttr, ::mlir::Attribute, detail::PositionConfigAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.pos_config";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static PositionConfigAttr get(::mlir::MLIRContext *context, ::mlir::afir::Position position, uint32_t depth, bool is_double_buffer);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pos_config"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::afir::Position getPosition() const;
  uint32_t getDepth() const;
  bool getIsDoubleBuffer() const;
};
namespace detail {

struct AscTensorGroupsAttrStorage;

} // namespace detail
class AscTensorGroupsAttr : public ::mlir::Attribute::AttrBase<AscTensorGroupsAttr, ::mlir::Attribute, detail::AscTensorGroupsAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.asc_tensor";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static AscTensorGroupsAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> vectorized_axis, ::llvm::ArrayRef<Attribute> vectorized_strides, int64_t tensor_id, int64_t reuse_id, PositionConfigAttr position_config, int64_t position_id);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"asc_tensor"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getVectorizedAxis() const;
  ::llvm::ArrayRef<Attribute> getVectorizedStrides() const;
  int64_t getTensorId() const;
  int64_t getReuseId() const;
  PositionConfigAttr getPositionConfig() const;
  int64_t getPositionId() const;
};
namespace detail {

struct AxisAttrStorage;

} // namespace detail
class AxisAttr : public ::mlir::Attribute::AttrBase<AxisAttr, ::mlir::Attribute, detail::AxisAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.axis";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static AxisAttr get(::mlir::MLIRContext *context, int64_t id, StringAttr name, ::mlir::afir::AxisType axis_type, bool bind_block, StringAttr size, StringAttr align, ::llvm::ArrayRef<int64_t> from);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"axis"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getId() const;
  StringAttr getName() const;
  ::mlir::afir::AxisType getAxisType() const;
  bool getBindBlock() const;
  StringAttr getSize() const;
  StringAttr getAlign() const;
  ::llvm::ArrayRef<int64_t> getFrom() const;
};
namespace detail {

struct TmpBufDescAttrStorage;

} // namespace detail
class TmpBufDescAttr : public ::mlir::Attribute::AttrBase<TmpBufDescAttr, ::mlir::Attribute, detail::TmpBufDescAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.tmp_buf_desc";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static TmpBufDescAttr get(::mlir::MLIRContext *context, StringAttr size, int64_t life_time_axis_id);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tmp_buf_desc"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getSize() const;
  int64_t getLifeTimeAxisId() const;
};
namespace detail {

struct AscGraphAttrGroupsAttrStorage;

} // namespace detail
class AscGraphAttrGroupsAttr : public ::mlir::Attribute::AttrBase<AscGraphAttrGroupsAttr, ::mlir::Attribute, detail::AscGraphAttrGroupsAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.asc_graph";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static AscGraphAttrGroupsAttr get(::mlir::MLIRContext *context, int64_t tiling_key, ::llvm::ArrayRef<AxisAttr> axes, ::mlir::afir::AscGraphType type, ::llvm::ArrayRef<StringAttr> size_var);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"asc_graph"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getTilingKey() const;
  ::llvm::ArrayRef<AxisAttr> getAxes() const;
  ::mlir::afir::AscGraphType getType() const;
  ::llvm::ArrayRef<StringAttr> getSizeVar() const;
};
namespace detail {

struct ScalarValueAttrStorage;

} // namespace detail
class ScalarValueAttr : public ::mlir::Attribute::AttrBase<ScalarValueAttr, ::mlir::Attribute, detail::ScalarValueAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "afir.value";
  static constexpr ::llvm::StringLiteral dialectName = "afir";
  static ScalarValueAttr get(::mlir::MLIRContext *context, StringAttr value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"value"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getValue() const;
};

} // namespace mlir::afir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::PositionConfigAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AscTensorGroupsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AxisAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::TmpBufDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::AscGraphAttrGroupsAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::afir::ScalarValueAttr)

#endif // GET_ATTRDEF_CLASSES

