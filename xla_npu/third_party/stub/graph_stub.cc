#include <iostream>

#include "xla/xla_npu/third_party/metadef/external/graph/graph.h"
#include "xla/xla_npu/third_party/metadef/external/graph/ascend_string.h"
#include "xla/xla_npu/third_party/metadef/external/graph/tensor.h"

namespace ge {
Graph::Graph(const char* name) {
  std::cout << "[STUB] ge Graph constructor" << std::endl;
}

Graph graph("test");

graphStatus Graph::AddOp(const ge::Operator &op) {
  return 0;
}

Graph& Graph::SetInputs(const std::vector<Operator>& inputs) {
  return graph;
}

Graph& Graph::SetOutputs(const std::vector<Operator>& outputs) {
  return graph;
}

Graph& Graph::SetOutputs(
    const std::vector<std::pair<Operator, std::vector<size_t>>>&
        output_indexs) {
  return graph;
}
AscendString::AscendString(const char* name) {}
bool AscendString::operator<(const AscendString& d) const {
  return true;
}
const char* AscendString::GetString() const {
  return 0;
}
bool AscendString::operator==(const AscendString& d) const {
  return true;
}

TensorDesc::TensorDesc() {}
TensorDesc::TensorDesc(const TensorDesc& desc) {}
TensorDesc::TensorDesc(TensorDesc&& desc) {}
TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}
void TensorDesc::SetShape(const Shape &shape) {}
void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer, const size_t &const_data_len) {}
Shape::Shape(const std::vector<int64_t>& dims) {}
int64_t Shape::GetShapeSize() const {
  return 0;
}
Tensor::Tensor() {}
Tensor::Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size) {}

std::vector<GNode> Graph::GetDirectNode() const {
  return {};
}

GNode::GNode() {}
graphStatus GNode::GetType(AscendString& type) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, int64_t& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, int32_t& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, float& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, AscendString& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, bool& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<int64_t>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<float>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<bool>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<std::vector<int64_t>>& attr_value) const {
  return 0;
}

std::vector<int64_t> Shape::GetDims() const {
  return {};
}
} // namespace ge
