#ifndef XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
#define XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_

#include "xla/xla_npu/third_party/air/external/ge/ge_api.h"
#include "xla/xla_npu/third_party/metadef/external/graph/graph.h"


namespace xla {
namespace npu {
enum class Placement {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
};


class GraphData {
 public:
  GraphData() {
    graph = nullptr;
    summary = nullptr;
  }
  ~GraphData() = default;
  uint32_t id = 0U;
  ge::GraphPtr graph = nullptr;
  std::map<ge::AscendString, ge::AscendString> compile_options;
  std::map<ge::AscendString, ge::AscendString> load_options;
  std::vector<Placement> input_placements;
  std::vector<std::vector<int64_t>> inputs_shape;
  std::vector<std::vector<int64_t>> outputs_shape;
  std::vector<ge::DataType> output_dtypes;
  std::shared_ptr<ge::CompiledGraphSummary> summary = nullptr;
};
}  // namespace npu
}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
