/**
 * Copyright (c)2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
#define XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_

#include "xla/xla_npu/dependency/header/ge/ge_api.h"
#include "xla/xla_npu/dependency/header/metadef/graph/graph.h"


namespace xla {

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
    // std::map<ge::AscendString, ge::AscendString> compile_options;
    // std::map<ge::AscendString, ge::AscendString> load_options;
    // std::vector<Placement> input_placements;
    std::vector<ge::Shape> inputs_shapes;
    std::vector<ge::DataType> input_dtypes;
    std::vector<ge::Shape> outputs_shapes;
    std::vector<ge::DataType> output_dtypes;
    std::shared_ptr<ge::CompiledGraphSummary> summary = nullptr;
};
}  // namespace xla

#endif  // XLA_XLA_NPU_PJRT_NPU_NPU_GE_GRAPH_DATA_H_
