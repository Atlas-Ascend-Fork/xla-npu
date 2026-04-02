/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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

#ifndef PJRT_NPU_NPU_TOOL_H_
#define PJRT_NPU_NPU_TOOL_H_

#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"
#include "xla/xla_npu/dependency/tool/log.h"

namespace xla {

class NpuTool {
public:
    static std::string GetShapeString(const ge::Shape &shape) {
        std::stringstream ss;
        auto dims = shape.GetDims();
        if (dims.empty()) {
            return "empty";
        }
        ss << "[";
        size_t index = 0U;
        for (; index < (dims.size() - 1U); index++) {
        ss << dims[index] << ", ";
        }
        ss << dims[index] << "]";
        
        return ss.str(); 
    }

    static ge::DataType ConvertXlaType2GeType(const PrimitiveType& type)
    {
        switch (type) {
            default:
                return ge::DataType::DT_FLOAT;
        };
    }

    static PrimitiveType ConvertGeType2XlaType(const ge::DataType& type)
    {
        switch (type) {
            default:
                return PrimitiveType::F32;
        };
    }

    static ge::Shape ConvertXlaShape2GeShape(const Shape& xla_data_shape)
    {
        std::vector<int64_t> ge_shape_vector;
        for (int i = 0; i < xla_data_shape.dimensions_size(); ++i) {
            const int64_t& xla_data_shape_dim = xla_data_shape.dimensions(i);
            ge_shape_vector.emplace_back(xla_data_shape_dim);
        }

        ge::Shape ge_data_shape(ge_shape_vector);
        NPU_LOG_FORMAT_PRINT(DEBUG, "convert xla shape to ge shape, input xla shape is %s, output ge shape is %s",
                             xla_data_shape.ToString().c_str(), GetShapeString(ge_data_shape).c_str());

        return ge_data_shape;
    }

    static Shape ConvertGeShape2XlaShape(const ge::Shape& ge_data_shape, const ge::DataType& ge_data_type)
    {
        const auto& xla_data_dims = ge_data_shape.GetDims();
        const auto& xla_data_type = ConvertGeType2XlaType(ge_data_type);
        const Shape& xla_data_shape = ShapeUtil::MakeShape(xla_data_type, xla_data_dims);

        NPU_LOG_FORMAT_PRINT(DEBUG, "convert ge shape to xla shape, input ge shape is %s which type is %d, output xla shape is %s",
                             GetShapeString(ge_data_shape).c_str(), static_cast<int>(xla_data_type),
                             xla_data_shape.ToString().c_str());

        return xla_data_shape;
    }
};

}
#endif  //PJRT_NPU_PJRT_COMMON_H_