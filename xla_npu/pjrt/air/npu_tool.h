/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PJRT_NPU_NPU_TOOL_H_
#define PJRT_NPU_NPU_TOOL_H_

#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

#include "xla/xla_npu/dependency/header/metadef/graph/tensor.h"

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
            // ge_data_shape.SetDim(i, xla_data_shape_dim);
        }

        ge::Shape ge_data_shape(ge_shape_vector);
        std::cout << "convert xla shape to ge shape,"
            << " input xla shape is " << xla_data_shape
            << ", output ge shape is " << GetShapeString(ge_data_shape)
            << std::endl;

        return ge_data_shape;
    }

    static Shape ConvertGeShape2XlaShape(const ge::Shape& ge_data_shape, const ge::DataType& ge_data_type)
    {
        const auto& xla_data_dims = ge_data_shape.GetDims();
        const auto& xla_data_type = ConvertGeType2XlaType(ge_data_type);
        const Shape& xla_data_shape = ShapeUtil::MakeShape(xla_data_type, xla_data_dims);

        std::cout << "convert ge shape to xla shape,"
            << " input ge shape is " << GetShapeString(ge_data_shape)
            << " which type is " << xla_data_type
            << ", output xla shape is " << xla_data_shape
            << std::endl;

        return xla_data_shape;
    }
};

}
#endif  //PJRT_NPU_PJRT_COMMON_H_