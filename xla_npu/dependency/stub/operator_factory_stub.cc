/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "xla/xla_npu/dependency/header/metadef/graph/operator_factory.h"

namespace ge {
Operator OperatorFactory::CreateOperator(
    const char* operator_name,
    const char* operator_type) {
  return Operator();
}

Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  return Operator();
}

OperatorCreatorRegister::OperatorCreatorRegister(
    const char* operator_type,
    const OpCreatorV2& op_creator) {}

bool OperatorFactory::IsExistOp(const char* operator_type) {
  return true;
}
} // namespace ge
