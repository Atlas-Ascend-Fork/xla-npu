/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_UTILS_UTILS_H_
#define AIR_UTILS_UTILS_H_

#include <cstdint>
#include <iostream>
#include <sstream>

namespace mlir::air {

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef std::int64_t int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef std::uint64_t uint64;

static void log(const std::string &str) {
        std::cout << str << std::endl;
    }

}  // namespace air

#endif  // AIR_UTILS_UTILS_H_
