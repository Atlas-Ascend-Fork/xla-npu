#ifndef REGISTER_ACLNNINTERFACE_H_
#define REGISTER_ACLNNINTERFACE_H_

#include "xla/xla_npu/third_party/aclnn/acl_meta.h"
#include "xla/xla_npu/third_party/register/functionLoader.h"

#define DECLARE_ACL_FUNC_IN_NNOPBASE(ret, func_name, args_type, args_name) \
    DECLARE_ACL_FUNC(libnnopbase, ret, func_name, args_type, args_name)

namespace npu {

DECLARE_ACL_FUNC_IN_NNOPBASE(aclTensor*, aclCreateTensor,
                             (const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                              const int64_t* stride, int64_t offset, aclFormat format, const int64_t* storageDims,
                              uint64_t storageDimsNum, void* tensorData),
                             (viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum,
                              tensorData))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclScalar*, aclCreateScalar, (void* value, aclDataType dataType), (value, dataType))

DECLARE_ACL_FUNC_IN_NNOPBASE(aclIntArray*, aclCreateIntArray, (const int64_t* value, uint64_t size), (value, size))

}  // namespace npu

#endif  // REGISTER_ACLNNINTERFACE_H_
