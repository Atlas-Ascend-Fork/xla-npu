#include "aclnnInterface.h"

#include "functionLoader.h"

namespace npu {

#define LOAD_FUNCTION_NNOPBASE(funcName) REGISTER_FUNCTION(libnnopbase, funcName)
#define GET_FUNC_NNOPBASE(funcName) GET_FUNCTION(libnnopbase, funcName)

REGISTER_LIBRARY(libnnopbase)
LOAD_FUNCTION_NNOPBASE(aclCreateTensor)

aclTensor *AclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                          const int64_t *stride, int64_t offset, aclFormat format,
                          const int64_t *storageDims, uint64_t storageDimsNum,
                          void *tensorData)
{
    typedef aclTensor* (*aclCreateTensorFunc)(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                          const int64_t *stride, int64_t offset, aclFormat format,
                          const int64_t *storageDims, uint64_t storageDimsNum,
                          void *tensorData);
    static aclCreateTensorFunc func = nullptr;
    if (func == nullptr) {
        func = (aclCreateTensorFunc)GET_FUNC_NNOPBASE(aclCreateTensor);
    }
    if (func != nullptr) {
        return func(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum, tensorData);
    }
    return nullptr;
}

} // namespace npu
