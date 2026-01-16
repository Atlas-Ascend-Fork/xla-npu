#include "aclnnInterface.h"

#include "functionLoader.h"

namespace npu {

#define LOAD_FUNCTION_NNOPBASE(funcName) REGISTER_FUNCTION(libnnopbase, funcName)
#define GET_FUNC_NNOPBASE(funcName) GET_FUNCTION(libnnopbase, funcName)

REGISTER_LIBRARY(libnnopbase)
LOAD_FUNCTION_NNOPBASE(aclCreateTensor)
LOAD_FUNCTION_NNOPBASE(aclCreateScalar)
LOAD_FUNCTION_NNOPBASE(aclCreateIntArray)

aclTensor* AclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride,
                           int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum,
                           void* tensorData) {
    typedef aclTensor* (*aclCreateTensorFunc)(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                              const int64_t* stride, int64_t offset, aclFormat format,
                                              const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData);
    static aclCreateTensorFunc func = nullptr;
    if (func == nullptr) {
        func = (aclCreateTensorFunc)GET_FUNC_NNOPBASE(aclCreateTensor);
    }
    if (func != nullptr) {
        return func(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum, tensorData);
    }
    return nullptr;
}

aclScalar* AclCreateScalar(void* value, aclDataType dataType) {
    typedef aclScalar* (*aclCreateScalarFunc)(void* value, aclDataType dataType);
    static aclCreateScalarFunc func = nullptr;
    if (func == nullptr) {
        func = (aclCreateScalarFunc)GET_FUNC_NNOPBASE(aclCreateScalar);
    }
    if (func != nullptr) {
        return func(value, dataType);
    }
    return nullptr;
}

aclIntArray* AclCreateIntArray(const int64_t* value, uint64_t size) {
    typedef aclIntArray* (*aclCreateIntArrayFunc)(const int64_t* value, uint64_t size);
    static aclCreateIntArrayFunc func = nullptr;
    if (func == nullptr) {
        func = (aclCreateIntArrayFunc)GET_FUNC_NNOPBASE(aclCreateIntArray);
    }
    if (func != nullptr) {
        return func(value, size);
    }
    return nullptr;
}

}  // namespace npu
