
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CLIPPED_SWIGLU_H_
#define ACLNN_CLIPPED_SWIGLU_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnClippedSwigluGetWorkspaceSize
 * parameters :
 * x : required
 * groupIndexOptional : optional
 * dim : optional
 * alpha : optional
 * limit : optional
 * bias : optional
 * interleaved : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClippedSwigluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t dim,
    double alpha,
    double limit,
    double bias,
    bool interleaved,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnClippedSwiglu
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClippedSwiglu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
