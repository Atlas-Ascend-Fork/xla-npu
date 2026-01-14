
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_PA_CACHE_H_
#define ACLNN_SCATTER_PA_CACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterPaCacheGetWorkspaceSize
 * parameters :
 * key : required
 * keyCacheRef : required
 * slotMapping : required
 * compressLensOptional : optional
 * compressSeqOffsetOptional : optional
 * seqLensOptional : optional
 * cacheModeOptional : optional
 * keyCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterPaCache
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterPaCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
