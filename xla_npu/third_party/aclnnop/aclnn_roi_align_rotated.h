
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROI_ALIGN_ROTATED_H_
#define ACLNN_ROI_ALIGN_ROTATED_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiAlignRotatedGetWorkspaceSize
 * parameters :
 * x : required
 * rois : required
 * pooledH : required
 * pooledW : required
 * spatialScale : required
 * samplingRatio : optional
 * aligned : optional
 * clockwise : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotatedGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rois,
    int64_t pooledH,
    int64_t pooledW,
    double spatialScale,
    int64_t samplingRatio,
    bool aligned,
    bool clockwise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiAlignRotated
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotated(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
