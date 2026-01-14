
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROI_ALIGN_ROTATED_GRAD_H_
#define ACLNN_ROI_ALIGN_ROTATED_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRoiAlignRotatedGradGetWorkspaceSize
 * parameters :
 * xGrad : required
 * rois : required
 * yGradShape : required
 * pooledH : required
 * pooledW : required
 * spatialScale : required
 * samplingRatio : required
 * aligned : required
 * clockwise : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotatedGradGetWorkspaceSize(
    const aclTensor *xGrad,
    const aclTensor *rois,
    const aclIntArray *yGradShape,
    int64_t pooledH,
    int64_t pooledW,
    double spatialScale,
    int64_t samplingRatio,
    bool aligned,
    bool clockwise,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRoiAlignRotatedGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRoiAlignRotatedGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
