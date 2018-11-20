#include "caffe/proto/caffe.pb.h"
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include <caffe/caffe.hpp>

#ifndef CPU_ONLY  // GPU
template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
                                      const bool normalized = true);

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
                                            const Dtype* bbox2);

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data);

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data);

template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int outer_num, const int channels,
    const int inner_num, Dtype* prob);

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data);

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob);

template <typename Dtype>
  void ComputeConfLossGPU(const Blob<Dtype>& conf_blob, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);
#endif  // !CPU_ONLY