#pragma once

#include "tools.hpp"
#include "vox_hess.hpp"
#include "octo_tree.hpp"
#include "hba.hpp"
#include <thread>

using OctoTreeMap = std::unordered_map<VOXEL_LOC, std::shared_ptr<OctoTreeRoot>>;
class VoxOptimizer {
 public:

  VoxOptimizer(const int win_size, const bool avg_thr);

  double DivideThread(vector<IMUST> &x_stats, VoxHess &voxhess, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT);

  double OnlyResidual(const vector<IMUST> &x_stats, const VoxHess &voxhess, const bool is_avg = false);

  void RemoveOutlier(const vector<IMUST> &x_stats, VoxHess &voxhess, const double ratio);

  void DampingIter(vector<IMUST> &x_stats, VoxHess &voxhess, double &residual, PLV(6) & hess_vec, size_t &mem_cost);
  void DampingIter(vector<IMUST> &x_stats, VoxHess &voxhess);
  static size_t CheckMem();

 private:
  int thd_num_ = 4;
  int win_size_, jac_leng_;
  bool avg_thr_{false};
};

void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<PointType> &feat_pt,
              Eigen::Quaterniond q, Eigen::Vector3d t, int fnum, double voxel_size, int window_size, float eigen_ratio);
void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<PointType> &pl_feat,
              const IMUST &x_key, int fnum, int win_size);

void ParallelComp(Layer &layer, int thread_id, Layer &next_layer);

void ParallelTail(Layer &layer, int thread_id, Layer &next_layer);

void GlobalBA(Layer &layer);

void DistributeThread(Layer &layer, Layer &next_layer);