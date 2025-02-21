#pragma once
#include "tools.hpp"
class VoxHess {
 public:
  explicit VoxHess(const int _win_size);
  ~VoxHess();
  void PushVoxel(const vector<VOX_FACTOR> *vec_orig, const VOX_FACTOR *fix) ;
  void RightAccEvaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                        double &residual) const;
  void LeftAccEvaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                       double &residual) const;
  void EvaluateOnlyResidual(const vector<IMUST> &xs, double &residual) const;
  std::vector<double> EvaluateResidual(const vector<IMUST> &xs) const;
  void RemoveResidual(const vector<IMUST> &xs, const double threshold, const double reject_num);
public:
  vector<const vector<VOX_FACTOR> *> plvec_voxels;
private:
  vector<const VOX_FACTOR *> sig_vecs_;
  vector<double> coeffs_;
  int win_size_;
};
