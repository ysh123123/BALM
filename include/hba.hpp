#pragma once

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "tools.hpp"

class Layer {
 public:
  int pose_size, layer_num, max_iter, part_length, left_size, left_h_size, j_upper, tail, thread_num, gap_num,
      last_win_size, left_gap_num;
  double downsample_size, voxel_size, eigen_ratio, reject_ratio;

  std::string data_path;
  vector<pose> pose_vec;
  std::vector<thread*> mthreads;
  std::vector<double> mem_costs;

  std::vector<VEC(6)> hessians;
  std::vector<pcl::PointCloud<PointType>::Ptr> pcds;

  Layer();

  void InitStorage(int total_layer_num_);

  void InitParameter(int pose_size_ = 0);
};

class HBA {
 public:
  int thread_num, total_layer_num;
  std::vector<Layer> layers;
  std::string data_path;
  HBA(){}
  HBA(int total_layer_num_, std::string data_path_, int thread_num_,vector<pose> pose_vec);
  HBA(int total_layer_num_, int thread_num_,vector<pose> pose_vec,std::vector<pcl::PointCloud<PointType>::Ptr> pc_vec);
  void UpdateNextLayerState(int cur_layer_num);

  std::vector<pose> PoseGraphOptimization() const;
};

