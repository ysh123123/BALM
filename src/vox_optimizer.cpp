#include "vox_optimizer.hpp"

VoxOptimizer::VoxOptimizer(const int win_size, const bool avg_thr)
    : win_size_(win_size), jac_leng_(DVEL * win_size_), avg_thr_(avg_thr) {}

double VoxOptimizer::DivideThread(vector<IMUST> &x_stats, VoxHess &voxhess, Eigen::MatrixXd &Hess,
                                  Eigen::VectorXd &JacT) {
  double residual = 0;
  Hess.setZero();
  JacT.setZero();
  PLM(-1) hessians(thd_num_);
  PLV(-1) jacobins(thd_num_);

  for (int i = 0; i < thd_num_; i++) {
    hessians[i].resize(jac_leng_, jac_leng_);
    jacobins[i].resize(jac_leng_);
  }

  int tthd_num = thd_num_;
  vector<double> resis(tthd_num, 0);
  int g_size = voxhess.plvec_voxels.size();
  if (g_size < tthd_num) tthd_num = 1;

  vector<thread *> mthreads(tthd_num);
  double part = 1.0 * g_size / tthd_num;
  for (int i = 0; i < tthd_num; i++)
    mthreads[i] = new thread(&VoxHess::RightAccEvaluate, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]),
                             ref(jacobins[i]), ref(resis[i]));

  for (int i = 0; i < tthd_num; i++) {
    mthreads[i]->join();
    Hess += hessians[i];
    JacT += jacobins[i];
    residual += resis[i];
    delete mthreads[i];
  }
  if (avg_thr_) {
    return residual / g_size;
  }

  return residual;
}

double VoxOptimizer::OnlyResidual(const vector<IMUST> &x_stats, const VoxHess &voxhess, const bool is_avg) {
  double residual2 = 0;
  voxhess.EvaluateOnlyResidual(x_stats, residual2);
  if (is_avg) return residual2 / voxhess.plvec_voxels.size();
  return residual2;
}

void VoxOptimizer::RemoveOutlier(const vector<IMUST> &x_stats, VoxHess &voxhess, const double ratio) {
  std::vector<double> residuals = voxhess.EvaluateResidual(x_stats);
  std::sort(residuals.begin(), residuals.end());  // sort in ascending order

  double threshold = residuals[std::floor((1 - ratio) * voxhess.plvec_voxels.size()) - 1];
  int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
  // std::cout << "vox_num before4 " << voxhess.plvec_voxels.size();
  // std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
  voxhess.RemoveResidual(x_stats, threshold, reject_num);
  // std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
}

void VoxOptimizer::DampingIter(vector<IMUST> &x_stats, VoxHess &voxhess, double &residual, PLV(6) & hess_vec,
                               size_t &mem_cost) {
  double u = 0.01, v = 2;
  Eigen::MatrixXd D(jac_leng_, jac_leng_), Hess(jac_leng_, jac_leng_), HessuD(jac_leng_, jac_leng_);
  Eigen::VectorXd JacT(jac_leng_), dxi(jac_leng_), new_dxi(jac_leng_);

  D.setIdentity();
  double residual1, residual2, q;
  bool is_calc_hess = true;
  vector<IMUST> x_stats_temp;

  vector<IMUST> x_ab(win_size_);
  x_ab[0] = x_stats[0];
  for (int i = 1; i < win_size_; i++) {
    x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
    x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
  }

  double hesstime = 0;
  double solvtime = 0;
  size_t max_mem = 0;
  double loop_num = 0;
  for (int i = 0; i < 10; i++) {
    if (is_calc_hess) {
      auto tm = std::chrono::high_resolution_clock::now();
      residual1 = DivideThread(x_stats, voxhess /*, x_ab*/, Hess, JacT);

      hesstime += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tm).count();
    }

    auto tm = std::chrono::high_resolution_clock::now();
    D.diagonal() = Hess.diagonal();
    HessuD = Hess + u * D;
    auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> A1_sparse(jac_leng_, jac_leng_);
    std::vector<Eigen::Triplet<double>> tripletlist;
    for (int a = 0; a < jac_leng_; a++)
      for (int b = 0; b < jac_leng_; b++)
        if (HessuD(a, b) != 0) {
          tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
          // A1_sparse.insert(a, b) = HessuD(a, b);
        }
    A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
    A1_sparse.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
    Solver_sparse.compute(A1_sparse);
    // 获取当前进程的内存使用量
    size_t temp_mem = CheckMem();
    // 更新最大内存使用量
    if (temp_mem > max_mem) max_mem = temp_mem;
    dxi = Solver_sparse.solve(-JacT);
    temp_mem = CheckMem();
    if (temp_mem > max_mem) max_mem = temp_mem;
    solvtime += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tm).count();
    // new_dxi = Solver_sparse.solve(-JacT);
    // printf("new solve time cost %f\n",std::chrono::high_resolution_clock::now() - t1);
    // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
    // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
    // std::cout<<"relative error "<<relative_err<<std::endl;
    // std::cout<<"absolute error "<<absolute_err<<std::endl;
    // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

    x_stats_temp = x_stats;
    for (int j = 0; j < win_size_; j++) {
      x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
      x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
    }

    double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
    if (avg_thr_) {
      residual2 = OnlyResidual(x_stats_temp, voxhess /*, x_ab*/, true);
      q1 /= voxhess.plvec_voxels.size();
    } else {
      residual2 = OnlyResidual(x_stats_temp, voxhess /*, x_ab*/);
    }

    residual = residual2;
    q = (residual1 - residual2);
    // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
    //        i, residual1, residual2, u, v, q/q1, q1, q);
    loop_num = i + 1;
    // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
    // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
    // if (double(max_mem / 1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem / 1048576.0));

    if (q > 0) {
      x_stats = x_stats_temp;
      q = q / q1;
      v = 2;
      q = 1 - pow(2 * q - 1, 3);
      u *= (q < one_three ? one_three : q);
      is_calc_hess = true;
    } else {
      u = u * v;
      v = 2 * v;
      is_calc_hess = false;
    }
    if (avg_thr_) {
      if ((fabs(residual1 - residual2) / residual1) < 0.05 || i == 9) {
        if (mem_cost < max_mem) mem_cost = max_mem;
        for (int j = 0; j < win_size_ - 1; j++)
          for (int k = j + 1; k < win_size_; k++)
            hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL * j, DVEL * k).diagonal().segment<DVEL>(0));
        break;
      }
    } else {
      if (fabs(residual1 - residual2) < 1e-9) break;
    }
  }
}
void VoxOptimizer::DampingIter(vector<IMUST> &x_stats, VoxHess &voxhess) {
  double u = 0.01, v = 2;
  Eigen::MatrixXd D(jac_leng_, jac_leng_), Hess(jac_leng_, jac_leng_), HessuD(jac_leng_, jac_leng_);
  Eigen::VectorXd JacT(jac_leng_), dxi(jac_leng_);

  D.setIdentity();
  double residual1, residual2, q;
  bool is_calc_hess = true;
  vector<IMUST> x_stats_temp = x_stats;
  vector<IMUST> x_stats_temp2 = x_stats;
  bool update_z = true;
  auto check_hessian_degeneracy = [&]() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hess);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();

    // std::cout << "Hessian Matrix Eigenvalues: \n" << eigenvalues.transpose() << std::endl;
    double cond_number = eigenvalues.maxCoeff() / eigenvalues.minCoeff();
    // std::cout << "Condition Number of Hessian: " << cond_number << std::endl;
    if (cond_number > 1e6) {
      std::cout << "Warning: Hessian matrix is ill-conditioned!" << std::endl;
      return false;
    }
    double min_eigenvalue = eigenvalues.minCoeff();
    if (min_eigenvalue < 1e-6) {
      std::cout << "Warning: Hessian matrix is degenerate!" << std::endl;
      return false;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hess);
    Eigen::VectorXd singularValues = svd.singularValues();

    // std::cout << "Hessian Matrix Singular Values: \n" << singularValues.transpose() << std::endl;

    double min_singular_value = singularValues.minCoeff();
    if (min_singular_value < 1e-6) {
      std::cout << "Warning: Hessian matrix rank deficiency detected!" << std::endl;
      return false;
    }
    return true;
  };
  for (int i = 0; i < 10; i++) {
    if (is_calc_hess) {
      residual1 = DivideThread(x_stats, voxhess /*, x_ab*/, Hess, JacT);

      if (check_degeneracy && !check_hessian_degeneracy()) {
        break;
      }
    }

    D.diagonal() = Hess.diagonal();
    // dxi = (Hess + u * D).ldlt().solve(-JacT);
    HessuD = Hess + u * D;
    auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> A1_sparse(jac_leng_, jac_leng_);
    std::vector<Eigen::Triplet<double>> tripletlist;
    for (int a = 0; a < jac_leng_; a++)
      for (int b = 0; b < jac_leng_; b++)
        if (HessuD(a, b) != 0) {
          tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
          // A1_sparse.insert(a, b) = HessuD(a, b);
        }
    A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
    A1_sparse.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
    Solver_sparse.compute(A1_sparse);
    dxi = Solver_sparse.solve(-JacT);

    for (int j = 0; j < win_size_; j++) {
      // right update
      x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
      auto d_t = dxi.block<3, 1>(DVEL * j + 3, 0);
      x_stats_temp[j].p = x_stats[j].p + d_t;

      // left update
      // Eigen::Matrix3d dR = Exp(dxi.block<3, 1>(DVEL*j, 0));
      // x_stats_temp[j].R = dR * x_stats[j].R;
      // x_stats_temp[j].p = dR * x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
    }
    double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);

    residual2 = OnlyResidual(x_stats_temp, voxhess /*, x_ab*/);

    q = (residual1 - residual2);
    printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);

    if (q > 0) {
      for (int j = 0; j < win_size_; j++) {
        auto stat_z = x_stats[j].p[2];
        auto stat_temp_z = x_stats_temp[j].p[2];
        auto delta_z = stat_z - stat_temp_z;
        if (delta_z > 0.04) {
          update_z = false;
          break;
        }
      }
      x_stats = x_stats_temp;
      q = q / q1;
      v = 2;
      q = 1 - pow(2 * q - 1, 3);
      u *= (q < one_three ? one_three : q);
      is_calc_hess = true;
    } else {
      u = u * v;
      v = 2 * v;
      is_calc_hess = false;
    }

    // if(iter_stop(dxi2, 1e-4))
    // if(iter_stop(dxi, 1e-6))
    //   break;

    if (fabs(residual1 - residual2) / residual1 < 1e-6) break;
  }
  if (!update_z) {
    for (int j = 0; j < win_size_; j++) {
      x_stats[j].p[2] = x_stats_temp2[j].p[2];
    }
  }
}
size_t VoxOptimizer::CheckMem() {
  FILE *file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = strlen(line);

      const char *p = line;
      for (; std::isdigit(*p) == false; ++p) {
      }

      line[len - 3] = 0;
      result = atoi(p);

      break;
    }
  }
  fclose(file);

  return result;
}
void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<PointType> &feat_pt, Eigen::Quaterniond q, Eigen::Vector3d t,
              int fnum, double voxel_size, int window_size, float eigen_ratio) {
  float loc_xyz[3];
  for (PointType &p_c : feat_pt.points) {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;

    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);
      if (iter->second->octo_state != 1) {
        iter->second->sig_orig[fnum].push(pvec_orig);
        iter->second->sig_tran[fnum].push(pvec_tran);
      }
    } else {
      const auto ot = std::make_shared<OctoTreeRoot>(window_size);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);

      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}
void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<PointType> &pl_feat, const IMUST &x_key, int fnum, int win_size) {
  float loc_xyz[3];

  for (PointType &p_c : pl_feat.points) {
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = x_key.R * pvec_orig + x_key.p;

    // 计算体素坐标
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;  // 体素坐标对齐
    }

    VOXEL_LOC position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
                       static_cast<int64_t>(loc_xyz[2]));

    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      // 体素已经存在，更新现有体素
      auto &octo_node = iter->second;

      // 减少重复访问
      auto &vec_orig = octo_node->vec_orig[fnum];
      auto &vec_tran = octo_node->vec_tran[fnum];
      auto &sig_orig = octo_node->sig_orig[fnum];
      auto &sig_tran = octo_node->sig_tran[fnum];

      // 更新点云数据
      vec_orig.push_back(pvec_orig);
      vec_tran.push_back(pvec_tran);

      if (octo_node->octo_state != 1) {
        sig_orig.push(pvec_orig);
        sig_tran.push(pvec_tran);
      }

      // 标记该体素需要优化
      octo_node->is2opt = true;
      octo_node->each_num[fnum]++;

    } else {
      // 体素不存在，创建新的体素
      auto ot = std::make_shared<OctoTreeRoot>(win_size);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);
      ot->each_num[fnum]++;

      // 设置体素中心和其他参数
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;

      // 将新体素加入到feat_map
      feat_map[position] = ot;
    }
  }
}

void ParallelComp(Layer &layer, int thread_id, Layer &next_layer) {
  int &part_length = layer.part_length;
  int &layer_num = layer.layer_num;
  for (int i = thread_id * part_length; i < (thread_id + 1) * part_length; i++) {
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE);
    raw_pc.resize(WIN_SIZE);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for (int j = 0; j < WIN_SIZE; j++) {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    if (layer_num != 1)
      for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++) {
      if (layer_num == 1)
        for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) {
          // if(j < pose_size) {
          if (loop == 0) {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            if (cloud_from_path) {
              loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            } else {
              pc = layer.pcds[j];
            }

            raw_pc[j - i * GAP] = pc;
          }
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
          // }
        }

      OctoTreeMap surf_map;

      for (size_t j = 0; j < WIN_SIZE; j++) {
        if (layer.downsample_size > 0) down_sampling_voxel(*src_pc[j], layer.downsample_size);
        CutVoxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE,
                 layer.eigen_ratio);
      }
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->Recut(WIN_SIZE);

      VoxHess voxhess(WIN_SIZE);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->TrasOpt(voxhess, WIN_SIZE);

      VoxOptimizer opt_lsv(WIN_SIZE, true);
      opt_lsv.RemoveOutlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.DampingIter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
        if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
          layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for (size_t j = 0; j < WIN_SIZE; j++) {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = append_cloud(pc_keyframe, *pc_oneframe);
    }
    down_sampling_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
}

void ParallelTail(Layer &layer, int thread_id, Layer &next_layer) {
  int &part_length = layer.part_length;
  int &layer_num = layer.layer_num;
  int &left_gap_num = layer.left_gap_num;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0, tran_t = 0, sol_t = 0, save_t = 0;

  if (layer.gap_num - (layer.thread_num - 1) * part_length + 1 != left_gap_num) printf("THIS IS WRONG!\n");

  for (uint i = thread_id * part_length; i < thread_id * part_length + left_gap_num; i++) {
    printf("parallel tail computing %d\n", i);
    chrono::time_point<chrono::system_clock, chrono::system_clock::duration> t0, t1;
    auto t_begin = std::chrono::high_resolution_clock::now();

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE);
    raw_pc.resize(WIN_SIZE);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for (int j = 0; j < WIN_SIZE; j++) {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    if (layer_num != 1) {
      t0 = std::chrono::high_resolution_clock::now();
      for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
      load_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++) {
      if (layer_num == 1) {
        t0 = std::chrono::high_resolution_clock::now();
        for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) {
          // if(j < pose_size) {
          if (loop == 0) {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            // loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            if (cloud_from_path) {
              loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            } else {
              pc = layer.pcds[j];
            }
            raw_pc[j - i * GAP] = pc;
          }
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
          // }
        }
        load_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      }

      OctoTreeMap surf_map;
      // std::cout<<"-----cut_voxel------"<<std::endl;
      for (size_t j = 0; j < WIN_SIZE; j++) {
        t0 = std::chrono::high_resolution_clock::now();
        // std::cout<<"-----111111-------"<<std::endl;
        if (layer.downsample_size > 0) down_sampling_voxel(*src_pc[j], layer.downsample_size);
        dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        // std::cout<<"-----111111-221111113333------"<<std::endl;
        CutVoxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE,
                 layer.eigen_ratio);
        // std::cout<<"-----111111-222222222222------"<<std::endl;
        cut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      }

      t0 = std::chrono::high_resolution_clock::now();
      // std::cout<<"-----recut------"<<std::endl;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->Recut(WIN_SIZE);
      recut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

      t0 = std::chrono::high_resolution_clock::now();
      VoxHess voxhess(WIN_SIZE);
      // std::cout<<"-----tras_opt------"<<std::endl;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->TrasOpt(voxhess, WIN_SIZE);
      tran_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

      VoxOptimizer opt_lsv(WIN_SIZE, true);
      t0 = std::chrono::high_resolution_clock::now();
      // std::cout<<"-----remove_outlier------"<<std::endl;
      opt_lsv.RemoveOutlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      // std::cout<<"-----damping_iter------"<<std::endl;
      opt_lsv.DampingIter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      // std::cout<<"-----delete------"<<std::endl;
      // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
        if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        if (i < thread_id * part_length + left_gap_num)
          for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
            layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for (size_t j = 0; j < WIN_SIZE; j++) {
      t1 = std::chrono::high_resolution_clock::now();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = append_cloud(pc_keyframe, *pc_oneframe);
      save_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();
    }
    t0 = std::chrono::high_resolution_clock::now();
    down_sampling_voxel(*pc_keyframe, 0.05);
    dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    next_layer.pcds[i] = pc_keyframe;
    save_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    total_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_begin).count();
  }
  if (layer.tail > 0) {
    int i = thread_id * part_length + left_gap_num;

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(layer.last_win_size);
    raw_pc.resize(layer.last_win_size);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    for (int j = 0; j < layer.last_win_size; j++) {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    if (layer_num != 1) {
      for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
    }

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++) {
      if (layer_num == 1)
        for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++) {
          // if(j < pose_size) {
          if (loop == 0) {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            // loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            if (cloud_from_path) {
              loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            } else {
              pc = layer.pcds[j];
            }
            raw_pc[j - i * GAP] = pc;
          }
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
          // }
        }

      OctoTreeMap surf_map;

      for (size_t j = 0; j < layer.last_win_size; j++) {
        if (layer.downsample_size > 0) down_sampling_voxel(*src_pc[j], layer.downsample_size);
        CutVoxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size,
                 layer.last_win_size, layer.eigen_ratio);
      }
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->Recut(layer.last_win_size);

      VoxHess voxhess(layer.last_win_size);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->TrasOpt(voxhess, layer.last_win_size);

      VoxOptimizer opt_lsv(layer.last_win_size, true);
      opt_lsv.RemoveOutlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6) hess_vec;
      opt_lsv.DampingIter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
        if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

        for (int j = 0; j < layer.last_win_size * (layer.last_win_size - 1) / 2; j++)
          layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for (size_t j = 0; j < layer.last_win_size; j++) {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = append_cloud(pc_keyframe, *pc_oneframe);
    }
    down_sampling_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
  printf("total time: %.2fs\n", total_t);
  printf(
      "load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
      "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% "
      "| "
      "save pcd %.2fs %.2f%%\n",
      load_t, load_t / total_t * 100, undis_t, undis_t / total_t * 100, dsp_t, dsp_t / total_t * 100, cut_t,
      cut_t / total_t * 100, recut_t, recut_t / total_t * 100, tran_t, tran_t / total_t * 100, sol_t,
      sol_t / total_t * 100, save_t, save_t / total_t * 100);
}

void GlobalBA(Layer &layer) {
  int window_size = layer.pose_vec.size();
  vector<IMUST> x_buf(window_size);
  for (int i = 0; i < window_size; i++) {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }

  vector<pcl::PointCloud<PointType>::Ptr> src_pc;
  src_pc.resize(window_size);
  for (int i = 0; i < window_size; i++) src_pc[i] = (*layer.pcds[i]).makeShared();

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0;
  chrono::time_point<chrono::system_clock, chrono::system_clock::duration> t0;
  for (int loop = 0; loop < layer.max_iter; loop++) {
    std::cout << "---------------------" << std::endl;
    std::cout << "Iteration " << loop << std::endl;

    OctoTreeMap surf_map;
    // std::cout << "CutVoxel " << loop << std::endl;
    // ct = true;
    for (int i = 0; i < window_size; i++) {
      t0 = std::chrono::high_resolution_clock::now();
      if (layer.downsample_size > 0) down_sampling_voxel(*src_pc[i], layer.downsample_size);
      dsp_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
      t0 = std::chrono::high_resolution_clock::now();
      CutVoxel(surf_map, *src_pc[i], Eigen::Quaterniond(x_buf[i].R), x_buf[i].p, i, layer.voxel_size, window_size,
               layer.eigen_ratio * 2);
      cut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    }
    // std::cout << "Recut " << loop << std::endl;
    t0 = std::chrono::high_resolution_clock::now();

    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->Recut(window_size);
    recut_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    // std::cout << "TrasOpt " << loop << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    VoxHess voxhess(window_size);
    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->TrasOpt(voxhess, window_size);
    tran_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    // std::cout << "RemoveOutlier " << loop << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    VoxOptimizer opt_lsv(window_size, true);
    opt_lsv.RemoveOutlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6) hess_vec;
    // std::cout << "DampingIter " << loop << std::endl;
    opt_lsv.DampingIter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    // for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

    cout << "Residual absolute: " << abs(residual_pre - residual_cur) << " | "
         << "percentage: " << abs(residual_pre - residual_cur) / abs(residual_cur) << endl;

    if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
      if (max_mem < mem_cost) max_mem = mem_cost;
#ifdef FULL_HESS
      for (int i = 0; i < window_size * (window_size - 1) / 2; i++) layer.hessians[i] = hess_vec[i];
#else
      for (int i = 0; i < window_size - 1; i++) {
        Matrix6d hess = Hess_cur.block(6 * i, 6 * i + 6, 6, 6);
        for (int row = 0; row < 6; row++)
          for (int col = 0; col < 6; col++) hessFile << hess(row, col) << ((row * col == 25) ? "" : " ");
        if (i < window_size - 2) hessFile << "\n";
      }
#endif
      break;
    }
    residual_pre = residual_cur;
  }
  for (int i = 0; i < window_size; i++) {
    layer.pose_vec[i].q = Eigen::Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }
  printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
}

void DistributeThread(Layer &layer, Layer &next_layer) {
  int &thread_num = layer.thread_num;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_num; i++)
    if (i < thread_num - 1)
      layer.mthreads[i] = new thread(ParallelComp, ref(layer), i, ref(next_layer));
    else
      layer.mthreads[i] = new thread(ParallelTail, ref(layer), i, ref(next_layer));
  // printf("Thread distribution time: %f\n", std::chrono::high_resolution_clock::now()-t0);

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < thread_num; i++) {
    layer.mthreads[i]->join();
    delete layer.mthreads[i];
  }
  // printf("Thread join time: %f\n", std::chrono::high_resolution_clock::now()-t0);
}