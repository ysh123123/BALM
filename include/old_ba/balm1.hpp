#ifndef BALM1_HPP
#define BALM1_HPP

#include <pcl/common/transforms.h>
#include <unordered_map>
// #include <opencv/cv.h>
#include "myso3/myso3.hpp"
#include <thread>
#include <mutex>
#include <fstream>

// namespace {
//
// }
typedef std::vector<Eigen::Vector3d> PL_VEC;
typedef pcl::PointXYZI PointType;
#define MIN_PS 7
using namespace std;
#define DEG_TO_RAD (M_PI / 180.0)
int layer_limit = 3;
// Key of hash table

class VOXEL_LOC {
 public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std {
template <>
struct hash<VOXEL_LOC> {
  size_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
  }
};
}  // namespace std

struct M_POINT {
  float xyz[3];
  int count = 0;
};

/**
 * @brief Similar with PCL voxelgrid filter
 *
 * @param pl_feat 点云
 * @param voxel_size 划分体素
 */
void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }
  // 创建体素地图
  unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    PointType &p_c = pl_feat[i];
    float loc_xyz[3];
    // 获取每个点的体素索引
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    // 找到点对应的体素
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    // 如果该点在已有体素中加入
    if (iter != feat_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.count++;
    }
    // 如果不在则新建一个体素
    else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.count = 1;
      feat_map[position] = anp;
    }
  }

  // 将点数设置为体素数
  plsize = feat_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  // 取每个体素中点的平均值
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}

void down_sampling_voxel(PL_VEC &pl_feat, double voxel_size) {
  unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d &p_c = pl_feat[i];
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.xyz[0] += p_c[0];
      iter->second.xyz[1] += p_c[1];
      iter->second.xyz[2] += p_c[2];
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c[0];
      anp.xyz[1] = p_c[1];
      anp.xyz[2] = p_c[2];
      anp.count = 1;
      feat_map[position] = anp;
    }
  }

  plsize = feat_map.size();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i][0] = iter->second.xyz[0] / iter->second.count;
    pl_feat[i][1] = iter->second.xyz[1] / iter->second.count;
    pl_feat[i][2] = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}

void plvec_trans_func(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R,
                      Eigen::Vector3d t) {
  uint orig_size = orig.size();
  tran.resize(orig_size);

  for (uint i = 0; i < orig_size; i++) {
    tran[i] = R * orig[i] + t;
  }
}

// P_fix in the paper
// Summation of P_fix
class SIG_VEC_CLASS {
 public:
  Eigen::Matrix3d sigma_vTv;
  Eigen::Vector3d sigma_vi;
  int sigma_size;

  SIG_VEC_CLASS() {
    sigma_vTv.setZero();
    sigma_vi.setZero();
    sigma_size = 0;
  }

  void tozero() {
    sigma_vTv.setZero();
    sigma_vi.setZero();
    sigma_size = 0;
  }
};

const double one_three = (1.0 / 3.0);
double feat_eigen_limit[2] = {3 * 3, 2 * 2};
double opt_feat_eigen_limit[2] = {4 * 4, 3 * 3};

// LM optimizer for map-refine
class LM_SLWD_VOXEL {
 public:
  int slwd_size,  // 滑窗大小
      filternum,  // 在滑窗中一帧点云中取几组点加入优化
      thd_num, jac_leng;
  int iter_max = 20;

  double corn_less;

  vector<SO3> so3_poses, so3_poses_temp;
  vector<Eigen::Vector3d> t_poses, t_poses_temp;

  vector<int> lam_types;  // 0 surf, 1 line
  vector<SIG_VEC_CLASS> sig_vecs;
  vector<vector<Eigen::Vector3d> *> plvec_voxels;
  vector<vector<int> *> slwd_nums;
  int map_refine_flag;
  mutex my_mutex;

  LM_SLWD_VOXEL(int ss, int fn, int thnum) : slwd_size(ss), filternum(fn), thd_num(thnum) {
    so3_poses.resize(ss);
    t_poses.resize(ss);
    so3_poses_temp.resize(ss);
    t_poses_temp.resize(ss);
    jac_leng = 6 * ss;
    corn_less = 0.1;
    map_refine_flag = 0;
  }

  // Used by "push_voxel"
  void downsample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame, vector<Eigen::Vector3d> &plvec_voxel,
                  vector<int> &slwd_num, int filternum2use) {
    // 记录滑动窗口中当前体素下其中一帧的点数
    uint plsize = plvec_orig.size();
    // 如果该点数较少，则直接全部加入
    if (plsize <= (uint)filternum2use) {
      for (uint i = 0; i < plsize; i++) {
        plvec_voxel.push_back(plvec_orig[i]);
        slwd_num.push_back(cur_frame);
      }
      return;
    }

    Eigen::Vector3d center;
    // 分成几部分
    double part = 1.0 * plsize / filternum2use;
    // 否则分成filternum2use个
    for (int i = 0; i < filternum2use; i++) {
      uint np = part * i;
      uint nn = part * (i + 1);
      center.setZero();
      for (uint j = np; j < nn; j++) {
        center += plvec_orig[j];
      }
      center = center / (nn - np);
      plvec_voxel.push_back(center);
      slwd_num.push_back(cur_frame);
    }
  }

  // Push voxel into optimizer
  void push_voxel(vector<vector<Eigen::Vector3d> *> &plvec_orig, SIG_VEC_CLASS &sig_vec, int lam_type) {
    int process_points_size = 0;
    // 记录有效滑窗的个数
    for (int i = 0; i < slwd_size; i++) {
      if (!plvec_orig[i]->empty()) {
        process_points_size++;
      }
    }

    // Only one scan
    // 如果只有一个滑窗有效则返回
    if (process_points_size <= 1) {
      return;
    }

    int filternum2use = filternum;
    // filternum为一帧中取几组点加入优化
    // process_points_size为一个有几帧
    // 所以这里判断加入优化的点数是否足够
    // 如果不够则增加每帧加入优化的点数
    if (filternum * process_points_size < MIN_PS) {
      filternum2use = MIN_PS / process_points_size + 1;
    }

    vector<Eigen::Vector3d> *plvec_voxel = new vector<Eigen::Vector3d>();
    // Frame num in sliding window for each point in "plvec_voxel"
    vector<int> *slwd_num = new vector<int>();
    // 单个体素中滑窗加入优化的点
    plvec_voxel->reserve(filternum2use * slwd_size);
    slwd_num->reserve(filternum2use * slwd_size);

    // retain one point for one scan (you can modify)
    for (int i = 0; i < slwd_size; i++) {
      if (!plvec_orig[i]->empty()) {
        // 降采样，将当前体素中的某一帧的所有点降采样要filternum2uset个点
        downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
      }
    }

    // for(int i=0; i<slwd_size; i++)
    // {
    //   for(uint j=0; j<plvec_orig[i]->size(); j++)
    //   {
    //     plvec_voxel->push_back(plvec_orig[i]->at(j));
    //     slwd_num->push_back(i);
    //   }
    // }

    plvec_voxels.push_back(plvec_voxel);  // Push a voxel into optimizer
    // 接下来三个不知道干啥的
    slwd_nums.push_back(slwd_num);
    lam_types.push_back(lam_type);
    sig_vecs.push_back(sig_vec);  // history points out of sliding window
  }

  // Calculate Hessian, Jacobian, residual
  void acc_t_evaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head, int end, Eigen::MatrixXd &Hess,
                      Eigen::VectorXd &JacT, double &residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::MatrixXd _jact(JacT);

    // In program, lambda_0 < lambda_1 < lambda_2
    // For plane, the residual is lambda_0
    // For line, the residual is lambda_0+lambda_1
    // We only calculate lambda_1 here
    for (int a = head; a < end; a++) {
      uint k = lam_types[a];  // 0 is surf, 1 is line
      SIG_VEC_CLASS &sig_vec = sig_vecs[a];
      vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
      // Position in slidingwindow for each point in "plvec_voxel"
      vector<int> &slwd_num = *slwd_nums[a];
      uint backnum = plvec_voxel.size();

      Eigen::Vector3d vec_tran;
      vector<Eigen::Vector3d> plvec_back(backnum);
      // derivative point to T (R, t)
      vector<Eigen::Matrix3d> point_xis(backnum);
      Eigen::Vector3d centor(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for (uint i = 0; i < backnum; i++) {
        vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i];
        // left multiplication instead of right muliplication in paper
        point_xis[i] = -SO3::hat(vec_tran);
        plvec_back[i] = vec_tran + t_ps[slwd_num[i]];  // after trans

        centor += plvec_back[i];
        covMat += plvec_back[i] * plvec_back[i].transpose();
      }

      double N_points = backnum + sig_vec.sigma_size;
      centor += sig_vec.sigma_vi;
      covMat += sig_vec.sigma_vTv;

      covMat = covMat - centor * centor.transpose() / N_points;
      covMat = covMat / N_points;
      centor = centor / N_points;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d eigen_value = saes.eigenvalues();

      Eigen::Matrix3d U = saes.eigenvectors();
      Eigen::Vector3d u[3];  // eigenvectors
      for (int j = 0; j < 3; j++) {
        u[j] = U.block<3, 1>(0, j);
      }

      // Jacobian matrix
      Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
      Eigen::Vector3d vec_Jt;
      for (uint i = 0; i < backnum; i++) {
        plvec_back[i] = plvec_back[i] - centor;
        vec_Jt = 2.0 / N_points * ukukT * plvec_back[i];
        _jact.block<3, 1>(6 * slwd_num[i] + 3, 0) += vec_Jt;
        _jact.block<3, 1>(6 * slwd_num[i], 0) -= point_xis[i] * vec_Jt;
      }

      // Hessian matrix
      Eigen::Matrix3d Hessian33;
      Eigen::Matrix3d C_k;
      vector<Eigen::Matrix3d> C_k_np(3);
      for (uint i = 0; i < 3; i++) {
        if (i == k) {
          C_k_np[i].setZero();
          continue;
        }
        Hessian33 = u[i] * u[k].transpose();
        // part of F matrix in paper
        C_k_np[i] = -1.0 / N_points / (eigen_value[i] - eigen_value[k]) * (Hessian33 + Hessian33.transpose());
      }

      Eigen::Matrix3d h33;
      uint rownum, colnum;
      for (uint j = 0; j < backnum; j++) {
        for (int f = 0; f < 3; f++) {
          C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];
        }
        C_k = U * C_k;
        colnum = 6 * slwd_num[j];
        // block matrix operation, half Hessian matrix
        for (uint i = j; i < backnum; i++) {
          Hessian33 = u[k] * (plvec_back[i]).transpose() * C_k + u[k].dot(plvec_back[i]) * C_k;

          rownum = 6 * slwd_num[i];
          if (i == j) {
            Hessian33 += (N_points - 1) / N_points * ukukT;
          } else {
            Hessian33 -= 1.0 / N_points * ukukT;
          }
          Hessian33 = 2.0 / N_points * Hessian33;  // Hessian matrix of lambda and point

          // Hessian matrix of lambda and pose
          if (rownum == colnum && i != j) {
            _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33 + Hessian33.transpose();

            h33 = -point_xis[i] * Hessian33;
            _hess.block<3, 3>(rownum, colnum + 3) += h33;
            _hess.block<3, 3>(rownum + 3, colnum) += h33.transpose();
            h33 = Hessian33 * point_xis[j];
            _hess.block<3, 3>(rownum + 3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum + 3) += h33.transpose();
            h33 = -point_xis[i] * h33;
            _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();
          } else {
            _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33;
            h33 = Hessian33 * point_xis[j];
            _hess.block<3, 3>(rownum + 3, colnum) += h33;
            _hess.block<3, 3>(rownum, colnum + 3) -= point_xis[i] * Hessian33;
            _hess.block<3, 3>(rownum, colnum) -= point_xis[i] * h33;
          }
        }
      }

      if (k == 1) {
        // add weight for line feature
        residual += corn_less * eigen_value[k];
        Hess += corn_less * _hess;
        JacT += corn_less * _jact;
      } else {
        residual += eigen_value[k];
        Hess += _hess;
        JacT += _jact;
      }
      _hess.setZero();
      _jact.setZero();
    }

    // Hessian is symmetric, copy to save time
    for (int j = 0; j < jac_leng; j += 6) {
      for (int i = j + 6; i < jac_leng; i += 6) {
        Hess.block<6, 6>(j, i) = Hess.block<6, 6>(i, j).transpose();
      }
    }
  }

  // Multithread for "acc_t_evaluate"
  void divide_thread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                     double &residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;

    vector<Eigen::MatrixXd> hessians(thd_num, Hess);
    vector<Eigen::VectorXd> jacobians(thd_num, JacT);
    vector<double> resis(thd_num, 0);

    uint gps_size = plvec_voxels.size();
    if (gps_size < (uint)thd_num) {
      acc_t_evaluate(so3_ps, t_ps, 0, gps_size, Hess, JacT, residual);
      Hess = hessians[0];
      JacT = jacobians[0];
      residual = resis[0];
      return;
    }

    vector<thread *> mthreads(thd_num);

    double part = 1.0 * (gps_size) / thd_num;
    for (int i = 0; i < thd_num; i++) {
      int np = part * i;
      int nn = part * (i + 1);

      mthreads[i] = new thread(&LM_SLWD_VOXEL::acc_t_evaluate, this, ref(so3_ps), ref(t_ps), np, nn, ref(hessians[i]),
                               ref(jacobians[i]), ref(resis[i]));
    }

    for (int i = 0; i < thd_num; i++) {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobians[i];
      residual += resis[i];
      delete mthreads[i];
    }
  }

  // Calculate residual
  void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual) {
    residual = 0;
    uint gps_size = plvec_voxels.size();
    Eigen::Vector3d vec_tran;

    for (uint a = 0; a < gps_size; a++) {
      uint k = lam_types[a];
      SIG_VEC_CLASS &sig_vec = sig_vecs[a];
      vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels[a];
      vector<int> &slwd_num = *slwd_nums[a];
      uint backnum = plvec_voxel.size();

      Eigen::Vector3d centor(Eigen::Vector3d::Zero());
      Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

      for (uint i = 0; i < backnum; i++) {
        vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i] + t_ps[slwd_num[i]];
        centor += vec_tran;
        covMat += vec_tran * vec_tran.transpose();
      }

      double N_points = backnum + sig_vec.sigma_size;
      centor += sig_vec.sigma_vi;
      covMat += sig_vec.sigma_vTv;

      covMat = covMat - centor * centor.transpose() / N_points;
      covMat = covMat / N_points;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
      Eigen::Vector3d eigen_value = saes.eigenvalues();

      if (k == 1) {
        residual += corn_less * eigen_value[k];
      } else {
        residual += eigen_value[k];
      }
    }
  }

  // LM process
  void damping_iter() {
    my_mutex.lock();
    map_refine_flag = 1;
    my_mutex.unlock();

    if (plvec_voxels.size() != slwd_nums.size() || plvec_voxels.size() != lam_types.size() ||
        plvec_voxels.size() != sig_vecs.size()) {
      printf("size is not equal\n");
      exit(0);
    }
    auto t_poses_temp2 = t_poses;
    bool update_z = true;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);

    Eigen::MatrixXd Hess2(jac_leng, jac_leng);
    Eigen::VectorXd JacT2(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;

    // Eigen::MatrixXd matA = Eigen::MatrixXd::Zero(jac_leng, jac_leng);
    // Eigen::VectorXd matB = Eigen::VectorXd::Zero(jac_leng);
    // Eigen::VectorXd matX = Eigen::VectorXd::Zero(jac_leng);
    auto check_hessian_degeneracy = [&]() {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hess);
      Eigen::VectorXd eigenvalues = solver.eigenvalues();

      // std::cout << "Hessian Matrix Eigenvalues: \n" << eigenvalues.transpose() << std::endl;
      double cond_number = eigenvalues.maxCoeff() / eigenvalues.minCoeff();
      // std::cout << "Condition Number of Hessian: " << cond_number << std::endl;
      if (cond_number > 1e5) {
        std::cout << "Warning: Hessian matrix is ill-conditioned!" << std::endl;
        return false;
      }
      double min_eigenvalue = eigenvalues.minCoeff();
      if (min_eigenvalue < 1e-5) {
        std::cout << "Warning: Hessian matrix is degenerate!" << std::endl;
        return false;
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(Hess);
      Eigen::VectorXd singularValues = svd.singularValues();

      // std::cout << "Hessian Matrix Singular Values: \n" << singularValues.transpose() << std::endl;

      double min_singular_value = singularValues.minCoeff();
      if (min_singular_value < 1e-5) {
        std::cout << "Warning: Hessian matrix rank deficiency detected!" << std::endl;
        return false;
      }
      return true;
    };
    for (int i = 0; i < iter_max; i++) {
      if (is_calc_hess) {
        // calculate Hessian, Jacobian, residual
        divide_thread(so3_poses, t_poses, Hess, JacT, residual1);
        // 计算 Hessian 是否退化
        if (!check_hessian_degeneracy()) {
          // **特征值分解，防止退化**
          // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hess);
          // Eigen::VectorXd eigenvalues = es.eigenvalues();
          // Eigen::MatrixXd eigenvectors = es.eigenvectors();
          //
          // // 设定特征值阈值，防止 Hessian 退化
          // double min_lambda = 1e-4;
          // for (int j = 0; j < eigenvalues.size(); j++) {
          //   if (eigenvalues(j) < min_lambda) {
          //     eigenvalues(j) = min_lambda; // 设定最小值，防止 Hessian 退化
          //   }
          // }
          //
          // // 重新构造修正后的 Hessian
          // Hess = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
          // if(!check_hessian_degeneracy()) {
          return;
          // }
        }
      }

      D = Hess.diagonal().asDiagonal();
      Hess2 = Hess + u * D;

      dxi = Hess2.ldlt().solve(-JacT);

      for (int j = 0; j < slwd_size; j++) {
        // left multiplication
        so3_poses_temp[j] = SO3::exp(dxi.block<3, 1>(6 * (j), 0)) * so3_poses[j];
        t_poses_temp[j] = t_poses[j] + dxi.block<3, 1>(6 * (j) + 3, 0);
      }

      // LM
      double q1 = 0.5 * (dxi.transpose() * (u * D * dxi - JacT))[0];
      // double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
      evaluate_only_residual(so3_poses_temp, t_poses_temp, residual2);

      q = (residual1 - residual2);
      printf("residual%d: %lf %lf u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);

      if (q > 0) {
        for (int j = 0; j < slwd_size; j++) {
          auto stat_z = t_poses[j][2];
          auto stat_temp_z = t_poses_temp[j][2];
          auto delta_z = stat_z - stat_temp_z;
          if (delta_z > 0.04) {
            update_z = false;
            break;
          }
        }
        so3_poses = so3_poses_temp;
        t_poses = t_poses_temp;
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

      if (fabs(residual1 - residual2) < 1e-6) {
        break;
      }
    }
    if (!update_z) {
      for (int j = 0; j < slwd_size; j++) {
        t_poses[j][2] = t_poses_temp2[j][2];
      }
    }
    my_mutex.lock();
    map_refine_flag = 2;
    my_mutex.unlock();
  }

  int read_refine_state() {
    int tem_flag;
    my_mutex.lock();
    tem_flag = map_refine_flag;
    my_mutex.unlock();
    return tem_flag;
  }

  void set_refine_state(int tem) {
    my_mutex.lock();
    map_refine_flag = tem;
    my_mutex.unlock();
  }

  void free_voxel() {
    uint a_size = plvec_voxels.size();
    for (uint i = 0; i < a_size; i++) {
      delete (plvec_voxels[i]);
      delete (slwd_nums[i]);
    }

    plvec_voxels.clear();
    slwd_nums.clear();
    sig_vecs.clear();
    lam_types.clear();
  }
};

class OCTO_TREE {
 public:
  static int voxel_windowsize;
  // 用于存放滑窗中需要优化的点云
  vector<PL_VEC *> plvec_orig;  // b系
  vector<PL_VEC *> plvec_tran;  // w系
  int octo_state;               // 0 is end of tree, 1 is not
  bool put_state{false};
  PL_VEC sig_vec_points;
  SIG_VEC_CLASS sig_vec;
  int ftype;
  int points_size, sw_points_size;
  double feat_eigen_ratio,  // 特征值比
      feat_eigen_ratio_test;
  pcl::PointXYZINormal ap_centor_direct;
  double voxel_center[3];  // x, y, z
  double quater_length;
  OCTO_TREE *leaves[8];
  bool is2opt;
  int capacity;
  pcl::PointCloud<PointType> root_centors;

  OCTO_TREE(int ft, int capa) : ftype(ft), capacity(capa) {
    octo_state = 0;
    for (int i = 0; i < 8; i++) {
      leaves[i] = nullptr;
    }

    for (int i = 0; i < capacity; i++) {
      plvec_orig.push_back(new PL_VEC());
      plvec_tran.push_back(new PL_VEC());
    }
    is2opt = true;
  }

  // Used by "recut"
  void calc_eigen() {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint asize;
    // 先将滑窗中待优化的点和点协方差求和
    for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
      // 遍历滑窗中每一帧
      asize = plvec_tran[i]->size();
      for (uint j = 0; j < asize; j++) {
        covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();
        center += (*plvec_tran[i])[j];
      }
    }

    // 再加上固定点的点和协方差
    covMat += sig_vec.sigma_vTv;
    center += sig_vec.sigma_vi;

    // 求得点平均值
    center /= points_size;

    // 求得协方差
    covMat = covMat / points_size - center * center.transpose();

    // 对协方差进行特征分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    // 计算特征比值 (从大到小)
    //         面：val[2]/val[0]
    //         线：val[2]/val[1]
    feat_eigen_ratio = saes.eigenvalues()[2] / saes.eigenvalues()[ftype];
    // 获取特征向量
    //         面的法线方向vec[0]
    //         线的法线方向vec[2]
    Eigen::Vector3d direct_vec = saes.eigenvectors().col(2 * ftype);

    ap_centor_direct.x = center.x();
    ap_centor_direct.y = center.y();
    ap_centor_direct.z = center.z();
    ap_centor_direct.normal_x = direct_vec.x();
    ap_centor_direct.normal_y = direct_vec.y();
    ap_centor_direct.normal_z = direct_vec.z();
  }

  // Cut root voxel into small pieces
  // frame_head: Position of newest scan in sliding window
  /**
   * @brief 用递归(dfs)对体素节点进行切割，从头节点开始
   *
   * @param layer 当前节点层数
   * @param frame_head 当前帧在滑窗中的位置(但是被赋予新的意义)
   * @param pl_feat_map 存放滑窗中的所有特征信息
   */
  void recut(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map) {
    // 1. 如果当前体素处于未分割状态(初始化)
    if (octo_state == 0) {
      points_size = 0;

      // 1.1 记录滑窗中当前体素中b系下所有点数(待优化)
      for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
        points_size += plvec_orig[i]->size();
      }

      // 1.2 再加上固定的点数(不需要优化)
      // 这些点已经删除了,所以只留下不确定性
      points_size += sig_vec.sigma_size;

      // 1.3 判断点数是否足够
      // feat_eigen_ratio 用于判断是否
      if (points_size < MIN_PS) {
        feat_eigen_ratio = -1;
        // 返回 继续去拟合其他体素的线/面
        return;
      }

      // 1.4 计算当前体素的线面特性
      calc_eigen();

      if (isnan(feat_eigen_ratio)) {
        feat_eigen_ratio = -1;
        return;
      }

      // 将满足线/面的阈值的线/面信息加入到地图中
      // 返回 继续去拟合其他体素的线/面
      if (feat_eigen_ratio >= feat_eigen_limit[ftype]) {
        if (ftype == 1) {
          Eigen::Vector3d v = {ap_centor_direct.normal_x, ap_centor_direct.normal_y, ap_centor_direct.normal_z};
          double vz = std::abs(v.z());  // 取 |v_z|
          double norm = v.norm();       // 计算向量模长 ||v||
          if (norm > 1e-6) {
            double cosTheta = vz / norm;                      // 计算 cos(θ)
            double theta = std::acos(cosTheta) / DEG_TO_RAD;  // 计算角度 θ
            if (theta < 20.0) {
              uint asize;
              for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
                // 遍历滑窗中每一帧
                asize = plvec_tran[i]->size();
                for (uint j = 0; j < asize; j++) {
                  auto p = (*plvec_tran[i])[j];
                  PointType pp;
                  pp.x = p[0];
                  pp.y = p[1];
                  pp.z = p[2];
                  pp.intensity = 2;
                  pl_feat_map.push_back(pp);
                }
              }
              put_state = true;
              return;
            }
          }
        } else {
          uint asize;
          for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
            // 遍历滑窗中每一帧
            asize = plvec_tran[i]->size();
            for (uint j = 0; j < asize; j++) {
              auto p = (*plvec_tran[i])[j];
              PointType pp;
              pp.x = p[0];
              pp.y = p[1];
              pp.z = p[2];
              pp.intensity = 2;
              pl_feat_map.push_back(pp);
            }
          }
          put_state = true;
          return;
        }
      }

      // if(layer == 3)
      // 如果达到根节点返回
      //
      if (layer == layer_limit) {
        return;
      }

      // 标记为需要分割
      octo_state = 1;
      // All points in slidingwindow should be put into subvoxel
      frame_head = 0;
    }

    int leafnum;
    uint a_size;

    // 2. 从当前帧在滑窗中的位置开始遍历
    for (int i = frame_head; i < OCTO_TREE::voxel_windowsize; i++) {
      a_size = plvec_tran[i]->size();
      for (uint j = 0; j < a_size; j++) {
        int xyz[3] = {0, 0, 0};
        for (uint k = 0; k < 3; k++) {
          if ((*plvec_tran[i])[j][k] > voxel_center[k]) {
            xyz[k] = 1;
          }
        }
        // 定位子节点位置
        leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        // 如果是新叶子节点
        if (leaves[leafnum] == nullptr) {
          // 新建
          leaves[leafnum] = new OCTO_TREE(ftype, capacity);
          // 存放节点信息
          leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
          leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
          leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
          leaves[leafnum]->quater_length = quater_length / 2;
        }
        // 将当前点存放入子节点中
        leaves[leafnum]->plvec_orig[i]->push_back((*plvec_orig[i])[j]);
        leaves[leafnum]->plvec_tran[i]->push_back((*plvec_tran[i])[j]);
      }
    }

    // 如果非头节点
    // 清空当前层的点(因为其要被分割乘八块)
    if (layer != 0) {
      for (int i = frame_head; i < OCTO_TREE::voxel_windowsize; i++) {
        if (plvec_orig[i]->size() != 0) {
          vector<Eigen::Vector3d>().swap(*plvec_orig[i]);
          vector<Eigen::Vector3d>().swap(*plvec_tran[i]);
        }
      }
    }

    //
    layer++;
    for (uint i = 0; i < 8; i++) {
      if (leaves[i] != nullptr) {
        leaves[i]->recut(layer, frame_head, pl_feat_map);
      }
    }
  }

  /**
   * @brief marginalize 5 scans in slidingwindow (assume margi_size is 5)
   *
   * @param layer
   * @param margi_size
   * @param q_poses
   * @param t_poses
   * @param window_base
   * @param pl_feat_map
   */
  void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses,
                   int window_base, pcl::PointCloud<PointType> &pl_feat_map) {
    // 如果层数是0 或 达到树的最深层
    if (octo_state != 1 || layer == 0) {
      // 如果达到最深层,update points by new poses
      if (octo_state != 1) {
        for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
          // Update points by new poses
          plvec_trans_func(*plvec_orig[i], *plvec_tran[i], q_poses[i + window_base].matrix(), t_poses[i + window_base]);
        }
      }

      // Push front 5 scans into P_fix
      uint a_size;
      if (feat_eigen_ratio > feat_eigen_limit[ftype]) {
        for (int i = 0; i < margi_size; i++) {
          sig_vec_points.insert(sig_vec_points.end(), plvec_tran[i]->begin(), plvec_tran[i]->end());
        }
        down_sampling_voxel(sig_vec_points, quater_length);

        a_size = sig_vec_points.size();
        sig_vec.tozero();
        sig_vec.sigma_size = a_size;
        for (uint i = 0; i < a_size; i++) {
          sig_vec.sigma_vTv += sig_vec_points[i] * sig_vec_points[i].transpose();
          sig_vec.sigma_vi += sig_vec_points[i];
        }
      }

      // Clear front 5 scans
      for (int i = 0; i < margi_size; i++) {
        PL_VEC().swap(*plvec_orig[i]);
        PL_VEC().swap(*plvec_tran[i]);
        // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
      }

      //
      if (layer == 0) {
        a_size = 0;
        for (int i = margi_size; i < OCTO_TREE::voxel_windowsize; i++) {
          a_size += plvec_orig[i]->size();
        }
        if (a_size == 0) {
          // Voxel has no points in slidingwindow
          is2opt = false;
        }
      }

      for (int i = margi_size; i < OCTO_TREE::voxel_windowsize; i++) {
        plvec_orig[i]->swap(*plvec_orig[i - margi_size]);
        plvec_tran[i]->swap(*plvec_tran[i - margi_size]);
      }

      if (octo_state != 1) {
        points_size = 0;
        for (int i = 0; i < OCTO_TREE::voxel_windowsize - margi_size; i++) {
          points_size += plvec_orig[i]->size();
        }
        points_size += sig_vec.sigma_size;
        if (points_size < MIN_PS) {
          feat_eigen_ratio = -1;
          return;
        }

        calc_eigen();

        if (isnan(feat_eigen_ratio)) {
          feat_eigen_ratio = -1;
          return;
        }
        if (feat_eigen_ratio >= feat_eigen_limit[ftype]) {
          // pl_feat_map.push_back(ap_centor_direct);
        }
      }
    }

    if (octo_state == 1) {
      layer++;
      for (int i = 0; i < 8; i++) {
        if (leaves[i] != nullptr) {
          leaves[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map);
        }
      }
    }
  }

  // Used by "traversal_opt"
  void traversal_opt_calc_eigen() {
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint asize;
    for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
      asize = plvec_tran[i]->size();
      for (uint j = 0; j < asize; j++) {
        covMat += (*plvec_tran[i])[j] * (*plvec_tran[i])[j].transpose();
        center += (*plvec_tran[i])[j];
      }
    }

    covMat -= center * center.transpose() / sw_points_size;
    covMat /= sw_points_size;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    feat_eigen_ratio_test = (saes.eigenvalues()[2] / saes.eigenvalues()[ftype]);
  }

  /**
   * @brief Push voxel into "opt_lsv" (LM optimizer)
   *
   * @param opt_lsv
   */
  void traversal_opt(LM_SLWD_VOXEL &opt_lsv) {
    // 如果该体素中的平面粗拟合成功
    if (octo_state != 1) {
      sw_points_size = 0;
      // 记录滑窗中该体素需要优化的点数
      for (int i = 0; i < OCTO_TREE::voxel_windowsize; i++) {
        sw_points_size += plvec_orig[i]->size();
      }
      // 判断点数是否足够
      if (sw_points_size < MIN_PS) {
        return;
      }

      // 通过更严格的方法拟合体素中待优化点的平面
      // 之前粗拟合的平面条件更宽松且还用了优化完成的点
      traversal_opt_calc_eigen();

      // 判断平面是否拟合成功
      if (isnan(feat_eigen_ratio_test)) {
        return;
      }
      // 如果拟合成功则加入优化器中
      if (put_state && feat_eigen_ratio_test > opt_feat_eigen_limit[ftype]) {
        opt_lsv.push_voxel(plvec_orig, sig_vec, ftype);
      }

    } else {
      for (int i = 0; i < 8; i++) {
        if (leaves[i] != nullptr) {
          leaves[i]->traversal_opt(opt_lsv);
        }
      }
    }
  }
};

int OCTO_TREE::voxel_windowsize = 0;

// Like "LM_SLWD_VOXEL"
// // Scam2map optimizer
// class VOXEL_DISTANCE
// {
// public:
//   SO3 so3_pose, so3_temp;
//   Eigen::Vector3d t_pose, t_temp;
//   PL_VEC surf_centor, surf_direct;
//   PL_VEC corn_centor, corn_direct;
//   PL_VEC surf_gather, corn_gather;
//   vector<double> surf_coeffs, corn_coeffs;
//
//   void push_surf(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
//   {
//     direct.normalize();
//     surf_direct.push_back(direct); surf_centor.push_back(centor);
//     surf_gather.push_back(orip); surf_coeffs.push_back(coeff);
//   }
//
//   void push_line(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
//   {
//     direct.normalize();
//     corn_direct.push_back(direct); corn_centor.push_back(centor);
//     corn_gather.push_back(orip); corn_coeffs.push_back(coeff);
//   }
//
//   void evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1>
//   &g, double &residual)
//   {
//     Hess.setZero(); g.setZero(); residual = 0;
//     uint a_size = surf_gather.size();
//     for(uint i=0; i<a_size; i++)
//     {
//       Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
//       Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
//       Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
//       vec_tran += t_p;
//
//       Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
//       Eigen::Vector3d d_vec = _jac * v_ac;
//       Eigen::Matrix<double, 3, 6> jacob;
//       jacob.block<3, 3>(0, 0) = _jac * point_xi;
//       jacob.block<3, 3>(0, 3) = _jac;
//
//       residual += surf_coeffs[i] * d_vec.dot(d_vec);
//       Hess += surf_coeffs[i] * jacob.transpose() * jacob;
//       g += surf_coeffs[i] * jacob.transpose() * d_vec;
//     }
//
//     a_size = corn_gather.size();
//     for(uint i=0; i<a_size; i++)
//     {
//       Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
//       Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
//       Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
//       vec_tran += t_p;
//
//       Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
//       Eigen::Vector3d d_vec = _jac * v_ac;
//       Eigen::Matrix<double, 3, 6> jacob;
//       jacob.block<3, 3>(0, 0) = _jac * point_xi;
//       jacob.block<3, 3>(0, 3) = _jac;
//
//       residual += corn_coeffs[i] * d_vec.dot(d_vec);
//       Hess += corn_coeffs[i] * jacob.transpose() * jacob;
//       g += corn_coeffs[i] * jacob.transpose() * d_vec;
//     }
//   }
//
//   void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual)
//   {
//     residual = 0;
//     uint a_size = surf_gather.size();
//     for(uint i=0; i<a_size; i++)
//     {
//       Eigen::Matrix3d _jac = surf_direct[i] * surf_direct[i].transpose();
//       Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather[i];
//       vec_tran += t_p;
//
//       Eigen::Vector3d v_ac = vec_tran - surf_centor[i];
//       Eigen::Vector3d d_vec = _jac * v_ac;
//
//       residual += surf_coeffs[i] * d_vec.dot(d_vec);
//     }
//
//     a_size = corn_gather.size();
//     for(uint i=0; i<a_size; i++)
//     {
//       Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct[i] * corn_direct[i].transpose();
//       Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather[i];
//       vec_tran += t_p;
//
//       Eigen::Vector3d v_ac = vec_tran - corn_centor[i];
//       Eigen::Vector3d d_vec = _jac * v_ac;
//
//       residual += corn_coeffs[i] * d_vec.dot(d_vec);
//     }
//
//   }
//
//   void damping_iter()
//   {
//     double u = 0.01, v = 2;
//     Eigen::Matrix<double, 6, 6> D; D.setIdentity();
//     Eigen::Matrix<double, 6, 6> Hess, Hess2;
//     Eigen::Matrix<double, 6, 1> g;
//     Eigen::Matrix<double, 6, 1> dxi;
//     double residual1, residual2;
//
//     cv::Mat matA(6, 6, CV_64F, cv::Scalar::all(0));
//     cv::Mat matB(6, 1, CV_64F, cv::Scalar::all(0));
//     cv::Mat matX(6, 1, CV_64F, cv::Scalar::all(0));
//
//     for(int i=0; i<20; i++)
//     {
//       evaluate_para(so3_pose, t_pose, Hess, g, residual1);
//       D = Hess.diagonal().asDiagonal();
//
//       // dxi = (Hess + u*D).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-g);
//
//       Hess2 = Hess + u*D;
//       for(int j=0; j<6; j++)
//       {
//         matB.at<double>(j, 0) = -g(j, 0);
//         for(int f=0; f<6; f++)
//         {
//           matA.at<double>(j, f) = Hess2(j, f);
//         }
//       }
//       cv::solve(matA, matB, matX, cv::DECOMP_QR);
//       for(int j=0; j<6; j++)
//       {
//         dxi(j, 0) = matX.at<double>(j, 0);
//       }
//
//       so3_temp = SO3::exp(dxi.block<3, 1>(0, 0)) * so3_pose;
//       t_temp = t_pose + dxi.block<3, 1>(3, 0);
//       evaluate_only_residual(so3_temp, t_temp, residual2);
//       double q1 = dxi.dot(u*D*dxi-g);
//       double q = residual1 - residual2;
//       // printf("residual: %lf u: %lf v: %lf q: %lf %lf %lf\n", residual1, u, v, q/q1, q1, q);
//       if(q > 0)
//       {
//         so3_pose = so3_temp;
//         t_pose = t_temp;
//         q = q / q1;
//         v = 2;
//         q = 1 - pow(2*q-1, 3);
//         u *= (q<one_three ? one_three:q);
//       }
//       else
//       {
//         u = u * v;
//         v = 2 * v;
//       }
//
//       if(fabs(residual1-residual2)<1e-9)
//       {
//         break;
//       }
//
//     }
//
//   }
//
// };


#endif

