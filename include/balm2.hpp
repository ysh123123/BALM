#ifndef BAVOXEL_HPP
#define BAVOXEL_HPP

#include <Eigen/Eigenvalues>
#include <thread>
#include "tools.hpp"
namespace BALM2 {
// 全局变量
inline int layer_limit = 2;
inline int layer_size[4] = {30, 30, 30, 30};
// float eigen_value_array[] = {1.0/4.0, 1.0/4.0, 1.0/4.0};
inline float eigen_value_array[4] = {1.0 / 16, 1.0 / 16, 1.0 / 16, 1.0 / 16};
inline int min_ps = 15;
inline double one_three = (1.0 / 3.0);

inline double voxel_size = 1;
inline int life_span = 1000;
inline int win_size = 30;
inline int fix_size = 5;

inline int merge_enable = 1;

class VOX_HESS {
 public:
  vector<PointCluster > sig_vecs;
  //
  vector<vector<PointCluster> > plvec_voxels;
  vector<double> coeffs, coeffs_back;

  vector<pcl::PointCloud<PointType>::Ptr> plptrs;

  void push_voxel(const vector<PointCluster> &vec_orig, const PointCluster &fix, double feat_eigen, int layer) {
    int process_size = 0;
    for (int i = 0; i < win_size; i++)
      if (vec_orig[i].N != 0) process_size++;

    if (process_size < 2) return;  // 改

    double coe = 1 - feat_eigen / eigen_value_array[layer];
    coe = coe * coe;
    coe = 1;
    coe = 0;
    for (int j = 0; j < win_size; j++) coe += vec_orig[j].N;
    // std::cout << vec_orig->size() << std::endl;
    plvec_voxels.push_back(vec_orig);

    sig_vecs.push_back(fix);
    coeffs.push_back(coe);
    pcl::PointCloud<PointType>::Ptr plptr(new pcl::PointCloud<PointType>());
    plptrs.push_back(plptr);
  }

  void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                     double &residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    const int kk = 0;

    PLV(3) viRiTuk(win_size);
    PLM(3) viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    for (int a = head; a < end; a++) {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      double coe = coeffs[a];

      PointCluster sig = sig_vecs[a];
      for (int i = 0; i < win_size; i++)
        if (sig_orig[i].N != 0) {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }

      const Eigen::Vector3d &vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d &lmbd = saes.eigenvalues();
      const Eigen::Matrix3d &U = saes.eigenvectors();
      int NN = sig.N;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for (int i = 0; i < 3; i++)
        if (i != kk) umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for (int i = 0; i < win_size; i++)
        // for(int i=1; i<win_size; i++)
        if (sig_orig[i].N != 0) {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat;
          vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat;
          RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();

          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6 * i, 0) += coe * jjt;

          const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
                                  2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() -
                                  0.5 * hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

          Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
        }

      for (int i = 0; i < win_size - 1; i++)
        // for(int i=1; i<win_size-1; i++)
        if (sig_orig[i].N != 0) {
          double ni = sig_orig[i].N;
          for (int j = i + 1; j < win_size; j++)
            if (sig_orig[j].N != 0) {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Hb;
            }
        }

      residual += coe * lmbd[kk];
    }

    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++) Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  void left_evaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                     double &residual) {
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    // vector<PointCluster> sig_tran(win_size);
    int l = 0;
    Eigen::Matrix<double, 3, 4> Sp;
    Sp.setZero();
    Sp.block<3, 3>(0, 0).setIdentity();
    Eigen::Matrix4d F;
    F.setZero();
    F(3, 3) = 1;

    PLM(4) T(win_size);
    for (int i = 0; i < win_size; i++) T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;

    vector<PLM(4) *> Cs;
    for (int a = 0; a < plvec_voxels.size(); a++) {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
      for (int i = 0; i < win_size; i++)
        Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      Cs.push_back(Co);
    }

    // double t0 = ros::Time::now().toSec();

    for (int a = head; a < end; a++) {
      // const vector<PointCluster> &sig_orig = *plvec_voxels[a];
      double coe = coeffs[a];

      // PLM(4) Co(win_size, Eigen::Matrix4d::Zero());
      Eigen::Matrix4d C;
      C.setZero();
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   Co[i] << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      //   C += T[i] * Co[i] * T[i].transpose();
      // }

      PLM(4) &Co = *Cs[a];
      for (int i = 0; i < win_size; i++)
        if ((int)Co[i](3, 3) > 0) C += T[i] * Co[i] * T[i].transpose();

      double NN = C(3, 3);
      C = C / NN;
      // Eigen::Vector4d CF = C.block<4, 1>(0, 3);
      // cout << CF << endl << endl;
      // cout << C*F << endl;
      // exit(0);

      Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d Uev = saes.eigenvectors();

      residual += coe * lmbd[l];

      Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
      Eigen::Matrix<double, 4, 6> U[3];

      PLV(-1) g_kl(3);
      for (int k = 0; k < 3; k++) {
        g_kl[k].resize(6 * win_size);
        g_kl[k].setZero();
        U[k].setZero();
        U[k].block<3, 3>(0, 0) = hat(u[k]);
        U[k].block<1, 3>(3, 3) = u[k];
      }

      for (int j = 0; j < win_size; j++)
        for (int k = 0; k < 3; k++)
          if (Co[j](3, 3) > 0.1) {
            Eigen::Matrix<double, 3, 4> SpTC = Sp * (T[j] - C * F) * Co[j] * T[j].transpose();
            Eigen::Matrix<double, 1, 6> g1, g2;
            g1 = u[l].transpose() * SpTC * U[k];
            g2 = u[k].transpose() * SpTC * U[l];

            g_kl[k].block<6, 1>(6 * j, 0) = (g1 + g2).transpose() / NN;
          }

      JacT += coe * g_kl[l];

      for (int i = 0; i < win_size; i++)
        if (Co[i](3, 3) > 0.1) {
          for (int j = 0; j < win_size; j++)
            if (Co[j](3, 3) > 0.1) {
              Eigen::Matrix4d Dij = Co[i] * F * Co[j];
              Eigen::Matrix<double, 6, 6> Hs = -2.0 / NN / NN * U[l].transpose() * T[i] * Dij * T[j].transpose() * U[l];

              if (i == j) {
                Hs += 2 / NN * U[l].transpose() * T[j] * Co[j] * T[j].transpose() * U[l];
                Eigen::Vector3d SpTC = Sp * T[j] * Co[j] * (T[j] - C * F).transpose() * Sp.transpose() * u[l];
                Eigen::Matrix3d h1 = hat(SpTC);
                Eigen::Matrix3d h2 = hat(u[l]);

                Hs.block<3, 3>(0, 0) += (h1 * h2 + h2 * h1) / NN;
              }

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Hs;
            }
        }

      for (int k = 0; k < 3; k++)
        if (k != l) Hess += coe * 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k] * g_kl[k].transpose();
    }

    // double t1 = ros::Time::now().toSec();
    // printf("t1: %lf\n", t1 - t0);

    // PLM(6) LL(win_size);
    // Eigen::Matrix3d zero33; zero33.setZero();
    // for(int i=0; i<win_size; i++)
    //   LL[i] << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;

    // for(int i=0; i<win_size; i++)
    // {
    //   JacT.block<6, 1>(6*i, 0) = LL[i].transpose() * JacT.block<6, 1>(6*i, 0);
    //   for(int j=0; j<win_size; j++)
    //   {
    //     Hess.block<6, 6>(6*i, 6*j) = LL[i].transpose() * Hess.block<6, 6>(6*i, 6*j) * LL[j];
    //   }
    // }

    // Eigen::Matrix3d zero33; zero33.setZero();
    // Eigen::MatrixXd LL(6*win_size, 6*win_size); LL.setZero();
    // for(int i=0; i<win_size; i++)
    // {
    //   LL.block<6, 6>(6*i, 6*i) << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;
    // }
    // JacT = LL.transpose() * JacT;
    // Hess = LL.transpose() * Hess * LL;
  }

  void left_evaluate_acc2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                          double &residual) {
    // std::cout << "---------------" << std::endl;
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    int l = 0;
    PLM(4) T(win_size);
    for (int i = 0; i < win_size; i++) T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;
    // std::cout << "------11111----" << std::endl;
    vector<PLM(4) *> Cs;
    for (int a = 0; a < plvec_voxels.size(); a++) {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
      for (int i = 0; i < win_size; i++)
        Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
      Cs.push_back(Co);
    }
    // std::cout << "---33333------" << std::endl;
    for (int a = head; a < end; a++) {
      double coe = coeffs[a];
      Eigen::Matrix4d C;
      C.setZero();

      vector<int> Ns(win_size);

      PLM(4) &Co = *Cs[a];
      PLM(4) TC(win_size), TCT(win_size);
      for (int j = 0; j < win_size; j++)
        if ((int)Co[j](3, 3) > 0) {
          TC[j] = T[j] * Co[j];
          TCT[j] = TC[j] * T[j].transpose();
          C += TCT[j];

          Ns[j] = Co[j](3, 3);
        }

      double NN = C(3, 3);
      C = C / NN;
      Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();
      Eigen::Matrix3d Uev = saes.eigenvectors();

      residual += coe * lmbd[l];

      Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
      Eigen::Matrix<double, 6, 4> U[3];
      PLV(6) g_kl[3];
      for (int k = 0; k < 3; k++) {
        g_kl[k].resize(win_size);
        U[k].setZero();
        U[k].block<3, 3>(0, 0) = hat(-u[k]);
        U[k].block<3, 1>(3, 3) = u[k];
      }

      PLV(6) UlTCF(win_size, Eigen::Matrix<double, 6, 1>::Zero());

      Eigen::VectorXd JacT_iter(6 * win_size);
      for (int i = 0; i < win_size; i++)
        if (Ns[i] != 0) {
          Eigen::Matrix<double, 3, 4> temp = T[i].block<3, 4>(0, 0);
          temp.block<3, 1>(0, 3) -= v_bar;
          Eigen::Matrix<double, 4, 3> TC_TCFSp = TC[i] * temp.transpose();
          for (int k = 0; k < 3; k++) {
            Eigen::Matrix<double, 6, 1> g1, g2;
            g1 = U[k] * TC_TCFSp * u[l];
            g2 = U[l] * TC_TCFSp * u[k];

            g_kl[k][i] = (g1 + g2) / NN;
          }

          UlTCF[i] = (U[l] * TC[i]).block<6, 1>(0, 3);
          JacT.block<6, 1>(6 * i, 0) += coe * g_kl[l][i];

          // Eigen::Matrix<double, 6, 6> Hb(2.0/NN * U[l] * TCT[i] * U[l].transpose());

          Eigen::Matrix<double, 6, 6> Ha(-2.0 / NN / NN * UlTCF[i] * UlTCF[i].transpose());

          Eigen::Matrix3d Ell = 1.0 / NN * hat(TC_TCFSp.block<3, 3>(0, 0) * u[l]) * hat(u[l]);
          Ha.block<3, 3>(0, 0) += Ell + Ell.transpose();

          for (int k = 0; k < 3; k++)
            if (k != l) Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][i].transpose();

          Hess.block<6, 6>(6 * i, 6 * i) += coe * Ha;
        }

      for (int i = 0; i < win_size; i++)
        if (Ns[i] != 0) {
          Eigen::Matrix<double, 6, 6> Hb = U[l] * TCT[i] * U[l].transpose();
          Hess.block<6, 6>(6 * i, 6 * i) += 2.0 / NN * coe * Hb;
        }

      for (int i = 0; i < win_size - 1; i++)
        if (Ns[i] != 0) {
          for (int j = i + 1; j < win_size; j++)
            if (Ns[j] != 0) {
              Eigen::Matrix<double, 6, 6> Ha = -2.0 / NN / NN * UlTCF[i] * UlTCF[j].transpose();

              for (int k = 0; k < 3; k++)
                if (k != l) Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][j].transpose();

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Ha;
            }
        }
    }
    // std::cout << "----22222--------" << std::endl;
    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++) Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  void evaluate_only_residual(const vector<IMUST> &xs, double &residual) const {
    residual = 0;
    vector<PointCluster> sig_tran(win_size);
    int kk = 0;  // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    vector<double> ress(gps_size);

    for (int a = 0; a < gps_size; a++) {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      PointCluster sig = sig_vecs[a];

      for (int i = 0; i < win_size; i++) {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }
      // std::cout << "sig.N = " << sig.N << std::endl;
      // std::cout << "sig.v = " << sig.v << std::endl;
      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += coeffs[a] * lmbd[kk];

      ress[a] = lmbd[kk];
    }

    // vector<double> ress_tem = ress;
    // sort(ress_tem.begin(), ress_tem.end());
    // double bound = 0.8;
    // bound = ress_tem[gps_size * bound];
    // coeffs = coeffs_back;

    // for(int a=0; a<gps_size; a++)
    //   if(ress[a] > bound)
    //     coeffs[a] = 0;
  }

  ~VOX_HESS() {
    int vsize = sig_vecs.size();
    // for(int i=0; i<vsize; i++)
    // {
    //   delete sig_vecs[i], sig_vecs[i] = nullptr;
    //   delete plvec_voxels[i], plvec_voxels[i] = nullptr;
    // }
  }
};

class OctoTreeNode {
 public:
  bool is_plane{false};
  vector<PLV(3)> vec_orig{}, vec_tran{};
  vector<PointCluster> sig_orig{}, sig_tran{};
  PointCluster fix_point{};
  PLV(3) vec_fix{};

  Eigen::Vector3d center{}, direct{}, value_vector{};  // temporal
  double decision{}, ref{};

  OctoTreeNode() {
    vec_orig.resize(win_size);
    vec_tran.resize(win_size);
    sig_orig.resize(win_size);
    sig_tran.resize(win_size);

    ref = 255.0 * rand() / (RAND_MAX + 1.0f);
  }

  bool JudgeEigen(const int win_count) {
    // 获取该voxel下的所有点
    PointCluster cov_mat = fix_point;
    for (int i = 0; i < win_count; i++) {
      cov_mat += sig_tran[i];
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat.cov());
    value_vector = saes.eigenvalues();
    center = cov_mat.v / cov_mat.N;
    direct = saes.eigenvectors().col(0);

    decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];
    return (decision < eigen_value_array[0]) && saes.eigenvalues()[2] / saes.eigenvalues()[1] < 10.;
  }

  void Recut(const int win_count) {
    // 如果不是mid_node
    if (!is_plane) {
      // 获取用于拟合的原始点数
      int point_size = fix_point.N;
      // std::cout<<"fix_point = "<<fix_point.v<<std::endl;
      for (int i = 0; i < win_count; i++) {
        // 获取新增点数
        point_size += sig_orig[i].N;
        // std::cout<<"sig_orig = "<<sig_orig[i].v<<std::endl;
      }
      // 判断点数是否足够
      if (point_size <= min_ps) {
        return;
      }
      // 拟合平面
      if (JudgeEigen(win_count)) {
        // 将当前voxel状态设置为平面
        if (point_size > layer_size[0]) {
          is_plane = true;
        }
      }
    }
  }

  void ToMargi(int mg_size, vector<IMUST> &x_poses, int win_count) {
    // 更新点
    if (!x_poses.empty())
      for (int i = 0; i < win_count; i++) {
        sig_tran[i].transform(sig_orig[i], x_poses[i]);
        plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
      }
    // PointType ap;
    // ap.intensity = ref;
    if (fix_point.N < 50) {
      for (int i = 0; i < mg_size; i++) {
        // for (auto pvec : vec_tran[i]) {
        //   ap.x = pvec.x();
        //   ap.y = pvec.y();
        //   ap.z = pvec.z();
        //   pl_feat.push_back(ap);
        // }

        fix_point += sig_tran[i];
        vec_fix.insert(vec_fix.end(), vec_tran[i].begin(), vec_tran[i].end());
      }
    }

    for (int i = mg_size; i < win_count; i++) {
      sig_orig[i - mg_size] = sig_orig[i];
      sig_tran[i - mg_size] = sig_tran[i];
      vec_orig[i - mg_size].swap(vec_orig[i]);
      vec_tran[i - mg_size].swap(vec_tran[i]);
    }

    for (int i = win_count - mg_size; i < win_count; i++) {
      sig_orig[i].clear();
      sig_tran[i].clear();
      vec_orig[i].clear();
      vec_tran[i].clear();
    }
  }

  void TrasDisplay(pcl::PointCloud<PointType> &pl_feat, const int win_count) {
    if (is_plane) {
      PointType ap;
      ap.intensity = ref;

      int tsize = 0;
      for (int i = 0; i < win_count; i++) tsize += vec_tran[i].size();
      if (tsize < 100) return;

      for (int i = 0; i < win_count; i++)
        for (Eigen::Vector3d pvec : vec_tran[i]) {
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          pl_feat.push_back(ap);
        }
    }
  }

  void TrasOpt(pcl::PointCloud<PointType> &pl_feat, VOX_HESS &vox_opt, vector<IMUST> &x_poses, const int win_count) {
    if (is_plane) {
      int points_size = 0;
      vector<PointCluster> cur_sig_orig;
      PointCluster cur_fix_point{};
      cur_sig_orig.resize(win_size);
      for (int i = 0; i < win_size; i++) {
        if (!vec_orig[i].empty()) {
          plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
          points_size += vec_orig[i].size();
          for (auto p : vec_orig[i]) {
            cur_sig_orig[i].push(p);
          }
          // std::cout<<"cur_sig_orig[i].N = "<<cur_sig_orig[i].N<<std::endl;
          // std::cout<<"cur_sig_orig[i].v = "<<cur_sig_orig[i].v<<std::endl;
          for (auto pvec : vec_tran[i]) {
            PointType ap;
            ap.intensity = ref;
            ap.x = pvec.x();
            ap.y = pvec.y();
            ap.z = pvec.z();
            pl_feat.push_back(ap);
          }
        }
      }
      if (!vec_fix.empty()) {
        for (auto p : vec_fix) {
          cur_fix_point.push(p);
        }
        // std::cout<<"cur_fix_point.N = "<<cur_fix_point.N<<std::endl;
        // std::cout<<"cur_fix_point.v = "<<cur_fix_point.v<<std::endl;
      }

      if (points_size < min_ps) {
        return;
      }
      vox_opt.push_voxel(cur_sig_orig, cur_fix_point, decision, 0);
    }
  }
};

class OctoTreeRoot : public OctoTreeNode {
 public:
  bool is2opt;
  vector<int> each_num;

  // 在基类的基础上增加
  OctoTreeRoot() {
    is2opt = true;
    each_num.resize(win_size);
    for (int i = 0; i < win_size; i++) each_num[i] = 0;
  }
  void AddPoint(const int fnum, const Eigen::Vector3d &pvec_orig, const Eigen::Vector3d &pvec_tran) {
    // 动态检查并调整 vector 大小
    if (fnum >= vec_orig.size()) {
      vec_orig.resize(fnum + 1);
      vec_tran.resize(fnum + 1);
      sig_orig.resize(fnum + 1);
      sig_tran.resize(fnum + 1);
    }

    vec_orig[fnum].push_back(pvec_orig);
    vec_tran[fnum].push_back(pvec_tran);

    sig_orig[fnum].push(pvec_orig);
    sig_tran[fnum].push(pvec_tran);

    each_num[fnum]++;
  }
  // 边缘化
  void Marginalize(const int mg_size, vector<IMUST> &x_poses, const int win_count) {
    ToMargi(mg_size, x_poses, win_count);

    int left_size = 0;
    for (int i = mg_size; i < win_count; i++) {
      each_num[i - mg_size] = each_num[i];
      left_size += each_num[i - mg_size];
    }

    if (left_size == 0) is2opt = false;

    for (int i = win_count - mg_size; i < win_count; i++) each_num[i] = 0;
  }
};

using OctoTreeRootPtr = std::shared_ptr<OctoTreeRoot>;
using OctoTreeMap = unordered_map<VOXEL_LOC, OctoTreeRootPtr>;

class BALM2 {
 public:
  BALM2() {}
  double DivideThreadLeft(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd &Hess,
                          Eigen::VectorXd &JacT) {
    // std::cout << "sssssss111111sssss" << std::endl;
    constexpr int thd_num = 4;
    double residual = 0;
    Hess.setZero();
    JacT.setZero();
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);
    // std::cout << "sssss333333sssssss" << std::endl;
    for (int i = 0; i < thd_num; i++) {
      hessians[i].resize(6 * win_size, 6 * win_size);
      jacobins[i].resize(6 * win_size);
    }
    // std::cout << "ssssssssssssssss" << std::endl;
    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    const int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num) tthd_num = 1;

    vector<thread *> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    for (int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::left_evaluate_acc2, &voxhess, x_stats, part * i, part * (i + 1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for (int i = 0; i < tthd_num; i++) {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double OnlyResidual(const vector<IMUST> &x_stats, const VOX_HESS &voxhess, vector<IMUST> &x_ab) {
    double residual1 = 0, residual2 = 0;

    voxhess.evaluate_only_residual(x_stats, residual2);
    return (residual1 + residual2);
  }

  void DampingIter(vector<IMUST> &x_stats, VOX_HESS &voxhess) {
    // 记录每一帧的平面数
    //std::cout << "-------44--------333333 = " << std::endl;
    vector<int> planes(x_stats.size(), 0);
    //std::cout << "----33333-----------333333 = " << x_stats.size() << " " << voxhess.plvec_voxels[0].size()
             // << std::endl;
//     for (auto ss : voxhess.plvec_voxels) {
//       if(ss == nullptr) {
//         std::cout << ss->size() << std::endl;
//       }
//
//     }
    if (!voxhess.plvec_voxels.empty()) {
      for (int i = 0; i < voxhess.plvec_voxels.size(); i++) {
        if (!voxhess.plvec_voxels[i].empty()) {
          for (int j = 0; j < voxhess.plvec_voxels[i].size(); j++) {
            if (voxhess.plvec_voxels[i].at(j).N != 0) {
              planes[j]++;
            }
          }
        } else {
          std::cout << "voxhess.plvec_voxels[i].empty()" << std::endl;
        }
      }
    } else {
      std::cout << "voxhess.plvec_voxels.empty()" << std::endl;
    }

    // std::cout << "----------11111-----333333 = " << std::endl;
    // 检查平面数释放满足条件
    sort(planes.begin(), planes.end());
    if (planes[0] < 20) {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      exit(0);
    }
    // std::cout << "----------2222-----333333 = " << std::endl;
    // LM参数设置
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(6 * win_size, 6 * win_size), Hess(6 * win_size, 6 * win_size);
    Eigen::VectorXd JacT(6 * win_size), dxi(6 * win_size);

    D.setIdentity();
    double residual1 = 0.,  // 优化前误差
        residual2 = 0.,     // 优化后误差
        q = 0.;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    for (int i = 1; i < win_size; i++) {
      x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
      x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
    }
    std::cout << "start calc hess----------" << std::endl;
    for (int i = 0; i < 10; i++) {
      if (is_calc_hess) {
        // 多线程加速计算hess
        // residual1 = DivideThreadRight(x_stats, voxhess, x_ab, Hess, JacT);
        // std::cout << "---------------333333 = " << std::endl;
        residual1 = DivideThreadLeft(x_stats, voxhess, x_ab, Hess, JacT);
        // std::cout << "----------------residual1 = " << residual1 << std::endl;
      }
      if (Hess.determinant() < 1e-6) {
        std::cerr << "Warning: Hessian is nearly singular!" << std::endl;
      }
      // std::cout << "hess ---------- " << i << " calced" << std::endl;
      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u * D).ldlt().solve(-JacT);
      // std::cout << dxi << std::endl;
      // 更新位姿
      for (int j = 0; j < win_size; j++) {
        // right update
        // x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
        // x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);

        // left update
        Eigen::Matrix3d dR = Exp(dxi.block<3, 1>(DVEL * j, 0));
        x_stats_temp[j].R = dR * x_stats[j].R;
        x_stats_temp[j].p = dR * x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
      }
      residual2 = OnlyResidual(x_stats_temp, voxhess, x_ab);
      // std::cout << "----------------residual2 = " << residual2 << std::endl;

      const double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
      // std::cout << "----------------q1 = " << q1 << std::endl;
      // 计算优化后残差

      // 计算残差变换量
      q = (residual1 - residual2);

      printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);

      if (q > 0) {
        // 如果误差缩小，则更新
        x_stats = x_stats_temp;
        // 更新LM参数
        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < one_three ? one_three : q);
        is_calc_hess = true;
      } else {
        // break;
        // 如果误差增大
        // 更新LM参数
        u = u * v;
        v = 2 * v;
        // 不计算hess
        is_calc_hess = false;
      }

      // if(IterStop(dxi2, 1e-4))
      if (IterStop(dxi, 1e-6)) break;

      if (fabs(residual1 - residual2) / residual1 < 1e-6) {
        break;
      }
    }

    IMUST es0 = x_stats[0];
    for (uint i = 0; i < x_stats.size(); i++) {
      x_stats[i].p = es0.R.transpose() * (x_stats[i].p - es0.p);
      x_stats[i].R = es0.R.transpose() * x_stats[i].R;
    }
  }

  bool IterStop(Eigen::VectorXd &dx, const double thre = 1e-7, int win_size = 0) {
    // int win_size = dx.rows() / 6;
    if (win_size == 0) win_size = dx.rows() / 6;

    double angErr = 0, tranErr = 0;
    for (int i = 0; i < win_size; i++) {
      angErr += dx.block<3, 1>(6 * i, 0).norm();
      tranErr += dx.block<3, 1>(6 * i + 3, 0).norm();
    }

    angErr /= win_size;
    tranErr /= win_size;
    return (angErr < thre) && (tranErr < thre);
  }

  /**
   * @brief 将当前帧点云加入地图
   * @param feat_map 初始的八叉树地图
   * @param pl_feat 雷达系下的点云
   * @param x_key 对应的位姿
   * @param fnum 对应的id
   */
  template <typename T>
  static void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<T> &pl_feat, const IMUST &x_key, const int fnum) {
    float loc_xyz[3];
    for (const auto &p_c : pl_feat) {
      // 获取雷达坐标系下点
      Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
      // 将点变换到世界坐标系下
      Eigen::Vector3d pvec_tran = x_key.R * pvec_orig + x_key.p;

      // 计算每个点对应的voxel位置
      for (int j = 0; j < 3; j++) {
        loc_xyz[j] = pvec_tran[j] / voxel_size;
        if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
      }

      // 存放当前点的voxel位置
      VOXEL_LOC position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
                         static_cast<int64_t>(loc_xyz[2]));
      // 通过位置找到对应栅格
      if (auto iter = feat_map.find(position); iter != feat_map.end()) {
        const auto &cur_feat = iter->second;
        // 如果是已有栅格
        cur_feat->AddPoint(fnum, pvec_orig, pvec_tran);
        cur_feat->is2opt = true;
      } else {
        // 构建新的八叉树栅格头
        const auto ot = std::make_shared<OctoTreeRoot>();
        ot->AddPoint(fnum, pvec_orig, pvec_tran);
        ot->is2opt = true;
        // 存放新栅格
        feat_map[position] = ot;
      }
    }
  }
};
}  // namespace BALM2

#endif

// #ifndef BAVOXEL_HPP
// #define BAVOXEL_HPP
//
// #include <Eigen/Eigenvalues>
// #include <thread>
// #include "tools.hpp"
// namespace BALM2 {
// // 全局变量
// inline int layer_limit = 2;
// inline int layer_size[4] = {30, 30, 30, 30};
// // float eigen_value_array[] = {1.0/4.0, 1.0/4.0, 1.0/4.0};
// inline float eigen_value_array[4] = {1.0 / 16, 1.0 / 16, 1.0 / 16, 1.0 / 16};
// inline int min_ps = 15;
// inline double one_three = (1.0 / 3.0);
//
// inline double voxel_size = 1;
// inline int life_span = 1000;
// inline int win_size = 30;
// inline int fix_size = 5;
//
// inline int merge_enable = 1;
//
// class VoxHess {
//  public:
//   vector<const PointCluster *> sig_vecs;
//   vector<const vector<PointCluster> *> plvec_voxels;
//   vector<double> coeffs, coeffs_back;
//
//   vector<pcl::PointCloud<PointType>::Ptr> plptrs;
//
//   void PushVoxel(const vector<PointCluster> *vec_orig, const PointCluster *fix, const double feat_eigen,
//                  const int layer) {
//     int process_size = 0;
//     for (int i = 0; i < win_size; i++)
//       if ((*vec_orig)[i].N != 0) process_size++;
//
//     if (process_size < 2) return;  // 改
//
//     double coe = 1 - feat_eigen / eigen_value_array[layer];
//     coe = coe * coe;
//     coe = 1;
//     coe = 0;
//     for (int j = 0; j < win_size; j++) coe += (*vec_orig)[j].N;
//
//     plvec_voxels.push_back(vec_orig);
//     sig_vecs.push_back(fix);
//     // 平面系数
//     coeffs.push_back(coe);
//     const pcl::PointCloud<PointType>::Ptr plptr(new pcl::PointCloud<PointType>());
//     plptrs.push_back(plptr);
//   }
//
//   void AccEvaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
//                     double &residual) const {
//     Hess.setZero();
//     JacT.setZero();
//     residual = 0;
//     vector<PointCluster> sig_tran(win_size);
//     const int kk = 0;
//
//     PLV(3) viRiTuk(win_size);
//     PLM(3) viRiTukukT(win_size);
//
//     vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
//     Eigen::Matrix3d umumT;
//
//     for (int a = head; a < end; a++) {
//       const vector<PointCluster> &sig_orig = *plvec_voxels[a];
//       double coe = coeffs[a];
//
//       PointCluster sig = *sig_vecs[a];
//       for (int i = 0; i < win_size; i++)
//         if (sig_orig[i].N != 0) {
//           sig_tran[i].transform(sig_orig[i], xs[i]);
//           sig += sig_tran[i];
//         }
//
//       const Eigen::Vector3d &vBar = sig.v / sig.N;
//       Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
//       const Eigen::Vector3d &lmbd = saes.eigenvalues();
//       const Eigen::Matrix3d &U = saes.eigenvectors();
//       int NN = sig.N;
//
//       Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
//
//       const Eigen::Vector3d &uk = u[kk];
//       Eigen::Matrix3d ukukT = uk * uk.transpose();
//       umumT.setZero();
//       for (int i = 0; i < 3; i++)
//         if (i != kk) umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();
//
//       for (int i = 0; i < win_size; i++)
//         // for(int i=1; i<win_size; i++)
//         if (sig_orig[i].N != 0) {
//           Eigen::Matrix3d Pi = sig_orig[i].P;
//           Eigen::Vector3d vi = sig_orig[i].v;
//           Eigen::Matrix3d Ri = xs[i].R;
//           double ni = sig_orig[i].N;
//
//           Eigen::Matrix3d vihat;
//           vihat << SKEW_SYM_MATRX(vi);
//           Eigen::Vector3d RiTuk = Ri.transpose() * uk;
//           Eigen::Matrix3d RiTukhat;
//           RiTukhat << SKEW_SYM_MATRX(RiTuk);
//
//           Eigen::Vector3d PiRiTuk = Pi * RiTuk;
//           viRiTuk[i] = vihat * RiTuk;
//           viRiTukukT[i] = viRiTuk[i] * uk.transpose();
//
//           Eigen::Vector3d ti_v = xs[i].p - vBar;
//           double ukTti_v = uk.dot(ti_v);
//
//           Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
//           Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
//           Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
//           Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
//           Auk[i] /= NN;
//
//           const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
//           JacT.block<6, 1>(6 * i, 0) += coe * jjt;
//
//           const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
//           Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
//           Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat -
//                                   2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() -
//                                   0.5 * hat(jjt.block<3, 1>(0, 0));
//           Hb.block<3, 3>(0, 3) += HRt;
//           Hb.block<3, 3>(3, 0) += HRt.transpose();
//           Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;
//
//           Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
//         }
//
//       for (int i = 0; i < win_size - 1; i++)
//         // for(int i=1; i<win_size-1; i++)
//         if (sig_orig[i].N != 0) {
//           double ni = sig_orig[i].N;
//           for (int j = i + 1; j < win_size; j++)
//             if (sig_orig[j].N != 0) {
//               double nj = sig_orig[j].N;
//               Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
//               Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
//               Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
//               Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
//               Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;
//
//               Hess.block<6, 6>(6 * i, 6 * j) += coe * Hb;
//             }
//         }
//
//       residual += coe * lmbd[kk];
//     }
//
//     for (int i = 1; i < win_size; i++)
//       for (int j = 0; j < i; j++) Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
//   }
//
//   void LeftEvaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
//                     double &residual) const {
//     Hess.setZero();
//     JacT.setZero();
//     residual = 0;
//     // vector<PointCluster> sig_tran(win_size);
//     int l = 0;
//     Eigen::Matrix<double, 3, 4> Sp;
//     Sp.setZero();
//     Sp.block<3, 3>(0, 0).setIdentity();
//     Eigen::Matrix4d F;
//     F.setZero();
//     F(3, 3) = 1;
//
//     PLM(4) T(win_size);
//     for (int i = 0; i < win_size; i++) T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;
//
//     vector<PLM(4) *> Cs;
//     for (int a = 0; a < plvec_voxels.size(); a++) {
//       const vector<PointCluster> &sig_orig = *plvec_voxels[a];
//       PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
//       for (int i = 0; i < win_size; i++)
//         Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
//       Cs.push_back(Co);
//     }
//
//     // double t0 = ros::Time::now().toSec();
//
//     for (int a = head; a < end; a++) {
//       // const vector<PointCluster> &sig_orig = *plvec_voxels[a];
//       double coe = coeffs[a];
//
//       // PLM(4) Co(win_size, Eigen::Matrix4d::Zero());
//       Eigen::Matrix4d C;
//       C.setZero();
//       // for(int i=0; i<win_size; i++)
//       // if(sig_orig[i].N != 0)
//       // {
//       //   Co[i] << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
//       //   C += T[i] * Co[i] * T[i].transpose();
//       // }
//
//       PLM(4) &Co = *Cs[a];
//       for (int i = 0; i < win_size; i++)
//         if ((int)Co[i](3, 3) > 0) C += T[i] * Co[i] * T[i].transpose();
//
//       double NN = C(3, 3);
//       C = C / NN;
//       // Eigen::Vector4d CF = C.block<4, 1>(0, 3);
//       // cout << CF << endl << endl;
//       // cout << C*F << endl;
//       // exit(0);
//
//       Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);
//
//       Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
//       Eigen::Vector3d lmbd = saes.eigenvalues();
//       Eigen::Matrix3d Uev = saes.eigenvectors();
//
//       residual += coe * lmbd[l];
//
//       Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
//       Eigen::Matrix<double, 4, 6> U[3];
//
//       PLV(-1) g_kl(3);
//       for (int k = 0; k < 3; k++) {
//         g_kl[k].resize(6 * win_size);
//         g_kl[k].setZero();
//         U[k].setZero();
//         U[k].block<3, 3>(0, 0) = hat(u[k]);
//         U[k].block<1, 3>(3, 3) = u[k];
//       }
//
//       for (int j = 0; j < win_size; j++)
//         for (int k = 0; k < 3; k++)
//           if (Co[j](3, 3) > 0.1) {
//             Eigen::Matrix<double, 3, 4> SpTC = Sp * (T[j] - C * F) * Co[j] * T[j].transpose();
//             Eigen::Matrix<double, 1, 6> g1, g2;
//             g1 = u[l].transpose() * SpTC * U[k];
//             g2 = u[k].transpose() * SpTC * U[l];
//
//             g_kl[k].block<6, 1>(6 * j, 0) = (g1 + g2).transpose() / NN;
//           }
//
//       JacT += coe * g_kl[l];
//
//       for (int i = 0; i < win_size; i++)
//         if (Co[i](3, 3) > 0.1) {
//           for (int j = 0; j < win_size; j++)
//             if (Co[j](3, 3) > 0.1) {
//               Eigen::Matrix4d Dij = Co[i] * F * Co[j];
//               Eigen::Matrix<double, 6, 6> Hs = -2.0 / NN / NN * U[l].transpose() * T[i] * Dij * T[j].transpose() *
//               U[l];
//
//               if (i == j) {
//                 Hs += 2 / NN * U[l].transpose() * T[j] * Co[j] * T[j].transpose() * U[l];
//                 Eigen::Vector3d SpTC = Sp * T[j] * Co[j] * (T[j] - C * F).transpose() * Sp.transpose() * u[l];
//                 Eigen::Matrix3d h1 = hat(SpTC);
//                 Eigen::Matrix3d h2 = hat(u[l]);
//
//                 Hs.block<3, 3>(0, 0) += (h1 * h2 + h2 * h1) / NN;
//               }
//
//               Hess.block<6, 6>(6 * i, 6 * j) += coe * Hs;
//             }
//         }
//
//       for (int k = 0; k < 3; k++)
//         if (k != l) Hess += coe * 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k] * g_kl[k].transpose();
//     }
//
//     // double t1 = ros::Time::now().toSec();
//     // printf("t1: %lf\n", t1 - t0);
//
//     // PLM(6) LL(win_size);
//     // Eigen::Matrix3d zero33; zero33.setZero();
//     // for(int i=0; i<win_size; i++)
//     //   LL[i] << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;
//
//     // for(int i=0; i<win_size; i++)
//     // {
//     //   JacT.block<6, 1>(6*i, 0) = LL[i].transpose() * JacT.block<6, 1>(6*i, 0);
//     //   for(int j=0; j<win_size; j++)
//     //   {
//     //     Hess.block<6, 6>(6*i, 6*j) = LL[i].transpose() * Hess.block<6, 6>(6*i, 6*j) * LL[j];
//     //   }
//     // }
//
//     // Eigen::Matrix3d zero33; zero33.setZero();
//     // Eigen::MatrixXd LL(6*win_size, 6*win_size); LL.setZero();
//     // for(int i=0; i<win_size; i++)
//     // {
//     //   LL.block<6, 6>(6*i, 6*i) << xs[i].R, zero33, hat(xs[i].p) * xs[i].R, xs[i].R;
//     // }
//     // JacT = LL.transpose() * JacT;
//     // Hess = LL.transpose() * Hess * LL;
//   }
//
//   void LeftEvaluateAcc2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
//                         double &residual) const {
//     // 海森矩阵
//     Hess.setZero();
//     // 雅可比矩阵转置
//     JacT.setZero();
//     residual = 0;
//     int l = 0;
//     // 变换矩阵
//     PLM(4) T(win_size);
//     for (int i = 0; i < win_size; i++) {
//       // 为变换矩阵赋值
//       T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;
//     }
//
//     // 将点云数据以增广矩阵的形式存储
//     vector<PLM(4) *> Cs;
//     for (int a = 0; a < plvec_voxels.size(); a++) {
//       // 点簇表示一个栅格中的点
//       const vector<PointCluster> &sig_orig = *plvec_voxels[a];
//       // 将voexl中的点按照所在的不同点云按顺序存放到Co中
//       PLM(4) *Co = new PLM(4)(win_size, Eigen::Matrix4d::Zero());
//       for (int i = 0; i < win_size; i++)
//         Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
//       // 将所有voxel的Co存放到Cs中
//       Cs.push_back(Co);
//     }
//     // 分块优化
//     for (int a = head; a < end; a++) {
//       // 用于计算残差的系数，目前设置为voxel中的点数
//       double coe = coeffs[a];
//       Eigen::Matrix4d C;
//       C.setZero();
//
//       vector<int> Ns(win_size);
//       // 获取第a个voxel中的点簇集
//       PLM(4) &Co = *Cs[a];
//       PLM(4) TC(win_size), TCT(win_size);
//       for (int j = 0; j < win_size; j++) {
//         if ((int)Co[j](3, 3) > 0) {
//           // 如果当前voxel中的点簇存在
//           TC[j] = T[j] * Co[j];
//           TCT[j] = TC[j] * T[j].transpose();
//           // 将点簇变换到世界坐标系下，并按照voxel叠加
//           C += TCT[j];
//           // 获取当前点簇的点数
//           Ns[j] = Co[j](3, 3);
//         }
//       }
//       // 获取当前voxel的点数
//       double NN = C(3, 3);
//       // 取平均
//       C = C / NN;
//       // 获取voxel的中心点
//       Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);
//       // 平面拟合
//       Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(C.block<3, 3>(0, 0) - v_bar * v_bar.transpose());
//       // 特征值
//       Eigen::Vector3d lmbd = saes.eigenvalues();
//       // 特征向量
//       Eigen::Matrix3d Uev = saes.eigenvectors();
//       // 残差计算
//       residual += coe * lmbd[l];
//       // 获取特征向量
//       Eigen::Vector3d u[3] = {Uev.col(0), Uev.col(1), Uev.col(2)};
//
//       Eigen::Matrix<double, 6, 4> U[3];
//       PLV(6) g_kl[3];
//
//       for (int k = 0; k < 3; k++) {
//         g_kl[k].resize(win_size);
//         U[k].setZero();
//         U[k].block<3, 3>(0, 0) = hat(-u[k]);
//         U[k].block<3, 1>(3, 3) = u[k];
//       }
//
//       PLV(6) UlTCF(win_size, Eigen::Matrix<double, 6, 1>::Zero());
//
//       Eigen::VectorXd JacT_iter(6 * win_size);
//       for (int i = 0; i < win_size; i++)
//         if (Ns[i] != 0) {
//           Eigen::Matrix<double, 3, 4> temp = T[i].block<3, 4>(0, 0);
//           temp.block<3, 1>(0, 3) -= v_bar;
//           Eigen::Matrix<double, 4, 3> TC_TCFSp = TC[i] * temp.transpose();
//           for (int k = 0; k < 3; k++) {
//             Eigen::Matrix<double, 6, 1> g1, g2;
//             g1 = U[k] * TC_TCFSp * u[l];
//             g2 = U[l] * TC_TCFSp * u[k];
//
//             g_kl[k][i] = (g1 + g2) / NN;
//           }
//
//           UlTCF[i] = (U[l] * TC[i]).block<6, 1>(0, 3);
//           // 计算雅可比
//           JacT.block<6, 1>(6 * i, 0) += coe * g_kl[l][i];
//
//           // Eigen::Matrix<double, 6, 6> Hb(2.0/NN * U[l] * TCT[i] * U[l].transpose());
//           // ha1
//           Eigen::Matrix<double, 6, 6> Ha(-2.0 / NN / NN * UlTCF[i] * UlTCF[i].transpose());
//           // ha2
//           Eigen::Matrix3d Ell = 1.0 / NN * hat(TC_TCFSp.block<3, 3>(0, 0) * u[l]) * hat(u[l]);
//           Ha.block<3, 3>(0, 0) += Ell + Ell.transpose();
//           // ha3
//           for (int k = 0; k < 3; k++) {
//             if (k != l) {
//               Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][i].transpose();
//             }
//           }
//           // 填充海森矩阵的对角上的第一个参数
//           Hess.block<6, 6>(6 * i, 6 * i) += coe * Ha;
//         }
//
//       for (int i = 0; i < win_size; i++)
//         if (Ns[i] != 0) {
//           Eigen::Matrix<double, 6, 6> Hb = U[l] * TCT[i] * U[l].transpose();
//           // 填充海森矩阵的对角上的第二个参数
//           Hess.block<6, 6>(6 * i, 6 * i) += 2.0 / NN * coe * Hb;
//         }
//       // 填充海森非对角上的值
//       for (int i = 0; i < win_size - 1; i++)
//         if (Ns[i] != 0) {
//           for (int j = i + 1; j < win_size; j++)
//             if (Ns[j] != 0) {
//               Eigen::Matrix<double, 6, 6> Ha = -2.0 / NN / NN * UlTCF[i] * UlTCF[j].transpose();
//
//               for (int k = 0; k < 3; k++)
//                 if (k != l) Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][j].transpose();
//               Hess.block<6, 6>(6 * i, 6 * j) += coe * Ha;
//             }
//         }
//     }
//     // 填充海森矩阵的对称位置
//     for (int i = 1; i < win_size; i++)
//       for (int j = 0; j < i; j++) {
//         Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
//       }
//   }
//
//   void EvaluateOnlyResidual(const vector<IMUST> &xs, double &residual) const {
//     residual = 0;
//     vector<PointCluster> sig_tran(win_size);
//     int kk = 0;  // The kk-th lambda value
//
//     int gps_size = plvec_voxels.size();
//
//     vector<double> ress(gps_size);
//
//     for (int a = 0; a < gps_size; a++) {
//       const vector<PointCluster> &sig_orig = *plvec_voxels[a];
//       PointCluster sig = *sig_vecs[a];
//
//       for (int i = 0; i < win_size; i++) {
//         sig_tran[i].transform(sig_orig[i], xs[i]);
//         sig += sig_tran[i];
//       }
//
//       Eigen::Vector3d vBar = sig.v / sig.N;
//       Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
//
//       Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
//       Eigen::Vector3d lmbd = saes.eigenvalues();
//
//       residual += coeffs[a] * lmbd[kk];
//
//       ress[a] = lmbd[kk];
//     }
//
//     // vector<double> ress_tem = ress;
//     // sort(ress_tem.begin(), ress_tem.end());
//     // double bound = 0.8;
//     // bound = ress_tem[gps_size * bound];
//     // coeffs = coeffs_back;
//
//     // for(int a=0; a<gps_size; a++)
//     //   if(ress[a] > bound)
//     //     coeffs[a] = 0;
//   }
//
//   ~VoxHess() {
//     int vsize = sig_vecs.size();
//     // for(int i=0; i<vsize; i++)
//     // {
//     //   delete sig_vecs[i], sig_vecs[i] = nullptr;
//     //   delete plvec_voxels[i], plvec_voxels[i] = nullptr;
//     // }
//   }
// };
//
// class OctoTreeNode {
//  public:
//   int octo_state{};  // 0(unknown), 1(mid node), 2(plane)
//   int layer{};
//   vector<PLV(3)> vec_orig{}, vec_tran{};        // 用于切割
//   vector<PointCluster> sig_orig{}, sig_tran{};  // 用于拟合平面
//   PointCluster fix_point{};
//   PLV(3) vec_fix{};
//
//   OctoTreeNode *leaves[8];  // 8个子节点
//   float voxel_center[3]{};
//   float quater_length{};
//
//   Eigen::Vector3d center{}, direct{}, value_vector{};  // temporal
//   double decision{}, ref{};
//
//   OctoTreeNode() {
//     octo_state = 0;
//     vec_orig.resize(win_size);
//     vec_tran.resize(win_size);
//     sig_orig.resize(win_size);
//     sig_tran.resize(win_size);
//     for (int i = 0; i < 8; i++) leaves[i] = nullptr;
//     ref = 255.0 * rand() / (RAND_MAX + 1.0f);
//     layer = 0;
//   }
//   /**
//    * @brief 拟合平面
//    * @param win_count
//    * @return
//    */
//   bool JudgeEigen(const int win_count) {
//     // 获取该voxel下的所有点
//     PointCluster cov_mat = fix_point;
//     for (int i = 0; i < win_count; i++) {
//       cov_mat += sig_tran[i];
//     }
//
//     const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat.cov());
//     value_vector = saes.eigenvalues();
//     center = cov_mat.v / cov_mat.N;
//     direct = saes.eigenvectors().col(0);
//
//     decision = saes.eigenvalues()[0] / saes.eigenvalues()[1];
//     return (decision < eigen_value_array[layer]) && saes.eigenvalues()[2] / saes.eigenvalues()[1] < 10.;
//   }
//
//   void CutFunc(const int ci) {
//     PLV(3) &pvec_orig = vec_orig[ci];
//     PLV(3) &pvec_tran = vec_tran[ci];
//
//     const uint a_size = pvec_tran.size();
//     for (uint j = 0; j < a_size; j++) {
//       // 计算当前点在当前节点下的哪个子节点下
//       int xyz[3] = {0, 0, 0};
//       for (uint k = 0; k < 3; k++) {
//         if (pvec_tran[j][k] > voxel_center[k]) {
//           xyz[k] = 1;
//         }
//       }
//       const int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
//       // 如果子节点不存在，则新建
//       if (leaves[leafnum] == nullptr) {
//         leaves[leafnum] = new OctoTreeNode();
//         leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
//         leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
//         leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
//         leaves[leafnum]->quater_length = quater_length / 2;
//         leaves[leafnum]->layer = layer + 1;
//       }
//       // 将对应点存放到子节点中
//       leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
//       leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);
//       if (leaves[leafnum]->octo_state != 1) {
//         leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
//         leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
//       }
//     }
//     // 清空主节点下的点云
//     PLV(3)().swap(pvec_orig);
//     PLV(3)().swap(pvec_tran);
//   }
//
//   void Recut(const int win_count) {
//     // 如果不是mid_node
//     if (octo_state != 2) {
//       // 获取用于拟合的原始点数
//       int point_size = fix_point.N;
//       for (int i = 0; i < win_count; i++) {
//         // 获取新增点数
//         point_size += sig_orig[i].N;
//       }
//       // 判断点数是否足够
//       if (point_size <= min_ps) {
//         return;
//       }
//       // 拟合平面
//       if (JudgeEigen(win_count)) {
//         // 将当前voxel状态设置为平面
//         if (octo_state == 0 && point_size > layer_size[layer]) {
//           octo_state = 2;
//         }
//         // 计算新增的点数
//         point_size -= fix_point.N;
//         return;
//       }
//       // 如果切割了两次，则认为是平面
//       if (layer == layer_limit) {
//         octo_state = 2;
//         return;
//       }
//
//       // 将当前voxel设置为mid_node
//       octo_state = 1;
//       // 因为确定是mid_node，不需要进行平面拟合，所以清空
//       // vector<PointCluster>().swap(sig_orig);
//       // vector<PointCluster>().swap(sig_tran);
//       // 切割当前voxel
//       // for (int i = 0; i < win_count; i++) {
//       //   CutFunc(i);
//       // }
//     } /*else {
//       // 如果是中间节点，切割最新的一帧
//       CutFunc(win_count - 1);
//     }*/
//
//     // for (int i = 0; i < 8; i++)
//     //   if (leaves[i] != nullptr) {
//     //     leaves[i]->Recut(win_count);
//     //   }
//   }
//
//   void ToMargi(int mg_size, vector<IMUST> &x_poses, int win_count) {
//     // if (octo_state == 2) {
//     // 更新点
//     if (!x_poses.empty())
//       for (int i = 0; i < win_count; i++) {
//         sig_tran[i].transform(sig_orig[i], x_poses[i]);
//         plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
//       }
//
//     if (fix_point.N < 50)
//       for (int i = 0; i < mg_size; i++) {
//         fix_point += sig_tran[i];
//         vec_fix.insert(vec_fix.end(), vec_tran[i].begin(), vec_tran[i].end());
//       }
//
//     for (int i = mg_size; i < win_count; i++) {
//       sig_orig[i - mg_size] = sig_orig[i];
//       sig_tran[i - mg_size] = sig_tran[i];
//       vec_orig[i - mg_size].swap(vec_orig[i]);
//       vec_tran[i - mg_size].swap(vec_tran[i]);
//     }
//
//     for (int i = win_count - mg_size; i < win_count; i++) {
//       sig_orig[i].clear();
//       sig_tran[i].clear();
//       vec_orig[i].clear();
//       vec_tran[i].clear();
//     }
//
//     // } /*else
//     //   for (int i = 0; i < 8; i++)
//     //     if (leaves[i] != nullptr) leaves[i]->ToMargi(mg_size, x_poses, win_count);*/
//   }
//
//   ~OctoTreeNode() {
//     for (int i = 0; i < 8; i++)
//       if (leaves[i] != nullptr) delete leaves[i];
//   }
//
//   void TrasDisplay(pcl::PointCloud<PointType> &pl_feat, const int win_count) {
//     if (octo_state != 1) {
//       PointType ap;
//       ap.intensity = ref;
//
//       int tsize = 0;
//       for (int i = 0; i < win_count; i++) tsize += vec_tran[i].size();
//       if (tsize < 100) return;
//
//       for (int i = 0; i < win_count; i++)
//         for (Eigen::Vector3d pvec : vec_tran[i]) {
//           ap.x = pvec.x();
//           ap.y = pvec.y();
//           ap.z = pvec.z();
//           pl_feat.push_back(ap);
//         }
//       // for (auto pvec : vec_fix) {
//       //   ap.x = pvec.x();
//       //   ap.y = pvec.y();
//       //   ap.z = pvec.z();
//       //   pl_feat.push_back(ap);
//       // }
//     } else {
//       // if(layer != layer_limit)
//       // {
//       //   PointType ap;
//       //   ap.x = voxel_center[0];
//       //   ap.y = voxel_center[1];
//       //   ap.z = voxel_center[2];
//       //   pl_cent.push_back(ap);
//       // }
//
//       for (int i = 0; i < 8; i++)
//         if (leaves[i] != nullptr) leaves[i]->TrasDisplay(pl_feat, win_count);
//     }
//   }
//
//   /**
//    * @brief 获取平面数据
//    * @param vox_opt
//    * @param win_count
//    */
//   void TrasOpt(VoxHess &vox_opt, const int win_count) const {
//     if (octo_state == 2) {
//       int points_size = 0;
//       for (int i = 0; i < win_count; i++) {
//         points_size += sig_orig[i].N;
//       }
//       if (points_size < min_ps) {
//         return;
//       }
//       vox_opt.PushVoxel(&sig_orig, &fix_point, decision, layer);
//     } else {
//       for (int i = 0; i < 8; i++)
//         if (leaves[i] != nullptr) {
//           leaves[i]->TrasOpt(vox_opt, win_count);
//         }
//     }
//   }
// };
//
// class OctoTreeRoot : public OctoTreeNode {
//  public:
//   bool is2opt;
//   vector<int> each_num;
//
//   // 在基类的基础上增加
//   OctoTreeRoot() {
//     is2opt = true;
//     each_num.resize(win_size);
//     for (int i = 0; i < win_size; i++) each_num[i] = 0;
//   }
//   void AddPoint(const int fnum, const Eigen::Vector3d &pvec_orig, const Eigen::Vector3d &pvec_tran) {
//     // 动态检查并调整 vector 大小
//     if (fnum >= vec_orig.size()) {
//       vec_orig.resize(fnum + 1);
//       vec_tran.resize(fnum + 1);
//       sig_orig.resize(fnum + 1);
//       sig_tran.resize(fnum + 1);
//     }
//
//     vec_orig[fnum].push_back(pvec_orig);
//     vec_tran[fnum].push_back(pvec_tran);
//     // if (octo_state != 1) {
//       try {
//         sig_orig[fnum].push(pvec_orig);
//         sig_tran[fnum].push(pvec_tran);
//       } catch (const std::exception &e) {
//         std::cerr << "Exception caught: " << e.what() << std::endl;
//       } catch (...) {
//         std::cerr << "Unknown exception caught while pushing points." << std::endl;
//       }
//     // }
//     each_num[fnum]++;
//   }
//   // 边缘化
//   void Marginalize(const int mg_size, vector<IMUST> &x_poses, const int win_count) {
//     ToMargi(mg_size, x_poses, win_count);
//
//     int left_size = 0;
//     for (int i = mg_size; i < win_count; i++) {
//       each_num[i - mg_size] = each_num[i];
//       left_size += each_num[i - mg_size];
//     }
//
//     if (left_size == 0) is2opt = false;
//
//     for (int i = win_count - mg_size; i < win_count; i++) each_num[i] = 0;
//   }
// };
//
// using OctoTreeRootPtr = std::shared_ptr<OctoTreeRoot>;
// using OctoTreeMap = unordered_map<VOXEL_LOC, OctoTreeRootPtr>;
//
// class BALM2 {
//  public:
//   BALM2() {}
//
//   static double DivideThreadRight(vector<IMUST> &x_stats, VoxHess &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd
//   &Hess,
//                                   Eigen::VectorXd &JacT) {
//     constexpr int thd_num = 4;
//     double residual = 0;
//     Hess.setZero();
//     JacT.setZero();
//     PLM(-1) hessians(thd_num);
//     PLV(-1) jacobins(thd_num);
//
//     for (int i = 0; i < thd_num; i++) {
//       hessians[i].resize(6 * win_size, 6 * win_size);
//       jacobins[i].resize(6 * win_size);
//     }
//
//     int tthd_num = thd_num;
//     vector<double> resis(tthd_num, 0);
//     int g_size = voxhess.plvec_voxels.size();
//     if (g_size < tthd_num) tthd_num = 1;
//
//     vector<thread *> mthreads(tthd_num);
//     double part = 1.0 * g_size / tthd_num;
//     for (int i = 0; i < tthd_num; i++)
//       mthreads[i] = new thread(&VoxHess::AccEvaluate2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]),
//                                ref(jacobins[i]), ref(resis[i]));
//
//     for (int i = 0; i < tthd_num; i++) {
//       mthreads[i]->join();
//       Hess += hessians[i];
//       JacT += jacobins[i];
//       residual += resis[i];
//       delete mthreads[i];
//     }
//
//     return residual;
//   }
//
//   static double DivideThreadLeft(vector<IMUST> &x_stats, VoxHess &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd
//   &Hess,
//                                  Eigen::VectorXd &JacT) {
//     constexpr int thd_num = 4;
//     double residual = 0;
//     Hess.setZero();
//     JacT.setZero();
//     PLM(-1) hessians(thd_num);
//     PLV(-1) jacobins(thd_num);
//
//     for (int i = 0; i < thd_num; i++) {
//       hessians[i].resize(6 * win_size, 6 * win_size);
//       jacobins[i].resize(6 * win_size);
//     }
//
//     int tthd_num = thd_num;
//     vector<double> resis(tthd_num, 0);
//     const int g_size = voxhess.plvec_voxels.size();
//     if (g_size < tthd_num) tthd_num = 1;
//
//     vector<thread *> mthreads(tthd_num);
//     double part = 1.0 * g_size / tthd_num;
//     for (int i = 0; i < tthd_num; i++)
//       mthreads[i] = new thread(&VoxHess::LeftEvaluateAcc2, &voxhess, x_stats, part * i, part * (i + 1),
//                                ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));
//
//     for (int i = 0; i < tthd_num; i++) {
//       mthreads[i]->join();
//       Hess += hessians[i];
//       JacT += jacobins[i];
//       residual += resis[i];
//       delete mthreads[i];
//     }
//
//     return residual;
//   }
//
//   static double OnlyResidual(const vector<IMUST> &x_stats, const VoxHess &voxhess, vector<IMUST> &x_ab) {
//     double residual1 = 0, residual2 = 0;
//
//     voxhess.EvaluateOnlyResidual(x_stats, residual2);
//     return (residual1 + residual2);
//   }
//
//   static void DampingIter(vector<IMUST> &x_stats, VoxHess &voxhess) {
//     // 记录每一帧的平面数
//     vector<int> planes(x_stats.size(), 0);
//     for (int i = 0; i < voxhess.plvec_voxels.size(); i++) {
//       for (int j = 0; j < voxhess.plvec_voxels[i]->size(); j++) {
//         if (voxhess.plvec_voxels[i]->at(j).N != 0) {
//           planes[j]++;
//         }
//       }
//     }
//     // 检查平面数释放满足条件
//     sort(planes.begin(), planes.end());
//     if (planes[0] < 20) {
//       printf("Initial error too large.\n");
//       printf("Please loose plane determination criteria for more planes.\n");
//       printf("The optimization is terminated.\n");
//       exit(0);
//     }
//     // LM参数设置
//     double u = 0.01, v = 2;
//     Eigen::MatrixXd D(6 * win_size, 6 * win_size), Hess(6 * win_size, 6 * win_size);
//     Eigen::VectorXd JacT(6 * win_size), dxi(6 * win_size);
//
//     D.setIdentity();
//     double residual1 = 0.,  // 优化前误差
//         residual2 = 0.,     // 优化后误差
//         q = 0.;
//     bool is_calc_hess = true;
//     vector<IMUST> x_stats_temp = x_stats;
//
//     vector<IMUST> x_ab(win_size);
//     x_ab[0] = x_stats[0];
//     for (int i = 1; i < win_size; i++) {
//       x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
//       x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
//     }
//
//     for (int i = 0; i < 10; i++) {
//       if (is_calc_hess) {
//         // 多线程加速计算hess
//         // residual1 = DivideThreadRight(x_stats, voxhess, x_ab, Hess, JacT);
//         residual1 = DivideThreadLeft(x_stats, voxhess, x_ab, Hess, JacT);
//       }
//
//       D.diagonal() = Hess.diagonal();
//       dxi = (Hess + u * D).ldlt().solve(-JacT);
//
//       // 更新位姿
//       for (int j = 0; j < win_size; j++) {
//         // right update
//         // x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL*j, 0));
//         // x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL*j+3, 0);
//
//         // left update
//         Eigen::Matrix3d dR = Exp(dxi.block<3, 1>(DVEL * j, 0));
//         x_stats_temp[j].R = dR * x_stats[j].R;
//         x_stats_temp[j].p = dR * x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
//       }
//
//       const double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
//
//       // 计算优化后残差
//       residual2 = OnlyResidual(x_stats_temp, voxhess, x_ab);
//       // 计算残差变换量
//       q = (residual1 - residual2);
//       printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.3lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);
//
//       if (q > 0) {
//         // 如果误差缩小，则更新
//         x_stats = x_stats_temp;
//         // 更新LM参数
//         q = q / q1;
//         v = 2;
//         q = 1 - pow(2 * q - 1, 3);
//         u *= (q < one_three ? one_three : q);
//         is_calc_hess = true;
//       } else {
//         // break;
//         // 如果误差增大
//         // 更新LM参数
//         u = u * v;
//         v = 2 * v;
//         // 不计算hess
//         is_calc_hess = false;
//       }
//
//       // if(IterStop(dxi2, 1e-4))
//       if (IterStop(dxi, 1e-6)) break;
//
//       if (fabs(residual1 - residual2) / residual1 < 1e-6) {
//         break;
//       }
//     }
//
//     IMUST es0 = x_stats[0];
//     for (uint i = 0; i < x_stats.size(); i++) {
//       x_stats[i].p = es0.R.transpose() * (x_stats[i].p - es0.p);
//       x_stats[i].R = es0.R.transpose() * x_stats[i].R;
//     }
//   }
//
//   static bool IterStop(Eigen::VectorXd &dx, const double thre = 1e-7, int win_size = 0) {
//     // int win_size = dx.rows() / 6;
//     if (win_size == 0) win_size = dx.rows() / 6;
//
//     double angErr = 0, tranErr = 0;
//     for (int i = 0; i < win_size; i++) {
//       angErr += dx.block<3, 1>(6 * i, 0).norm();
//       tranErr += dx.block<3, 1>(6 * i + 3, 0).norm();
//     }
//
//     angErr /= win_size;
//     tranErr /= win_size;
//     return (angErr < thre) && (tranErr < thre);
//   }
//
//   /**
//    * @brief 将当前帧点云加入地图
//    * @param feat_map 初始的八叉树地图
//    * @param pl_feat 雷达系下的点云
//    * @param x_key 对应的位姿
//    * @param fnum 对应的id
//    */
//   template <typename T>
//   static void CutVoxel(OctoTreeMap &feat_map, pcl::PointCloud<T> &pl_feat, const IMUST &x_key, const int fnum) {
//     float loc_xyz[3];
//     // std::cout << "---------------" << pl_feat.size() << std::endl;
//     for (const auto &p_c : pl_feat) {
//       // 获取雷达坐标系下点
//       Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
//       // 将点变换到世界坐标系下
//       Eigen::Vector3d pvec_tran = x_key.R * pvec_orig + x_key.p;
//
//       // 计算每个点对应的voxel位置
//       for (int j = 0; j < 3; j++) {
//         loc_xyz[j] = pvec_tran[j] / voxel_size;
//         if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
//       }
//
//       // 存放当前点的voxel位置
//       VOXEL_LOC position(static_cast<int64_t>(loc_xyz[0]), static_cast<int64_t>(loc_xyz[1]),
//                          static_cast<int64_t>(loc_xyz[2]));
//       // 通过位置找到对应栅格
//       if (auto iter = feat_map.find(position); iter != feat_map.end()) {
//         const auto &cur_feat = iter->second;
//         // 如果是已有栅格
//         cur_feat->AddPoint(fnum, pvec_orig, pvec_tran);
//         cur_feat->is2opt = true;
//       } else {
//         // 构建新的八叉树栅格头
//         const auto ot = std::make_shared<OctoTreeRoot>();
//         ot->AddPoint(fnum, pvec_orig, pvec_tran);
//         ot->is2opt = true;
//         // ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
//         // ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
//         // ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
//         // ot->quater_length = voxel_size / 4.0;
//         // ot->layer = 0;
//         // 存放新栅格
//         feat_map[position] = ot;
//       }
//     }
//     // std::cout<<"---------11111111------"<<std::endl;
//   }
// };
// }
//
// #endif