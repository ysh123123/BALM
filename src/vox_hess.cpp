#include "vox_hess.hpp"
VoxHess::VoxHess(const int _win_size) : win_size_(_win_size) {}
VoxHess::~VoxHess() { vector<const vector<VOX_FACTOR> *>().swap(plvec_voxels); }
void VoxHess::PushVoxel(const vector<VOX_FACTOR> *vec_orig, const VOX_FACTOR *fix) {
  int process_size = 0;
  for (int i = 0; i < win_size_; i++)
    if ((*vec_orig)[i].N != 0) process_size++;

  if (process_size < 2) return;  // 改

  double coe = 0;
  for (int j = 0; j < win_size_; j++) coe += (*vec_orig)[j].N;

  plvec_voxels.push_back(vec_orig);
  sig_vecs_.push_back(fix);
  coeffs_.push_back(coe);
}
void VoxHess::RightAccEvaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                               double &residual) const {
  // 1.初始化
  Hess.setZero();
  JacT.setZero();
  residual = 0;
  // 声明一个 PointCluster 类型的 sig_tran 数组，用来存储每个窗口（win_size）对应的点簇变换结果。
  vector<VOX_FACTOR> sig_tran(win_size_);
  const int kk = 0;

  PLV(3) viRiTuk(win_size_);
  PLM(3) viRiTukukT(win_size_);

  vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size_);
  Eigen::Matrix3d umumT;

  for (int a = head; a < end; a++) {
    const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
    double coe = coeffs_[a];

    VOX_FACTOR sig = *sig_vecs_[a];
    for (int i = 0; i < win_size_; i++)
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

    for (int i = 0; i < win_size_; i++)
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
                                2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * hat(jjt.block<3, 1>(0, 0));
        Hb.block<3, 3>(0, 3) += HRt;
        Hb.block<3, 3>(3, 0) += HRt.transpose();
        Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

        Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
      }

    for (int i = 0; i < win_size_ - 1; i++)
      // for(int i=1; i<win_size-1; i++)
      if (sig_orig[i].N != 0) {
        double ni = sig_orig[i].N;
        for (int j = i + 1; j < win_size_; j++)
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

  for (int i = 1; i < win_size_; i++)
    for (int j = 0; j < i; j++) Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
}
void VoxHess::LeftAccEvaluate(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT,
                              double &residual) const {
  Hess.setZero();
  JacT.setZero();
  residual = 0;
  int l = 0;
  PLM(4) T(win_size_);
  for (int i = 0; i < win_size_; i++) T[i] << xs[i].R, xs[i].p, 0, 0, 0, 1;

  vector<PLM(4) *> Cs;
  for (int a = 0; a < plvec_voxels.size(); a++) {
    const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
    PLM(4) *Co = new PLM(4)(win_size_, Eigen::Matrix4d::Zero());
    for (int i = 0; i < win_size_; i++)
      Co->at(i) << sig_orig[i].P, sig_orig[i].v, sig_orig[i].v.transpose(), sig_orig[i].N;
    Cs.push_back(Co);
  }

  for (int a = head; a < end; a++) {
    double coe = coeffs_[a];
    Eigen::Matrix4d C;
    C.setZero();

    vector<int> Ns(win_size_);

    PLM(4) &Co = *Cs[a];
    PLM(4) TC(win_size_), TCT(win_size_);
    for (int j = 0; j < win_size_; j++)
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
      g_kl[k].resize(win_size_);
      U[k].setZero();
      U[k].block<3, 3>(0, 0) = hat(-u[k]);
      U[k].block<3, 1>(3, 3) = u[k];
    }

    PLV(6) UlTCF(win_size_, Eigen::Matrix<double, 6, 1>::Zero());

    Eigen::VectorXd JacT_iter(6 * win_size_);
    for (int i = 0; i < win_size_; i++)
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

    for (int i = 0; i < win_size_; i++)
      if (Ns[i] != 0) {
        Eigen::Matrix<double, 6, 6> Hb = U[l] * TCT[i] * U[l].transpose();
        Hess.block<6, 6>(6 * i, 6 * i) += 2.0 / NN * coe * Hb;
      }

    for (int i = 0; i < win_size_ - 1; i++)
      if (Ns[i] != 0) {
        for (int j = i + 1; j < win_size_; j++)
          if (Ns[j] != 0) {
            Eigen::Matrix<double, 6, 6> Ha = -2.0 / NN / NN * UlTCF[i] * UlTCF[j].transpose();

            for (int k = 0; k < 3; k++)
              if (k != l) Ha += 2.0 / (lmbd[l] - lmbd[k]) * g_kl[k][i] * g_kl[k][j].transpose();

            Hess.block<6, 6>(6 * i, 6 * j) += coe * Ha;
          }
      }
  }

  for (int i = 1; i < win_size_; i++)
    for (int j = 0; j < i; j++) Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
}
void VoxHess::EvaluateOnlyResidual(const vector<IMUST> &xs, double &residual) const {
  residual = 0;
  vector<VOX_FACTOR> sig_tran(win_size_);
  int kk = 0;  // The kk-th lambda value

  int gps_size = plvec_voxels.size();

  vector<double> ress(gps_size);

  for (int a = 0; a < gps_size; a++) {
    const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
    VOX_FACTOR sig = *sig_vecs_[a];

    for (int i = 0; i < win_size_; i++) {
      sig_tran[i].transform(sig_orig[i], xs[i]);
      sig += sig_tran[i];
    }

    Eigen::Vector3d vBar = sig.v / sig.N;
    Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
    Eigen::Vector3d lmbd = saes.eigenvalues();

    residual += coeffs_[a] * lmbd[kk];

    ress[a] = lmbd[kk];
  }

  // vector<double> ress_tem = ress;
  // sort(ress_tem.begin(), ress_tem.end());
  // double bound = 0.8;
  // bound = ress_tem[gps_size * bound];
  // coeffs_ = coeffs_back;

  // for(int a=0; a<gps_size; a++)
  //   if(ress[a] > bound)
  //     coeffs_[a] = 0;
}
std::vector<double> VoxHess::EvaluateResidual(const vector<IMUST> &xs) const {
  /* for outlier removal usage */
  std::vector<double> residuals;
  vector<VOX_FACTOR> sig_tran(win_size_);
  int kk = 0;  // The kk-th lambda value
  int gps_size = plvec_voxels.size();

  for (int a = 0; a < gps_size; a++) {
    const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
    VOX_FACTOR sig;

    for (int i = 0; i < win_size_; i++) {
      sig_tran[i].transform(sig_orig[i], xs[i]);
      sig += sig_tran[i];
    }

    Eigen::Vector3d vBar = sig.v / sig.N;
    Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
    Eigen::Vector3d lmbd = saes.eigenvalues();

    residuals.push_back(lmbd[kk]);
  }

  return residuals;
}
void VoxHess::RemoveResidual(const vector<IMUST> &xs, const double threshold, const double reject_num) {
  vector<VOX_FACTOR> sig_tran(win_size_);
  int kk = 0;  // The kk-th lambda value
  int rej_cnt = 0;
  size_t i = 0;
  for (; i < plvec_voxels.size();) {
    const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[i];
    VOX_FACTOR sig;

    for (int j = 0; j < win_size_; j++) {
      sig_tran[j].transform(sig_orig[j], xs[j]);
      sig += sig_tran[j];
    }

    Eigen::Vector3d vBar = sig.v / sig.N;
    Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
    Eigen::Vector3d lmbd = saes.eigenvalues();

    if (lmbd[kk] >= threshold) {
      plvec_voxels.erase(plvec_voxels.begin() + i);
      rej_cnt++;
      continue;
    }
    i++;
    if (rej_cnt == reject_num) break;
    }
  }