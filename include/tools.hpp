#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#define USE_PATC_API
#ifdef USE_PATC_API
#include <patc/patc_api.h>
#include "mvt_msg_pointcloud.pb.h"
#include "mvt_msg_keyframe.pb.h"
#include "mvt_msg_plot.pb.h"
#else
#include "not_use_patc.h"
#endif
using namespace std;

#define HASH_P 116101
#define MAX_N 10000000019
#define SMALL_EPS 1e-10
#define G_m_s2 9.81
#define DIMU 18
#define DIM 15
#define DNOI 12
#define NMATCH 5
#define DVEL 6
#define WIN_SIZE 10
#define GAP 5
#define FULL_HESS
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define PLM(a) vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>
#define VEC(a) Eigen::Matrix<double, a, 1>

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> vector_quad;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef pcl::PointXYZI PointType;

inline Eigen::Matrix3d I33(Eigen::Matrix3d::Identity());
inline Eigen::Matrix<double, DIMU, DIMU> I_imu(Eigen::Matrix<double, DIMU, DIMU>::Identity());
inline Eigen::Matrix<double, 12, 12> I12(Eigen::Matrix<double, 12, 12>::Identity());
inline int layer_limit = 3;
inline int layer_size[] = {10, 10, 10, 10, 10};
inline float eigen_value_array[5] = {1.0 / 30, 1.0 / 30, 1.0 / 30, 1.0 / 30, 1.0 / 30};
inline int min_ps = 5;
inline double one_three = (1.0 / 3.0);
inline double voxel_size = 1;
inline bool check_degeneracy = true;
inline int pcd_name_fill_num = 0;
inline bool cloud_from_path = false;
class VOXEL_LOC {
 public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const { return (x == other.x && y == other.y && z == other.z); }
};

namespace std {
template <>
struct hash<VOXEL_LOC> {
  size_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    // return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
    long long index_x, index_y, index_z;
    double cub_len = 0.125;
    index_x = int(round(floor((s.x) / cub_len + SMALL_EPS)));
    index_y = int(round(floor((s.y) / cub_len + SMALL_EPS)));
    index_z = int(round(floor((s.z) / cub_len + SMALL_EPS)));
    return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
  }
};
}  // namespace std

inline Eigen::Matrix3d Exp(const Eigen::Vector3d &ang) {
  double ang_norm = ang.norm();
  // if (ang_norm >= 0.0000001)
  if (ang_norm >= 1e-11) {
    Eigen::Vector3d r_axis = ang / ang_norm;
    Eigen::Matrix3d K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return I33 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }

  return I33;
}

inline Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt) {
  double ang_vel_norm = ang_vel.norm();
  if (ang_vel_norm > 0.0000001) {
    Eigen::Vector3d r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix3d K;

    K << SKEW_SYM_MATRX(r_axis);
    double r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return I33 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  }

  return I33;
}

inline Eigen::Vector3d Log(const Eigen::Matrix3d &R) {
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Vector3d K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

inline Eigen::Matrix3d hat(const Eigen::Vector3d &v) {
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

inline Eigen::Matrix3d jr(Eigen::Vector3d vec) {
  double ang = vec.norm();

  if (ang < 1e-9) {
    return I33;
  } else {
    vec /= ang;
    double ra = sin(ang) / ang;
    return ra * I33 + (1 - ra) * vec * vec.transpose() - (1 - cos(ang)) / ang * hat(vec);
  }
}

inline Eigen::Matrix3d jr_inv(const Eigen::Matrix3d &rotR) {
  Eigen::AngleAxisd rot_vec(rotR);
  Eigen::Vector3d axi = rot_vec.axis();
  double ang = rot_vec.angle();

  if (ang < 1e-9) {
    return I33;
  } else {
    double ctt = ang / 2 / tan(ang / 2);
    return ctt * I33 + (1 - ctt) * axi * axi.transpose() + ang / 2 * hat(axi);
  }
}

struct IMUST {
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d g;

  IMUST() { setZero(); }

  IMUST(double _t, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_p, const Eigen::Vector3d &_v,
        const Eigen::Vector3d &_bg, const Eigen::Vector3d &_ba,
        const Eigen::Vector3d &_g = Eigen::Vector3d(0, 0, -G_m_s2))
      : t(_t), R(_R), p(_p), v(_v), bg(_bg), ba(_ba), g(_g) {}

  IMUST &operator+=(const Eigen::Matrix<double, DIMU, 1> &ist) {
    this->R = this->R * Exp(ist.block<3, 1>(0, 0));
    this->p += ist.block<3, 1>(3, 0);
    this->v += ist.block<3, 1>(6, 0);
    this->bg += ist.block<3, 1>(9, 0);
    this->ba += ist.block<3, 1>(12, 0);
    this->g += ist.block<3, 1>(15, 0);
    return *this;
  }

  Eigen::Matrix<double, DIMU, 1> operator-(const IMUST &b) {
    Eigen::Matrix<double, DIMU, 1> a;
    a.block<3, 1>(0, 0) = Log(b.R.transpose() * this->R);
    a.block<3, 1>(3, 0) = this->p - b.p;
    a.block<3, 1>(6, 0) = this->v - b.v;
    a.block<3, 1>(9, 0) = this->bg - b.bg;
    a.block<3, 1>(12, 0) = this->ba - b.ba;
    a.block<3, 1>(15, 0) = this->g - b.g;
    return a;
  }

  IMUST &operator=(const IMUST &b) {
    this->R = b.R;
    this->p = b.p;
    this->v = b.v;
    this->bg = b.bg;
    this->ba = b.ba;
    this->g = b.g;
    this->t = b.t;
    return *this;
  }

  void setZero() {
    t = 0;
    R.setIdentity();
    p.setZero();
    v.setZero();
    bg.setZero();
    ba.setZero();
    g << 0, 0, -G_m_s2;
  }
};
inline void assign_qt(Eigen::Quaterniond &q, Eigen::Vector3d &t, const Eigen::Quaterniond &q_,
                      const Eigen::Vector3d &t_) {
  q.w() = q_.w();
  q.x() = q_.x();
  q.y() = q_.y();
  q.z() = q_.z();
  t(0) = t_(0);
  t(1) = t_(1);
  t(2) = t_(2);
}

inline void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size) {
  // std::cout<<"-----111111--555555555-----"<<std::endl;
  if (voxel_size < 0.001) return;

  unordered_map<VOXEL_LOC, PointType> feat_map;
  float loc_xyz[3];
  for (PointType &p_c : pl_feat.points) {
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter == feat_map.end()) {
      PointType pp = p_c;
      // pp.curvature = 1;
      feat_map[position] = pp;
    } else {
      // PointType &pp = iter->second;
      // pp.x = (pp.x * pp.curvature + p_c.x) / (pp.curvature + 1);
      // pp.y = (pp.y * pp.curvature + p_c.y) / (pp.curvature + 1);
      // pp.z = (pp.z * pp.curvature + p_c.z) / (pp.curvature + 1);
      // pp.curvature += 1;
    }
  }

  // pl_feat.clear();
  pcl::PointCloud<PointType> pl_feat2;
  pl_feat.swap(pl_feat2);
  pl_feat.reserve(feat_map.size());
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) pl_feat.push_back(iter->second);
  // std::cout<<"-----111111--66666666666-----"<<std::endl;
}

inline void down_sampling_serie(pcl::PointCloud<PointType> &pl_feat, int num) {
  if (num < 1) num = 1;

  pcl::PointCloud<PointType> pl_down;
  int psize = pl_feat.size();
  pl_down.reserve(psize);
  for (int i = 0; i < psize; i += num) pl_down.push_back(pl_feat[i]);
  pl_feat.swap(pl_down);
}

inline void pl_transform(pcl::PointCloud<PointType> &pl1, const Eigen::Matrix3d &rr, const Eigen::Vector3d &tt) {
  for (PointType &ap : pl1.points) {
    Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    pvec = rr * pvec + tt;
    ap.x = pvec[0];
    ap.y = pvec[1];
    ap.z = pvec[2];
  }
}

inline void pl_transform(pcl::PointCloud<PointType> &pl1, const IMUST &xx) {
  for (PointType &ap : pl1.points) {
    Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    pvec = xx.R * pvec + xx.p;
    ap.x = pvec[0];
    ap.y = pvec[1];
    ap.z = pvec[2];
  }
}

inline void plvec_trans(PLV(3) & porig, PLV(3) & ptran, IMUST &stat) {
  uint asize = porig.size();
  ptran.resize(asize);
  for (uint i = 0; i < asize; i++) ptran[i] = stat.R * porig[i] + stat.p;
}

// bool time_compare(PointType &x, PointType &y) {return (x.curvature < y.curvature);}

class VOX_FACTOR {
 public:
  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  int N;

  VOX_FACTOR() {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void clear() {
    P.setZero();
    v.setZero();
    N = 0;
  }

  void push(const Eigen::Vector3d &vec) {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  Eigen::Matrix3d cov() {
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }

  VOX_FACTOR &operator+=(const VOX_FACTOR &sigv) {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  void transform(const VOX_FACTOR &sigv, const IMUST &stat) {
    N = sigv.N;
    v = stat.R * sigv.v + N * stat.p;
    Eigen::Matrix3d rp = stat.R * sigv.v * stat.p.transpose();
    P = stat.R * sigv.P * stat.R.transpose() + rp + rp.transpose() + N * stat.p * stat.p.transpose();
  }

  void transform(const VOX_FACTOR &sigv, const Eigen::Matrix3d &R, const Eigen::Vector3d &p) {
    N = sigv.N;
    v = R * sigv.v + N * p;
    Eigen::Matrix3d rp = R * sigv.v * p.transpose();
    P = R * sigv.P * R.transpose() + rp + rp.transpose() + N * p * p.transpose();
  }
};

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/

const double threshold = 0.1;
inline bool esti_plane(Eigen::Vector4d &pca_result, const pcl::PointCloud<PointType> &point) {
  Eigen::Matrix<double, NMATCH, 3> A;
  Eigen::Matrix<double, NMATCH, 1> b;
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < NMATCH; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < NMATCH; j++) {
    if (fabs(normvec.dot(A.row(j)) + 1.0) > threshold) return false;
  }

  double n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;
  return true;
}
inline void read_file(vector<IMUST> &x_buf, string &data_path, int &start, int &endd) {
  std::ifstream file(data_path + "pose.json");
  double tx, ty, tz, qw, qx, qy, qz;

  while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
    static int i = 0;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    IMUST iumst;
    iumst.R = q.toRotationMatrix();
    // iumst.q = q;
    iumst.p = t;
    if (i >= start) x_buf.push_back(iumst);
    if (i >= endd) break;
    i++;
  }
}
// inline void read_file(vector<pose> &x_buf, string &data_path, int &start, int &endd) {
//   std::ifstream file(data_path + "pose.json");
//   double tx, ty, tz, qw, qx, qy, qz;
//
//   while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
//     static int i = 0;
//     Eigen::Quaterniond q(qw, qx, qy, qz);
//     Eigen::Vector3d t(tx, ty, tz);
//     IMUST iumst;
//     iumst.R = q.toRotationMatrix();
//     // iumst.q = q;
//     iumst.p = t;
//     if (i >= start) x_buf.push_back(iumst);
//     if (i >= endd) break;
//     i++;
//   }
// }
inline void save_file(vector<IMUST> &x_buf, string &data_path) {
  std::ofstream file(data_path + "pose.json");
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << data_path + "pose.json" << std::endl;
    return;
  }
  bool first = true;
  for (const auto &pose : x_buf) {
    Eigen::Quaterniond q = Eigen::Quaterniond(pose.R);
    if (first) {
      first = false;
    } else {
      file << std::endl;
    }
    file << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " " << q.w() << " " << q.x() << " " << q.y() << " "
         << q.z();
  }
  file.close();
}
struct Point {
  double x = 0;
  double y = 0;
  double z = 0;
};
using Points = vector<Point>;
#ifdef USE_PATC_API
inline void FillPointCloudMsg(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, mvt::protocol::PointCloudXYZI *msg) {
  msg->mutable_point()->Clear();
  msg->mutable_point()->Reserve((int)cloud->size());

  for (auto &pt : cloud->points) {
    auto p = msg->mutable_point()->Add();
    p->set_x(pt.x);
    p->set_y(pt.y);
    p->set_z(pt.z);
    p->set_intensity(pt.intensity);
  }
}
template <typename T>
void FillPointCloudMsg(const pcl::PointCloud<T> &cloud, mvt::protocol::PointCloudXYZI *msg) {
  msg->mutable_point()->Clear();
  msg->mutable_point()->Reserve((int)cloud.size());

  for (auto &pt : cloud) {
    auto p = msg->mutable_point()->Add();
    p->set_x(pt.x);
    p->set_y(pt.y);
    p->set_z(pt.z);
    p->set_intensity(pt.intensity);
  }
}
inline void PublishLines(const Points &points, const std::string &sensor_name) {
  mvt::protocol::PlotLines constraint_lines;
  constraint_lines.mutable_header()->set_time(PATC_GetTime);
  constraint_lines.mutable_header()->set_name(sensor_name);
  for (const auto &[x, y, z] : points) {
    // 对所要发布的线进行一些设置
    const auto line_s = constraint_lines.mutable_lines()->Add();
    line_s->mutable_start()->set_x(x);
    line_s->mutable_start()->set_y(y);
    line_s->mutable_start()->set_z(z);
    line_s->mutable_end()->set_x(x);
    line_s->mutable_end()->set_y(y);
    line_s->mutable_end()->set_z(0);
  }
  PATC_Pub(constraint_lines);
}
#endif
inline void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls) {
  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++) {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for (int i = 0; i < winsize; i++) {
    {
      static int index = 0;
      mvt::protocol::KeyFrame key_frame;
      key_frame.set_id(index);
      key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
      key_frame.mutable_header()->set_name("opt_frame");
      auto pos = key_frame.mutable_pose()->mutable_position();
      key_frame.mutable_header()->set_time(PATC_GetTime);
      auto t = x_buf[i].p;
      pos->set_x(t[0]);
      pos->set_y(t[1]);
      pos->set_z(t[2]);
      Eigen::Quaterniond q(x_buf[i].R);
      key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
      key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
      key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
      key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
      auto pc = key_frame.mutable_pc();
      FillPointCloudMsg(*pl_fulls[i], pc);
      PATC_Pub(key_frame);
      index++;
    }
  }

  // pub_pl_func(pl_path, pub_path);
}

struct pose {
  pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0), Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0))
      : q(_q), t(_t) {}
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};

inline void loadPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr &pc, int num,
                    std::string prefix = "") {
  std::stringstream ss;
  if (pcd_fill_num > 0)
    ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
  else
    ss << num;
  const auto cloud_path = filePath + prefix + ss.str() + ".pcd";
  // cout << "cloud_path = " << cloud_path.c_str() << endl;
  pcl::io::loadPCDFile(cloud_path, *pc);
}

inline void savdPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr &pc, int num) {
  std::stringstream ss;
  if (pcd_fill_num > 0)
    ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
  else
    ss << num;
  pcl::io::savePCDFileBinary(filePath + ss.str() + ".pcd", *pc);
}

inline std::vector<pose> read_pose(std::string filename, Eigen::Quaterniond qe = Eigen::Quaterniond(1, 0, 0, 0),
                                   Eigen::Vector3d te = Eigen::Vector3d(0, 0, 0)) {
  std::vector<pose> pose_vec;
  std::fstream file;
  file.open(filename);
  double tx, ty, tz, w, x, y, z;
  while (!file.eof()) {
    file >> tx >> ty >> tz >> w >> x >> y >> z;
    Eigen::Quaterniond q(w, x, y, z);
    Eigen::Vector3d t(tx, ty, tz);
    pose_vec.push_back(pose(qe * q, qe * t + te));
  }
  file.close();
  return pose_vec;
}

inline void transform_pointcloud(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out,
                                 Eigen::Vector3d t, Eigen::Quaterniond q) {
  size_t size = pc_in.points.size();
  pt_out.points.resize(size);
  for (size_t i = 0; i < size; i++) {
    Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
    Eigen::Vector3d pt_to;
    // if(pt_cur.norm()<0.3) continue;
    pt_to = q * pt_cur + t;
    pt_out.points[i].x = pt_to.x();
    pt_out.points[i].y = pt_to.y();
    pt_out.points[i].z = pt_to.z();
    // pt_out.points[i].r = pc_in.points[i].r;
    // pt_out.points[i].g = pc_in.points[i].g;
    // pt_out.points[i].b = pc_in.points[i].b;
  }
}

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr append_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1,
                                                           pcl::PointCloud<pcl::PointXYZRGB> pc2) {
  size_t size1 = pc1->points.size();
  size_t size2 = pc2.points.size();
  pc1->points.resize(size1 + size2);
  for (size_t i = size1; i < size1 + size2; i++) {
    pc1->points[i].x = pc2.points[i - size1].x;
    pc1->points[i].y = pc2.points[i - size1].y;
    pc1->points[i].z = pc2.points[i - size1].z;
    pc1->points[i].r = pc2.points[i - size1].r;
    pc1->points[i].g = pc2.points[i - size1].g;
    pc1->points[i].b = pc2.points[i - size1].b;
    // pc1->points[i].intensity = pc2.points[i-size1].intensity;
  }
  return pc1;
}

inline pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1,
                                                    pcl::PointCloud<PointType> pc2) {
  size_t size1 = pc1->points.size();
  size_t size2 = pc2.points.size();
  pc1->points.resize(size1 + size2);
  for (size_t i = size1; i < size1 + size2; i++) {
    pc1->points[i].x = pc2.points[i - size1].x;
    pc1->points[i].y = pc2.points[i - size1].y;
    pc1->points[i].z = pc2.points[i - size1].z;
    // pc1->points[i].r = pc2.points[i-size1].r;
    // pc1->points[i].g = pc2.points[i-size1].g;
    // pc1->points[i].b = pc2.points[i-size1].b;
    // pc1->points[i].intensity = pc2.points[i-size1].intensity;
  }
  return pc1;
}

inline double compute_inlier_ratio(std::vector<double> residuals, double ratio) {
  std::set<double> dis_vec;
  for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
    dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));

  return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
}

inline void write_pose(std::vector<pose> &pose_vec, std::string path) {
  std::ofstream file;
  file.open(path + "hba_pose.json", std::ofstream::trunc);
  file.close();
  Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
  Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
  file.open(path + "hba_pose.json", std::ofstream::app);

  for (size_t i = 0; i < pose_vec.size(); i++) {
    pose_vec[i].t << q0.inverse() * (pose_vec[i].t - t0);
    pose_vec[i].q.w() = (q0.inverse() * pose_vec[i].q).w();
    pose_vec[i].q.x() = (q0.inverse() * pose_vec[i].q).x();
    pose_vec[i].q.y() = (q0.inverse() * pose_vec[i].q).y();
    pose_vec[i].q.z() = (q0.inverse() * pose_vec[i].q).z();
    file << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " " << pose_vec[i].q.w() << " "
         << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " " << pose_vec[i].q.z();
    if (i < pose_vec.size() - 1) file << "\n";
  }
  file.close();
}

inline void writeEVOPose(std::vector<double> &lidar_times, std::vector<pose> &pose_vec, std::string path) {
  std::ofstream file;
  file.open(path + "evo_pose.txt", std::ofstream::trunc);
  for (size_t i = 0; i < pose_vec.size(); i++) {
    file << std::setprecision(18) << lidar_times[i] << " " << std::setprecision(6) << pose_vec[i].t(0) << " "
         << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " " << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " "
         << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
    if (i < pose_vec.size() - 1) file << "\n";
  }
  file.close();
  }