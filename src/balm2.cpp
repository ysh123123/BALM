#include "toolss.hpp"
// #include <ros/ros.h>
#include <Eigen/Eigenvalues>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
// #include <tf/transform_broadcaster.h>
#include "baa.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>
#define USE_PATC_API
#ifdef USE_PATC_API
#include <patc/patc_api.h>
#include "mvt_msg_pointcloud.pb.h"
#include "mvt_msg_keyframe.pb.h"
#else
#include "not_use_patc.h"
#endif
using namespace std;

#ifdef USE_PATC_API
static void FillPointCloudMsg(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, mvt::protocol::PointCloudXYZI *msg) {
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
static void FillPointCloudMsg(const pcl::PointCloud<T> &cloud, mvt::protocol::PointCloudXYZI *msg) {
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
#endif

string data_path = "/mnt/d/proto_files/HBA/kitti07/";
std::string cloud_dir = data_path + "pcd/";
int pcd_name_fill_num = 5;

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename) {
  std::ifstream file(data_path + "pose.json");
  double tx, ty, tz, qw, qx, qy, qz;
  int start = 0, end = 20;
  while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
    static int i = 0;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    IMUST iumst;
    iumst.R = q.toRotationMatrix();
    // iumst.q = q;
    iumst.p = t;
    if (i >= start) x_buf.push_back(iumst);
    if (i >= end) break;
    i++;
  }

  for (int m = 0; m < x_buf.size(); m++) {
    string filename = prename + "full" + to_string(m + start) + ".pcd";
    std::stringstream ss;
    if (pcd_name_fill_num > 0)
      ss << std::setw(pcd_name_fill_num) << std::setfill('0') << m + start;
    else
      ss << m + start;
    // pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
    filename = cloud_dir + ss.str() + ".pcd";
    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
    for (pcl::PointXYZI &pp : pl_tem.points) {
      PointType ap;
      ap.x = pp.x;
      ap.y = pp.y;
      ap.z = pp.z;
      ap.intensity = pp.intensity;
      if (/*ap.z < 5 && */ ap.z >= -2) pl_ptr->push_back(ap);
    }

    pl_fulls.push_back(pl_ptr);
    {
      static int index = 0;
      mvt::protocol::KeyFrame key_frame;
      key_frame.set_id(index);
      key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
      key_frame.mutable_header()->set_name("odom_frame");
      auto pos = key_frame.mutable_pose()->mutable_position();
      key_frame.mutable_header()->set_time(PATC_GetTime);
      auto t = x_buf[m].p;
      pos->set_x(t[0]);
      pos->set_y(t[1]);
      pos->set_z(t[2]);
      Eigen::Quaterniond q(x_buf[m].R);
      key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
      key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
      key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
      key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
      auto pc = key_frame.mutable_pc();
      FillPointCloudMsg(*pl_ptr, pc);
      PATC_Pub(key_frame);
      index++;
    }

    // IMUST curr;
    // curr.R = rots[m]; curr.p = poss[m]; curr.t = tims[m];
    // x_buf.push_back(curr);
  }
}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls) {
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

int main(int argc, char **argv) {
  PATC_SetTheadName("main thread");

  string prename, ofname;
  vector<IMUST> x_buf;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;
  int merge_size = 5;

  cout << " win_size = " << win_size << endl;
  cout << " merge_size = " << merge_size << endl;

  string file_path;
  read_file(x_buf, pl_fulls, file_path);
  win_size = x_buf.size();
  cout << " win_size = " << win_size << endl;
  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++) {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }


  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path;
  // for (int iterCount = 0; iterCount < 1; iterCount++) {
  unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

  eigen_value_array[0] = 1.0 / 64;
  eigen_value_array[1] = 1.0 / 64;
  eigen_value_array[2] = 1.0 / 64;

  for (int i = 0; i < win_size; i++) {
    cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);
  }

  pcl::PointCloud<PointType> pl_send;
  // pub_pl_func(pl_send, pub_show);

  pcl::PointCloud<PointType> pl_cent;
  pl_send.clear();
  VOX_HESS voxhess;
  for (auto iter = surf_map.begin(); iter != surf_map.end() /*&& n.ok()*/; iter++) {
    // std::cout<<"start recut"<<std::endl;
    iter->second->recut(win_size);
    // std::cout<<"start opt"<<std::endl;
    iter->second->tras_opt(voxhess, win_size);
    // std::cout<<"start display"<<std::endl;
    iter->second->tras_display(pl_send, win_size);
  }
  // cout << pl_send.size() << endl;
  {
    static int index = 0;
    mvt::protocol::KeyFrame key_frame;
    key_frame.set_id(index);
    key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
    key_frame.mutable_header()->set_name("recuted_frame");
    auto pos = key_frame.mutable_pose()->mutable_position();
    key_frame.mutable_header()->set_time(PATC_GetTime);
    pos->set_x(0);
    pos->set_y(0);
    pos->set_z(0);
    key_frame.mutable_pose()->mutable_orientation()->set_x(0);
    key_frame.mutable_pose()->mutable_orientation()->set_y(0);
    key_frame.mutable_pose()->mutable_orientation()->set_z(0);
    key_frame.mutable_pose()->mutable_orientation()->set_w(0);
    auto pc = key_frame.mutable_pc();
    down_sampling_voxel(pl_send, 0.2);
    FillPointCloudMsg(pl_send, pc);
    PATC_Pub(key_frame);
    index++;
  }
  // pub_pl_func(pl_send, pub_cute);
  printf("\nThe planes (point association) cut by adaptive voxelization.\n");
  printf("If the planes are too few, the optimization will be degenerated and fail.\n");
  printf("If no problem, input '1' to continue or '0' to exit...\n");
  // int a;
  // cin >> a;
  // if (a == 0) exit(0);
  // pl_send.clear();
  // pub_pl_func(pl_send, pub_cute);

  if (voxhess.plvec_voxels.size() < 3 * x_buf.size()) {
    printf("Initial error too large.\n");
    printf("Please loose plane determination criteria for more planes.\n");
    printf("The optimization is terminated.\n");
    exit(0);
  }

  BALM2 opt_lsv;
  opt_lsv.damping_iter(x_buf, voxhess);

  for (auto iter = surf_map.begin(); iter != surf_map.end();) {
    delete iter->second;
    surf_map.erase(iter++);
  }
  surf_map.clear();

  malloc_trim(0);
  // }

  printf("\nRefined point cloud is publishing...\n");
  malloc_trim(0);
  data_show(x_buf, pl_fulls);
  printf("\nRefined point cloud is published.\n");

  // ros::spin();
  return 0;
}

// #include "bavoxel.hpp"
// #include "balm2.hpp"
//
// #include <random>
// #include <ctime>
// #include <malloc.h>
//
// #include <Eigen/Eigenvalues>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/io/pcd_io.h>
//
// #define USE_PATC_API
// #ifdef USE_PATC_API
// #include <patc/patc_api.h>
// #include "mvt_msg_pointcloud.pb.h"
// #include "mvt_msg_keyframe.pb.h"
// #else
// #include "not_use_patc.h"
// #endif
//
// using namespace std;
// int pcd_name_fill_num = 0;
// // double voxel_size[2] = {1, 1};
// string data_path = "/mnt/d/proto_files/HBA/wq/";
// std::string cloud_dir = data_path + "pcd/";
// struct Pose {
//   Eigen::Matrix3d R;
//   Eigen::Quaterniond q;
//   Eigen::Vector3d t;
// };
// #ifdef USE_PATC_API
// static void FillPointCloudMsg(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, mvt::protocol::PointCloudXYZI *msg)
// {
//   msg->mutable_point()->Clear();
//   msg->mutable_point()->Reserve((int)cloud->size());
//
//   for (auto &pt : cloud->points) {
//     auto p = msg->mutable_point()->Add();
//     p->set_x(pt.x);
//     p->set_y(pt.y);
//     p->set_z(pt.z);
//     p->set_intensity(pt.intensity);
//   }
// }
// template <typename T>
// static void FillPointCloudMsg(const pcl::PointCloud<T> &cloud, mvt::protocol::PointCloudXYZI *msg) {
//   msg->mutable_point()->Clear();
//   msg->mutable_point()->Reserve((int)cloud.size());
//
//   for (auto &pt : cloud) {
//     auto p = msg->mutable_point()->Add();
//     p->set_x(pt.x);
//     p->set_y(pt.y);
//     p->set_z(pt.z);
//     p->set_intensity(pt.intensity);
//   }
// }
// #endif
//
// int main(int argc, char **argv) {
//   PATC_SetTheadName("main thread");
//
//   std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_vec;
//   std::vector<Pose> init_pose_vec;
//
//   std::ifstream file(data_path + "pose.json");
//   double tx, ty, tz, qw, qx, qy, qz;
//   while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
//     Eigen::Quaterniond q(qw, qx, qy, qz);
//     Eigen::Vector3d t(tx, ty, tz);
//     Pose iumst;
//     iumst.R = q.toRotationMatrix();
//     iumst.q = q;
//     iumst.t = t;
//     init_pose_vec.push_back(iumst);
//   }
//
//   // std::cout << "cloud_vec.size = " << cloud_vec.size() << std::endl;
//   std::cout << "init_pose_vec.size = " << init_pose_vec.size() << std::endl;
//   // double surf_filter_length = 0.4;
//
//   int window_size = 30;
//   int margi_size = 5;
//   int filter_num = 1;
//   int thread_num = 4;
//
//   // BALM2::eigen_value_array[0] = 1.0 / 32;
//   // BALM2::eigen_value_array[1] = 1.0 / 32;
//   // BALM2::eigen_value_array[2] = 1.0 / 32;
//   // BALM2::eigen_value_array[3] = 1.0 / 32;
//   // BALM2::layer_size[0] = 30;
//   // BALM2::layer_size[1] = 30;
//   // BALM2::layer_size[2] = 30;
//   // BALM2::layer_size[3] = 30;
//   // BALM2::min_ps = 10;
//   // BALM2::voxel_size = 0.5;
//   // BALM2::win_size = init_pose_vec.size();
//   // BALM2::fix_size = margi_size;
//
//   vector<Eigen::Quaterniond> q_poses;
//   vector<Eigen::Vector3d> t_poses;
//   vector<BALM2::IMUST> x_buf(window_size);
//
//   Eigen::Quaterniond q_odom, q_last(1, 0, 0, 0);
//   Eigen::Vector3d t_odom, t_last(0, 0, 0);
//
//   int plcount = 0, window_base = 0;
//
//   BALM2::OctoTreeMap surf_map;
//   Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
//   // for (int i = 0; i < init_pose_vec.size(); ++i) {
//   for (int i = 0; i < 40; ++i) {
//     // pl_surf = cloud_vec[i];
//     pcl::PointCloud<pcl::PointXYZI>::Ptr pl_surf(new pcl::PointCloud<pcl::PointXYZI>);
//     // 读取点云数据
//     {
//       std::stringstream ss;
//       if (pcd_name_fill_num > 0)
//         ss << std::setw(pcd_name_fill_num) << std::setfill('0') << i;
//       else
//         ss << i;
//       // pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
//       std::string filename = cloud_dir + ss.str() + ".pcd";
//       if (pcl::io::loadPCDFile(filename, *pl_surf) == -1) {
//         std::cout << "文件不存在，停止读取" << std::endl;
//         break;  // 文件不存在，停止读取
//       }
//       cloud_vec.push_back(pl_surf);
//     }
//     // 更新位姿
//     {
//       // 获取里程计位姿
//       auto cur_iumst = init_pose_vec[i];
//       q_odom = cur_iumst.q;
//       t_odom = cur_iumst.t;
//
//       // 获取里程计的相对位姿
//       Eigen::Vector3d delta_t(q_last.matrix().transpose() * (t_odom - t_last));
//       Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
//       q_last = q_odom;
//       t_last = t_odom;
//
//       //
//       if (plcount == 0) {
//         q_poses.push_back(delta_q);
//         t_poses.push_back(delta_t);
//       } else {
//         t_poses.push_back(t_poses[plcount - 1] + q_poses[plcount - 1] * delta_t);
//         q_poses.push_back(q_poses[plcount - 1] * delta_q);
//       }
//     }
//     // 点云可视化
//     {
//       mvt::protocol::KeyFrame key_frame;
//       key_frame.set_id(i);
//       key_frame.mutable_header()->set_stamp(int64_t(i * 1e9));
//       key_frame.mutable_header()->set_name("odom_frame");
//       auto pos = key_frame.mutable_pose()->mutable_position();
//       key_frame.mutable_header()->set_time(PATC_GetTime);
//       // auto t = t_poses.back();
//       auto t = t_odom;
//       pos->set_x(t[0]);
//       pos->set_y(t[1]);
//       pos->set_z(t[2]);
//       // auto q = q_poses.back();
//       auto q = q_odom;
//       key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
//       key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
//       key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
//       key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
//       auto pc = key_frame.mutable_pc();
//       FillPointCloudMsg(pl_surf, pc);
//       PATC_Pub(key_frame);
//     }
//
//     plcount++;
//     int wind_count = plcount - window_base;
//     int frame_head = plcount - 1 - window_base;
//
//     BALM2::IMUST cur_imu;
//     cur_imu.R = q_poses.back().toRotationMatrix();
//     cur_imu.p = t_poses.back();
//     BALM2::BALM2::CutVoxel(surf_map, *pl_surf, cur_imu, frame_head);
//
//     // std::cout << "frame_head = " << frame_head << std::endl;
//     // std::cout << "surf_map size = " << surf_map.size() << std::endl;
//
//     if (plcount >= window_base + window_size) {
//       int recut_size = 0;
//
//       for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
//         if (iter->second->is2opt) {
//           recut_size++;
//
//           iter->second->Recut(wind_count);
//         }
//       }
//       std::cout << "recut_size = " << recut_size << std::endl;
//       // 更新滑窗中的位姿
//       for (int j = 0; j < window_size; j++) {
//         BALM2::IMUST imu;
//         imu.R = q_poses[window_base + j].toRotationMatrix();
//         imu.p = t_poses[window_base + j];
//         x_buf[j] = imu;
//       }
//
//       //
//       if (window_base != 0) {
//         BALM2::VOX_HESS voxhess;
//         pcl::PointCloud<BALM2::PointType> pl_send;
//         for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
//           if (iter->second->is2opt) {
//             iter->second->TrasOpt(pl_send,voxhess, x_buf,frame_head + 1);
//           }
//         }
//         {
//           static int index = 0;
//           mvt::protocol::KeyFrame key_frame;
//           key_frame.set_id(index);
//           key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
//           key_frame.mutable_header()->set_name("recuted_frame");
//           auto pos = key_frame.mutable_pose()->mutable_position();
//           key_frame.mutable_header()->set_time(PATC_GetTime);
//           pos->set_x(0);
//           pos->set_y(0);
//           pos->set_z(0);
//           key_frame.mutable_pose()->mutable_orientation()->set_x(0);
//           key_frame.mutable_pose()->mutable_orientation()->set_y(0);
//           key_frame.mutable_pose()->mutable_orientation()->set_z(0);
//           key_frame.mutable_pose()->mutable_orientation()->set_w(0);
//           auto pc = key_frame.mutable_pc();
//           FillPointCloudMsg(pl_send, pc);
//           PATC_Pub(key_frame);
//           index++;
//         }
//         printf("Begin to optimize...\n");
//         // Eigen::MatrixXd Rcov(6 * window_size, 6 * window_size);
//         // Rcov.setZero();
//         BALM2::BALM2 opt;
//         opt.DampingIter(x_buf, voxhess);
//       }
//       for (int j = 0; j < window_size; j++) {
//         q_poses[window_base + j] = Eigen::Quaterniond(x_buf[j].R);
//         t_poses[window_base + j] = x_buf[j].p;
//       }
//       pcl::PointCloud<BALM2::PointType> pl_send;
//       for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
//         if (iter->second->is2opt) {
//           // iter->second->TrasDisplay(pl_send, frame_head + 1);
//           iter->second->Marginalize(margi_size, x_buf, frame_head + 1);
//         }
//       }
//       // 发布切割后的点云
//
//       window_base += margi_size;
//     }
//   }
//   return 0;
// }
