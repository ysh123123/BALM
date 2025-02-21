//
// Created by seer on 25-2-17.
//

// #include <ros/ros.h>
#include <Eigen/Eigenvalues>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
// #include <geometry_msgs/PoseArray.h>
#include <random>
#include <ctime>
// #include <tf/transform_broadcaster.h>
#include "old_ba/balm1.hpp"


#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>
#include "CSF/CSF.h"
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
double voxel_size[2] = {1, 1};
struct Point {
  double x = 0;
  double y = 0;
  double z = 0;
};
struct IMUST {
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};
void pl_transform(pcl::PointCloud<PointType> &pl1, const Eigen::Quaterniond &q, const Eigen::Vector3d &t) {
  for (PointType &ap : pl1.points) {
    Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    pvec = q * pvec + t;
    ap.x = pvec[0];
    ap.y = pvec[1];
    ap.z = pvec[2];
  }
}
// struct Line {
//   Point front;
//   Point back;
// };
using Points = vector<Point>;
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
static void PublishLines(const Points &points, const std::string &sensor_name) {
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
int iter = 5;
string data_path = "/mnt/d/proto_files/HBA/hagong-1/";
std::string cloud_dir = data_path + "pcd/";
int pcd_name_fill_num = 0;
int start = 0, endd = 160;
void read_file(vector<IMUST> &x_buf) {
  std::ifstream file(data_path + "pose.json");
  double tx, ty, tz, qw, qx, qy, qz;

  while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
    static int i = 0;
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    IMUST iumst;
    iumst.q = q;
    // iumst.q = q;
    iumst.t = t;
    if (i >= start) x_buf.push_back(iumst);
    // if (i >= endd) break;
    i++;
  }
}
void save_file(vector<IMUST> &x_buf) {
  std::ofstream file(data_path + "pose.json");
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << data_path + "pose.json" << std::endl;
    return;
  }
  bool first = true;
  for (const auto &pose : x_buf) {
    Eigen::Quaterniond q = pose.q;
    if (first) {
      first = false;
    } else {
      file << std::endl;
    }
    file << pose.t.x() << " " << pose.t.y() << " " << pose.t.z() << " " << q.w() << " " << q.x() << " " << q.y() << " "
         << q.z();
  }
  file.close();
}
void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls) {
  IMUST es0 = x_buf[0];
  for (uint i = 0; i < x_buf.size(); i++) {
    x_buf[i].t = es0.q.toRotationMatrix().transpose() * (x_buf[i].t - es0.t);
    x_buf[i].q = es0.q.toRotationMatrix().transpose() * x_buf[i].q;
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
      auto t = x_buf[i].t;
      pos->set_x(t[0]);
      pos->set_y(t[1]);
      pos->set_z(t[2]);
      Eigen::Quaterniond q(x_buf[i].q);
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
void CutVoxel(unordered_map<VOXEL_LOC, OCTO_TREE *> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat,
               Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity) {
  uint plsize = pl_feat->size();
  for (uint i = 0; i < plsize; i++) {
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = R_p * pvec_orig + t_p;

    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      iter->second->plvec_tran[fnum]->push_back(pvec_tran);
      iter->second->is2opt = true;
    } else {
      OCTO_TREE *ot = new OCTO_TREE(feattype, capacity);
      ot->plvec_orig[fnum]->push_back(pvec_orig);
      ot->plvec_tran[fnum]->push_back(pvec_tran);

      ot->voxel_center[0] = (0.5 + position.x) * voxel_size[feattype];
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size[feattype];
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size[feattype];
      ot->quater_length = voxel_size[feattype] / 4.0;
      feat_map[position] = ot;
    }
  }
}

bool use_surf = true;
bool use_corn = true;
bool down_sample = false;
int main(int argc, char **argv) {
  PATC_SetTheadName("main thread");
  vector<IMUST> x_odom, x_ba;

  string file_path;
  // 读取里程计数据
  read_file(x_odom);

  // cout << " win_size = " << win_size << endl;
  // 对里程计数据进行矫正
  IMUST es0 = x_odom[0];
  for (uint i = 0; i < x_odom.size(); i++) {
    x_odom[i].t = es0.q.toRotationMatrix().transpose() * (x_odom[i].t - es0.t);
    x_odom[i].q = es0.q.toRotationMatrix().transpose() * x_odom[i].q;
  }

  double surf_filter_length = 0.4;
  double corn_filter_length = 0.2;

  int win_size = 40;
  int margi_size = 10;
  int filter_num = 1;
  int thread_num = 4;
  int skip_num = 3;
  cout << " win_size = " << win_size << endl;
  cout << " margi_size = " << margi_size << endl;

  LM_SLWD_VOXEL opt_lsv(win_size, filter_num, thread_num);

  pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);

  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  unordered_map<VOXEL_LOC, OCTO_TREE *> surf_map, corn_map;

  int jump_flag = skip_num;
  printf("%d\n", skip_num);

  vector<Eigen::Quaterniond> q_poses;
  vector<Eigen::Vector3d> t_poses;
  Eigen::Quaterniond q_odom, q_ba, q_gather_pose(1, 0, 0, 0), q_last(1, 0, 0, 0);
  Eigen::Vector3d t_odom, t_ba,t_gather_pose(0, 0, 0), t_last(0, 0, 0);
  int plcount = 0, window_head = 0;
  Points odom_lines, ba_lines;

  for (int m = 0; m < x_odom.size(); m++) {
    // 获取点云数据
    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    {
      std::stringstream ss;
      if (pcd_name_fill_num > 0)
        ss << std::setw(pcd_name_fill_num) << std::setfill('0') << m + start;
      else
        ss << m + start;
      // pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
      string filename = cloud_dir + ss.str() + ".pcd";

      pcl::PointCloud<pcl::PointXYZI> pl_tem;
      pcl::io::loadPCDFile(filename, pl_tem);
      for (pcl::PointXYZI &pp : pl_tem.points) {
        PointType ap;
        ap.x = pp.x;
        ap.y = pp.y;
        // if(ap.z < 1)
        ap.z = pp.z;
        ap.intensity = pp.intensity;
        /*if (ap.z < 5)  */ pl_ptr->push_back(ap);
      }
    }
    // CSF csf;
    // csf.params.interations = 600;
    // csf.params.time_step = 0.95;
    // csf.params.cloth_resolution = 3;
    // csf.params.bSloopSmooth = false;
    // csf.setPointCloud(*pl_ptr);
    // std::vector<int> groundIndexes,offGroundIndexes;
    // // pcl::PointCloud<pcl::PointXYZI>::Ptr groundFrame(new pcl::PointCloud<pcl::PointXYZI>());
    // // pcl::PointCloud<pcl::PointXYZI>::Ptr offGroundFrame(new pcl::PointCloud<pcl::PointXYZI>());
    // csf.do_filtering(groundIndexes,offGroundIndexes);
    // pcl::copyPointCloud(*pl_ptr,groundIndexes,*pl_surf);
    // pcl::copyPointCloud(*pl_ptr,offGroundIndexes,*pl_corn);

    q_odom = x_odom[m].q;
    t_odom = x_odom[m].t;
    odom_lines.push_back({t_odom[0], t_odom[1], t_odom[2]});
    // 发布里程计数据
    {
      static int index = 0;
      mvt::protocol::KeyFrame key_frame;
      mvt::protocol::PlotLines constraint_lines;
      key_frame.set_id(index);
      key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
      key_frame.mutable_header()->set_name("odom_frame");
      auto pos = key_frame.mutable_pose()->mutable_position();
      key_frame.mutable_header()->set_time(PATC_GetTime);
      auto t = t_odom;
      pos->set_x(t[0]);
      pos->set_y(t[1]);
      pos->set_z(t[2]);
      Eigen::Quaterniond q(q_odom);
      key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
      key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
      key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
      key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
      auto pc = key_frame.mutable_pc();
      FillPointCloudMsg(*pl_ptr, pc);
      PATC_Pub(key_frame);
      index++;
    }

    Eigen::Vector3d delta_t(q_last.matrix().transpose() * (t_odom - t_last));
    Eigen::Quaterniond delta_q(q_last.matrix().transpose() * q_odom.matrix());
    q_last = q_odom;
    t_last = t_odom;
    t_gather_pose = t_gather_pose + q_gather_pose * delta_t;
    q_gather_pose = q_gather_pose * delta_q;

    if (jump_flag < skip_num) {
      jump_flag++;
      continue;
    }
    jump_flag = 0;

    if (plcount == 0) {
      IMUST ba_pose;
      // ba_pose.R = delta_q.toRotationMatrix();
      // ba_pose.p = delta_t;
      // x_ba.push_back(ba_pose);
      q_poses.push_back(q_gather_pose);
      t_poses.push_back(t_gather_pose);
    } else {
      auto last_ba_q = q_poses.back();
      auto last_ba_t = t_poses.back();
      // IMUST ba_pose;
      // ba_pose.R = last_ba_pose.R * delta_q;
      // ba_pose.p = last_ba_pose.R * delta_t + last_ba_pose.p;
      // x_ba.push_back(ba_pose);
      q_poses.push_back(last_ba_q * q_gather_pose);
      t_poses.push_back(last_ba_q * t_gather_pose + last_ba_t);
    }

    plcount++;


    pl_corn = pl_ptr;
    pl_surf = pl_ptr;
    if(down_sample) {
      down_sampling_voxel(*pl_corn, corn_filter_length);
      down_sampling_voxel(*pl_surf, surf_filter_length);
    }

    pl_fulls.push_back(pl_ptr);
    int win_id = plcount - window_head - 1;
    int win_count = win_id + 1;
    q_gather_pose.setIdentity();
    t_gather_pose.setZero();
    OCTO_TREE::voxel_windowsize = win_count;
    if (use_corn) {
      CutVoxel(corn_map, pl_corn, q_poses.back().toRotationMatrix(), t_poses.back(), 1, win_id, win_size);
    }
    if (use_surf) {
      CutVoxel(surf_map, pl_surf, q_poses.back().toRotationMatrix(), t_poses.back(), 0, win_id, win_size);
    }

    if (win_count >= win_size) {
      for (int i = 0; i < win_size; i++) {
        // Eigen::Quaterniond q(x_ba[window_head + i].R);
        opt_lsv.so3_poses[i].setQuaternion(q_poses[window_head + i]);
        opt_lsv.t_poses[i] = t_poses[window_head + i];
      }
      pcl::PointCloud<PointType> pl_corn_send, pl_surf_send;
      if (use_corn) {
        for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter) {
          if (iter->second->is2opt) {
            iter->second->root_centors.clear();
            iter->second->recut(0, win_id, iter->second->root_centors);
            pl_corn_send += iter->second->root_centors;
            iter->second->traversal_opt(opt_lsv);
          }
        }
      }
      if (use_surf) {
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
          if (iter->second->is2opt) {
            iter->second->root_centors.clear();
            iter->second->recut(0, win_id, iter->second->root_centors);
            pl_surf_send += iter->second->root_centors;
            iter->second->traversal_opt(opt_lsv);
          }
        }
      }
      // 发布特征点云
      {
        if(!pl_corn_send.empty()) {
          static int index = 0;
          mvt::protocol::KeyFrame key_frame;
          key_frame.set_id(-1);
          // key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
          key_frame.mutable_header()->set_name("pl_corn_send");
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
          down_sampling_voxel(pl_corn_send, 0.2);
          FillPointCloudMsg(pl_corn_send, pc);
          PATC_Pub(key_frame);
          index++;
        }
        if(!pl_surf_send.empty()) {
          static int index = 0;
          mvt::protocol::KeyFrame key_frame;
          key_frame.set_id(-1);
          // key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
          key_frame.mutable_header()->set_name("pl_surf_send");
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
          down_sampling_voxel(pl_surf_send, 0.2);
          FillPointCloudMsg(pl_surf_send, pc);
          PATC_Pub(key_frame);
          index++;
        }
      }
      opt_lsv.damping_iter();

      pcl::PointCloud<PointType> pl_send;

      for (int i = 0; i < win_size; ++i) {
        q_poses[window_head + i] = opt_lsv.so3_poses[i].unit_quaternion();
        t_poses[window_head + i] = opt_lsv.t_poses[i];
        // q_poses[window_head + i] = r;
        if (i < margi_size) {
          q_ba = q_poses[window_head + i];
          t_ba = t_poses[window_head + i];
          auto pl = *pl_fulls[window_head + i];
          ba_lines.push_back({t_ba[0], t_ba[1], t_ba[2]});
          {
            static int index = 0;
            mvt::protocol::KeyFrame key_frame;
            key_frame.set_id(index);
            key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
            key_frame.mutable_header()->set_name("ba_frame");
            auto pos = key_frame.mutable_pose()->mutable_position();
            key_frame.mutable_header()->set_time(PATC_GetTime);
            auto t = t_ba;
            pos->set_x(t[0]);
            pos->set_y(t[1]);
            pos->set_z(t[2]);
            Eigen::Quaterniond q(q_ba);
            key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
            key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
            key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
            key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
            auto pc = key_frame.mutable_pc();
            // down_sampling_voxel(pl_send, 0.2);
            FillPointCloudMsg(pl, pc);
            PATC_Pub(key_frame);
            index++;
          }
          pl_transform(pl, q_ba, t_ba);
          pl_send += pl;
        }
      }
      if (use_corn) {
        for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter) {
          if (iter->second->is2opt) {
            iter->second->root_centors.clear();
            iter->second->marginalize(0, margi_size, q_poses, t_poses, window_head, iter->second->root_centors);
          }
        }
      }
      if (use_surf) {
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
          if (iter->second->is2opt) {
            iter->second->root_centors.clear();
            iter->second->marginalize(0, margi_size, q_poses, t_poses, window_head, iter->second->root_centors);
          }
        }
      }

      window_head += margi_size;
      opt_lsv.free_voxel();
    }
  }
  for (int i = window_head; i < x_ba.size(); ++i) {
    // static int index = 0;
    mvt::protocol::KeyFrame key_frame;
    key_frame.set_id(i);
    key_frame.mutable_header()->set_stamp(int64_t(i * 1e9));
    key_frame.mutable_header()->set_name("ba_frame");
    auto pos = key_frame.mutable_pose()->mutable_position();
    key_frame.mutable_header()->set_time(PATC_GetTime);
    auto t = x_ba[i].t;
    ba_lines.push_back({t[0], t[1], t[2]});
    pos->set_x(t[0]);
    pos->set_y(t[1]);
    pos->set_z(t[2]);
    Eigen::Quaterniond q(x_ba[i].q);
    key_frame.mutable_pose()->mutable_orientation()->set_x(q.x());
    key_frame.mutable_pose()->mutable_orientation()->set_y(q.y());
    key_frame.mutable_pose()->mutable_orientation()->set_z(q.z());
    key_frame.mutable_pose()->mutable_orientation()->set_w(q.w());
    auto pc = key_frame.mutable_pc();
    // down_sampling_voxel(pl_send, 0.2);
    FillPointCloudMsg(*pl_fulls[i], pc);
    PATC_Pub(key_frame);
    // index++;
  }
  for (auto iter = surf_map.begin(); iter != surf_map.end();) {
    delete iter->second;
    surf_map.erase(iter++);
  }
  surf_map.clear();
  for (auto iter = corn_map.begin(); iter != corn_map.end();) {
    delete iter->second;
    corn_map.erase(iter++);
  }
  corn_map.clear();
  malloc_trim(0);
  PublishLines(odom_lines, "odom_lines");
  PublishLines(ba_lines, "ba_lines");
  // save_file(x_ba);
  return 0;
}