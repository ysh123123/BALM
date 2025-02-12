#include "bavoxel.hpp"
#include "balm2.hpp"

#include <random>
#include <ctime>
#include <malloc.h>

#include <Eigen/Eigenvalues>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#define USE_PATC_API
#ifdef USE_PATC_API
#include <patc/patc_api.h>
#include "mvt_msg_pointcloud.pb.h"
#include "mvt_msg_keyframe.pb.h"
#else
#include "not_use_patc.h"
#endif

using namespace std;
int pcd_name_fill_num = 0;
double voxel_size[2] = {1, 1};
string data_path = "/mnt/d/proto_files/HBA/hagong-1/";
std::string cloud_dir = data_path + "pcd/";
bool use_balm1 = true;
struct Pose {
  Eigen::Matrix3d R;
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};
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
#endif
void cut_voxel(unordered_map<BALM1::VOXEL_LOC, BALM1::OCTO_TREE *> &feat_map,
               const pcl::PointCloud<pcl::PointXYZI>::Ptr &pl_feat, Eigen::Matrix3d R_p, Eigen::Vector3d t_p,
               int feattype, int fnum, int capacity) {
  uint plsize = pl_feat->size();
  for (uint i = 0; i < plsize; i++) {
    auto &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = R_p * pvec_orig + t_p;

    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    BALM1::VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      iter->second->plvec_tran[fnum]->push_back(pvec_tran);
      iter->second->is2opt = true;
    } else {
      BALM1::OCTO_TREE *ot = new BALM1::OCTO_TREE(feattype, capacity);
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

int main(int argc, char **argv) {
  PATC_SetTheadName("main thread");

  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_vec;
  std::vector<Pose> init_pose_vec;

  std::ifstream file(data_path + "pose.json");
  double tx, ty, tz, qw, qx, qy, qz;
  while (file >> tx >> ty >> tz >> qw >> qx >> qy >> qz) {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    Pose iumst;
    iumst.R = q.toRotationMatrix();
    iumst.q = q;
    iumst.t = t;
    init_pose_vec.push_back(iumst);
  }

  // std::cout << "cloud_vec.size = " << cloud_vec.size() << std::endl;
  std::cout << "init_pose_vec.size = " << init_pose_vec.size() << std::endl;
  double surf_filter_length = 0.4;

  int window_size = 20;
  int margi_size = 5;
  int filter_num = 1;
  int thread_num = 4;
  int skip_num = 0;
  int pub_skip = 1;

  BALM2::eigen_value_array[0] = 1.0 / 16;
  BALM2::eigen_value_array[1] = 1.0 / 16;
  BALM2::eigen_value_array[2] = 1.0 / 9;

  int jump_flag = skip_num;
  printf("%d\n", skip_num);
  BALM1::LM_SLWD_VOXEL opt_lsv(window_size, filter_num, thread_num);

  vector<Eigen::Quaterniond> q_poses;
  vector<Eigen::Vector3d> t_poses;

  vector<BALM2::IMUST> x_buf(window_size);

  Eigen::Quaterniond q_odom, q_gather_pose(1, 0, 0, 0), q_last(1, 0, 0, 0);
  Eigen::Vector3d t_odom, t_gather_pose(0, 0, 0), t_last(0, 0, 0);

  int plcount = 0, window_base = 0;

  unordered_map<BALM1::VOXEL_LOC, BALM1::OCTO_TREE *> surf_map;
  BALM2::OctoTreeMap surf_map2;
  Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());

  for (int i = 0; i < init_pose_vec.size(); ++i) {
    // pl_surf = cloud_vec[i];
    pcl::PointCloud<pcl::PointXYZI>::Ptr pl_surf(new pcl::PointCloud<pcl::PointXYZI>);
    std::stringstream ss;
    if (pcd_name_fill_num > 0)
      ss << std::setw(pcd_name_fill_num) << std::setfill('0') << i;
    else
      ss << i;
    // pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
    std::string filename = cloud_dir + ss.str() + ".pcd";
    if (pcl::io::loadPCDFile(filename, *pl_surf) == -1) {
      std::cout << "文件不存在，停止读取" << std::endl;
      break;  // 文件不存在，停止读取
    }
    cloud_vec.push_back(pl_surf);
    auto cur_iumst = init_pose_vec[i];
    q_odom = cur_iumst.q;
    t_odom = cur_iumst.t;
    mvt::protocol::KeyFrame keyframe1;
    keyframe1.set_id(i);
    keyframe1.mutable_header()->set_stamp(int64_t(i * 1e9));
    keyframe1.mutable_header()->set_name("keyframes22");
    auto pos = keyframe1.mutable_pose()->mutable_position();
    keyframe1.mutable_header()->set_time(PATC_GetTime);
    // auto t = t_odom;
    pos->set_x(t_odom[0]);
    pos->set_y(t_odom[1]);
    pos->set_z(t_odom[2]);
    // auto q = opt_lsv.so3_poses[j].unit_quaternion();
    keyframe1.mutable_pose()->mutable_orientation()->set_x(q_odom.x());
    keyframe1.mutable_pose()->mutable_orientation()->set_y(q_odom.y());
    keyframe1.mutable_pose()->mutable_orientation()->set_z(q_odom.z());
    keyframe1.mutable_pose()->mutable_orientation()->set_w(q_odom.w());
    auto pc = keyframe1.mutable_pc();
    FillPointCloudMsg(pl_surf, pc);
    PATC_Pub(keyframe1);

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
      q_poses.push_back(q_gather_pose);
      t_poses.push_back(t_gather_pose);

    } else {
      t_poses.push_back(t_poses[plcount - 1] + q_poses[plcount - 1] * t_gather_pose);
      q_poses.push_back(q_poses[plcount - 1] * q_gather_pose);
    }

    plcount++;
    BALM1::OCTO_TREE::voxel_windowsize = plcount - window_base;
    q_gather_pose.setIdentity();
    t_gather_pose.setZero();

    // BALM1::down_sampling_voxel(*pl_surf, surf_filter_length);

    int frame_head = plcount - 1 - window_base;

    cut_voxel(surf_map, pl_surf, q_poses.back().toRotationMatrix(), t_poses.back(), 0, frame_head, window_size);

    std::cout << "frame_head = " << frame_head << std::endl;

    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
      if (iter->second->is2opt) {
        iter->second->root_centors.clear();
        iter->second->recut(0, frame_head, iter->second->root_centors);
      }
    }

    std::cout << "plcount = " << plcount << std::endl;
    std::cout << "window_base = " << window_base << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    if (plcount >= window_base + window_size) {
      printf("Begin to BA...\n");

      std::cout << "surf_map.size()" << surf_map.size() << std::endl;
      for (int j = 0; j < window_size; j++) {
        opt_lsv.so3_poses[j].setQuaternion(q_poses[window_base + j]);
        opt_lsv.t_poses[j] = t_poses[window_base + j];
      }

      if (window_base != 0) {
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
          if (iter->second->is2opt) {
            iter->second->traversal_opt(opt_lsv);
          }
        }
        opt_lsv.damping_iter();
      }
      // pcl::PointCloud<pcl::PointXYZI> pl_send;

      for (int j = 0; j < margi_size; j += pub_skip) {
        static int id = 0;
        BALM2::IMUST imust;

        mvt::protocol::KeyFrame keyframe;
        keyframe.set_id(id);
        keyframe.mutable_header()->set_stamp(int64_t(id * 1e9));
        keyframe.mutable_header()->set_name("keyframes");
        auto pos = keyframe.mutable_pose()->mutable_position();
        keyframe.mutable_header()->set_time(PATC_GetTime);
        Eigen::Vector3d t;

        t = opt_lsv.t_poses[j];

        pos->set_x(t[0]);
        pos->set_y(t[1]);
        pos->set_z(t[2]);
        Eigen::Quaterniond q;

        q = opt_lsv.so3_poses[j].unit_quaternion();

        keyframe.mutable_pose()->mutable_orientation()->set_x(q.x());
        keyframe.mutable_pose()->mutable_orientation()->set_y(q.y());
        keyframe.mutable_pose()->mutable_orientation()->set_z(q.z());
        keyframe.mutable_pose()->mutable_orientation()->set_w(q.w());
        auto pc = keyframe.mutable_pc();
        FillPointCloudMsg(cloud_vec[window_base + j], pc);
        PATC_Pub(keyframe);
        id++;
        // auto pos1 = keyframe.mutable_pose()->mutable_position();
        // trans.block<3, 3>(0, 0) = opt_lsv.so3_poses[j].matrix();
        // trans.block<3, 1>(0, 3) = opt_lsv.t_poses[j];
        //
        // pcl::PointCloud<pcl::PointXYZI> pcloud;
        // pcl::transformPointCloud(*cloud_vec[window_base + j], pcloud, trans);
        // pl_send += pcloud;
      }

      // pub_func(pl_send, pub_full, ct);

      for (int j = 0; j < window_size; j++) {
        q_poses[window_base + j] = opt_lsv.so3_poses[j].unit_quaternion();
        t_poses[window_base + j] = opt_lsv.t_poses[j];
      }

      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) {
        if (iter->second->is2opt) {
          iter->second->root_centors.clear();
          iter->second->marginalize(0, margi_size, q_poses, t_poses, window_base, iter->second->root_centors);
        }
      }

      window_base += margi_size;

      opt_lsv.free_voxel();
    }
  }
  return 0;
}
