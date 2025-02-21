#include <Eigen/Eigenvalues>
#include <random>
#include <ctime>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>

#include "vox_optimizer.hpp"
#include "octo_tree.hpp"
#include "tools.hpp"
#include "hba.hpp"
using namespace std;
int hba_iter = 3;
string data_path = "/mnt/d/proto_files/HBA/wq3/";
std::string cloud_dir = data_path + "pcd/";
bool down_sample = true;
int start = 0, endd = 120000;
bool need_hba = true;
bool only_global_ba = false;
int main(int argc, char **argv) {
  PATC_SetTheadName("main thread");

  pcd_name_fill_num = 0;
  voxel_size = 0.8;
  eigen_value_array[0] = 1. / 15;
  eigen_value_array[1] = 1. / 15;
  eigen_value_array[2] = 1. / 15;
  eigen_value_array[3] = 1. / 15;
  constexpr int total_layer_num = 3, thread_num = 16;
  vector<IMUST> x_odom, x_ba;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;
  int margi_size = 5;
  int win_size = 20;
  int skip_num = 3;
  cout << " win_size = " << win_size << endl;
  cout << " margi_size = " << margi_size << endl;

  string file_path;
  // 读取里程计数据
  read_file(x_odom, data_path, start, endd);

  // 对里程计数据进行矫正
  IMUST es0 = x_odom[0];
  for (uint i = 0; i < x_odom.size(); i++) {
    x_odom[i].p = es0.R.transpose() * (x_odom[i].p - es0.p);
    x_odom[i].R = es0.R.transpose() * x_odom[i].R;
  }

  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path;
  OctoTreeMap surf_map;

  int jump_flag = skip_num;
  printf("%d\n", skip_num);
  Eigen::Quaterniond q_odom, q_ba, q_gather_pose(1, 0, 0, 0), q_last(1, 0, 0, 0);
  Eigen::Vector3d t_odom, t_ba, t_gather_pose(0, 0, 0), t_last(0, 0, 0);
  int plcount = 0, window_head = 0;
  Points odom_lines, ba_lines;
  for (int m = 0; m < x_odom.size(); m++) {
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
        auto dis = std::sqrt(std::pow(pp.x, 2) + std::pow(pp.y, 2) + std::pow(pp.z, 2));
        if (dis > 40) continue;
        ap.x = pp.x;
        ap.y = pp.y;
        // if(ap.z < 1)
        ap.z = pp.z;
        ap.intensity = pp.intensity;
        /*if (ap.z < 5)  */ pl_ptr->push_back(ap);
      }
    }
    // if(pl_ptr->size() > 1e4) {
    if (down_sample) {
      down_sampling_voxel(*pl_ptr, 0.2);
    }
    // }

    q_odom = x_odom[m].R;
    t_odom = x_odom[m].p;

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

    if (plcount == 0) {
      IMUST ba_pose;
      ba_pose.R = q_gather_pose.toRotationMatrix();
      ba_pose.p = t_gather_pose;
      x_ba.push_back(ba_pose);
    } else {
      auto last_ba_pose = x_ba.back();
      IMUST ba_pose;
      ba_pose.R = last_ba_pose.R * q_gather_pose;
      ba_pose.p = last_ba_pose.R * t_gather_pose + last_ba_pose.p;
      x_ba.push_back(ba_pose);
    }

    plcount++;
    pl_fulls.push_back(pl_ptr);

    int win_id = plcount - window_head - 1;
    int win_count = win_id + 1;
    q_gather_pose.setIdentity();
    t_gather_pose.setZero();
    if (only_global_ba) {
      continue;
    }
    CutVoxel(surf_map, *pl_ptr, x_ba.back(), win_id, win_size);
    if (win_count >= win_size) {
      vector<IMUST> x_buf;
      for (int i = 0; i < win_size; ++i) {
        x_buf.push_back(x_ba[window_head + i]);
      }
      VoxHess voxhess(win_size);

      for (auto iter = surf_map.begin(); iter != surf_map.end() /*&& n.ok()*/; iter++) {
        // std::cout<<"start recut"<<std::endl;
        if (iter->second->is2opt) {
          iter->second->Recut(win_count);
          iter->second->TrasOpt(voxhess, win_count);
        }
      }

      // BALM2 opt(win_size);
      // opt.damping_iter(x_buf, voxhess);
      VoxOptimizer opt_lsv(win_size, false);
      // PLV(6) hess_vec;
      // double residual_cur = 0;
      // size_t mem_cost = 0;
      opt_lsv.DampingIter(x_buf, voxhess);

      pcl::PointCloud<PointType> pl_send;
      for (auto iter = surf_map.begin(); iter != surf_map.end() /*&& n.ok()*/; iter++) {
        if (iter->second->is2opt) {
          iter->second->TrasDisplay(pl_send, win_count);
          iter->second->Marginalize(margi_size, x_buf, win_count);
        } /*else {
          delete iter->second;
          surf_map.erase(iter++);
        }*/
      }
      {
        static int index = 0;
        mvt::protocol::KeyFrame key_frame;
        key_frame.set_id(-1);
        // key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
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
      pl_send.clear();
      for (int i = 0; i < win_size; ++i) {
        x_ba[window_head + i] = x_buf[i];
        if (i < margi_size) {
          q_ba = x_buf[i].R;
          t_ba = x_buf[i].p;
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
          pl_transform(pl, x_buf[i]);
          pl_send += pl;
        }
      }
      // 发布ba数据
      // {
      //   static int index = 0;
      //   mvt::protocol::KeyFrame key_frame;
      //   key_frame.set_id(-1);
      //   // key_frame.mutable_header()->set_stamp(int64_t(index * 1e9));
      //   key_frame.mutable_header()->set_name("ba_submap_frame");
      //   auto pos = key_frame.mutable_pose()->mutable_position();
      //   key_frame.mutable_header()->set_time(PATC_GetTime);
      //   // auto t = t_ba;
      //   pos->set_x(0);
      //   pos->set_y(0);
      //   pos->set_z(0);
      //   // Eigen::Quaterniond q(q_ba);
      //   key_frame.mutable_pose()->mutable_orientation()->set_x(0);
      //   key_frame.mutable_pose()->mutable_orientation()->set_y(0);
      //   key_frame.mutable_pose()->mutable_orientation()->set_z(0);
      //   key_frame.mutable_pose()->mutable_orientation()->set_w(0);
      //   auto pc = key_frame.mutable_pc();
      //   down_sampling_voxel(pl_send, 0.2);
      //   FillPointCloudMsg(pl_send, pc);
      //   PATC_Pub(key_frame);
      //   index++;
      // }
      window_head += margi_size;
    }
  }

  PublishLines(odom_lines, "odom_lines");
  if (!only_global_ba) {
    for (int i = window_head; i < x_ba.size(); ++i) {
      // static int index = 0;
      mvt::protocol::KeyFrame key_frame;
      key_frame.set_id(i);
      key_frame.mutable_header()->set_stamp(int64_t(i * 1e9));
      key_frame.mutable_header()->set_name("ba_frame");
      auto pos = key_frame.mutable_pose()->mutable_position();
      key_frame.mutable_header()->set_time(PATC_GetTime);
      auto t = x_ba[i].p;
      ba_lines.push_back({t[0], t[1], t[2]});
      pos->set_x(t[0]);
      pos->set_y(t[1]);
      pos->set_z(t[2]);
      Eigen::Quaterniond q(x_ba[i].R);
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
    PublishLines(ba_lines, "ba_lines");
  }

  if (need_hba) {
    std::vector<pose> pose_ba;

    for (auto x : x_ba) {
      pose p;
      p.q = x.R;
      p.t = x.p;
      pose_ba.emplace_back(p);
    }
    auto pose_hba = pose_ba;

    for (int j = 0; j < hba_iter; ++j) {
      HBA hba(total_layer_num, thread_num, pose_hba, pl_fulls);
      for (int i = 0; i < total_layer_num - 1; i++) {
        std::cout << "---------------------" << std::endl;
        // // 按层进行线程加速
        // cout<<"i = "<<i<<"| path = "<<hba.layers[i].data_path<<endl;
        DistributeThread(hba.layers[i], hba.layers[i + 1]);

        hba.UpdateNextLayerState(i);
      }
      GlobalBA(hba.layers[total_layer_num - 1]);
      pose_hba = hba.PoseGraphOptimization();
      {
        for (int i = 0; i < pl_fulls.size(); i++) {
          // if (cloud_vec.size() > 800 && i > 100 && i < cloud_vec.size() - 100) {
          //   continue;
          // }
          mvt::protocol::KeyFrame keyframe2;
          keyframe2.set_id(i);
          keyframe2.mutable_header()->set_stamp(int64_t(i * 1e9));
          keyframe2.mutable_header()->set_name("hba_frame"+to_string(j));
          keyframe2.mutable_header()->set_time(PATC_GetTime);
          auto pos2 = keyframe2.mutable_pose()->mutable_position();
          auto opt_pose = pose_hba[i];
          pos2->set_x(opt_pose.t[0]);
          pos2->set_y(opt_pose.t[1]);
          pos2->set_z(opt_pose.t[2]);
          auto q2 = opt_pose.q;
          keyframe2.mutable_pose()->mutable_orientation()->set_x(q2.x());
          keyframe2.mutable_pose()->mutable_orientation()->set_y(q2.y());
          keyframe2.mutable_pose()->mutable_orientation()->set_z(q2.z());
          keyframe2.mutable_pose()->mutable_orientation()->set_w(q2.w());
          auto pc2 = keyframe2.mutable_pc();
          FillPointCloudMsg(pl_fulls[i], pc2);
          PATC_Pub(keyframe2);
        }
      }
    }
    printf("iteration complete\n");

  } else {
    ct = true;
    if (ct) cout<<"-------------start global ba---------------"<<endl;
    OctoTreeMap global_surf_map;
    win_size = x_ba.size();
    if (ct) cout<<"-------------CutVoxel---------------"<<endl;
    for (int i = 0; i < x_ba.size(); ++i) {
      CutVoxel(global_surf_map, *pl_fulls[i], x_ba[i], i, win_size);
    }
   if (ct)  cout<<"-------------Recut---------------"<<endl;

    for (const auto &[id, voxel] : global_surf_map) {
      voxel->Recut(win_size);
    }
    if (ct) cout<<"-------------TrasOpt---------------"<<endl;
    VoxHess voxhess(win_size);
    for (const auto &[id, voxel] : global_surf_map) {
      voxel->TrasOpt(voxhess, win_size);
    }
    if (ct) cout<<"-------------RemoveOutlier---------------"<<endl;
    VoxOptimizer opt_lsv(win_size, true);
    opt_lsv.RemoveOutlier(x_ba, voxhess, 0.5);
    PLV(6) hess_vec;
    // std::cout << "DampingIter " << loop << std::endl;
    cout<<"-------------DampingIter---------------"<<endl;
    opt_lsv.DampingIter(x_ba, voxhess);
    cout<<"-------------data_show---------------"<<endl;
    data_show(x_ba,pl_fulls);
  }
  return 0;
}