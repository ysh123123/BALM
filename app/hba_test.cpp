#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
// #include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
// #include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <geometry_msgs/PoseArray.h>
// #include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <pcl_conversions/pcl_conversions.h>

#include "vox_optimizer.hpp"
#include "hba.hpp"
#include "tools.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  PATC_SetTheadName("main thread");

  const int total_layer_num = 3, thread_num = 16;
  string data_path = "/mnt/d/proto_files/HBA/wq3/";
  pcd_name_fill_num = 0;
  cloud_from_path = true;
  cout << "cloud_from_path = " << cloud_from_path << endl;
  std::vector<pcl::PointCloud<PointType>::Ptr> cloud_vec;
  auto pose_vec = read_pose(data_path + "pose.json");
  cout << "pose_vec = " << pose_vec.size() << endl;
  if (!cloud_from_path) {
    for (int i = 0; i < pose_vec.size(); ++i) {
      pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
      loadPCD(data_path, pcd_name_fill_num, pc, i, "pcd/");
      cloud_vec.emplace_back(pc);
    }
    cout << "cloud_vec = " << cloud_vec.size() << endl;
  }
  auto pose_hba = pose_vec;
  for (int j = 0; j < 1; ++j) {
    HBA hba;
    if (cloud_from_path) {
      HBA hba2(total_layer_num, data_path, thread_num, pose_hba);
      hba = hba2;
    } else {
      HBA hba2(total_layer_num, thread_num, pose_hba, cloud_vec);
      hba = hba2;
    }

    for (int i = 0; i < total_layer_num - 1; i++) {
      std::cout << "---------------------" << std::endl;
      // // 按层进行线程加速
      // cout<<"i = "<<i<<"| path = "<<hba.layers[i].data_path<<endl;
      DistributeThread(hba.layers[i], hba.layers[i + 1]);

      hba.UpdateNextLayerState(i);
    }
    GlobalBA(hba.layers[total_layer_num - 1]);
    pose_hba = hba.PoseGraphOptimization();
  }
  printf("iteration complete\n");
  {
    for (int i = 0; i < cloud_vec.size(); i++) {
      // if (cloud_vec.size() > 800 && i > 100 && i < cloud_vec.size() - 100) {
      //   continue;
      // }
      mvt::protocol::KeyFrame keyframe1, keyframe2;
      keyframe1.set_id(i);
      keyframe1.mutable_header()->set_stamp(int64_t(i * 1e9));
      keyframe1.mutable_header()->set_name("init_frame");
      keyframe1.mutable_header()->set_time(PATC_GetTime);
      auto pos1 = keyframe1.mutable_pose()->mutable_position();
      auto init_pose = pose_vec[i];
      pos1->set_x(init_pose.t[0]);
      pos1->set_y(init_pose.t[1]);
      pos1->set_z(init_pose.t[2]);
      auto q1 = init_pose.q;
      keyframe1.mutable_pose()->mutable_orientation()->set_x(q1.x());
      keyframe1.mutable_pose()->mutable_orientation()->set_y(q1.y());
      keyframe1.mutable_pose()->mutable_orientation()->set_z(q1.z());
      keyframe1.mutable_pose()->mutable_orientation()->set_w(q1.w());
      auto pc1 = keyframe1.mutable_pc();
      FillPointCloudMsg(cloud_vec[i], pc1);
      PATC_Pub(keyframe1);

      keyframe2.set_id(i);
      keyframe2.mutable_header()->set_stamp(int64_t(i * 1e9));
      keyframe2.mutable_header()->set_name("hba_frame");
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
      FillPointCloudMsg(cloud_vec[i], pc2);
      PATC_Pub(keyframe2);
    }
  }
}