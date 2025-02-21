#pragma once
#include "tools.hpp"
#include "vox_hess.hpp"
#include <memory>  // 添加头文件，用于智能指针

inline bool ct = false;

class OctoTreeNode {
public:
 explicit OctoTreeNode(const int _win_size);
 ~OctoTreeNode();

 bool JudgeEigen(const int win_count);
 void CutFunc(const int ci);

 void Recut(const int win_count);
 void ToMargi(const int mg_size, vector<IMUST> &x_poses, const int win_count);
 void TrasDisplay(pcl::PointCloud<PointType> &pl_feat, const int win_count);

 void TrasOpt(VoxHess &vox_opt, const int win_count) const;

public:
 vector<PLV(3)> vec_orig{}, vec_tran{};  // 用于存储原始和转换的点云
 vector<VOX_FACTOR> sig_orig{}, sig_tran{};  // 用于存储原始和转换的特征
 float voxel_center[3]{0, 0, 0};  // 体素中心
 int layer{0};  // 当前层级
 float quater_length{0};  // 每个体素的边长
 int octo_state{0};  // 体素状态，0: 未知，1: 中间节点，2: 平面节点

protected:
 int win_size_{10};  // 窗口大小

private:
 int push_state_{0};  // 判断当前体素是否可以加入BA进行优化
 VOX_FACTOR fix_point_{};  // 固定点
 PLV(3) vec_fix_{};  // 固定点的点云
 std::array<std::unique_ptr<OctoTreeNode>, 8> leaves_;  // 使用智能指针管理子节点
 Eigen::Vector3d center_{0., 0., 0.}, direct_{0., 0., 0.}, value_vector_{0., 0., 0.};  // 临时变量
 double decision_{0}, ref_{0};  // 体素决策信息
};

class OctoTreeRoot : public OctoTreeNode {
public:
 explicit OctoTreeRoot(const int &_win_size);
 void Marginalize(const int &mg_size, vector<IMUST> &x_poses, const int &win_count);

public:
 bool is2opt;
 vector<int> each_num;
};
