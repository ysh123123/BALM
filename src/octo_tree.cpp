#include "octo_tree.hpp"

OctoTreeNode::OctoTreeNode(const int _win_size) : win_size_(_win_size) {
  vec_orig.resize(win_size_);
  vec_tran.resize(win_size_);
  sig_orig.resize(win_size_);
  sig_tran.resize(win_size_);
  for (int i = 0; i < 8; i++) leaves_[i] = nullptr;  // 智能指针已自动管理，无需手动删除
  ref_ = 255.0 * rand() / (RAND_MAX + 1.0f);  // 随机数生成
  layer = 0;
}

OctoTreeNode::~OctoTreeNode() {
  // 智能指针会自动管理内存，无需手动删除
}

bool OctoTreeNode::JudgeEigen(const int win_count) {
  VOX_FACTOR covMat = fix_point_;
  for (int i = 0; i < win_count; i++) covMat += sig_tran[i];

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
  value_vector_ = saes.eigenvalues();
  center_ = covMat.v / covMat.N;
  direct_ = saes.eigenvectors().col(0);

  decision_ = saes.eigenvalues()[0] / saes.eigenvalues()[1];

  if (saes.info() != Eigen::Success) {  // 计算特征值失败时返回
    std::cerr << "Eigen decomposition failed!" << std::endl;
    return false;
  }

  return (decision_ < eigen_value_array[layer]) && ((saes.eigenvalues()[2] / saes.eigenvalues()[1]) < 5);
}
void OctoTreeNode::CutFunc(const int ci) {
  PLV(3) &pvec_orig = vec_orig[ci];
  PLV(3) &pvec_tran = vec_tran[ci];

  uint a_size = pvec_tran.size();
  for (uint j = 0; j < a_size; j++) {
    int xyz[3] = {0, 0, 0};
    for (uint k = 0; k < 3; k++)
      if (pvec_tran[j][k] > voxel_center[k]) xyz[k] = 1;
    const int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

    // 检查 leafnum 是否越界
    if (leafnum < 0 || leafnum >= 8) {
      std::cerr << "Leaf number out of bounds: " << leafnum << std::endl;
      continue;  // 跳过不合法的 leafnum
    }

    if (leaves_[leafnum] == nullptr) {
      leaves_[leafnum] = std::make_unique<OctoTreeNode>(win_size_);
      leaves_[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
      leaves_[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
      leaves_[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
      leaves_[leafnum]->quater_length = quater_length / 2;
      leaves_[leafnum]->layer = layer + 1;
    }

    leaves_[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
    leaves_[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);

    if (leaves_[leafnum]->octo_state != 1) {
      leaves_[leafnum]->sig_orig[ci].push(pvec_orig[j]);
      leaves_[leafnum]->sig_tran[ci].push(pvec_tran[j]);
    }
  }
  // 使用clear()而非swap来清空容器
  vec_orig[ci].clear();
  vec_tran[ci].clear();
}
void OctoTreeNode::Recut(const int win_count) {
  if (octo_state != 1) {
    int point_size = fix_point_.N;
    for (int i = 0; i < win_count; i++) point_size += sig_orig[i].N;

    push_state_ = 0;
    if (point_size <= min_ps) {
      return;
    }

    if (JudgeEigen(win_count)) {
      if (octo_state == 0 && point_size > layer_size[layer]) octo_state = 2;

      point_size -= fix_point_.N;
      if (point_size > min_ps) push_state_ = 1;
      return;
    } else if (layer == layer_limit) {
      octo_state = 2;
      return;
    }

    octo_state = 1;
    sig_orig.clear();  // 使用 clear() 替代 swap
    sig_tran.clear();
    for (int i = 0; i < win_count; i++) CutFunc(i);
  } else {
    if (win_count > 0) CutFunc(win_count - 1);
  }

  for (int i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) leaves_[i]->Recut(win_count);
  }
}
void OctoTreeNode::ToMargi(const int mg_size, vector<IMUST> &x_poses, const int win_count) {
  if (octo_state != 1) {
    if (!x_poses.empty())
      for (int i = 0; i < win_count; i++) {
        sig_tran[i].transform(sig_orig[i], x_poses[i]);
        plvec_trans(vec_orig[i], vec_tran[i], x_poses[i]);
      }

    if (fix_point_.N < 50 && push_state_ == 1)
      for (int i = 0; i < mg_size; i++) {
        fix_point_ += sig_tran[i];
        vec_fix_.insert(vec_fix_.end(), vec_tran[i].begin(), vec_tran[i].end());
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
  } else
    for (int i = 0; i < 8; i++)
      if (leaves_[i] != nullptr) leaves_[i]->ToMargi(mg_size, x_poses, win_count);
}


void OctoTreeNode::TrasDisplay(pcl::PointCloud<PointType> &pl_feat, const int win_count) {
  if (octo_state != 1) {
    if (push_state_ != 1) return;

    // ap.intensity = ref;

    // int tsize = 0;
    // for (int i = 0; i < win_count; i++) tsize += vec_tran[i].size();
    // if (tsize < 100) return;

    for (int i = 0; i < win_count; i++)
      for (Eigen::Vector3d pvec : vec_tran[i]) {
        PointType ap;
        ap.x = pvec.x();
        ap.y = pvec.y();
        ap.z = pvec.z();
        ap.intensity = ref_;
        // ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
        // ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
        // ap.normal_z = sqrt(value_vector[0]);
        // ap.normal_x = voxel_center[0];
        // ap.normal_y = voxel_center[1];
        // ap.normal_z = voxel_center[2];
        // ap.curvature = quater_length * 4;

        pl_feat.push_back(ap);
      }

  } else {
    // if(layer != layer_limit)
    // {
    //   PointType ap;
    //   ap.x = voxel_center[0];
    //   ap.y = voxel_center[1];
    //   ap.z = voxel_center[2];
    //   pl_cent.push_back(ap);
    // }

    for (int i = 0; i < 8; i++)
      if (leaves_[i] != nullptr) leaves_[i]->TrasDisplay(pl_feat, win_count);
  }
}

void OctoTreeNode::TrasOpt(VoxHess &vox_opt, const int win_count) const {
  if (octo_state != 1) {
    int points_size = 0;
    for (int i = 0; i < win_count; i++) points_size += sig_orig[i].N;
    if (points_size < min_ps) return;
    if (push_state_ == 1 && octo_state == 2) {
      vox_opt.PushVoxel(&sig_orig, &fix_point_);
    }
  } else {
    for (int i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) leaves_[i]->TrasOpt(vox_opt, win_count);
    }
  }
}

OctoTreeRoot::OctoTreeRoot(const int &_win_size) : OctoTreeNode(_win_size) {
  is2opt = true;
  each_num.resize(win_size_);
  for (int i = 0; i < win_size_; i++) each_num[i] = 0;
}

void OctoTreeRoot::Marginalize(const int &mg_size, vector<IMUST> &x_poses, const int &win_count) {
  ToMargi(mg_size, x_poses, win_count);

  int left_size = 0;
  for (int i = mg_size; i < win_count; i++) {
    each_num[i - mg_size] = each_num[i];
    left_size += each_num[i - mg_size];
  }

  if (left_size == 0) is2opt = false;

  for (int i = win_count - mg_size; i < win_count; i++) each_num[i] = 0;
}