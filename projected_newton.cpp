#include "projected_newton.hpp"

#include <iostream>
#include <igl/Timer.h>

namespace jakob {

#include "autodiff_jakob.h"
DECLARE_DIFFSCALAR_BASE();

double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                   Eigen::RowVector4d &local_grad,
                                   Eigen::Matrix4d &local_hessian) {
#ifdef NOHESSIAN
  using DScalar = DScalar1<double, Eigen::Vector4d>;
#else
  using DScalar = DScalar2<double, Eigen::Vector4d, Eigen::Matrix4d>;
#endif
  DiffScalarBase::setVariableCount(4);
  DScalar a(0, J(0));
  DScalar b(1, J(1));
  DScalar c(2, J(2));
  DScalar d(3, J(3));
  auto sd = symmetric_dirichlet_energy_t(a, b, c, d);

  local_grad = sd.getGradient();
#ifndef NOHESSIAN
  local_hessian = sd.getHessian();
#endif
  DiffScalarBase::setVariableCount(0);
  return sd.getValue();
}
}  // namespace jakob

namespace desai {
#include "desai_symmd.c"
double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                   Eigen::RowVector4d &local_grad,
                                   Eigen::Matrix4d &local_hessian) {
  double energy = symmetric_dirichlet_energy_t(J(0),J(1),J(2),J(3));
  double grad[4], hessian[10];
  reverse_diff(J.data(), 1, local_grad.data());
#ifndef NOHESSIAN
  reverse_hessian(J.data(), 1, local_hessian.data());
#endif
  return energy;
  }

Eigen::VectorXd gradient_and_hessian_from_J_vec(const Eigen::Matrix<double, -1, 4, Eigen::RowMajor> &J,
Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &grad,
Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &hessian) {
  reverse_diff(J.data(), J.rows(), grad.data());
#ifndef NOHESSIAN
  reverse_hessian(J.data(), J.rows(), hessian.data());
  return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3));
#endif
  return Eigen::VectorXd();
}
}  // namespace desai

double compute_energy_from_jacobian(const Xd &J, const Vd &area) {
  return 
  symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3)).dot(area) / area.sum();
}

extern long global_autodiff_time;
double grad_and_hessian_from_jacobian(const Vd &area, const Xd &jacobian,
                                      Xd &total_grad, spXd &hessian) {
  int f_num = area.rows();
  total_grad.resize(f_num, 4);
  total_grad.setZero();
  double energy = 0;
  hessian.resize(4 * f_num, 4 * f_num);
  std::vector<Eigen::Triplet<double>> IJV;
  IJV.reserve(16 * f_num);
  double total_area = area.sum();

  std::vector<Eigen::Matrix4d> all_hessian(f_num);
  igl::Timer timer;timer.start();
#ifndef AD_ENGINE
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> half_hessian(f_num,16);
  Eigen::Matrix<double, -1, -1, Eigen::RowMajor> local_grad(f_num, 4);
  Vd energy_vec = desai::gradient_and_hessian_from_J_vec(jacobian, local_grad, half_hessian);
#ifndef NOHESSIAN
  energy = energy_vec.dot(area) / total_area;
  total_grad = area.asDiagonal()*local_grad / total_area;
  half_hessian = area.asDiagonal()*half_hessian/total_area;
  for(int i=0; i<f_num; i++) {
      auto hessian = half_hessian.row(i);
      all_hessian[i] << hessian[0], hessian[1], hessian[2], hessian[3], 
      hessian[1], hessian[4], hessian[5], hessian[6], 
      hessian[2], hessian[5], hessian[7], hessian[8], 
      hessian[3], hessian[6], hessian[8], hessian[9];
  }
#endif
#else
  for (int i = 0; i < f_num; i++) {
    Eigen::RowVector4d J = jacobian.row(i);
    Eigen::Matrix4d local_hessian;
    Eigen::RowVector4d local_grad;
    energy += AD_ENGINE::gradient_and_hessian_from_J(J, local_grad, local_hessian);
    #ifndef NOHESSIAN
    local_grad *= area(i) / total_area;
    local_hessian *= area(i) / total_area;
    all_hessian[i] = local_hessian;
    total_grad.row(i) = local_grad;
    #endif
  }
#endif
global_autodiff_time = timer.getElapsedTimeInMicroSec();
  
#ifndef NOHESSIAN
  hessian.reserve(Eigen::VectorXi::Constant(4*f_num,4));
  for (int i = 0; i < f_num; i++) {
    Eigen::Matrix4d local_hessian = all_hessian[i];

    project_hessian(local_hessian);
    for (int v1 = 0; v1 < 4; v1++)
      for (int v2 = 0; v2 < v1+1; v2++)
        hessian.insert(v1 * f_num + i, v2 * f_num + i) = local_hessian(v1, v2);
  }
  hessian.makeCompressed();
#endif
   return energy;
}

void jacobian_from_uv(const spXd &G, const Xd &uv, Xd &Ji) {
  Vd altJ = G * Eigen::Map<const Vd>(uv.data(), uv.size());
  Ji = (Xd)Eigen::Map<Xd>(altJ.data(), G.rows() / 4, 4);
}

Vd vec(Xd &M2) {
  Vd v = Eigen::Map<Vd>(M2.data(), M2.size());
  return v;
}

double get_grad_and_hessian(const spXd &G, const Vd &area, const Xd &uv,
                            Vd &grad, spXd &hessian) {
  int f_num = area.rows();
  Xd Ji, total_grad;
  jacobian_from_uv(G, uv, Ji);
  double energy;
  energy = grad_and_hessian_from_jacobian(area, Ji, total_grad, hessian);

  Vd vec_grad = vec(total_grad);
  hessian = G.transpose() * hessian.selfadjointView<Eigen::Lower>() * G;
  grad = vec_grad.transpose() * G;

  return energy;
}
