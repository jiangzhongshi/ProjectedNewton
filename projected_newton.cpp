#include "projected_newton.hpp"

#include <igl/local_basis.h>

namespace jakob {

#include "autodiff_jakob.h"
DECLARE_DIFFSCALAR_BASE();
using DScalar = DScalar2<double, Vd, Xd>;

double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                   Eigen::RowVector4d &local_grad,
                                   Eigen::Matrix4d &local_hessian) {
  DiffScalarBase::setVariableCount(4);
  DScalar a(0, J(0));
  DScalar b(1, J(1));
  DScalar c(2, J(2));
  DScalar d(3, J(3));
  auto sd = symmetric_dirichlet_energy_t(a, b, c, d);

  local_grad = sd.getGradient();
  local_hessian = sd.getHessian();
  DiffScalarBase::setVariableCount(0);
  return sd.getValue();
}
}  // namespace jakob

namespace desai {
#include "desai_symmd.c"
double gradient_and_hessian_from_J(const Eigen::RowVector4d &J,
                                   Eigen::RowVector4d &local_grad,
                                   Eigen::Matrix4d &local_hessian) {
  double values[4]={J(0),J(1),J(2),J(3)};
  double energy = symmetric_dirichlet_energy_t(J(0),J(1),J(2),J(3));
  double grad[4], hessian[10];
  reverse_diff(values, 1, grad);
  reverse_hessian(values, 1, hessian);
  local_grad << grad[0], grad[1], grad[2], grad[3];
  local_hessian << hessian[0], hessian[1], hessian[2], hessian[3], hessian[1], hessian[4], hessian[5], hessian[6], hessian[2], hessian[5], hessian[7], hessian[8], hessian[3], hessian[6], hessian[8], hessian[9];
  return energy;

  }
}  // namespace desai

double compute_energy_from_jacobian(const Xd &J, const Vd &area) {
  double e = 0;
  for (int i = 0; i < J.rows(); i++) {
    e += area(i) *
         symmetric_dirichlet_energy_t(J(i, 0), J(i, 1), J(i, 2), J(i, 3));
  }
  return e / area.sum();
}

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
  for (int i = 0; i < f_num; i++) {
    Eigen::RowVector4d J = jacobian.row(i);
    Eigen::Matrix4d local_hessian;
    Eigen::RowVector4d local_grad;
    energy += desai::gradient_and_hessian_from_J(J, local_grad, local_hessian);
    local_grad *= area(i) / total_area;
    local_hessian *= area(i) / total_area;

    total_grad.row(i) = local_grad;
    project_hessian(local_hessian);
    for (int v1 = 0; v1 < 4; v1++)
      for (int v2 = 0; v2 < 4; v2++)
        IJV.push_back(Eigen::Triplet<double>(v1 * f_num + i, v2 * f_num + i,
                                             local_hessian(v1, v2)));
  }
  hessian.setFromTriplets(IJV.begin(), IJV.end());
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

  double energy = grad_and_hessian_from_jacobian(area, Ji, total_grad, hessian);

  Vd vec_grad = vec(total_grad);
  hessian = G.transpose() * hessian * G;
  grad = vec_grad.transpose() * G;

  return energy;
}