#include "projected_newton.hpp"

#include <iostream>

#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/grad.h>
#include <igl/harmonic.h>
#include <igl/local_basis.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/matrix_to_list.h>
#include <igl/read_triangle_mesh.h>
#include <igl/serialize.h>
#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <igl/Timer.h>

using Jtype = Eigen::Matrix<double, -1, 4, Eigen::RowMajor>;
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
  double energy = symmetric_dirichlet_energy_t(J(0), J(1), J(2), J(3));
  reverse_diff(J.data(), 1, local_grad.data());
#ifndef NOHESSIAN
  reverse_hessian(J.data(), 1, local_hessian.data());
#endif
  return energy;
}

Eigen::VectorXd gradient_and_hessian_from_J_vec(
    const Jtype &J,
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &grad,
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &hessian) {
  reverse_diff(J.data(), J.rows(), grad.data());
#ifndef NOHESSIAN
  reverse_hessian(J.data(), J.rows(), hessian.data());
#endif
  return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3));
}
}  // namespace desai

template <int type>
double grad_and_hessian_from_jacobian(const Vd &area, const Jtype &jacobian,
                                      Xd &total_grad, spXd &hessian) {
  double energy = 0;
  int f_num = area.rows();
  igl::Timer timer;
  timer.start();
  if constexpr (type == 0) {
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> hessian(f_num, 10);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> local_grad(f_num, 4);
    Vd energy_vec = desai::gradient_and_hessian_from_J_vec(jacobian, local_grad,
                                                           hessian);
  } else {
    for (int i = 0; i < f_num; i++) {
      Eigen::RowVector4d J = jacobian.row(i);
      Eigen::Matrix4d local_hessian;
      Eigen::RowVector4d local_grad;
      if constexpr (type == 1)
        energy +=
            desai::gradient_and_hessian_from_J(J, local_grad, local_hessian);
      else
        energy +=
            jakob::gradient_and_hessian_from_J(J, local_grad, local_hessian);
    }
  }
  std::cout << "AD Time" << timer.getElapsedTimeInMicroSec() << std::endl;

  return 0;
}

void prepare(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, spXd &Dx,
             spXd &Dy) {
  Eigen::MatrixXd F1, F2, F3;
  igl::local_basis(V, F, F1, F2, F3);
  Eigen::SparseMatrix<double> G;
  igl::grad(V, F, G);
  auto face_proj = [](Eigen::MatrixXd &F) {
    std::vector<Eigen::Triplet<double>> IJV;
    int f_num = F.rows();
    for (int i = 0; i < F.rows(); i++) {
      IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
      IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
      IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
    }
    Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
    P.setFromTriplets(IJV.begin(), IJV.end());
    return P;
  };

  Dx = face_proj(F1) * G;
  Dy = face_proj(F2) * G;
}

spXd combine_Dx_Dy(const spXd &Dx, const spXd &Dy) {
  // [Dx, 0; Dy, 0; 0, Dx; 0, Dy]
  spXd hstack = igl::cat(1, Dx, Dy);
  spXd empty(hstack.rows(), hstack.cols());
  // gruesom way for Kronecker product.
  return igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
}


void jacobian_from_uv(const spXd &G, const Xd &uv, Eigen::Matrix<double,-1,4,Eigen::RowMajor> &Ji) {
  Vd altJ = G * Eigen::Map<const Vd>(uv.data(), uv.size());
  Ji = (Xd)Eigen::Map<Xd>(altJ.data(), G.rows() / 4, 4);
}
int main(int argc, char *argv[]) {
  Xd V;
  Xi F;
  Xd uv_init;
  Eigen::VectorXi bnd;
  Xd bnd_uv;
  double mesh_area;

  igl::read_triangle_mesh(argv[1], V, F);
  igl::boundary_loop(F, bnd);
  igl::map_vertices_to_circle(V, bnd, bnd_uv);
  igl::harmonic(V, F, bnd, bnd_uv, 1, uv_init);
  Vd dblarea;
  igl::doublearea(V, F, dblarea);
  dblarea *= 0.5;
  mesh_area = dblarea.sum();

  // timing_slim(V,F,uv_init);

  spXd Dx, Dy, G;
  prepare(V, F, Dx, Dy);
  G = combine_Dx_Dy(Dx, Dy);
  auto cur_uv = uv_init;

  int f_num = dblarea.rows();
  spXd hessian;
  Xd total_grad;
  Jtype Ji;
  jacobian_from_uv(G, uv_init, Ji);
  double energy;
  constexpr int iteration = 10;
  std::cout<<"Vec"<<std::endl;
  for (int i = 0; i < iteration; i++) {
    energy =
        grad_and_hessian_from_jacobian<0>(dblarea, Ji, total_grad, hessian);
  }
  std::cout<<"Desai"<<std::endl;
  for (int i = 0; i < iteration; i++) {
    energy =
        grad_and_hessian_from_jacobian<1>(dblarea, Ji, total_grad, hessian);
  }
  std::cout<<"Jakob"<<std::endl;
  for (int i = 0; i < iteration; i++) {
    energy =
        grad_and_hessian_from_jacobian<2>(dblarea, Ji, total_grad, hessian);
  }
}
