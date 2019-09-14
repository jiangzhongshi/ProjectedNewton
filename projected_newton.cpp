#include "projected_newton.hpp"

#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/matrix_to_list.h>
#include <igl/read_triangle_mesh.h>
#include <igl/serialize.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <igl/flip_avoiding_line_search.h>
#include <igl/local_basis.h>
#include <igl/grad.h>

#include "wenzel_autodiff.h"

DECLARE_DIFFSCALAR_BASE();

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
    DiffScalarBase::setVariableCount(4);
    Eigen::RowVector4d J = jacobian.row(i);

    auto sd = eval_energy(J) * area(i) / total_area;
    energy += sd.getValue();
    total_grad.row(i) = sd.getGradient();

    Eigen::Matrix4d local_hessian = sd.getHessian();
    project_hessian(local_hessian);
    for (int v1 = 0; v1 < 4; v1++)
      for (int v2 = 0; v2 < 4; v2++)
        IJV.push_back(Eigen::Triplet<double>(v1 * f_num + i, v2 * f_num + i,
                                             local_hessian(v1, v2)));

    DiffScalarBase::setVariableCount(0);
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

DScalar eval_energy(const Eigen::RowVector4d &J_rowvec) {
  DScalar a(0, J_rowvec(0));
  DScalar b(1, J_rowvec(1));
  DScalar c(2, J_rowvec(2));
  DScalar d(3, J_rowvec(3));
  return symmetric_dirichlet_energy_t(a, b, c, d);
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