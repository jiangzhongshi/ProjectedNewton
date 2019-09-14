#ifndef PROJECTED_NEWTON_HPP
#define PROJECTED_NEWTON_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

using Xd = Eigen::MatrixXd;
using Vd = Eigen::VectorXd;
using Xi = Eigen::MatrixXi;
using spXd = Eigen::SparseMatrix<double>;

template <typename T>
T symmetric_dirichlet_energy_t(T a, T b, T c, T d) {
  auto det = a * d - b * c;
  auto frob2 = a * a + b * b + c * c + d * d;
  return frob2 * (1 + 1 / (det * det));
}


template <typename DerivedH>
void project_hessian(Eigen::MatrixBase<DerivedH> &local_hessian) {
  Eigen::SelfAdjointEigenSolver<DerivedH> es(local_hessian);
  Eigen::MatrixXd D = es.eigenvalues();
  Eigen::MatrixXd U = es.eigenvectors();
  for (int i = 0; i < D.rows(); i++) D(i) = (D(i) < 0) ? 0 : D(i);
  local_hessian = U * D.asDiagonal() * U.inverse();
}

double compute_energy_from_jacobian(const Eigen::MatrixXd &J, const Eigen::VectorXd &area);

double grad_and_hessian_from_jacobian(const Vd &area, const Xd &jacobian,
                                      Xd &total_grad, spXd &hessian);

void jacobian_from_uv(const spXd &G, const Xd &uv, Xd &Ji);

Vd vec(Xd &M2);

double get_grad_and_hessian(const spXd &G, const Vd &area, const Xd &uv,
                            Vd &grad, spXd &hessian);
#endif