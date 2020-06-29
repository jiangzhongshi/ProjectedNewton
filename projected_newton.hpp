#ifndef PROJECTED_NEWTON_HPP
#define PROJECTED_NEWTON_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <iostream>

using Xd = Eigen::MatrixXd;
using Vd = Eigen::VectorXd;
using Xi = Eigen::MatrixXi;
using spXd = Eigen::SparseMatrix<double>;

template <typename T>
T symmetric_dirichlet_energy_t(T a, T b, T c, T d)
{
  auto det = a * d - b * c;
  auto frob2 = a * a + b * b + c * c + d * d;
  return frob2 * (1 + 1 / (det * det));
}

template <typename Derived>
inline auto symmetric_dirichlet_energy(const Eigen::MatrixBase<Derived> &a,
                                       const Eigen::MatrixBase<Derived> &b, const Eigen::MatrixBase<Derived> &c, const Eigen::MatrixBase<Derived> &d)
{
  auto det = a.array() * d.array() - b.array() * c.array();
  auto frob2 = a.array().abs2() + b.array().abs2() + c.array().abs2() + d.array().abs2();
  return (frob2 * (1 + (det).abs2().inverse())).matrix();
}

template <typename DerivedH>
void project_hessian(Eigen::MatrixBase<DerivedH> &local_hessian)
{
  Eigen::LLT<DerivedH> llt;
  llt.compute(local_hessian);
  if (llt.info() == Eigen::Success)
  {
    return;
  };
  Eigen::SelfAdjointEigenSolver<DerivedH> es(local_hessian);
  Eigen::Matrix<typename DerivedH::Scalar, -1, 1> D = es.eigenvalues();
  DerivedH U = es.eigenvectors();
  bool clamped = false;
  for (int i = 0; i < D.size(); i++)
  {
    if (D(i) < 0)
    {
      D(i) = 0;
      clamped = true;
    }
  }
  if (clamped)
    local_hessian = U * D.asDiagonal() * U.transpose();
}

double compute_energy_from_jacobian(const Eigen::MatrixXd &J, const Eigen::VectorXd &area);

double grad_and_hessian_from_jacobian(const Vd &area, const Xd &jacobian,
                                      Xd &total_grad, spXd &hessian);

void jacobian_from_uv(const spXd &G, const Xd &uv, Xd &Ji);

Vd vec(Xd &M2);

double get_grad_and_hessian(const spXd &G, const Vd &area, const Xd &uv,
                            Vd &grad, spXd &hessian);

double grad_and_hessian_from_jacobian(const Vd &area, const Xd &jacobian,
                                      Xd &total_grad);

double wolfe_linesearch(
    const Eigen::MatrixXi F,
    Eigen::MatrixXd &cur_v,
    Eigen::MatrixXd &dst_v,
    std::function<double(Eigen::MatrixXd &)> energy,
    Eigen::VectorXd &grad0,
    double energy0);
#endif