#include "projected_newton.hpp"

#include <iostream>

#include <igl/flip_avoiding_line_search.h>
#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/read_triangle_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/matrix_to_list.h>
#include <igl/serialize.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <igl/local_basis.h>
#include <igl/grad.h>


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

#include <igl/slim.h>
Xd timing_slim(const Xd &V, const Xi &F, const Xd &uv) {
  igl::SLIMData data;
  Eigen::VectorXi b;
  Xd bc;
  igl::Timer timer;
  timer.start();
  igl::slim_precompute(V, F, uv, data,
                       igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b, bc, 0.);
  for (int i=0; i<100; i++){
   igl::slim_solve(data, 1);
   std::cout << "SLIM e="<<data.energy<<"\tTimer:"<<timer.getElapsedTime()<<std::endl;
  }
  return data.V_o;
}

int main(int argc, char* argv[]) {
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
  G = combine_Dx_Dy(Dx,Dy);
  auto cur_uv = uv_init;

  auto compute_energy = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
    Xd Ji;
    jacobian_from_uv(G, aaa, Ji);
    return compute_energy_from_jacobian(Ji, dblarea) * mesh_area;
  };

  double energy = compute_energy(cur_uv) / mesh_area;
  std::cout << "Start Energy" << energy << std::endl;
  auto uv3 = uv_init;
  uv3.conservativeResize(V.rows(), 3);
  igl::Timer timer;
  timer.start();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  for (int ii = 0; ii < 500; ii++) {
    spXd hessian;
    Vd grad;
    double e1 = get_grad_and_hessian(G, dblarea, cur_uv, grad, hessian);
    if (ii==0) solver.analyzePattern(hessian);
    solver.factorize(hessian);

    Xd newton = solver.solve(grad);
    // std::cout<<"newton"<<timer.getElapsedTime()-time0<<std::endl;
    if (solver.info() != Eigen::Success) {
      exit(1);
    }
    Xd dest_res = cur_uv - Eigen::Map<Xd>(newton.data(), V.rows(), 2);
    energy = igl::flip_avoiding_line_search(F, cur_uv, dest_res, compute_energy,
                                            energy * mesh_area) /
             mesh_area;
    std::cout << std::setprecision(25) << "Energy"
              << compute_energy(cur_uv) / mesh_area << "\tTimer"<<timer.getElapsedTime()<< std::endl;
  }
  uv3 *= 0;
  uv3.leftCols(2) = cur_uv;
}

