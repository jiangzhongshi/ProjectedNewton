#include "projected_newton.hpp"

void main() {
  Xd V;
  Xi F;
  Xd uv_init;
  Eigen::VectorXi bnd;
  Xd bnd_uv;
  double mesh_area;

  igl::read_triangle_mesh(
      "/home/zhongshi/Workspace/Scaffold-Map/models/camel_b.obj", V, F);
  igl::boundary_loop(F, bnd);
  igl::map_vertices_to_circle(V, bnd, bnd_uv);
  igl::harmonic(V, F, bnd, bnd_uv, 1, uv_init);
  Vd dblarea;
  igl::doublearea(V, F, dblarea);
  dblarea *= 0.5;
  mesh_area = dblarea.sum();

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
  igl::writeOBJ("camel_after_slim.obj", uv3, F);
  for (int ii = 0; ii < 100; ii++) {
    spXd hessian;
    Vd grad;
    double e1 = get_grad_and_hessian(G, dblarea, cur_uv, grad, hessian);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(hessian);
    Xd newton = solver.solve(grad);
    if (solver.info() != Eigen::Success) {
      exit(1);
    }
    Xd dest_res = cur_uv - Eigen::Map<Xd>(newton.data(), V.rows(), 2);
    energy = igl::flip_avoiding_line_search(F, cur_uv, dest_res, compute_energy,
                                            energy * mesh_area) /
             mesh_area;
    std::cout << std::setprecision(25) << "Energy"
              << compute_energy(cur_uv) / mesh_area << std::endl;
  }
  uv3 *= 0;
  uv3.leftCols(2) = cur_uv;
  igl::writeOBJ("camel_after_newton.obj", uv3, F);
}

