cmake_minimum_required(VERSION 3.1)
project(projnewton)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# set(SEISM_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/external/)

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "" OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           OFF)
option(LIBIGL_WITH_TRIANGLE            "Use OpenGL"         OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)

find_package(LIBIGL QUIET)

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -flto")
  endif()
endif()

# add_executable(projnewton_bin main.cpp projected_newton.cpp)
# target_link_libraries(projnewton_bin igl::core)

add_executable(jakob_bin main.cpp projected_newton.cpp)
target_link_libraries(jakob_bin igl::core)
target_compile_definitions(jakob_bin PUBLIC AD_ENGINE=jakob)

# add_executable(desai_bin main.cpp projected_newton.cpp)
# target_link_libraries(desai_bin igl::core)
# target_compile_definitions(desai_bin PUBLIC AD_ENGINE=desai)


# add_executable(test_bin profile.cpp)
# target_link_libraries(test_bin igl::core)
# target_compile_features(test_bin PUBLIC cxx_std_17)