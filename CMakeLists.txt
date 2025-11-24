cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_BUILD_TYPE Release)

# mpc_solver.cpp 为静态库
add_library(mpc_solver STATIC src/mpc_solver.cpp)
target_include_directories(mpc_solver PRIVATE include ${EIGEN3_INCLUDE_DIR} /usr/local/include OpenMP::OpenMP_CXX)
target_link_libraries(mpc_solver PRIVATE osqp pthread)

# main.cpp 作为可执行文件，链接 mpc_solver 库
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include ${EIGEN3_INCLUDE_DIR} /usr/local/include OpenMP::OpenMP_CXX)
target_link_libraries(main PRIVATE mpc_solver osqp pthread)