#include "mpc_solver.hpp"
#include "common.hpp"
#include "Timer.hpp"
#include <iostream>
#include <thread>

int main() {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::STATE_DIM);
    Q.diagonal() << 1, 2, 100, 5, 500, 10;
    Eigen::MatrixXd F = Q;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(wl_mpc::CONTROL_DIM, wl_mpc::CONTROL_DIM);
    R(0, 0) = 25;  // wheel
    R(1, 1) = 10;   // hip

    int N = 100;
    double DT = 0.0025;
    float L = 0.2;
    wl_mpc::MPCSolver solver(Q, F, R, N, DT);
    solver.setConstraints(5.0, 15.0);
    solver.init();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::STATE_DIM);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::CONTROL_DIM);

// 定义矩阵A（6x6）
A << 0,          1,          0,          0,          0,          0,
     265.9556,   0,          0,          0,          80.6327,    0,
     0,          0,          0,          1,          0,          0,
     -25.4562,   0,          0,          0,          1.8637,     0,
     0,          0,          0,          0,          0,          1,
     156.6952,   0,          0,          0,          183.0614,   0;

// 定义矩阵B（6x2）
B << 0,          0,
     -15.1389,   13.8563,
     0,          0,
     2.1208,     -0.7158,
     0,          0,
     -4.2238,    16.8001;

    solver.updateModel(A, B);

    Eigen::VectorXd x_curr = Eigen::VectorXd::Zero(wl_mpc::STATE_DIM);
    Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(wl_mpc::STATE_DIM);

    x_curr << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // 当前状态示例值
    x_ref  << 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;   // 参考状态为全零

    Timer timer;
    for (int i = 0; i < 10; ++i) { // 运行1000次
        timer.reset();
        Eigen::VectorXd u = solver.solve(x_curr, x_ref);
        double cost = timer.elapsed("MPC求解");
        std::cout << "MPC输出: " << u.transpose() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1000Hz
    }
    return 0;
}
