#include "mpc_solver.hpp"
#include "common.hpp"
#include "Timer.hpp"
#include <iostream>
#include <thread>
#include <cmath>

int main() {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::STATE_DIM);
    Q.diagonal() << 1, 2, 100, 5, 500, 10;
    Eigen::MatrixXd F = Q;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(wl_mpc::CONTROL_DIM, wl_mpc::CONTROL_DIM);
    R(0, 0) = 25;  
    R(1, 1) = 10;   

    int N = 30;
    double DT = 0.005; 
    
    wl_mpc::MPCSolver solver(Q, F, R, N, DT);
    solver.setConstraints(5.0, 15.0);
    solver.init(); // 必须调用 init

    // 模拟 LTV 系统：A 和 B 随时间变化
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::STATE_DIM);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(wl_mpc::STATE_DIM, wl_mpc::CONTROL_DIM);

    // 基础 A, B
    A << 0, 1, 0, 0, 0, 0,
         265, 0, 0, 0, 80, 0,
         0, 0, 0, 1, 0, 0,
         -25, 0, 0, 0, 1.8, 0,
         0, 0, 0, 0, 0, 1,
         156, 0, 0, 0, 183, 0;

    B << 0, 0,
         -15, 13,
         0, 0,
         2, -0.7,
         0, 0,
         -4, 16;

    Eigen::VectorXd x_curr = Eigen::VectorXd::Zero(wl_mpc::STATE_DIM);
    Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(wl_mpc::STATE_DIM);
    x_ref(1) = 1.0; // 目标速度

    Timer timer;
    double total_time = 0;
    int steps = 1000;

    std::cout << "Start LTV MPC Loop..." << std::endl;

    for (int i = 0; i < steps; ++i) {
        timer.reset();

        // 模拟 LTV：稍微改变 A 矩阵 (例如模拟摆杆角度变化导致的线性化误差)
        double angle_sim = 0.1 * std::sin(i * 0.1);
        A(1, 4) = 80.0 + angle_sim * 10.0; // 修改矩阵元素
        
        // 1. 更新模型 (耗时部分)
        solver.updateModel(A, B);

        // 2. 求解 (耗时部分)
        Eigen::VectorXd u = solver.solve(x_curr, x_ref);
        
        double cost = timer.elapsed(""); // 此时 elapsed 中不再打印，仅返回时间
        total_time += cost;
        
        // 简单的系统更新模拟 (x_next = x + (Ax+Bu)*dt)
        x_curr = x_curr + (A * x_curr + B * u) * DT;

        // std::cout << "Step " << i << " Cost: " << cost * 1000 << "ms | U: " << u.transpose() << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::cout << "Average Time per Step: " << (total_time / steps) * 1000 << " ms" << std::endl;
    std::cout << "Frequency: " << 1.0 / (total_time / steps) << " Hz" << std::endl;

    return 0;
}