#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "common.hpp"
#include <osqp/osqp.h> 

namespace wl_mpc {

class MPCSolver {
public:
    MPCSolver(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& F, 
              const Eigen::MatrixXd& R, int N, double dt);
    ~MPCSolver();

    void init();
    // LTV 系统核心：每次控制循环调用，更新模型矩阵 A, B
    void updateModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
    
    void setConstraints(double max_wheel_T, double max_hip_T);
    
    // 求解函数 (极速版)
    Eigen::VectorXd solve(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_ref);

private:
    // 将 Eigen 矩阵转换为 OSQP CSC 格式 (首次分配内存)
    void castToCsc(const Eigen::MatrixXd& mat, OSQPCscMatrix** csc_mat);
    
    // 原地更新 CSC 矩阵的数值 (不分配内存，用于 LTV 加速)
    void updateCscData(const Eigen::MatrixXd& mat, OSQPCscMatrix* csc_mat);
    
    void freeCsc(OSQPCscMatrix* mat);

    Eigen::MatrixXd Q_, F_, R_;
    int N_;
    double dt_;
    Eigen::Vector2d u_min_, u_max_;
    Eigen::VectorXd x_min_, x_max_;

    // 预测模型矩阵 (预分配内存)
    Eigen::MatrixXd Ad_, Bd_;
    Eigen::MatrixXd E_, G_, H_; 
    Eigen::MatrixXd M_, C_; 
    Eigen::MatrixXd Q_bar_, R_bar_;
    
    // 临时缓冲区，避免栈溢出
    Eigen::MatrixXd A_cons_dense_buffer_; 

    OSQPSettings* settings_ = nullptr;
    OSQPSolver* solver_ = nullptr; 
    
    OSQPCscMatrix* P_csc_ = nullptr;
    OSQPCscMatrix* A_constraints_csc_ = nullptr;
    
    OSQPFloat* q_vec_ = nullptr;
    OSQPFloat* l_vec_ = nullptr;
    OSQPFloat* u_vec_ = nullptr;

    bool is_initialized_ = false;
    bool osqp_solver_created_ = false;
};

}