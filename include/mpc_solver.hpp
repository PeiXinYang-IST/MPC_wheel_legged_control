#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "common.hpp"
#include <osqp/osqp.h> // Include only the main header

namespace wl_mpc {

class MPCSolver {
public:
    MPCSolver(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& F, 
              const Eigen::MatrixXd& R, int N, double dt);
    ~MPCSolver();

    void init();
    void updateModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
    void setConstraints(double max_wheel_T, double max_hip_T);
    Eigen::VectorXd solve(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_ref);

private:
    // Helper to convert Eigen to OSQP CSC Matrix
    void castToCsc(const Eigen::MatrixXd& mat, OSQPCscMatrix** csc_mat);
    void freeCsc(OSQPCscMatrix* mat); // Helper to free CSC memory

    Eigen::MatrixXd Q_, F_, R_;
    int N_;
    double dt_;
    Eigen::Vector2d u_min_, u_max_;

    Eigen::MatrixXd Ad_, Bd_;
    Eigen::MatrixXd E_, G_, H_; 
    Eigen::MatrixXd M_, C_; 
    Eigen::MatrixXd Q_bar_, R_bar_;

    // OSQP v1.0 Structures
    OSQPSettings* settings_ = nullptr;
    OSQPSolver* solver_ = nullptr; // v1.0 的工作空间/求解器句柄
    
    // Matrix buffers
    OSQPCscMatrix* P_csc_ = nullptr;
    OSQPCscMatrix* A_constraints_csc_ = nullptr;
    
    OSQPFloat* q_vec_ = nullptr;
    OSQPFloat* l_vec_ = nullptr;
    OSQPFloat* u_vec_ = nullptr;

    bool is_initialized_ = false;
    bool osqp_solver_created_ = false;
};

}