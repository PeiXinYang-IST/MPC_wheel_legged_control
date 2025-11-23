#include "mpc_solver.hpp"
#include <iostream>
#include <vector>
#include <cstdlib> // For standard malloc/free
#include <cstring> // For memcpy

namespace wl_mpc {

MPCSolver::MPCSolver(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& F, 
                     const Eigen::MatrixXd& R, int N, double dt)
    : Q_(Q), F_(F), R_(R), N_(N), dt_(dt) {
    
    u_min_ << -100.0, -100.0;
    u_max_ << 100.0, 100.0;
    
    // Allocate settings (使用标准 malloc)
    settings_ = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    if (settings_) {
        osqp_set_default_settings(settings_);
        settings_->verbose = 0;
        // 修正: 'warm_start' -> 'warm_starting'
        settings_->warm_starting = 1; 
        settings_->eps_abs = 1e-4;
        settings_->eps_rel = 1e-4;
    }
}

MPCSolver::~MPCSolver() {
    // Cleanup OSQP v1.0
    if (solver_) osqp_cleanup(solver_);
    if (settings_) free(settings_);
    
    // Free matrices and vectors
    freeCsc(P_csc_);
    freeCsc(A_constraints_csc_);
    if (q_vec_) free(q_vec_);
    if (l_vec_) free(l_vec_);
    if (u_vec_) free(u_vec_);
}

void MPCSolver::setConstraints(double max_wheel_T, double max_hip_T) {
    u_max_ << max_wheel_T, max_hip_T;
    u_min_ << -max_wheel_T, -max_hip_T;
}

void MPCSolver::init() {
    // 1. Build Q_bar
    Q_bar_.resize(STATE_DIM * (N_ + 1), STATE_DIM * (N_ + 1));
    Q_bar_.setZero();
    for (int i = 0; i < N_; ++i) {
        Q_bar_.block(i * STATE_DIM, i * STATE_DIM, STATE_DIM, STATE_DIM) = Q_;
    }
    Q_bar_.block(N_ * STATE_DIM, N_ * STATE_DIM, STATE_DIM, STATE_DIM) = F_;

    // 2. Build R_bar
    R_bar_.resize(N_ * CONTROL_DIM, N_ * CONTROL_DIM);
    R_bar_.setZero();
    for (int i = 0; i < N_; ++i) {
        R_bar_.block(i * CONTROL_DIM, i * CONTROL_DIM, CONTROL_DIM, CONTROL_DIM) = R_;
    }

    // 3. Allocate vectors
    int n_vars = N_ * CONTROL_DIM;
    int n_cons = N_ * CONTROL_DIM;

    // 使用 OSQPFloat 进行分配
    q_vec_ = (OSQPFloat*)malloc(n_vars * sizeof(OSQPFloat));
    l_vec_ = (OSQPFloat*)malloc(n_cons * sizeof(OSQPFloat));
    u_vec_ = (OSQPFloat*)malloc(n_cons * sizeof(OSQPFloat));

    // 4. Build constraint matrix A (Identity, for input bounds)
    Eigen::MatrixXd A_cons_dense = Eigen::MatrixXd::Identity(n_cons, n_vars);
    castToCsc(A_cons_dense, &A_constraints_csc_);

    // 5. Fill bounds
    for (int i = 0; i < N_; ++i) {
        l_vec_[2*i]     = u_min_(0);
        l_vec_[2*i + 1] = u_min_(1);
        u_vec_[2*i]     = u_max_(0);
        u_vec_[2*i + 1] = u_max_(1);
    }

    is_initialized_ = true;
}

void MPCSolver::updateModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (!is_initialized_) init();

    Ad_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) + A * dt_;
    Bd_ = B * dt_;
    
    // M 矩阵
    M_.resize(STATE_DIM * (N_ + 1), STATE_DIM);
    M_.setZero();
    
    // C 矩阵
    C_.resize(STATE_DIM * (N_ + 1), CONTROL_DIM * N_);
    C_.setZero();

    Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    M_.block(0, 0, STATE_DIM, STATE_DIM) = temp;

    for (int i = 1; i <= N_; ++i) {
        M_.block(i * STATE_DIM, 0, STATE_DIM, STATE_DIM) = Ad_ * M_.block((i-1)*STATE_DIM, 0, STATE_DIM, STATE_DIM);
        for (int j = 0; j < i; ++j) {
            if (j == i - 1) {
                C_.block(i * STATE_DIM, j * CONTROL_DIM, STATE_DIM, CONTROL_DIM) = Bd_;
            } else {
                C_.block(i * STATE_DIM, j * CONTROL_DIM, STATE_DIM, CONTROL_DIM) = 
                    Ad_ * C_.block((i-1) * STATE_DIM, j * CONTROL_DIM, STATE_DIM, CONTROL_DIM);
            }
        }
    }

    H_ = C_.transpose() * Q_bar_ * C_ + R_bar_;
    // 保证H_为严格对称上三角
    H_ = H_.triangularView<Eigen::Upper>(); // 只保留上三角，下三角强制为0
    E_ = M_.transpose() * Q_bar_ * C_;
    G_ = Q_bar_ * C_;
}

Eigen::VectorXd MPCSolver::solve(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_ref) {
    if (!is_initialized_) return Eigen::VectorXd::Zero(CONTROL_DIM);

    int n_vars = N_ * CONTROL_DIM;
    int n_cons = N_ * CONTROL_DIM;

    // 1. Calculate q vector
    Eigen::VectorXd Xd((N_ + 1) * STATE_DIM);
    for (int i = 0; i <= N_; ++i) Xd.block(i * STATE_DIM, 0, STATE_DIM, 1) = x_ref;
    Eigen::VectorXd q_eigen = E_.transpose() * x_curr - G_.transpose() * Xd;
    for(int i=0; i<n_vars; ++i) q_vec_[i] = q_eigen(i);

    // 2. Convert H to CSC
    freeCsc(P_csc_);
    castToCsc(H_, &P_csc_);

    // 3. Solve
    if (!osqp_solver_created_) {
        // OSQP 1.0 初始化：显式指定维度类型为 OSQPInt
        OSQPInt exitflag = osqp_setup(&solver_, P_csc_, q_vec_, A_constraints_csc_, 
                                    l_vec_, u_vec_, (OSQPInt)n_cons, (OSQPInt)n_vars, settings_);
        
        if (exitflag == 0) {
            osqp_solver_created_ = true;
            osqp_solve(solver_);
        } else {
             std::cerr << "OSQP Setup Failed: " << exitflag << std::endl;
        }
    } else {
        // 1. 更新 P 矩阵（仅更新非零元素值，结构不变）
        osqp_update_data_mat(solver_, P_csc_->x, nullptr, 0, nullptr, nullptr, 0);
        // 2. 更新线性项 q 向量
        osqp_update_data_vec(solver_, q_vec_, nullptr, nullptr);
        // 3. 更新约束边界 l 和 u
        osqp_update_data_vec(solver_, nullptr, l_vec_, u_vec_);
        
        osqp_solve(solver_);
    }

    // 4. Retrieve solution（保持不变）
    Eigen::VectorXd U_opt(CONTROL_DIM);
    
    if (solver_ && (solver_->info->status_val == OSQP_SOLVED || solver_->info->status_val == OSQP_SOLVED_INACCURATE)) {
        U_opt(0) = solver_->solution->x[0];
        U_opt(1) = solver_->solution->x[1];
    } else {
        if (solver_) {
            std::cerr << "OSQP Failed with status: " << solver_->info->status_val << std::endl;
        } else {
            std::cerr << "OSQP Solver not created successfully." << std::endl;
        }
        U_opt.setZero();
    }

    return U_opt;
}

void MPCSolver::freeCsc(OSQPCscMatrix* mat) {
    if (mat) {
        // CSC 结构中的三个数组需要单独释放
        if (mat->p) free(mat->p);
        if (mat->i) free(mat->i);
        if (mat->x) free(mat->x);
        // 最后释放结构体本身
        free(mat);
    }
}

void MPCSolver::castToCsc(const Eigen::MatrixXd& mat, OSQPCscMatrix** csc_mat) {
    Eigen::SparseMatrix<double> sparse_mat = mat.sparseView();
    sparse_mat.makeCompressed();
    
    // 使用 OSQPInt 作为计数器类型
    OSQPInt n_rows = sparse_mat.rows();
    OSQPInt n_cols = sparse_mat.cols();
    OSQPInt nnz = sparse_mat.nonZeros();

    // 1. Allocate OSQPCscMatrix struct
    *csc_mat = (OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
    
    // 2. Allocate arrays (修正: 使用 OSQPInt 和 OSQPFloat)
    (*csc_mat)->m = n_rows;
    (*csc_mat)->n = n_cols;
    (*csc_mat)->nz = -1; // -1 for CSC format
    (*csc_mat)->nzmax = nnz;
    (*csc_mat)->p = (OSQPInt*)malloc((n_cols + 1) * sizeof(OSQPInt));
    (*csc_mat)->i = (OSQPInt*)malloc(nnz * sizeof(OSQPInt));
    (*csc_mat)->x = (OSQPFloat*)malloc(nnz * sizeof(OSQPFloat));

    // 3. Copy data
    // Eigen 的内部指针类型
    const int* p_eigen = sparse_mat.outerIndexPtr();
    const int* i_eigen = sparse_mat.innerIndexPtr();
    const double* x_eigen = sparse_mat.valuePtr();

    // 显式转换并复制数据
    for(int k=0; k <= n_cols; k++) (*csc_mat)->p[k] = (OSQPInt)p_eigen[k];
    for(int k=0; k < nnz; k++)     (*csc_mat)->i[k] = (OSQPInt)i_eigen[k];
    for(int k=0; k < nnz; k++)     (*csc_mat)->x[k] = (OSQPFloat)x_eigen[k];
}

}