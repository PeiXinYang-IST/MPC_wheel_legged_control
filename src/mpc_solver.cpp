#include "mpc_solver.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <omp.h> 

namespace wl_mpc {

MPCSolver::MPCSolver(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& F, 
                     const Eigen::MatrixXd& R, int N, double dt)
    : Q_(Q), F_(F), R_(R), N_(N), dt_(dt) {
    
    u_min_ << -100.0, -100.0;
    u_max_ << 100.0, 100.0;
    
    settings_ = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    if (settings_) {
        osqp_set_default_settings(settings_);
        settings_->verbose = 0; // 关闭打印以提高速度
        settings_->warm_starting = 1; 
        settings_->eps_abs = 1e-4; // 适当放宽精度以换取速度
        settings_->eps_rel = 1e-4;
        settings_->max_iter = 200; // 限制最大迭代次数
        settings_->polishing = 0;     // 关闭 polish
    }
}

MPCSolver::~MPCSolver() {
    if (solver_) osqp_cleanup(solver_);
    if (settings_) free(settings_);
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
    // 预分配大矩阵内存
    Q_bar_.resize(STATE_DIM * (N_ + 1), STATE_DIM * (N_ + 1));
    Q_bar_.setZero();
    for (int i = 0; i < N_; ++i) {
        Q_bar_.block(i * STATE_DIM, i * STATE_DIM, STATE_DIM, STATE_DIM) = Q_;
    }
    Q_bar_.block(N_ * STATE_DIM, N_ * STATE_DIM, STATE_DIM, STATE_DIM) = F_;

    R_bar_.resize(N_ * CONTROL_DIM, N_ * CONTROL_DIM);
    R_bar_.setZero();
    for (int i = 0; i < N_; ++i) {
        R_bar_.block(i * CONTROL_DIM, i * CONTROL_DIM, CONTROL_DIM, CONTROL_DIM) = R_;
    }

    // 初始化状态约束
    x_min_.resize(STATE_DIM);
    x_max_.resize(STATE_DIM);
    double inf_val = 1e10; 
    x_min_.setConstant(-inf_val);
    x_max_.setConstant(inf_val);

    // 硬编码约束 (LTV 也可以在这里修改，如果约束本身不变)
    x_min_(2) = -1.3;  x_max_(2) = 1.3;
    x_min_(4) = -0.87; x_max_(4) = 0.87;

    int n_vars = N_ * CONTROL_DIM;
    int n_cons = N_ * CONTROL_DIM + (N_ + 1) * STATE_DIM;

    q_vec_ = (OSQPFloat*)malloc(n_vars * sizeof(OSQPFloat));
    l_vec_ = (OSQPFloat*)malloc(n_cons * sizeof(OSQPFloat));
    u_vec_ = (OSQPFloat*)malloc(n_cons * sizeof(OSQPFloat));
    
    // 预分配矩阵缓冲区
    M_.resize(STATE_DIM * (N_ + 1), STATE_DIM);
    C_.resize(STATE_DIM * (N_ + 1), CONTROL_DIM * N_);
    A_cons_dense_buffer_.resize(n_cons, n_vars);

    is_initialized_ = true;
}

// 核心加速函数：仅更新 CSC 矩阵的数值部分
void MPCSolver::updateCscData(const Eigen::MatrixXd& mat, OSQPCscMatrix* csc_mat) {
    // 假设 mat 是 Dense 且结构稳定（上三角或满块），直接利用 SparseView 提取值
    // 使用 Eigen 压缩并复制值。由于是 LTV，A/B 变化会导致数值变化，但不会改变 Dense MPC 的非零结构
    Eigen::SparseMatrix<double> sparse_mat = mat.sparseView();
    sparse_mat.makeCompressed(); 
    // 直接内存拷贝，极快
    memcpy(csc_mat->x, sparse_mat.valuePtr(), sparse_mat.nonZeros() * sizeof(OSQPFloat));
}

void MPCSolver::updateModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (!is_initialized_) init();

    // 1. 更新离散模型
    Ad_ = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM) + A * dt_;
    Bd_ = B * dt_;
    
    // 2. 更新预测矩阵 M (递归，难以完全并行，保持串行)
    M_.setZero();
    M_.block(0, 0, STATE_DIM, STATE_DIM).setIdentity();
    for (int i = 1; i <= N_; ++i) {
        M_.block(i * STATE_DIM, 0, STATE_DIM, STATE_DIM).noalias() = 
            Ad_ * M_.block((i-1)*STATE_DIM, 0, STATE_DIM, STATE_DIM);
    }

    // 3. 更新预测矩阵 C (耗时操作，OpenMP 并行化)
    C_.setZero();
    
    // C 矩阵的计算具有一定的独立性，尤其是列块
    // 外层循环依赖 Ad，但内层填充可以并行
    // 注意：OpenMP 的效果取决于 N 的大小，N 较小时多线程开销可能大于收益
    for (int i = 1; i <= N_; ++i) {
        // C 的对角线下方块： C(i, i-1) = Bd
        C_.block(i * STATE_DIM, (i-1) * CONTROL_DIM, STATE_DIM, CONTROL_DIM) = Bd_;
        
        // C 的其余块： C(i, j) = Ad * C(i-1, j)
        // 使用 OpenMP 并行填充当前行的不同列块
        #pragma omp parallel for if(i > 5) 
        for (int j = 0; j < i - 1; ++j) {
             C_.block(i * STATE_DIM, j * CONTROL_DIM, STATE_DIM, CONTROL_DIM).noalias() = 
                Ad_ * C_.block((i-1) * STATE_DIM, j * CONTROL_DIM, STATE_DIM, CONTROL_DIM);
        }
    }

    // 4. 计算 QP 矩阵 (密集矩阵乘法，Eigen 内部可能已有优化，这里显式并行可能冲突，通常交给 Eigen)
    // 使用 noalias() 避免临时变量
    H_.noalias() = C_.transpose() * Q_bar_ * C_ + R_bar_;
    H_ = H_.triangularView<Eigen::Upper>(); 
    E_.noalias() = M_.transpose() * Q_bar_ * C_;
    G_.noalias() = Q_bar_ * C_;

    // 5. 准备约束矩阵 A_constraints
    int n_vars = N_ * CONTROL_DIM;
    int n_u_cons = N_ * CONTROL_DIM;
    int n_x_cons = (N_ + 1) * STATE_DIM;
    
    A_cons_dense_buffer_.setZero();
    A_cons_dense_buffer_.block(0, 0, n_u_cons, n_vars).setIdentity();
    A_cons_dense_buffer_.block(n_u_cons, 0, n_x_cons, n_vars) = C_;

    // 6. 更新 OSQP 矩阵结构
    if (!osqp_solver_created_) {
        // 第一次：创建结构
        freeCsc(P_csc_);
        freeCsc(A_constraints_csc_);
        castToCsc(H_, &P_csc_);
        castToCsc(A_cons_dense_buffer_, &A_constraints_csc_);
    } else {
        // 后续 LTV：原地更新数值 (Hot Update)
        updateCscData(H_, P_csc_);
        updateCscData(A_cons_dense_buffer_, A_constraints_csc_);
    }
}

Eigen::VectorXd MPCSolver::solve(const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_ref) {
    if (!is_initialized_) return Eigen::VectorXd::Zero(CONTROL_DIM);

    int n_vars = N_ * CONTROL_DIM;
    int n_u_cons = N_ * CONTROL_DIM;
    int n_x_cons = (N_ + 1) * STATE_DIM;
    int n_cons_total = n_u_cons + n_x_cons;

    // 1. 计算线性项 q (Vector math)
    Eigen::VectorXd Xd((N_ + 1) * STATE_DIM);
    for (int i = 0; i <= N_; ++i) Xd.block(i * STATE_DIM, 0, STATE_DIM, 1) = x_ref;
    
    Eigen::VectorXd q_eigen = E_.transpose() * x_curr - G_.transpose() * Xd;
    // 并行拷贝
    #pragma omp parallel for
    for(int i=0; i<n_vars; ++i) q_vec_[i] = q_eigen(i);

    // 2. 更新约束边界 l 和 u
    // 填充控制约束
    #pragma omp parallel for
    for (int i = 0; i < N_; ++i) {
        l_vec_[2*i] = u_min_(0); l_vec_[2*i+1] = u_min_(1);
        u_vec_[2*i] = u_max_(0); u_vec_[2*i+1] = u_max_(1);
    }

    // 填充状态约束 (减去初始状态影响 M*x0)
    Eigen::VectorXd Mx0 = M_ * x_curr;
    
    #pragma omp parallel for
    for (int i = 0; i <= N_; ++i) {
        for (int j = 0; j < STATE_DIM; ++j) {
            int idx_in_cons = n_u_cons + i * STATE_DIM + j;
            double offset = Mx0(i * STATE_DIM + j);
            l_vec_[idx_in_cons] = x_min_(j) - offset;
            u_vec_[idx_in_cons] = x_max_(j) - offset;
        }
    }

    // 3. 求解 (Hot Start)
    if (!osqp_solver_created_) {
        OSQPInt exitflag = osqp_setup(&solver_, P_csc_, q_vec_, A_constraints_csc_, 
                                    l_vec_, u_vec_, (OSQPInt)n_cons_total, (OSQPInt)n_vars, settings_);
        if (exitflag == 0) {
            osqp_solver_created_ = true;
            osqp_solve(solver_);
        }
    } else {
        // 极速更新：通知 OSQP 矩阵数值已变
        // P_csc_->x 和 A_constraints_csc_->x 已经在 updateModel 中被原地修改了
        osqp_update_data_mat(solver_, 
                             P_csc_->x, nullptr, 0, 
                             A_constraints_csc_->x, nullptr, 0);

        osqp_update_data_vec(solver_, q_vec_, nullptr, nullptr);
        osqp_update_data_vec(solver_, nullptr, l_vec_, u_vec_);
        
        osqp_solve(solver_);
    }

    Eigen::VectorXd U_opt(CONTROL_DIM);
    if (solver_ && (solver_->info->status_val == OSQP_SOLVED || 
                    solver_->info->status_val == OSQP_SOLVED_INACCURATE ||
                    solver_->info->status_val == OSQP_MAX_ITER_REACHED)) { // 接受 max iter 结果
        U_opt(0) = solver_->solution->x[0];
        U_opt(1) = solver_->solution->x[1];
    } else {
        U_opt.setZero();
    }
    return U_opt;
}

void MPCSolver::freeCsc(OSQPCscMatrix* mat) {
    if (mat) {
        if (mat->p) free(mat->p);
        if (mat->i) free(mat->i);
        if (mat->x) free(mat->x);
        free(mat);
    }
}

void MPCSolver::castToCsc(const Eigen::MatrixXd& mat, OSQPCscMatrix** csc_mat) {
    Eigen::SparseMatrix<double> sparse_mat = mat.sparseView();
    sparse_mat.makeCompressed();
    
    OSQPInt n_rows = sparse_mat.rows();
    OSQPInt n_cols = sparse_mat.cols();
    OSQPInt nnz = sparse_mat.nonZeros();

    *csc_mat = (OSQPCscMatrix*)malloc(sizeof(OSQPCscMatrix));
    (*csc_mat)->m = n_rows;
    (*csc_mat)->n = n_cols;
    (*csc_mat)->nz = -1;
    (*csc_mat)->nzmax = nnz;
    (*csc_mat)->p = (OSQPInt*)malloc((n_cols + 1) * sizeof(OSQPInt));
    (*csc_mat)->i = (OSQPInt*)malloc(nnz * sizeof(OSQPInt));
    (*csc_mat)->x = (OSQPFloat*)malloc(nnz * sizeof(OSQPFloat));

    const int* p_eigen = sparse_mat.outerIndexPtr();
    const int* i_eigen = sparse_mat.innerIndexPtr();
    const double* x_eigen = sparse_mat.valuePtr();

    // 手动拷贝以适应不同类型的 OSQPInt/Float 定义
    for(int k=0; k <= n_cols; k++) (*csc_mat)->p[k] = (OSQPInt)p_eigen[k];
    for(int k=0; k < nnz; k++)     (*csc_mat)->i[k] = (OSQPInt)i_eigen[k];
    for(int k=0; k < nnz; k++)     (*csc_mat)->x[k] = (OSQPFloat)x_eigen[k];
}

}