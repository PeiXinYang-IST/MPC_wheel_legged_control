#pragma once

#include <Eigen/Dense>

namespace wl_mpc {
    constexpr int STATE_DIM = 6;
    // 输入: [T_wheel(轮力矩), T_hip(髋关节虚拟力矩)]
    constexpr int CONTROL_DIM = 2;

    struct RobotParams {
        double m_w = 0.5;    // 轮子质量
        double m_p = 5.0;    // 摆杆质量
        double M_b = 10.0;   // 机体质量 (上层机构)
        double R = 0.1;      // 轮半径
        double g = 9.81;     
        // 五连杆杆长 (参考文档图4)
        double l1 = 0.15, l2 = 0.15, l3 = 0.15, l4 = 0.15, l5 = 0.10; 
        
        double max_wheel_torque = 5.0; // Nm (根据实际电机设定)
        double max_hip_torque = 15.0;  // Nm (虚拟力矩上限)
    };
}