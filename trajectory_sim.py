import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import minimize

def load_path(filename='frieren_path.npy'):
    """Loads reference path coordinates from a NumPy file."""
    path_data = np.load(filename)
    return path_data[:, 0], path_data[:, 1]

def run_simulation(gains, x_ref, y_ref, dt, sigma_process, sigma_sensor):
    """
    Runs the 2D trajectory tracking simulation with Kalman Filter and PID control.
    gains: tuple or list of (Kp, Ki, Kd)
    """
    Kp, Ki, Kd = gains
    N = len(x_ref)
    
    # Sliding window for Integral term (prevents windup)
    window_size = 50
    error_window = deque(maxlen=window_size)
    
    # State-Space Matrices (SUVAT kinematics)
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    B = np.array([
        [0.5 * dt**2, 0.0        ],
        [0.0,         0.5 * dt**2],
        [dt,          0.0        ],
        [0.0,         dt         ]
    ])
    
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    Q = np.eye(4) * (sigma_process**2)
    R = np.eye(2) * (sigma_sensor**2)
    
    # Initial state estimates & uncertainty
    x_est = np.array([x_ref[0], y_ref[0], 0.0, 0.0], dtype=float).reshape(4, 1)
    P = np.eye(4) * 1.0
    
    # True and Open-loop states
    x_true = np.array([x_ref[0], y_ref[0], 0.0, 0.0], dtype=float).reshape(4, 1)
    x_open = np.array([x_ref[0], y_ref[0], 0.0, 0.0], dtype=float).reshape(4, 1)
    
    # History logs
    true_states = []
    estimated_states = []
    noisy_measurements = []
    open_loop_states = []
    
    for k in range(N - 1):
        # 1. Target Position
        target_pos = np.array([[x_ref[k]], [y_ref[k]]])
        est_pos = H @ x_est
        pos_error = target_pos - est_pos
        
        # 2. Rolling Window Integral Error
        error_window.append(pos_error.copy())
        error_integral = np.sum(np.array(error_window), axis=0) * dt
        
        # 3. Target Velocity (Finite difference)
        if k < N - 2:
            target_vel = np.array([
                [ (x_ref[k+1] - x_ref[k]) / dt ],
                [ (y_ref[k+1] - y_ref[k]) / dt ]
            ])
        else:
            target_vel = np.array([[0.0], [0.0]])
            
        est_vel = x_est[2:]
        vel_error = target_vel - est_vel
        
        # 4. Target Acceleration Feedforward (Path curvature anticipation)
        if k < N - 3:
            target_accel = np.array([
                [ (x_ref[k+2] - 2*x_ref[k+1] + x_ref[k]) / (dt**2) ],
                [ (y_ref[k+2] - 2*y_ref[k+1] + y_ref[k]) / (dt**2) ]
            ])
        else:
            target_accel = np.zeros((2, 1))
            
        # 5. Total Control Command (Feedforward + PID Feedback)
        u = target_accel + (Kp * pos_error) + (Kd * vel_error) + (Ki * error_integral)
        
        # 6. Physics Updates
        process_noise = np.random.normal(0, sigma_process, size=(4, 1))
        x_true = F @ x_true + B @ u + process_noise
        x_open = F @ x_open + B @ u + process_noise
        
        sensor_noise = np.random.normal(0, sigma_sensor, size=(2, 1))
        z = (H @ x_true) + sensor_noise
        
        # 7. Kalman Filter: Predict & Update
        x_pred = F @ x_est + B @ u
        P_pred = F @ P @ F.T + Q
        
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x_est = x_pred + K @ (z - (H @ x_pred))
        P = (np.eye(4) - K @ H) @ P_pred
        
        # Save history
        true_states.append(x_true.copy())
        estimated_states.append(x_est.copy())
        noisy_measurements.append(z.copy())
        open_loop_states.append(x_open.copy())
        
    return (np.array(true_states).squeeze(axis=2),
            np.array(estimated_states).squeeze(axis=2),
            np.array(open_loop_states).squeeze(axis=2),
            np.array(noisy_measurements).squeeze(axis=2))

def optimize_gains(x_ref, y_ref, dt, sigma_process, sigma_sensor):
    """Uses Nelder-Mead optimization to find PID gains that minimize tracking RMSE."""
    def objective_function(params):
        Kp, Ki, Kd = params
        if Kp < 0 or Ki < 0 or Kd < 0:
            return 1e6  # Penalty for negative gains
        
        _, est_arr, _, _ = run_simulation((Kp, Ki, Kd), x_ref, y_ref, dt, sigma_process, sigma_sensor)
        ref_trimmed_x = x_ref[:len(est_arr)]
        ref_trimmed_y = y_ref[:len(est_arr)]
        
        rmse = np.sqrt(np.mean((est_arr[:, 0] - ref_trimmed_x)**2 + (est_arr[:, 1] - ref_trimmed_y)**2))
        return rmse

    print("Running scipy optimization for PID gains...")
    initial_guess = [40.0, 2.0, 15.0]
    result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 25, 'disp': True})
    print(f"Optimal Gains Found -> Kp: {result.x[0]:.2f}, Ki: {result.x[1]:.2f}, Kd: {result.x[2]:.2f}")
    return result.x

def main():
    dt = 0.02
    sigma_process = 10
    sigma_sensor = 5
    
    x_ref, y_ref = load_path('frieren_path.npy')
    
    optimal_gains = optimize_gains(x_ref, y_ref, dt, sigma_process, sigma_sensor)
    #optimal_gains = [40,2,15]
    print(f"Optimal gains: {optimal_gains}")
    # Run final simulation
    true_arr, est_arr, open_arr, meas_arr = run_simulation(optimal_gains, x_ref, y_ref, dt, sigma_process, sigma_sensor)
    
    # Calculate RMSE
    ref_trimmed_x = x_ref[:len(true_arr)]
    ref_trimmed_y = y_ref[:len(true_arr)]
    
    rmse_open = np.sqrt(np.mean((open_arr[:, 0] - ref_trimmed_x)**2 + (open_arr[:, 1] - ref_trimmed_y)**2))
    rmse_kalman = np.sqrt(np.mean((est_arr[:, 0] - ref_trimmed_x)**2 + (est_arr[:, 1] - ref_trimmed_y)**2))
    
    print(f"\n--- Performance Metrics ---")
    print(f"Open-Loop Tracking RMSE:  {rmse_open:.4f} metres")
    print(f"Kalman + PID Tracking RMSE: {rmse_kalman:.4f} metres")
    
    # Visualisation
    plt.figure(figsize=(10, 8), dpi=120)
    plt.plot(x_ref, y_ref, 'k--', linewidth=2.0, label='Reference Path (Fourier)')
    plt.plot(open_arr[:, 0], open_arr[:, 1], color='orange', linestyle='-', linewidth=1.5, alpha=0.7, label='Uncompensated Open-Loop')
    plt.plot(est_arr[:, 0], est_arr[:, 1], color='blue', linestyle='-', linewidth=2.0, label='Kalman + PID Estimated Path')
    plt.scatter(meas_arr[:, 0], meas_arr[:, 1], color='gray', s=5, alpha=0.5, label='Noisy Sensor Measurements (Z)', zorder=5)
    
    plt.title('Project 2A: 2D Autonomous Trajectory Tracking (Feedforward + Kalman)', fontsize=14, fontweight='bold')
    plt.xlabel('X Position (metres)', fontsize=12)
    plt.ylabel('Y Position (metres)', fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    
    plt.savefig('trajectory_tracking_results.png', bbox_inches='tight')
    print("Plot saved successfully as 'trajectory_tracking_results.png'.")
    plt.show()

if __name__ == '__main__':
    main()