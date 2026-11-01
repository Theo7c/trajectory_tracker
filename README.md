# trajectory_tracker
Discrete-time state-space simulation of a 2D mobile robot tracing parametric Fourier trajectories under Gaussian noise, featuring a custom Linear Kalman Filter and closed-loop PID controller with RMSE evaluation.
# 2D Autonomous Trajectory Tracker & State Estimator

A rigorous Python-based simulation framework implementing a discrete-time **Linear Kalman Filter** combined with a **Closed-Loop PID Controller** and **Path-Curvature Feedforward Acceleration** to execute autonomous 2D trajectory tracking along complex parametric paths (e.g., high-resolution vector outline extractions).

---

## 🚀 Key Features & Technical Highlights

* **State-Space Estimation (Linear Kalman Filter):** Manages sensor noise and process uncertainty using a $4 \times 1$ state vector $\mathbf{x} = [x, y, \dot{x}, \dot{y}]^T$ based on discrete SUVAT kinematic models.
* **Advanced PID Control with Rolling-Window Anti-Windup:** Utilizes a bounded queue (`collections.deque`) to implement a sliding-window integral term, completely eliminating runaway integral windup without memory bloat.
* **Feedforward Path Acceleration:** Anticipates path curvature by computing second-order finite differences from reference coordinates, reducing trajectory phase lag to near zero.
* **Automated Gain Optimization:** Integrates `scipy.optimize` (Nelder-Mead simplex algorithm) to automatically tune $K_p$, $K_i$, and $K_d$ gains for minimum Root-Mean-Square Error (RMSE).
* **Comparative Benchmarking:** Automatically contrasts uncompensated Open-Loop dead reckoning against the fully filtered Kalman + PID closed-loop controller.

---

## 📊 Mathematical Architecture

1. **Prediction Step (Time Update):**
   $$\mathbf{x}_{\text{pred}} = \mathbf{F} \mathbf{x}_{\text{est}} + \mathbf{B} \mathbf{u}$$
   $$\mathbf{P}_{\text{pred}} = \mathbf{F} \mathbf{P} \mathbf{F}^T + \mathbf{Q}$$

2. **Update Step (Measurement Update):**
   $$\mathbf{S} = \mathbf{H} \mathbf{P}_{\text{pred}} \mathbf{H}^T + \mathbf{R}$$
   $$\mathbf{K} = \mathbf{P}_{\text{pred}} \mathbf{H}^T \mathbf{S}^{-1}$$
   $$\mathbf{x}_{\text{est}} = \mathbf{x}_{\text{pred}} + \mathbf{K}(\mathbf{z} - \mathbf{H}\mathbf{x}_{\text{pred}})$$

3. **Control Input ($\mathbf{u}$):**
   $$\mathbf{u} = \mathbf{a}_{\text{target\_ff}} + K_p \mathbf{e}_{\text{pos}} + K_d \mathbf{e}_{\text{vel}} + K_i \int_{\text{window}} \mathbf{e}_{\text{pos}} dt$$

---

## 📂 Repository Structure

```text
├── tracker_sim.py            # Main simulation, Kalman filter, and plotting script
├── frieren_path.npy          # Reference trajectory coordinate array (X-Y points)
├── trajectory_tracking_results.png  # Generated visual comparison plot
└── README.md                 # Project documentation