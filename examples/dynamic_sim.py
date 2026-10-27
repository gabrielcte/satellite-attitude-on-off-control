# dynamic_simulation


import os

import numpy as np

from satellite_attitude_on_off_control.actuators import actuator_dynamics
from satellite_attitude_on_off_control.integrators import RK4
from satellite_attitude_on_off_control.non_linearities import (
    deadzone,
    hysteresis,
    saturation,
)
from satellite_attitude_on_off_control.plants import satellite_dynamics
from satellite_attitude_on_off_control.plots import (
    plot_phase_flow_satellite,
    pole_plot,
    time_plots,
)
from satellite_attitude_on_off_control.sensors import sensor_dynamics


def system_dynamics(t, x, u, FM, FD, I, TC, omega, zeta):
    """Derivadas dos estados: theta, theta_dot, theta_s, theta_s_dot e torque."""
    theta, theta_dot, theta_s, theta_s_dot, torque = x
    actuator = actuator_dynamics(torque, u, FM, TC)
    satellite = satellite_dynamics(torque, FD, I)
    sensor = sensor_dynamics(theta, theta_s, theta_s_dot, omega, zeta)
    return np.array([
        theta_dot,
        satellite,
        theta_s_dot,
        sensor,
        actuator,
    ], dtype=float)



if __name__ == "__main__":


    # PARAMETERS
    theta_c = 0.09             # (rad) Command pitch angle
    deadzone_threshold = 0.1
    delta_on =  0.08           # (rad) Hysteresis activation threshold
    delta_off = 0.04           # (rad) Hysteresis deactivation threshold
    TD = 0.025                 # (s) Time delay
    TC = 0.050                 # (s) Actuator time constant
    I = 100                    # (kg.m²) Spacecraft inertia
    omega = 94.24778           # (rad/s) Sensor natural frequency
    zeta = 0.7                 # Sensor damping ratio
    KR = 0.5                   # (s) Rate feedback constant
    FM = 25                    # (N.m) Actuator torque
    FD = -12.1875              # (N.m) Disturbance torque

    # Used by the phase-flow plotting function
    theta_c_global = theta_c

    # CREATE OUTPUT DIRECTORY
    os.makedirs("docs", exist_ok=True)

    # POLES OF THE LINEAR OPEN-LOOP MODEL
    # The poles are the actuator pole, the satellite integrator, and
    # the two sensor poles. This avoids requiring the python-control package.
    actuator_pole = -1.0 / TC
    satellite_pole = 0.0
    sensor_poles = np.roots([1.0, 2.0 * zeta * omega, omega**2])
    poles = np.concatenate(([actuator_pole, satellite_pole], sensor_poles))
    pole_plot(poles)


    wn = [abs(p) for p in poles]
    wn_max = max(wn)
    f_max = wn_max / (2 * np.pi)
    f_nyquist = 2 * f_max
    fs = 10 * f_nyquist
    dt = 1 / fs

    print("\nPoles:")
    print(poles)
    print("\nMaximum Frequency (rad/s):", wn_max)
    print("\nNyquist Frequency (Hz):", f_max)
    print("\nSampling Frequency (Hz):", fs)
    print("\nSimulation Step dt (s):", dt)


    # SIMULATION PARAMETERS
    tf = 10                  # (s) Final simulation time
    N = int(tf / dt)
    delay_steps = int(TD / dt)

    # STATES
    thet_rk = 0
    thet_d_rk = 0
    thet_s_rk = 0
    thet_s_d_rk = 0
    f_rk = 0
    # State vector for the coupled RK4 integration:
    # [theta, theta_dot, theta_sensor, theta_sensor_dot, actuator_torque]
    x_rk = np.array([
        thet_rk,
        thet_d_rk,
        thet_s_rk,
        thet_s_d_rk,
        f_rk,
    ], dtype=float)

    # CONTROL VARIABLES
    up_rk = 0
    u_rk = 0
    u_buffer_rk = [0] * delay_steps

    # STORAGE VECTORS
    time_hist = []
    thet_rk_hist = []
    thet_d_rk_hist = []
    thet_s_rk_hist = []
    u_rk_hist = []
    f_rk_hist = []

    # SIMULATION LOOP
    for k in range(N):
        t = k * dt

        # CONTROL
        thet_e_rk = theta_c - thet_s_rk - KR * thet_d_rk
        error_processed = deadzone(thet_e_rk, deadzone_threshold)
        up_rk = hysteresis(thet_e_rk, up_rk, delta_on, delta_off)
        up_rk = saturation(up_rk, u_min=-1, u_max=1)

        # TIME DELAY
        u_buffer_rk.append(up_rk)
        u_rk = u_buffer_rk.pop(0)

        # DYNAMICS AND RK4 INTEGRATION
        # The delayed control u_rk is held constant during this time step.
        x_rk = RK4(
            lambda t_stage, x_stage: system_dynamics(
                t_stage, x_stage, u_rk, FM, FD, I, TC, omega, zeta  # noqa: B023
            ),
            t,
            x_rk,
            dt,
        )

        # Recover the states from the integrated state vector.
        thet_rk = x_rk[0]
        thet_d_rk = x_rk[1]
        thet_s_rk = x_rk[2]
        thet_s_d_rk = x_rk[3]
        f_rk = x_rk[4]

        # SAVE DATA
        time_hist.append(t)
        thet_rk_hist.append(thet_rk)
        thet_d_rk_hist.append(thet_d_rk)
        thet_s_rk_hist.append(thet_s_rk)
        u_rk_hist.append(u_rk)
        f_rk_hist.append(f_rk)

    # ORIGINAL TIME PLOTS
    time_plots(time_hist, thet_rk_hist, thet_s_rk_hist, theta_c, thet_d_rk_hist, u_rk_hist, f_rk_hist)
    plot_phase_flow_satellite(thet_rk_hist, thet_d_rk_hist, theta_c,)