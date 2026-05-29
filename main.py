#
# dynamic_simulation
#

import control as ctrl
import numpy as np
import matplotlib.pyplot as plt


def Euler(x, x_d, dt):
    return x + x_d*dt


def RK4(x, x_d, dt):
    k1 = x_d
    k2 = x_d
    k3 = x_d
    k4 = x_d

    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def actuator_dynamics(f, u, FM, TC):
    return (FM*u - f)/TC


def satellite_dynamics(f, FD, I):
    return (f - FD)/I


def sensor_dynamics(thet, thet_s, thet_s_d, omega, zeta):
    return (omega**2*(thet - thet_s) - 2*zeta*omega*thet_s_d)


if __name__ == '__main__':

    # PARAMETERS
    theta_c = 0.09          # (rad) Command pitch angle
    delta = 0.008           # (rad) Dead zone half-width
    TD = 0.025              # (s) Time delay
    TC = 0.050              # (s) Actuator time constant
    I = 100                 # (kg.m²) Spacecraft inertia
    omega = 94.24778        # (rad/s) Sensor natural frequency
    zeta = 0.7              # Sensor damping ratio
    KR = 0.5                # (s) Rate feedback constant
    FM = 25                 # (N.m) Actuator torque
    FD = -12.1875           # (N.m) Disturbance torque

    # TRANSFER FUNCTIONS
    s = ctrl.TransferFunction.s
    G_act = FM/(TC*s + 1)
    G_sat = 1/(I*s)
    G_sensor = omega**2/(s**2 + 2*zeta*omega*s + omega**2)
    G_open = G_act*G_sat*G_sensor

    # SYSTEM FREQUENCY
    poles = ctrl.poles(G_open)
    wn = []
    ctrl.pzmap(G_open, grid=True)

    for p in poles:
        wn.append(abs(p))

    wn_max = max(wn)
    f_max = wn_max/(2*np.pi)
    f_nyquist = 2*f_max
    fs = 10*f_nyquist
    dt = 1/fs

    print('\nPoles:')
    print(poles)
    print('\nMaximum Frequency (rad/s):', wn_max)
    print('\nMaximum Frequency (Hz):', f_max)
    print('\nNyquist Frequency (Hz):', f_nyquist)
    print('\nSampling Frequency (Hz):', fs)
    print('\nSimulation Step dt (s):', dt)

    # SIMULATION PARAMETERS
    tf = 10                 # (s) Final simulation time
    N = int(tf/dt)          # Number of simulation steps
    delay_steps = int(TD/dt)

    # EULER STATES
    thet_eu = 0             # (rad) Satellite attitude angle
    thet_d_eu = 0           # (rad/s) Satellite attitude rate
    thet_s_eu = 0           # (rad) Angle sensor output
    thet_s_d_eu = 0         # (rad/s) Angle sensor derivative
    f_eu = 0                # (N.m) Actuator torque

    # RK4 STATES
    thet_rk = 0             # (rad) Satellite attitude angle
    thet_d_rk = 0           # (rad/s) Satellite attitude rate
    thet_s_rk = 0           # (rad) Angle sensor output
    thet_s_d_rk = 0         # (rad/s) Angle sensor derivative
    f_rk = 0                # (N.m) Actuator torque

    # CONTROL VARIABLES
    up_eu = 0               # Controller switch output
    up_rk = 0               # Controller switch output

    u_eu = 0                # Delayed switch output
    u_rk = 0                # Delayed switch output

    # DELAY BUFFER
    u_buffer_eu = [0]*delay_steps
    u_buffer_rk = [0]*delay_steps

    # STORAGE VECTORS
    time_hist = []

    thet_eu_hist = []
    thet_rk_hist = []

    thet_d_eu_hist = []
    thet_d_rk_hist = []

    thet_s_eu_hist = []
    thet_s_rk_hist = []

    u_eu_hist = []
    u_rk_hist = []

    f_eu_hist = []
    f_rk_hist = []


    # SIMULATION LOOP
    for k in range(N):

        t = k*dt

        # EULER CONTROL
        thet_e_eu = theta_c - thet_s_eu - KR*thet_d_eu
        if thet_e_eu > delta:
            up_eu = 1

        elif thet_e_eu < -delta:
            up_eu = -1

        else:
            up_eu = 0

        u_buffer_eu.append(up_eu)
        u_eu = u_buffer_eu.pop(0)

        # EULER DYNAMICS
        f_d_eu = actuator_dynamics(f_eu, u_eu, FM, TC)
        thet_dd_eu = satellite_dynamics(f_eu, FD, I)
        thet_s_dd_eu = sensor_dynamics(thet_eu,thet_s_eu,thet_s_d_eu, omega,zeta)

        # EULER INTEGRATION
        f_eu = Euler(f_eu, f_d_eu, dt)
        thet_d_eu = Euler(thet_d_eu, thet_dd_eu, dt)
        thet_eu = Euler(thet_eu, thet_d_eu, dt)
        thet_s_d_eu = Euler(thet_s_d_eu, thet_s_dd_eu, dt)
        thet_s_eu = Euler(thet_s_eu, thet_s_d_eu, dt)

        # RK4 CONTROL
        thet_e_rk = theta_c - thet_s_rk - KR*thet_d_rk
        if thet_e_rk > delta:
            up_rk = 1

        elif thet_e_rk < -delta:
            up_rk = -1

        else:
            up_rk = 0

        u_buffer_rk.append(up_rk)
        u_rk = u_buffer_rk.pop(0)

        # RK4 DYNAMICS
        f_d_rk = actuator_dynamics(f_rk, u_rk, FM, TC)
        thet_dd_rk = satellite_dynamics(f_rk, FD, I)
        thet_s_dd_rk = sensor_dynamics(thet_rk, thet_s_rk, thet_s_d_rk, omega, zeta)

        # RK4 INTEGRATION
        f_rk = RK4(f_rk, f_d_rk, dt)
        thet_d_rk = RK4(thet_d_rk, thet_dd_rk, dt)
        thet_rk = RK4(thet_rk, thet_d_rk, dt)
        thet_s_d_rk = RK4(thet_s_d_rk, thet_s_dd_rk, dt)
        thet_s_rk = RK4(thet_s_rk, thet_s_d_rk, dt)

        # SAVE DATA
        time_hist.append(t)

        thet_eu_hist.append(thet_eu)
        thet_rk_hist.append(thet_rk)

        thet_d_eu_hist.append(thet_d_eu)
        thet_d_rk_hist.append(thet_d_rk)

        thet_s_eu_hist.append(thet_s_eu)
        thet_s_rk_hist.append(thet_s_rk)

        u_eu_hist.append(u_eu)
        u_rk_hist.append(u_rk)

        f_eu_hist.append(f_eu)
        f_rk_hist.append(f_rk)

    # PLOTS
    plt.figure(figsize=(12,10))

    # ANGLE
    plt.subplot(4,1,1)
    plt.plot(time_hist, thet_eu_hist,color='blue', label='thet Euler')
    plt.plot(time_hist, thet_rk_hist,color='red', label='thet RK4')
    plt.plot(time_hist, thet_s_eu_hist, '--',color='blue', label='thet_s Euler')
    plt.plot(time_hist, thet_s_rk_hist, '--',color='red', label='thet_s RK4')
    plt.axhline(theta_c, linestyle=':',color='black', label='theta_c')
    plt.ylabel('Angle (rad)')
    plt.grid()
    plt.legend()

    # RATE
    plt.subplot(4,1,2)
    plt.plot(time_hist, thet_d_eu_hist,color='blue', label='Euler')
    plt.plot(time_hist, thet_d_rk_hist,color='red', label='RK4')
    plt.ylabel('Rate (rad/s)')
    plt.grid()
    plt.legend()

    # CONTROL
    plt.subplot(4,1,3)
    plt.plot(time_hist, u_eu_hist,color='blue', label='Euler')
    plt.plot(time_hist, u_rk_hist,color='red', label='RK4')
    plt.ylabel('Control')
    plt.grid()
    plt.legend()

    # TORQUE
    plt.subplot(4,1,4)
    plt.plot(time_hist, f_eu_hist,color='blue', label='Euler')
    plt.plot(time_hist, f_rk_hist,color='red', label='RK4')
    plt.ylabel('Torque (N.m)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.show()