#
# dynamic_simulation
#

import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

def Euler(x, x_d, dt):
    return x + x_d*dt

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
    FD = 12.1875            # (N.m) Disturbance torque

    # TRANSFER FUNCTIONS
    s = ctrl.TransferFunction.s

    G_act = FM/(TC*s + 1)
    G_sat = 1/(I*s)
    G_sensor = omega**2/(s**2 + 2*zeta*omega*s + omega**2)
    G_open = G_act*G_sat*G_sensor

    # SYSTEM FREQUENCY
    poles = ctrl.poles(G_open)

    wn = []

    for p in poles:
        wn.append(abs(p))

    wn_max = max(wn)
    f_max = wn_max/(2*np.pi)
    f_nyquist = 2*f_max
    fs = 20*f_max
    dt = 1/fs
    print('\nPoles:')
    print(poles)

    print('\nMaximum Frequency (rad/s):', wn_max)
    print('\nMaximum Frequency (Hz):', f_max)
    print('\nNyquist Frequency (Hz):', f_nyquist)
    print('\nSampling Frequency (Hz):', fs)
    print('\nSimulation Step dt (s):', dt)
    # SIMULATION PARAMETERS
    tf = 20                 # (s) Final simulation time
    N = int(tf/dt)          # Number of simulation steps
    delay_steps = int(TD/dt)

    # PROBLEM VARIABLES
    thet = 0                # (rad) Satellite attitude angle
    thet_d = 0              # (rad/s) Satellite attitude rate
    thet_dd = 0             # (rad/s²) Satellite attitude acceleration
    thet_s = 0              # (rad) Angle sensor output
    thet_s_d = 0            # (rad/s) Angle sensor output derivative
    thet_s_dd = 0           # (rad/s²) Angle sensor output second derivative
    thet_e = 0              # (rad) Control error
    up = 0                  # Controller switch output
    u = 0                   # Delayed switch output
    f = 0                   # (N.m) Actuator torque
    f_d = 0                 # (N.m/s) Actuator torque derivative

    # DELAY BUFFER
    u_buffer = [0]*delay_steps

    # STORAGE VECTORS
    time_hist = []
    thet_hist = []
    thet_d_hist = []
    thet_s_hist = []
    thet_e_hist = []
    u_hist = []
    f_hist = []

    # SIMULATION LOOP
    for k in range(N):
        t = k*dt

        # CONTROL ERROR
        thet_e = theta_c - thet_s - KR*thet_d

        # ON-OFF CONTROL
        if thet_e > delta:
            up = 1

        elif thet_e < -delta:
            up = -1

        else:
            up = 0

        # TIME DELAY
        u_buffer.append(up)
        u = u_buffer.pop(0)

        # DYNAMICS
        f_d = (FM*u - f)/TC
        thet_dd = (f - FD)/I
        thet_s_dd = (omega**2*(thet - thet_s) - 2*zeta*omega*thet_s_d)

        # INTEGRATION
        f = Euler(f, f_d, dt)
        thet_d = Euler(thet_d, thet_dd, dt)
        thet = Euler(thet, thet_d, dt)
        thet_s_d = Euler(thet_s_d, thet_s_dd, dt)
        thet_s = Euler(thet_s, thet_s_d, dt)

        # SAVE DATA
        time_hist.append(t)
        thet_hist.append(thet)
        thet_d_hist.append(thet_d)
        thet_s_hist.append(thet_s)
        thet_e_hist.append(thet_e)
        u_hist.append(u)
        f_hist.append(f)

    # PLOTS
    plt.figure(figsize=(12,10))

    # ANGLE
    plt.subplot(4,1,1)
    plt.plot(time_hist, thet_hist, label='thet')
    plt.plot(time_hist, thet_s_hist, label='thet_s')
    plt.axhline(theta_c, linestyle='--', label='theta_c')
    plt.ylabel('Angle (rad)')
    plt.grid()
    plt.legend()

    # RATE
    plt.subplot(4,1,2)
    plt.plot(time_hist, thet_d_hist)
    plt.ylabel('Rate (rad/s)')
    plt.grid()

    # CONTROL
    plt.subplot(4,1,3)
    plt.plot(time_hist, u_hist)
    plt.ylabel('Control')
    plt.grid()

    # TORQUE
    plt.subplot(4,1,4)
    plt.plot(time_hist, f_hist)
    plt.ylabel('Torque (N.m)')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.show()