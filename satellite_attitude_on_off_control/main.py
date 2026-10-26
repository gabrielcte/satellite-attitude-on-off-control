# dynamic_simulation


import os

import matplotlib.pyplot as plt
import numpy as np


def RK4(f, t, x, dt):

    k1 = f(t, x)

    k2 = f(t + dt/2, x + dt*k1/2)

    k3 = f(t + dt/2, x + dt*k2/2)

    k4 = f(t + dt, x + dt*k3)


    return x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


def actuator_dynamics(f, u, FM, TC):

    return (FM * u - f) / TC


def satellite_dynamics(f, FD, I):

    return (f - FD) / I


def sensor_dynamics(thet, thet_s, thet_s_d, omega, zeta):

    return omega**2 * (thet - thet_s) - 2 * zeta * omega * thet_s_d


def system_dynamics(t, x, u, FM, FD, I, TC, omega, zeta):
    """Derivadas dos estados: theta, theta_dot, theta_s, theta_s_dot e torque."""
    theta, theta_dot, theta_s, theta_s_dot, torque = x

    return np.array([
        theta_dot,
        (torque - FD) / I,
        theta_s_dot,
        omega**2 * (theta - theta_s) - 2 * zeta * omega * theta_s_dot,
        (FM * u - torque) / TC,
    ], dtype=float)


def hysteresis(error, u_prev, delta_on, delta_off):

    if u_prev == 0:

        if error > delta_on:

            u = 1

        elif error < -delta_on:

            u = -1

        else:

            u = 0


    elif u_prev == 1:

        if error < -delta_on:

            u = -1

        elif error < delta_off:

            u = 0

        else:

            u = 1


    elif u_prev == -1:

        if error > delta_on:

            u = 1

        elif error > -delta_off:

            u = 0

        else:

            u = -1


    return u


def plot_phase_flow_satellite(

    ax,

    theta_range,

    rate_range,

    control_value,

    FM,

    FD,

    I,

    theta_hist,

    rate_hist,

):

    """

    Campo vetorial aproximado no plano (theta, theta_dot), para um

    comando de controle fixo. Supoe torque de atuador em regime:

    f = FM * control_value.

    """

    theta_grid = np.linspace(theta_range[0], theta_range[1], 21)

    rate_grid = np.linspace(rate_range[0], rate_range[1], 21)

    THETA, RATE = np.meshgrid(theta_grid, rate_grid)


    # No plano de fase:

    # theta_dot = rate

    # rate_dot = (f - FD) / I

    torque_equilibrium = FM * control_value

    THETA_D = RATE

    RATE_D = np.full_like(THETA, (torque_equilibrium - FD) / I)


    # Normalizar apenas para exibir a direcao das setas com clareza.

    magnitude = np.sqrt(THETA_D**2 + RATE_D**2)

    magnitude[magnitude == 0] = 1.0

    THETA_D_NORM = THETA_D / magnitude

    RATE_D_NORM = RATE_D / magnitude


    ax.quiver(

        THETA,

        RATE,

        THETA_D_NORM,

        RATE_D_NORM,

        angles="xy",

        width=0.002,

        headwidth=3.0,

        headlength=4.0,

        alpha=0.55,

        color="gray",

    )


    # Trajetoria real simulada, sobreposta ao campo aproximado.

    ax.plot(

        theta_hist,

        rate_hist,

        color="tab:blue",

        linewidth=1.6,

        label="Trajetoria simulada",

        zorder=3,

    )


    if len(theta_hist) > 0:

        ax.plot(

            theta_hist[0], rate_hist[0],

            marker="s", color="tab:green", markersize=5,

            linestyle="None", label="Inicio", zorder=4

        )

        ax.plot(

            theta_hist[-1], rate_hist[-1],

            marker="X", color="tab:red", markersize=6,

            linestyle="None", label="Fim", zorder=4

        )


    ax.axvline(

        theta_c_global,

        linestyle=":",

        linewidth=1.0,

        color="black",

        label=r"$\theta_c$",

    )

    ax.set_xlim(theta_range)

    ax.set_ylim(rate_range)

    ax.set_xlabel("Angulo (rad)")

    ax.set_ylabel("Velocidade angular (rad/s)")

    ax.set_title(f"Campo de fase para u = {control_value:+d}")

    ax.grid(True)

    ax.legend(fontsize=8, loc="best")


if __name__ == "__main__":


    # PARAMETERS

    theta_c = 0.09          # (rad) Command pitch angle

    delta_on =  0.008       # (rad) Hysteresis activation threshold

    delta_off = 0.004       # (rad) Hysteresis deactivation threshold

    TD = 0.025              # (s) Time delay

    TC = 0.050              # (s) Actuator time constant

    I = 100                  # (kg.m²) Spacecraft inertia

    omega = 94.24778         # (rad/s) Sensor natural frequency

    zeta = 0.7               # Sensor damping ratio

    KR = 0.5                 # (s) Rate feedback constant

    FM = 25                  # (N.m) Actuator torque

    FD = -12.1875            # (N.m) Disturbance torque


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

    plt.figure(figsize=(6, 5))
    plt.scatter(np.real(poles), np.imag(poles), marker="x")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginaria")
    plt.title("Polos do modelo linear em malha aberta")
    plt.grid(True)
    plt.tight_layout()

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

        up_rk = hysteresis(thet_e_rk, up_rk, delta_on, delta_off)

        up_rk = np.clip(up_rk, -1, 1)


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


    # POINCARE SECTION: upward crossings of theta_c

    poincare_rate = []

    poincare_torque = []


    for k in range(1, len(thet_rk_hist)):

        if (

            thet_rk_hist[k - 1] < theta_c

            and thet_rk_hist[k] >= theta_c

            and thet_d_rk_hist[k] > 0

        ):

            poincare_rate.append(thet_d_rk_hist[k])

            poincare_torque.append(f_rk_hist[k])


    # PHASE FLOW: trajectory and vector fields for fixed control values

    theta_min = min(thet_rk_hist)

    theta_max = max(thet_rk_hist)

    rate_min = min(thet_d_rk_hist)

    rate_max = max(thet_d_rk_hist)


    # Add a margin so the plotted trajectory does not touch the frame.

    theta_span = max(theta_max - theta_min, 1e-6)

    rate_span = max(rate_max - rate_min, 1e-6)

    theta_range = (theta_min - 0.05 * theta_span, theta_max + 0.05 * theta_span)

    rate_range = (rate_min - 0.05 * rate_span, rate_max + 0.05 * rate_span)


    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)


    for ax, control_value in zip(axes, [-1, 0, 1]):

        plot_phase_flow_satellite(

            ax=ax,

            theta_range=theta_range,

            rate_range=rate_range,

            control_value=control_value,

            FM=FM,

            FD=FD,

            I=I,

            theta_hist=thet_rk_hist,

            rate_hist=thet_d_rk_hist,

        )


    fig.suptitle("Trajetoria no espaco de fase e campos vetoriais aproximados")

    plt.tight_layout()

    plt.savefig("docs/phase_flow.png", dpi=300, bbox_inches="tight")

    print("\nGrafico salvo como 'docs/phase_flow.png'")


    # ORIGINAL TIME PLOTS

    plt.figure(figsize=(12, 10))


    # ANGLE

    plt.subplot(4, 1, 1)

    plt.plot(time_hist, thet_rk_hist, color="red", label="thet RK4")

    plt.plot(time_hist, thet_s_rk_hist, "--", color="red", label="thet_s RK4")

    plt.axhline(theta_c, linestyle=":", color="black", label="theta_c")

    plt.ylabel("Angle (rad)")

    plt.grid()

    plt.legend()


    # RATE

    plt.subplot(4, 1, 2)

    plt.plot(time_hist, thet_d_rk_hist, color="red", label="RK4")

    plt.ylabel("Rate (rad/s)")

    plt.grid()

    plt.legend()


    # CONTROL

    plt.subplot(4, 1, 3)

    plt.plot(time_hist, u_rk_hist, color="red", label="RK4")

    plt.ylabel("Control")

    plt.grid()

    plt.legend()


    # TORQUE

    plt.subplot(4, 1, 4)

    plt.plot(time_hist, f_rk_hist, color="red", label="RK4")

    plt.ylabel("Torque (N.m)")

    plt.xlabel("Time (s)")

    plt.grid()

    plt.legend()


    plt.tight_layout()

    plt.savefig(

        "docs/satellite_simulation_results.png",

        dpi=300,

        bbox_inches="tight",

    )

    print("\nGrafico salvo como 'docs/satellite_simulation_results.png'")


    plt.show()
