def sensor_dynamics(thet, thet_s, thet_s_d, omega, zeta):

    return omega**2 * (thet - thet_s) - 2 * zeta * omega * thet_s_d