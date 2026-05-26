#
# dynamic_simulation
#

import control as ctrl
import time
from pathlib                import Path
import numpy                as np
from enum                   import Enum
import matplotlib.pyplot    as plt
import matplotlib.image     as mpimg
import pandas               as pd

if __name__ == '__main__':
    # Parameters
    theta_c = 0.09 # (rad) Command pitch angle
    delta = 0.008 # (rad) Dead zone half-width
    TD = 0.025 # (s) Time Delay
    TC = 0.050 # (s) Actuator Time Constant
    I = 100 # (kg m²) Spacecraft Inertia
    omega = 94.24778 # (rad/s) Sensor Natural Frequency
    zeta = 0.7 # Sensor Damping Ratio
    KR = 0.5 # (s) Rate feedback constant
    FM = 25 # (kg m² / s²) Actuator Torque
    FD = 12.1875 # (kg m² / s²) Disturbance Torque 

    # Problem Variables
    thet = 0 # (rad) Satellite Attitude Angle
    thet_d = 0 # (rad/s) Satellite Attidude Angle Rate
    that_s = 0 # (rad) Angle Sensor Output
    thet_p = 0 # (rad/s) Angle Sensor Output Derivative
    thet_e = 0 # (rad) Error
    thet_s_d = 0 # (rad/s) Rate Sensor Output
    thet_s_p = 0 # (rad/s²) Rate Sebsir Output Derivative
    up = 0 # Controler Switch Output
    u = 0 # Delayed Switch Output
    f = 0 # (kg m²/s²)

    s = ctrl.TransferFunction.s

    G = 1/s
    H1 = 1/s
    H2 = 1/s

    # malha interna
    inner = ctrl.feedback(G, H1)

    # malha externa
    system = ctrl.feedback(inner, H2)

    print(system)