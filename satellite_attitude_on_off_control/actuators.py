def actuator_dynamics(f, u, FM, TC):

    return (FM * u - f) / TC

