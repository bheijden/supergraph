import flax.struct as struct
import jax.numpy as jnp
import numpy as onp


@struct.dataclass
class dFilter:
    samplingRate: float
    cutoffFreq: float


@struct.dataclass
class PidObject:
    """https://github.com/bitcraze/crazyflie-firmware/blob/22fb171c87b6fb78e6e524770d5dcc3544a97abd/src/modules/src/pid.c"""
    desired: float  # set point
    output: float  # previous output
    prevMeasured: float  # previous measurement
    prevError: float  # previous error
    integ: float  # integral
    deriv: float  # derivative
    kp: float  # proportional gain
    ki: float  # integral gain
    kd: float  # derivative gain
    kff: float  # feedforward gain
    outP: float  # proportional output (debugging)
    outI: float  # integral output (debugging)
    outD: float  # derivative output (debugging)
    outFF: float  # feedforward output (debugging)
    iLimit: float  # integral limit, absolute value. '0' means no limit.
    outputLimit: float  # total PID output limit, absolute value. '0' means no limit.
    dt: float  # delta-time dt
    dFilter: dFilter  # filter for D term
    enableDFilter: bool = struct.field(pytree_node=False) # filter for D term enable flag

    @classmethod
    def pidInit(cls, kp: float, ki: float, kd: float, outputLimit: float, iLimit: float,
                dt: float, samplingRate: float, cutoffFreq: float, enableDFilter: bool) -> "PidObject":
        dfilter = dFilter(samplingRate, cutoffFreq)
        return cls(
            desired=0.0,
            output=0.0,
            prevMeasured=0.0,
            prevError=0.0,
            integ=0.0,
            deriv=0.0,
            kp=kp,
            ki=ki,
            kd=kd,
            kff=0.0,
            outP=0.0,
            outI=0.0,
            outD=0.0,
            outFF=0.0,
            iLimit=iLimit,
            outputLimit=outputLimit,
            dt=dt,
            dFilter=dfilter,
            enableDFilter=enableDFilter
        )

    def pidReset(self) -> "PidObject":
        return self.replace(
            desired=0.0,
            output=0.0,
            prevMeasured=0.0,
            prevError=0.0,
            integ=0.0,
            deriv=0.0,
            kff=0.0,
            outP=0.0,
            outI=0.0,
            outD=0.0,
        )

    def pidUpdate(self, desired: float, measured: float) -> "PidObject":
        output = 0.0

        # Calculate error
        error = desired - measured

        # Proportional term
        outP = self.kp * error
        output = output + outP

        # Derivative term
        deriv = (error - self.prevError) / self.dt
        if self.enableDFilter:
            raise NotImplementedError("DFilter not implemented")
        deriv = jnp.nan_to_num(deriv, nan=0.0)
        outD = self.kd * deriv
        output = output + outD

        # Integral term
        integ = self.integ + error * self.dt
        integ_constrained = jnp.clip(integ, -self.iLimit, self.iLimit)
        integ = jnp.where(self.iLimit > 0, integ_constrained, integ)
        outI = self.ki * integ
        output = output + outI

        # constrain output
        output = jnp.nan_to_num(output, nan=0.0)
        output_constrained = jnp.clip(output, -self.outputLimit, self.outputLimit)
        output = jnp.where(self.outputLimit > 0, output_constrained, output)
        return self.replace(
            desired=desired,
            output=output,
            prevMeasured=measured,
            prevError=error,
            integ=integ,
            deriv=deriv,
            outP=outP,
            outI=outI,
            outD=outD,
            outFF=0.0
        )


if __name__ == "__main__":
    UINT16_MAX = 65535  # max value of uint16_t
    thrustScale = 1000
    thrustBase = 36000  #  Approximate throttle needed when in perfect hover. More weight/older battery can use a higher value
    thrustMin = 20000  # Minimum thrust value to output
    velMaxOverhead = 1.1
    zVelMax = 1.0  # m/s

    # z PID controller
    kp = 2.0
    ki = 0.5
    kd = 0.0
    rate = 100.
    dt = 1. / rate
    outputLimit = max(0.5, zVelMax) * velMaxOverhead
    pidZ = PidObject.pidInit(kp, ki, kd, outputLimit, 5000., dt, 100.0, 20.0, False)

    # vz PID controller
    kp = 25.0
    ki = 15.0
    kd = 0.0
    rate = 100.
    dt = 1. / rate
    outputLimit = UINT16_MAX / 2 / thrustScale
    pidVz = PidObject.pidInit(kp, ki, kd, 5000., outputLimit, dt, 100.0, 20.0, False)

    # Reset PID controllers
    pidZ = pidZ.pidReset()
    pidVz = pidVz.pidReset()

    # Run position controller
    z = 1.0   # Current
    vz = 1.0  # Current
    z_desired = 1.05
    pidZ = pidZ.pidUpdate(z_desired, z, True)
    vz_desired = pidZ.output

    # Run velocity controller
    pidVz = pidVz.pidUpdate(vz_desired, vz, True)
    thrustRaw = pidVz.output

    #Scale the thrust and add feed forward term
    thrust = thrustRaw * thrustScale + thrustBase
    thrust = jnp.clip(thrust, thrustMin, UINT16_MAX)


