import flax.struct as struct
import jax.numpy as jnp
import numpy as onp

@struct.dataclass
class dFilter:
    samplingRate: float
    cutoffFreq: float


@struct.dataclass
class LPFObject:
    """A 2-Pole Low-Pass Filter implementation.
    https://github.com/bitcraze/crazyflie-firmware/blob/master/src/utils/src/filter.c
    """
    input: float  # current input to the filter
    output: float  # current output of the filter
    a1: float  # first feedback coefficient
    a2: float  # second feedback coefficient
    b0: float  # feedforward coefficient (current input)
    b1: float  # first feedforward coefficient (first delay)
    b2: float  # second feedforward coefficient (second delay)
    delay_element_1: float  # first delay element
    delay_element_2: float  # second delay element
    dFilter: dFilter  # associated dFilter data for the cutoff frequency and sampling rate

    @classmethod
    def lpfInit(cls, samplingRate: float, cutoffFreq: float) -> "LPFObject":
        dfilter = dFilter(samplingRate, cutoffFreq)
        # Initialize filter coefficients
        fr = samplingRate / cutoffFreq
        ohm = jnp.tan(jnp.pi / fr)
        c = 1.0 + 2.0 * jnp.cos(jnp.pi / 4.0) * ohm + ohm**2
        b0 = ohm**2 / c
        b1 = 2.0 * b0
        b2 = b0
        a1 = 2.0 * (ohm**2 - 1.0) / c
        a2 = (1.0 - 2.0 * jnp.cos(jnp.pi / 4.0) * ohm + ohm**2) / c
        return cls(
            input=0.0,
            output=0.0,
            a1=a1,
            a2=a2,
            b0=b0,
            b1=b1,
            b2=b2,
            delay_element_1=0.0,
            delay_element_2=0.0,
            dFilter=dfilter
        )

    def lpfReset(self) -> "LPFObject":
        return self.replace(
            input=0.0,
            output=0.0,
            delay_element_1=0.0,
            delay_element_2=0.0,
        )

    def lpfUpdate(self, new_input: float) -> "LPFObject":
        delay_element_0 = (new_input - self.delay_element_1 * self.a1 -
                           self.delay_element_2 * self.a2)
        output = (delay_element_0 * self.b0 +
                  self.delay_element_1 * self.b1 +
                  self.delay_element_2 * self.b2)

        return self.replace(
            input=new_input,
            output=output,
            delay_element_2=self.delay_element_1,
            delay_element_1=delay_element_0
        )
