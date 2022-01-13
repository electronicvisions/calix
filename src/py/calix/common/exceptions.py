class CalibrationError(Exception):
    """
    Base class for Exceptions raised from Calibrations.
    """


class CalibrationNotSuccessful(CalibrationError):
    """
    Raised if a calibration fails to run successfully.

    This may happen if actions during the prelude fail, which may be
    a result of not satisfying the calibration's requirements.
    """


class CalibrationNotSupported(CalibrationError):
    """
    Raised if a calibration is not possible with the given algorithm
    or calibration routine.
    """


class TooFewSamplesError(CalibrationError):
    """
    The number of received MADC samples is significantly lower than
    expected. This is likely an FPGA-related problem. A longer wait
    between neurons can help to mitigate the issue.
    """


class AlgorithmError(Exception):
    """
    Base class for Exceptions raised from Algorithms.
    """


class ExcessiveNoiseError(AlgorithmError):
    """
    Raised if an algorithm that starts with noise is used with an
    excessive amount of noise requested on the initial parameters.

    If the amount of noise requested leads to hitting the boundaries
    of the dynamic range of the calibration within the first step,
    we raise this error as applying this amount of noise is not sensible.
    """
