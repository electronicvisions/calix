class CalibrationNotSuccessful(Exception):
    """
    Raised if a calibration fails to run successfully.

    This may happen if actions during the prelude fail, which may be
    a result of not satisfying the calibration's requirements.
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
