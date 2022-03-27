"""Handle `argparse` options formatting."""


import argparse


def float_zero_one(v: float) -> float:
    """Check that the value is between zero and one.

    Parameters
    ----------
    v : float
        Value to test.

    Returns
    -------
    v : float
        Value if between zero and one.

    Raises
    ------
    `argparse.ArgumentTypeError`
        If the entry is not between zero and one.
    """
    if 1. >= v >= 0.:
        return v
    raise argparse.ArgumentTypeError("Argument must be between 0 and 1.")
