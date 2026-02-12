"""
Helper for generating short, human-readable experiment IDs.
"""

from datetime import datetime
import random
import string


def make_exp_id(prefix: str = "mof") -> str:
    """
    Generate a new experiment identifier.

    The format is:
        <prefix>-YYYYMMDD-NNNN

    Example:
        mof-20260212-8342
    """
    date = datetime.now().strftime("%Y%m%d")
    suffix = "".join(random.choices(string.digits, k=4))
    return f"{prefix}-{date}-{suffix}"
