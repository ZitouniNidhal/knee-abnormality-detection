"""Tools for the RSNA Knee Abnormality Detection competition.

The package deliberately keeps report text out of the inference path: reports
are only a training-time source of weak labels, while test-time predictions
come from MRI series and permitted metadata.
"""

from .constants import TARGETS

__all__ = ["TARGETS"]
