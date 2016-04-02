import six

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import logging
log = logging.getLogger(__name__)

from .util import Available_Solver_lists

Solvers_Presents = Available_Solver_lists()