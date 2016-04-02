import six

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import logging
log = logging.getLogger(__name__)

from .util import Available_Solver_lists

Solvers_Presents = Available_Solver_lists()

if Solvers_Presents['GLPK']:
    from .glpk_interface import Prob_Model, Prob_Variable, Prob_Constraint, Prob_Objective
elif Solvers_Presents['CPLEX']:
    from .cplex_interface import Prob_Model, Prob_Variable, Prob_Constraint, Prob_Objective
else:
    log.error('no solvers ar present in your system')

if Solvers_Presents['GLPK']:
    if six.PY3:
        from . import glpk_interface
    else:
        import glpk_interface
if Solvers_Presents['CPLEX']:
    if six.PY3:
        from . import cplex_interface
    else:
        import cplex_interface