
"""this is a Solver interface IBM ILOG CPLEX solver.

this will Wrap the GLPK solver by subclassing and extending :class:`Prob_Model`,
:class:`Prob_Variable`, and :class:`Prob_Constraint` from :mod:`interface`.
"""
import logging

import six
from six.moves import StringIO
import tempfile
import sympy
import sys
import collections
import cplex


from sympy.core.singleton import S

from sympy.core.mul import _unevaluated_Mul
log = logging.getLogger(__name__)
from sympy.core.add import _unevaluated_Add

from optimizelp import interface

Zero = S.Zero
One = S.One
