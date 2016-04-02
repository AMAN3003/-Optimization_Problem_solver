"""Abstract solver interface definitions (:class:`Prob_Model`, :class:`Prob_Variable`,
:class:`Prob_Constraint`, :class:`Prob_Objective`) intended to be subclassed and
extended for individual solvers.
"""
import inspect

import logging
import random
import uuid
import six

import collections

import sys
from optlang.exceptions import Indi_Constr_No_Support

log = logging.getLogger(__name__)

import sympy
from sympy.core.singleton import S
from sympy.core.logic import fuzzy_bool

from .container import Container

OPTIMAL_VAL = 'optimal'
UNDEFINED_VAL = 'undefined'
Feasible_Val = 'feasible'
InFeasible_Val = 'infeasible'
Notfeasible_Val = 'nofeasible'
Unbounded_Val = 'unbounded'
Infeasible_or_Unbounded_Val = 'infeasible_or_unbouned'
#till here
LOADED = 'loaded'
CUTOFF = 'cutoff'
ITERATION_LIMIT = 'iteration_limit'
Memory_Lim_Val = 'memory_limit'
Node_Lim_Val = 'node_limit'
Time_Limit_Val = 'time_limit'
SOLUTION_LIMIT = 'solution_limit'
INTERRUPTED = 'interrupted'
Numeric_Val = 'numeric'
Suboptimal_Val = 'suboptimal'
Inprogress_Val = 'in_progress'
Aborted_Val = 'aborted'
Special_Val = 'check_original_solver_status'
