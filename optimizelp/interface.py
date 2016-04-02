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



class Prob_Variable(sympy.Symbol):
    """Class for the Optimization LP_Vars to be implemented.

    this class is used to Extends the sympy with optimization specific parameters and methods.

    parameters
    ----------
    name: str
        The variable's name.
    Lower_Bound: float or None, optional
        The lower bound, if None then -inf.
    Upper_Bound: float or None, optional
        The upper bound, if None then inf.
    Prob_Type: str, optional
        DEFINES THE TYPE OF THE VARIABLE WHETHER 'continuous' or 'integer' or 'binary'.
    LP_Problem: Prob_Model or None, optional
        OPTIMIZATION MODEL REFRENCES

    Examples
    --------
    >>> Prob_Variable('x', Lower_Bound=-10, Upper_Bound=10)
    '-10 <= x <= 10'
    """

    @staticmethod
    def Valid_Lower_Bound(Prob_Type, Var_Value, name):
        if Var_Value is not None:
            if Prob_Type == 'integer' and Var_Value % 1 != 0.:
                raise ValueError(
                    'The provided lower bound %g cannot be assigned to integer variable %s (%g mod 1 != 0).' % (
                        Var_Value, name, Var_Value))
        if Prob_Type == 'binary' and (Var_Value is None or Var_Value != 0):
            raise ValueError(
                'The provided lower bound %s cannot be assigned to binary variable %s.' % (Var_Value, name))

    @staticmethod
    def Valid_Upper_Bound(Prob_Type, Var_Value, name):
        if Var_Value is not None:
            if Prob_Type == 'integer' and Var_Value % 1 != 0.:
                raise ValueError(
                    'The provided upper bound %s cannot be assigned to integer variable %s (%s mod 1 != 0).' % (
                        Var_Value, name, Var_Value))
        if Prob_Type == 'binary' and (Var_Value is None or Var_Value != 1):
            raise ValueError(
                'The provided upper bound %s cannot be assigned to binary variable %s.' % (Var_Value, name))

    @classmethod
    def Funct_Cloning(cls, variable, **kwargs):
        return cls(variable.name, Lower_Bound=variable.Lower_Bound, Upper_Bound=variable.Upper_Bound, Prob_Type=variable.Prob_Type, **kwargs)

    def __new__(cls, name, **assumptions):

        if assumptions.get('zero', False):
            return S.Zero
        check_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if check_commutative is None:
            raise ValueError(
                '''commutativity symbol must take values eighter a  True or False.''')
        assumptions['commutative'] = check_commutative
        for key in assumptions.keys():
            assumptions[key] = bool(assumptions[key])
        return sympy.Symbol.__xnew__(cls, name, uuid=str(int(round(1e16*random.random()))), **assumptions) 

    def __init__(self, name, Lower_Bound=None, Upper_Bound=None, Prob_Type="continuous", LP_Problem=None, *args, **kwargs):

        # make sure name is string and nit a binary codec as some solvers only support string Prob_Type in Python 2.
        if six.PY2:
            name = str(name)

        for char in name:
            if char.isspace():
                raise ValueError(
                    'Prob_Variable contains the white space its a error. "%s" contains whitespace character "%s".' % (
                        name, char))
        sympy.Symbol.__init__(name, *args, **kwargs)  
        self.Low_bound = Lower_Bound
        self.Up_bound = Upper_Bound
        if self.Low_bound is None and Prob_Type == 'binary':
            self.Low_bound = 0.
        if self.Up_bound is None and Prob_Type == 'binary':
            self.Up_bound = 1.
        self.Valid_Lower_Bound(Prob_Type, self.Low_bound, name)
        self.Valid_Upper_Bound(Prob_Type, self.Up_bound, name)
        self._Probtype = Prob_Type
        self.LP_Problem = LP_Problem
