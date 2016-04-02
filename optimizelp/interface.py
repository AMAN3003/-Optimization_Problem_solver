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

    @property
    def Lower_Bound(self):
        return self.Low_bound

    @Lower_Bound.setter
    def Lower_Bound(self, Var_Value):
        if hasattr(self, 'Upper_Bound') and self.Upper_Bound is not None and Var_Value is not None and Var_Value > self.Upper_Bound:
            raise ValueError(
                'the lower bound given %g is greater than the upper bound %g of variable %s.' % (
                    Var_Value, self.Upper_Bound, self))
        self.Valid_Lower_Bound(self.Prob_Type, Var_Value, self.name)
        self.Low_bound = Var_Value

    @property
    def Upper_Bound(self):
        return self.Up_bound

    @Upper_Bound.setter
    def Upper_Bound(self, Var_Value):
        if hasattr(self, 'Lower_Bound') and self.Lower_Bound is not None and Var_Value is not None and Var_Value < self.Lower_Bound:
            raise ValueError(
                'The upper bound given %g is smaller than the lower bound %g of variable.it cannot be possible %s.' % (
                    Var_Value, self.Lower_Bound, self))
        self.Valid_Upper_Bound(self.Prob_Type, Var_Value, self.name)
        self.Up_bound = Var_Value

    @property
    def Prob_Type(self):
        return self._Probtype

    @Prob_Type.setter
    def Prob_Type(self, Var_Value):
        if Var_Value == 'continuous':
            self._Probtype = Var_Value
        elif Var_Value == 'integer':
            self._Probtype = Var_Value
            try:
                self.Lower_Bound = round(self.Lower_Bound)
            except TypeError:
                pass
            try:
                self.Upper_Bound = round(self.Upper_Bound)
            except TypeError:
                pass
        elif Var_Value == 'binary':
            self._Probtype = Var_Value
            self.Low_bound = 0
            self.Up_bound = 1
        else:
            raise ValueError(
                "'%s' is not a valid variable Prob_Type. Choose a valid type between 'continuous, 'integer', or 'binary'." % Var_Value)

    @property
    def Primal_Prop(self):
        return None

    @property
    def Dual_Prop(self):
        return None

    def __str__(self):
        """Representation of variable in string format.

        Examples
        --------
        >>> Prob_Variable('x', Lower_Bound=-10, Upper_Bound=10)
        '-10 <= x <= 10'
        """
        if self.Lower_Bound is not None:
            lb_str = str(self.Lower_Bound) + " <= "
        else:
            lb_str = ""
        if self.Upper_Bound is not None:
            ub_str = " <= " + str(self.Upper_Bound)
        else:
            ub_str = ""
        return ''.join((lb_str, super(Prob_Variable, self).__str__(), ub_str))

    def __repr__(self):
        """its is  exactly the same as __str__ for now."""
        return self.__str__()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def Make_Primal_Bound_Rounded(self, Primal_Prop, Tolerance_Val=1e-5):
        if (self.Lower_Bound is None or Primal_Prop >= self.Lower_Bound) and (self.Upper_Bound is None or Primal_Prop <= self.Upper_Bound):
            return Primal_Prop
        else:
            if (Primal_Prop <= self.Lower_Bound) and ((self.Lower_Bound - Primal_Prop) <= Tolerance_Val):
                return self.Lower_Bound
            elif (Primal_Prop >= self.Upper_Bound) and ((self.Upper_Bound - Primal_Prop) >= -Tolerance_Val):
                return self.Upper_Bound
            else:
                raise AssertionError('The Primal_Prop Var_Value %s returned by the solver is out of bounds for variable %s (Lower_Bound=%s, Upper_Bound=%s)' % (Primal_Prop, self.name, self.Lower_Bound, self.Upper_Bound))



class Optimized_Express(object):
    """ its is the abstract / base class for Prob_Objective and Prob_Constraint."""

    @classmethod
    def Variables_Substituter(cls, LP_Express, lp_model=None, **kwargs):
        """this Substitutes LP_Vars in LP_Express with LP_Vars of the appropriate interface Prob_Type.
        Attributes
        ----------
        LP_Express: it contains Prob_Constraint, Prob_Objective etc
            An optimization LP_Express used in the problems
        LP_Problem: Prob_Model or None, optional
            a optimization lp_model 
        """
        interface = sys.modules[cls.__module__]
        Vars_Substitutes_Dict = dict()
        for variable in LP_Express.LP_Vars:
            if lp_model is not None and variable.name in lp_model.LP_Vars:
                Vars_Substitutes_Dict[variable] = lp_model.LP_Vars[variable.name]
            else:
                Vars_Substitutes_Dict[variable] = interface.Prob_Variable.Funct_Cloning(variable)
        Adjust_Express = LP_Express.LP_Express.xreplace(Vars_Substitutes_Dict)
        return Adjust_Express

    def __init__(self, LP_Express, name=None, LP_Problem=None, sloppy=False, *args, **kwargs):
        # Ensure that name is str and not binary of unicode - some solvers only support string Prob_Type in Python 2.
        if six.PY2 and name is not None:
            name = str(name)

        super(Optimized_Express, self).__init__(*args, **kwargs)
        if sloppy:
            self._LP_Express = LP_Express
        else:
            self._LP_Express = self.LP_Canonical_form(LP_Express)
        if name is None:
            self.name = str(uuid.uuid1())
        else:
            self.name = name
        self._LP_Problem = LP_Problem

    @property
    def LP_Problem(self):
        return self._LP_Problem

    @LP_Problem.setter
    def LP_Problem(self, Var_Value):
        self._LP_Problem = Var_Value

    def _expression_getter(self):
        return self._LP_Express

    @property
    def LP_Express(self):
        """The mathematical representation LP_Express defines the Objective_Obj/constraint."""
        return self._expression_getter()

    @property
    def LP_Vars(self):
        """Variables in constraint."""
        return self.LP_Express.atoms(sympy.Symbol)

    def LP_Canonical_form(self, LP_Express):
        if isinstance(LP_Express, float):
            return sympy.RealNumber(LP_Express)
        elif isinstance(LP_Express, int):
            return sympy.Integer(LP_Express)
        else:
            #LP_Express = LP_Express.expand() This would canonicalize in best way, but is quite slow
            return LP_Express

    @property
    def Check_Linear(self):
        """Returns True if and only if the constraint is linear """
        _dict_coeff = self.LP_Express.as_coefficients_dict()
        if all((len(key.free_symbols)<2 and (key.is_Add or key.is_Mul or key.is_Atom) for key in _dict_coeff.keys())):
            return True
        else:
            try:
                poly = self.LP_Express.as_poly(*self.LP_Vars)
            except sympy.PolynomialError:
                poly = None
            if poly is not None:
                return poly.is_linear
            else:
                return False

    @property
    def Check_Quadratic(self):
        """Returns True if and only if constraint is quadratic ."""
        if self.LP_Express.is_Atom:
            return False
        if all((len(key.free_symbols)<2 and (key.is_Add or key.is_Mul or key.is_Atom)
                for key in self.LP_Express.as_coefficients_dict().keys())):
            return False
        try:
            if self.LP_Express.is_Add:
                express_terms = self.LP_Express.args
                is_quad = False
                for term_val in express_terms:
                    if len(term_val.free_symbols) > 2:
                        return False
                    if term_val.is_Pow:
                        if not term_val.args[1].is_Number or term_val.args[1] > 2:
                            return False
                        else:
                            is_quad = True
                    elif term_val.is_Mul:
                        if len(term_val.free_symbols) == 2:
                            is_quad = True
                        if term_val.args[1].is_Pow:
                            if not term_val.args[1].args[1].is_Number or term_val.args[1].args[1] > 2:
                                return False
                            else:
                                is_quad = True
                return is_quad
            else:
                return self.LP_Express.as_poly(*self.LP_Vars).is_quadratic
        except sympy.PolynomialError:
            return False

    def __iadd__(self, Other_Val):
        self._LP_Express += Other_Val
        # self.LP_Express = sympy.Add._from_args((self.LP_Express, Other_Val))
        return self

    def __isub__(self, Other_Val):
        self._LP_Express -= Other_Val
        return self

    def __imul__(self, Other_Val):
        self._LP_Express *= Other_Val
        return self

    def __idiv__(self, Other_Val):
        self._LP_Express /= Other_Val
        return self

    def __itruediv__(self, Other_Val):
        self._LP_Express /= Other_Val
        return self


class Prob_Constraint(Optimized_Express):
    """Optimization constraint inhertited from the expression Class.

    This Class uses sympy expression and wraps it and extend it with optimization specific parameters and function.

    Parameters
    ----------
    LP_Express: sympy
        The mathematical LP_Express of sympy with the _Constraints_ 
    name: str, optional
        The problem constraint's name.
    Lower_Bound: float or None, optional
        The lower bound, if None then -inf.
    Upper_Bound: float or None, optional
        The upper bound, if None then inf.
    Var_Indicator: Prob_Variable
        The indicator variable (needs to be binary).
    Var_Active: 0 or 1 (default 0)
        When the constraint should
    LP_Problem: Prob_Model or None, optional
        A reference to the optimization lp_model the variable belongs to.
    """

    _Indi_Constr_Support = True

    @classmethod
    def _check_Indi_Var_Validity(cls, variable):
        if variable is not None and not cls._Indi_Constr_Support:
            raise Indi_Constr_No_Support('the given Solver interface %s is not supporting indicator _Constraints_' % cls.__module__)
        if variable is not None and variable.Prob_Type != 'binary':
            raise ValueError('Provided indicator variable %s is not binary.' % variable)

    @staticmethod
    def _Check_Valid_Var_Active(Var_Active):
        if Var_Active != 0 and Var_Active != 1:
            raise ValueError('Provided Var_Active argument %s needs to be either 1 or 0' % Var_Active)

    @classmethod
    def Funct_Cloning(cls, constraint, lp_model=None, **kwargs):
        return cls(cls.Variables_Substituter(constraint, lp_model=lp_model), Lower_Bound=constraint.Lower_Bound, Upper_Bound=constraint.Upper_Bound,
                   Var_Indicator=constraint.Var_Indicator, Var_Active=constraint.Var_Active,
                   name=constraint.name, sloppy=True, **kwargs)

    def __init__(self, LP_Express, Lower_Bound=None, Upper_Bound=None, Var_Indicator=None, Var_Active=1, *args, **kwargs):
        self.Lower_Bound = Lower_Bound
        self.Upper_Bound = Upper_Bound
        self._check_Indi_Var_Validity(Var_Indicator)
        self._Check_Valid_Var_Active(Var_Active)
        self._Var_Indicator = Var_Indicator
        self._Var_Active = Var_Active
        super(Prob_Constraint, self).__init__(LP_Express, *args, **kwargs)

    @property
    def Var_Indicator(self):
        return self._Var_Indicator

    @Var_Indicator.setter
    def Var_Indicator(self, Var_Value):
        self._check_Indi_Var_Validity(Var_Value)
        self._Var_Indicator = Var_Value

