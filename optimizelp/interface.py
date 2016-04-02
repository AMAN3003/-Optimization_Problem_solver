
#creator aman omkar
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
from optimizelp.exceptions import Indi_Constr_No_Support

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

    @property
    def Var_Active(self):
        return self._Var_Active

    @Var_Indicator.setter
    def Var_Indicator(self, Var_Value):
        self._Check_Valid_Var_Active(Var_Value)
        self._Var_Active = Var_Value

    def __str__(self):
        if self.Lower_Bound is not None:
            lhs = str(self.Lower_Bound) + ' <= '
        else:
            lhs = ''
        if self.Upper_Bound is not None:
            rhs = ' <= ' + str(self.Upper_Bound)
        else:
            rhs = ''
        if self.Var_Indicator is not None:
            lhs = self.Var_Indicator.name + ' = ' + str(self.Var_Active) + ' -> ' + lhs
        return str(self.name) + ": " + lhs + self.LP_Express.__str__() + rhs

    def LP_Canonical_form(self, LP_Express):
        LP_Express = super(Prob_Constraint, self).LP_Canonical_form(LP_Express)
        if LP_Express.is_Atom or LP_Express.is_Mul:
            return LP_Express
        Alone_Coeff = [arg for arg in LP_Express.args if arg.is_Number]
        if not Alone_Coeff:
            return LP_Express
        assert len(Alone_Coeff) == 1
        coeff = Alone_Coeff[0]
        if self.Lower_Bound is None and self.Upper_Bound is None:
            raise ValueError(
                "%s cannot be shaped into canonical form if you are not giving lower or upper constraint bounds."
                % LP_Express
            )
        else:
            LP_Express = LP_Express - coeff
            if self.Lower_Bound is not None:
                self.Lower_Bound = self.Lower_Bound - coeff
            if self.Upper_Bound is not None:
                self.Upper_Bound = self.Upper_Bound - coeff
        return LP_Express

    @property
    def Primal_Prop(self):
        return None

    @property
    def Dual_Prop(self):
        return None


class Prob_Objective(Optimized_Express):
    """Prob_Objective function. used to make the Objective_Obj function

    Attributes
    ----------
    LP_Express: sympy
        The mathematical LP_Express defining the Objective_Obj of the problems
    name: str, optional
        The name of the constraint which is being maked as objectives
    Max_Or_Min_type: 'max' or 'min'
        The optimization type whether max or min
    Var_Value: float, read-only
        The current Objective_Obj Var_Value.
    LP_Problem: solver
        The low-level solver object for the problems

    """

    @classmethod
    def Funct_Cloning(cls, Objective_Obj, lp_model=None, **kwargs):
        return cls(cls.Variables_Substituter(Objective_Obj, lp_model=lp_model), name=Objective_Obj.name,
                   Max_Or_Min_type=Objective_Obj.Max_Or_Min_type, sloppy=True, **kwargs)

    def __init__(self, LP_Express, Var_Value=None, Max_Or_Min_type='max', *args, **kwargs):
        self._value = Var_Value
        self._Max_Or_Min_type = Max_Or_Min_type
        super(Prob_Objective, self).__init__(LP_Express, *args, **kwargs)

    @property
    def Var_Value(self):
        return self._value

    def __str__(self):
        return {'max': 'Maximize', 'min': 'Minimize'}[self.Max_Or_Min_type] + '\n' + str(self.LP_Express)
        # return ' '.join((self.Max_Or_Min_type, str(self.LP_Express)))

    def LP_Canonical_form(self, LP_Express):
        """For example, changes x + y to canonical form representation 1.*x + 1.*y"""
        LP_Express = super(Prob_Objective, self).LP_Canonical_form(LP_Express)
        LP_Express *= 1.
        return LP_Express

    @property
    def Max_Or_Min_type(self):
        """The Max_Or_Min_type of optimization. Either 'min' or 'max'."""
        return self._Max_Or_Min_type

    @Max_Or_Min_type.setter
    def Max_Or_Min_type(self, Var_Value):
        if Var_Value not in ['max', 'min']:
            raise ValueError("Provided optimization Max_Or_Min_type %s is neither 'min' or 'max'." % Var_Value)
        self._Max_Or_Min_type = Var_Value


class Prob_Configure(object):
    """Optimization solver configuration class used to configure the solvers."""

    @classmethod
    def Funct_Cloning(cls, __config, LP_Problem=None, **kwargs):
        _Properties = (k for k, v in inspect.getmembers(cls, _predicate=inspect.isdatadescriptor) if not k.startswith('__'))
        parameters = {property: getattr(__config, property) for property in _Properties if hasattr(__config, property)}
        return cls(LP_Problem=LP_Problem, **parameters)

    def __init__(self, LP_Problem=None, *args, **kwargs):
        self.LP_Problem = LP_Problem

    @property
    def Verbosity_Level(self):
        """Verbosity level.

        0: no output
        1: error and warning messages only
        2: normal output
        4: full output
        """
        raise NotImplementedError

    @Verbosity_Level.setter
    def Verbosity_Level(self, Var_Value):
        raise NotImplementedError

    @property
    def Times_Up(self):
        raise NotImplementedError

    @Times_Up.setter
    def Times_Up(self):
        raise NotImplementedError


class MathProgConfiguration(Prob_Configure):
    def __init__(self, *args, **kwargs):
        super(MathProgConfiguration, self).__init__(*args, **kwargs)

    @property
    def Presolve_Prop(self):
        raise NotImplementedError

    @Presolve_Prop.setter
    def Presolve_Prop(self, Var_Value):
        raise NotImplementedError


class EvolutionOptimizedConfiguration(Prob_Configure):
    """Heuristic Optimization is the future optimization to be used"""

    def __init__(self, *args, **kwargs):
        super(EvolutionOptimizedConfiguration, self).__init__(*args, **kwargs)


class Prob_Model(object):
    """Optimization LP_Problem.

    Attributes
    ----------
    Objective_Obj: str
        The Objective_Obj function.
    name: str, optional
        The name of the optimization LP_Problem.
    LP_Vars: Container, read-only
        Contains the LP_Vars of the optimization LP_Problem.
        The keys are the variable names and values are the actual LP_Vars.
    _Constraints_: Container, read-only
         Contains the LP_Vars of the optimization LP_Problem.
         The keys are the constraint names and values are the actual _Constraints_.
    Lp_Status: str, read-only
        The Lp_Status of the optimization LP_Problem.

    Examples
    --------


    """

    @classmethod
    def Funct_Cloning(cls, lp_model):
        interface = sys.modules[cls.__module__]
        new_model = cls()
        for variable in lp_model.LP_Vars:
            new_variable = interface.Prob_Variable.Funct_Cloning(variable)
            new_model.Add_Variable_Prob(new_variable)
        for constraint in lp_model._Constraints_:
            new_constraint = interface.Prob_Constraint.Funct_Cloning(constraint, lp_model=new_model)
            new_model.Constraint_Adder(new_constraint)
        if lp_model.Objective_Obj is not None:
            new_model.Objective_Obj = interface.Prob_Objective.Funct_Cloning(lp_model.Objective_Obj, lp_model=new_model)
        new_model.configuration = interface.Prob_Configure.Funct_Cloning(lp_model.configuration, LP_Problem=new_model)
        return new_model

    def __init__(self, name=None, Objective_Obj=None, LP_Vars=None, _Constraints_=None, *args, **kwargs):
        super(Prob_Model, self).__init__(*args, **kwargs)
        self.objective_var = Objective_Obj
        self._LP_Vars = Container()
        self._constraints = Container()
        self.Vars_To_Constr_Map = dict()
        self._status = None
        self.name = name
        if LP_Vars is not None:
            self.add(LP_Vars)
        if _Constraints_ is not None:
            self.add(_Constraints_)

    @property
    def interface(self):
        return sys.modules[self.__module__]

    @property
    def Objective_Obj(self):
        return self.objective_var

    @Objective_Obj.setter
    def Objective_Obj(self, Var_Value):
        try:
            for atom in Var_Value.LP_Express.atoms(sympy.Symbol):
                if isinstance(atom, Prob_Variable) and (atom.LP_Problem is None or atom.LP_Problem != self):
                    self.Add_Variable_Prob(atom)
        except AttributeError as e:
            if isinstance(Var_Value.LP_Express, six.types.FunctionType) or isinstance(Var_Value.LP_Express, float):
                pass
            else:
                raise AttributeError(e)
        self.objective_var = Var_Value
        self.objective_var.LP_Problem = self

    @property
    def LP_Vars(self):
        return self._LP_Vars

    @property
    def _Constraints_(self):
        return self._constraints

    @property
    def Lp_Status(self):
        return self._status

    @property
    def Primal_Val(self):
        return collections.OrderedDict([(variable.name, variable.Primal_Prop) for variable in self.LP_Vars])

    @property
    def Cost_Reducer(self):
        return collections.OrderedDict([(variable.name, variable.Dual_Prop) for variable in self.LP_Vars])

    @property
    def Dual_Val(self):
        return collections.OrderedDict([(constraint.name, constraint.Primal_Prop) for constraint in self.constraint])

    @property
    def Shadow_Pricing(self):
        return collections.OrderedDict([(constraint.name, constraint.Dual_Prop) for constraint in self.constraint])

    def __str__(self):
        return '\n'.join((
            str(self.Objective_Obj),
            "subject to",
            '\n'.join([str(constr) for constr in self._Constraints_]),
            'Bounds',
            '\n'.join([str(var) for var in self.LP_Vars])
        ))

    def add(self, Lp_Attributes):
        """Add LP_Vars and _Constraints_.

        Parameters
        ----------
        Lp_Attributes : iterable, Prob_Variable, Prob_Constraint
            Either an iterable containing LP_Vars and _Constraints_ or a single variable or constraint.

        Returns
        -------
        None


        """
        if isinstance(Lp_Attributes, collections.Iterable):
            for Element in Lp_Attributes:
                self.add(Element)
        elif isinstance(Lp_Attributes, Prob_Variable):
            if Lp_Attributes.__module__ != self.__module__:
                raise TypeError("Cannot add Prob_Variable %s of interface Prob_Type %s to lp_model of Prob_Type %s." % (
                    Lp_Attributes, Lp_Attributes.__module__, self.__module__))
            self.Add_Variable_Prob(Lp_Attributes)
        elif isinstance(Lp_Attributes, Prob_Constraint):
            if Lp_Attributes.__module__ != self.__module__:
                raise TypeError("Cannot add Prob_Constraint %s of interface Prob_Type %s to lp_model of Prob_Type %s." % (
                    Lp_Attributes, Lp_Attributes.__module__, self.__module__))
            self.Constraint_Adder(Lp_Attributes)
        elif isinstance(Lp_Attributes, Prob_Objective):
            if Lp_Attributes.__module__ != self.__module__:
                raise TypeError("Cannot set Prob_Objective %s of interface Prob_Type %s to lp_model of Prob_Type %s." % (
                    Lp_Attributes, Lp_Attributes.__module__, self.__module__))
            self.Objective_Obj = Lp_Attributes
        else:
            raise TypeError("Cannot add %s. It is neither a Prob_Variable, Prob_Constraint, or Prob_Objective." % Lp_Attributes)

    def remove(self, Lp_Attributes):
        """Remove LP_Vars and _Constraints_.

        Parameters
        ----------
        Lp_Attributes : iterable, str, Prob_Variable, Prob_Constraint
            Either an iterable containing LP_Vars and _Constraints_ to be removed from the lp_model or a single variable or contstraint (or their names).

        Returns
        -------
        None
        """
        if isinstance(Lp_Attributes, str):
            try:
                variable = self.LP_Vars[Lp_Attributes]
                self._Variable_Remove(variable)
            except KeyError:
                try:
                    constraint = self._Constraints_[Lp_Attributes]
                    self.Constraint_Remove_funct(constraint)
                except KeyError:
                    raise LookupError(
                        "%s is neither a variable nor a constraint in the current solver instance." % Lp_Attributes)
        elif isinstance(Lp_Attributes, Prob_Variable):
            self._Variable_Remove(Lp_Attributes)
        elif isinstance(Lp_Attributes, Prob_Constraint):
            self.Constraint_Remove_funct(Lp_Attributes)
        elif isinstance(Lp_Attributes, collections.Iterable):
            Elementent_types = set((Element.__class__ for Element in Lp_Attributes))
            if len(Elementent_types) == 1:
                Elementent_type = Elementent_types.pop()
                if issubclass(Elementent_type, Prob_Variable):
                    self.Remove_Variables_Prob(Lp_Attributes)
                elif issubclass(Elementent_type, Prob_Constraint):
                    self.Constraints_Remover(Lp_Attributes)
                else:
                    raise TypeError("Cannot remove %s. It is neither a variable nor a constraint." % Lp_Attributes)
            else:
                for Element in Lp_Attributes:
                    self.remove(Element)
        elif isinstance(Lp_Attributes, Prob_Objective):
            raise TypeError(
                "Cannot remove Objective_Obj %s. Use lp_model.Objective_Obj = Prob_Objective(...) to change the current Objective_Obj." % Lp_Attributes)
        else:
            raise TypeError(
                "Cannot remove %s. It neither a variable or constraint." % Lp_Attributes)

    def optimize_funct(self):
        """Solve the optimization LP_Problem.

        Returns
        -------
        Lp_Status: str
            Solution Lp_Status.
        """
        raise NotImplementedError(
            "You're using the high level interface to optimizelp. Problems cannot be optimized in this mode. Choose from one of the solver specific interfaces.")

    def Add_Variable_Prob(self, variable):
        self.LP_Vars.append(variable)
        self.Vars_To_Constr_Map[variable.name] = set([])
        variable.LP_Problem = self

        return variable

    def Remove_Variables_Prob(self, LP_Vars):

        for variable in LP_Vars:
            try:
                var = self.LP_Vars[variable.name]
            except KeyError:
                raise LookupError("Prob_Variable %s not in solver" % var)

        Constr_Ids = set()
        for variable in LP_Vars:
            Constr_Ids.update(self.Vars_To_Constr_Map[variable.name])
            del self.Vars_To_Constr_Map[variable.name]
            variable.LP_Problem = None
            del self.LP_Vars[variable.name]

        replacements = dict([(variable, 0) for variable in LP_Vars])
        for Constr__Id in Constr_Ids:
            constraint = self._Constraints_[Constr__Id]
            constraint._LP_Express = constraint.LP_Express.xreplace(replacements)

        self.Objective_Obj._LP_Express = self.Objective_Obj.LP_Express.xreplace(replacements)

    def _Variable_Remove(self, variable):
        self.Remove_Variables_Prob([variable])

    def Constraint_Adder(self, constraint, sloppy=False):
        Constr__Id = constraint.name
        if sloppy is False:
            LP_Vars = constraint.LP_Vars
            if constraint.Var_Indicator is not None:
                LP_Vars.add(constraint.Var_Indicator)
            for var in LP_Vars:
                if var.LP_Problem is not self:
                    self.Add_Variable_Prob(var)
                try:
                    self.Vars_To_Constr_Map[var.name].add(Constr__Id)
                except KeyError:
                    self.Vars_To_Constr_Map[var.name] = set([Constr__Id])
        self._Constraints_.append(constraint)
        constraint._LP_Problem = self

    def Constraints_Remover(self, _Constraints_):
        keys = [constraint.name for constraint in _Constraints_]
        if len(_Constraints_) > 350:  # figure a best threshold till here
            self._constraints = self._Constraints_.From_Keys(set(self._Constraints_.keys()).difference(set(keys)))
        else:
            for constraint in _Constraints_:
                try:
                    del self._Constraints_[constraint.name]
                except KeyError:
                    raise LookupError("Prob_Constraint %s not in solver" % constraint)
                else:
                    constraint.LP_Problem = None

    def Constraint_Remove_funct(self, constraint):
        self.Constraints_Remover([constraint])

    def _set_linear_objective_term(self, variable, coefficient):
        # TODO: make fast for the objectives with many terms
        if variable in self.Objective_Obj.LP_Express.atoms(sympy.Symbol):
            a = sympy.Wild('a', exclude=[variable])
            (new_expression, map) = self.Objective_Obj.LP_Express.replace(lambda expr: expr.match(a*variable), lambda expr: coefficient*variable, simultaneous=False, map=True)
            self.Objective_Obj.LP_Express = new_expression
        else:
            self.Objective_Obj.LP_Express = sympy.Add._from_args((self.Objective_Obj.LP_Express, sympy.Mul._from_args((sympy.RealNumber(coefficient), variable))))

if __name__ == '__main__':
    # Example 

    x1 = Prob_Variable('x1', Lower_Bound=0)
    x2 = Prob_Variable('x2', Lower_Bound=0)
    x3 = Prob_Variable('x3', Lower_Bound=0)
    c1 = Prob_Constraint(x1 + x2 + x3, Upper_Bound=100)
    c2 = Prob_Constraint(10 * x1 + 4 * x2 + 5 * x3, Upper_Bound=600)
    c3 = Prob_Constraint(2 * x1 + 2 * x2 + 6 * x3, Upper_Bound=300)
    obj = Prob_Objective(10 * x1 + 6 * x2 + 4 * x3, Max_Or_Min_type='max')
    lp_model = Prob_Model(name='Simple lp_model')
    lp_model.Objective_Obj = obj
    lp_model.add([c1, c2, c3])

    try:
        sol = lp_model.optimize_funct()
    except NotImplementedError as e:
        print(e)

    print(lp_model)
    print(lp_model.LP_Vars)

    # lp_model.remove(x1)

    import optimizelp

    lp_model.interface = optimizelp.glpk_interface
