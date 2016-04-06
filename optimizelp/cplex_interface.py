
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

Cplex_to_status = {
    cplex.Cplex.solution.Lp_Status.MIP_abort_feasible: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.MIP_abort_infeasible: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.MIP_dettime_limit_feasible: interface.Time_Limit_Val,
    cplex.Cplex.solution.Lp_Status.MIP_dettime_limit_infeasible: interface.Time_Limit_Val,
    cplex.Cplex.solution.Lp_Status.MIP_feasible: interface.Feasible_Val,
    cplex.Cplex.solution.Lp_Status.MIP_feasible_relaxed_inf: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_feasible_relaxed_quad: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_feasible_relaxed_sum: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_infeasible: interface.InFeasible_Val,
    cplex.Cplex.solution.Lp_Status.MIP_infeasible_or_unbounded: interface.Infeasible_or_Unbounded_Val,
    cplex.Cplex.solution.Lp_Status.MIP_optimal: interface.OPTIMAL_VALs,
    cplex.Cplex.solution.Lp_Status.MIP_optimal_infeasible: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_optimal_relaxed_inf: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_optimal_relaxed_sum: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.MIP_time_limit_feasible: interface.Time_Limit_Val,
    cplex.Cplex.solution.Lp_Status.MIP_time_limit_infeasible: interface.Time_Limit_Val,
    cplex.Cplex.solution.Lp_Status.MIP_unbounded: interface.Unbounded_Val,
    cplex.Cplex.solution.Lp_Status.abort_dettime_limit: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_dual_obj_limit: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_iteration_limit: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_obj_limit: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_primal_obj_limit: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_relaxed: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.abort_time_limit: interface.Time_Limit_Val,
    cplex.Cplex.solution.Lp_Status.abort_user: interface.Aborted_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_contradiction: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_dettime_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_iteration_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_memory_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_node_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_obj_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_time_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_abort_user: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_feasible: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.conflict_minimal: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.fail_feasible: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.fail_feasible_no_tree: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.fail_infeasible: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.fail_infeasible_no_tree: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.feasible: interface.Feasible_Val,
    cplex.Cplex.solution.Lp_Status.feasible_relaxed_inf: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.feasible_relaxed_quad: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.feasible_relaxed_sum: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.first_order: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.infeasible: interface.InFeasible_Val,
    cplex.Cplex.solution.Lp_Status.infeasible_or_unbounded: interface.Infeasible_or_Unbounded_Val,
    cplex.Cplex.solution.Lp_Status.mem_limit_feasible: interface.Memory_Lim_Val,
    cplex.Cplex.solution.Lp_Status.mem_limit_infeasible: interface.Memory_Lim_Val,
    cplex.Cplex.solution.Lp_Status.node_limit_feasible: interface.Node_Lim_Val,
    cplex.Cplex.solution.Lp_Status.node_limit_infeasible: interface.Node_Lim_Val,
    cplex.Cplex.solution.Lp_Status.num_best: interface.Numeric_Val,
    cplex.Cplex.solution.Lp_Status.optimal: interface.OPTIMAL_VAL,
    cplex.Cplex.solution.Lp_Status.optimal_face_unbounded: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_infeasible: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_populated: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_populated_tolerance: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_relaxed_inf: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_relaxed_quad: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_relaxed_sum: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.optimal_tolerance: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.populate_solution_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.solution_limit: interface.Special_Val,
    cplex.Cplex.solution.Lp_Status.unbounded: interface.Unbounded_Val,
    cplex.Cplex.solution.Lp_Status.relaxation_unbounded: interface.Unbounded_Val,
}

Linear_Programming_Methods = ["auto", "Primal_Prop", "Dual_Prop", "network", "barrier", "sifting", "concurrent"]

Solution_type_target = ("auto", "convex", "local", "global")

Methods_QP = ("auto", "Primal_Prop", "Dual_Prop", "network", "barrier")

_CPLEX_LPTYPE_TO_LPTYPE = {'C': 'continuous', 'I': 'integer', 'B': 'binary'}
# working for semi and semi continuous

dICT_CPLEX_LPTYPE = dict(
    [(val, key) for key, val in six.iteritems(_CPLEX_LPTYPE_TO_LPTYPE)]
)


def constraint_lb_ub_to_rhs_range_val(Lower_Bound, Upper_Bound):
    """Helper function which is used by Prob_Constraint and Prob_Model"""
    if Lower_Bound is None and Upper_Bound is None:
        # FIX: use cplex.infinity to fix the upper and lowe bound problems
        raise Exception(" Free constraint is not possible ...")
    elif Lower_Bound is None:
        eq_Sense = 'L'
        rhs = float(Upper_Bound)
        range_bound_value = 0.
    elif Upper_Bound is None:
        eq_Sense = 'G'
        rhs = float(Lower_Bound)
        range_bound_value = 0.
    elif Lower_Bound == Upper_Bound:
        eq_Sense = 'E'
        rhs = float(Lower_Bound)
        range_bound_value = 0.
    elif Lower_Bound > Upper_Bound:
        raise ValueError("Lower bound canot be larger than the upper bound.")
    else:
        eq_Sense = 'R'
        rhs = float(Lower_Bound)
        range_bound_value = float(Upper_Bound - Lower_Bound)
    return eq_Sense, rhs, range_bound_value


class Prob_Variable(interface.Prob_Variable):
    """CPLEX variable interface used for solving sympy expression to cplex expression"""

    def __init__(self, name, *args, **kwargs):
        super(Prob_Variable, self).__init__(name, **kwargs)

    @interface.Prob_Variable.Lower_Bound.setter
    def Lower_Bound(self, Var_Value):
        super(Prob_Variable, self.__class__).Lower_Bound.fset(self, Var_Value)
        if self.LP_Problem is not None:
            self.LP_Problem.LP_Problem.LP_Vars.set_lower_bounds(self.name, Var_Value)

    @interface.Prob_Variable.Upper_Bound.setter
    def Upper_Bound(self, Var_Value):
        super(Prob_Variable, self.__class__).Upper_Bound.fset(self, Var_Value)
        if self.LP_Problem is not None:
            self.LP_Problem.LP_Problem.LP_Vars.set_upper_bounds(self.name, Var_Value)

    @interface.Prob_Variable.Prob_Type.setter
    def Prob_Type(self, Var_Value):
        if self.LP_Problem is not None:
            try:
                cplex_kind = dICT_CPLEX_LPTYPE[Var_Value]
            except KeyError:
                raise Exception("CPLEX is unable to deal with variable of Prob_Type %s. \
                            only the following variables are supported:\n" +
                                " ".join(dICT_CPLEX_LPTYPE.keys()))
            self.LP_Problem.LP_Problem.LP_Vars.set_types(self.name, cplex_kind)
        super(Prob_Variable, self).__setattr__('Prob_Type', Var_Value)


    @property
    def Primal_Prop(self):
        if self.LP_Problem:
            solver_primal = self.LP_Problem.LP_Problem.solution.get_values(self.name)
            return self.Make_Primal_Bound_Rounded(solver_primal)
        else:
            return None

    @property
    def Dual_Prop(self):
        if self.LP_Problem is not None:
            if self.LP_Problem.LP_Problem.get_problem_type() != self.LP_Problem.LP_Problem.problem_type.LP: # cplex cannot determine reduced costs for MILP problems ...
                return None
            return self.LP_Problem.LP_Problem.solution.get_reduced_costs(self.name)
        else:
            return None


class Prob_Constraint(interface.Prob_Constraint):
    """CPLEX solver interface to solve constraints using the cplex"""

    _Indi_Constr_Support = True

    def __init__(self, LP_Express, *args, **kwargs):
        super(Prob_Constraint, self).__init__(LP_Express, *args, **kwargs)
        if self.Upper_Bound is not None and self.Lower_Bound is not None and self.Lower_Bound > self.Upper_Bound:
            raise ValueError(
                " ohh eroors encountered Lower bound %f is larger than upper bound %f in constraint  %s" %
                (self.Lower_Bound, self.Upper_Bound, self)
            )


    # TODO: need to implement LP_Express from solver structure for the cplex
    def _expression_getter(self):
        if self.LP_Problem is not None:
            cplex_lp_problem = self.LP_Problem.LP_Problem
            cplex_lp_row = cplex_lp_problem.linear_constraints.get_rows(self.name)
            LP_Vars = self.LP_Problem.LP_Vars
            LP_Express = sympy.Add._from_args([sympy.Mul._from_args((sympy.RealNumber(cplex_lp_row.val[i]), LP_Vars[ind])) for i, ind in enumerate(cplex_lp_row.ind)])
            self._LP_Express = LP_Express
        return self._LP_Express

    def Coefficient_TO_LOWLEVEL_Setter(self, variables_coefficients_dict):
        self_Name = self.name
        if self.Check_Linear:
            cplex_format_STANDARD = [(self_Name, variable.name, coefficient) for variable, coefficient in six.iteritems(variables_coefficients_dict)]
            self.LP_Problem.LP_Problem.linear_constraints.set_coefficients(cplex_format_STANDARD)
        else:
            raise Exception('Coefficient_TO_LOWLEVEL_Setter works only with linear constraints are in the cplex interface.')

    @property
    def LP_Problem(self):
        return self._LP_Problem

    @LP_Problem.setter
    def LP_Problem(self, Var_Value):
        if Var_Value is None:
            # Update LP_Express from solver instance for the last time 
            self._expression_getter()
            self._LP_Problem = None
        else:
            self._LP_Problem = Var_Value

    @property
    def Primal_Prop(self):
        if self.LP_Problem is not None:
            return self.LP_Problem.LP_Problem.solution.get_activity_levels(self.name)
        else:
            return None

    @property
    def Dual_Prop(self):
        if self.LP_Problem is not None:
            return self.LP_Problem.LP_Problem.solution.get_dual_values(self.name)
        else:
            return None

    # TODO: add refactor to properties
    def __setattr__(self, name, Var_Value):
        try:
            Previous_Name = self.name  
        except AttributeError:
            pass
        super(Prob_Constraint, self).__setattr__(name, Var_Value)
        if getattr(self, 'LP_Problem', None):

            if name == 'name':
                if self.Var_Indicator is not None:
                    raise NotImplementedError("Unfortunately, in the CPLEX changing an indicator constraint's name is not possible")
                else:
                    # TODO:deal with quadratic _Constraints_
                    self.LP_Problem.LP_Problem.linear_constraints.set_names(Previous_Name, Var_Value)

            elif name == 'Lower_Bound' or name == 'Upper_Bound':
                if self.Var_Indicator is not None:
                    raise NotImplementedError("Unfortunately, in the CPLEX changing an indicator constraint's bounds is not supported")
                if name == 'Lower_Bound':
                    if Var_Value > self.Upper_Bound:
                        raise ValueError(
                            "Lower bound %f is larger than upper bound %f in constraint %s" %
                            (Var_Value, self.Upper_Bound, self)
                        )
                    eq_Sense, rhs, range_bound_value = constraint_lb_ub_to_rhs_range_val(Var_Value, self.Upper_Bound)
                elif name == 'Upper_Bound':
                    if Var_Value < self.Lower_Bound:
                        raise ValueError(
                            "ohh this is error Upper bound %f is less than lower bound %f in constraint %s" %
                            (Var_Value, self.Lower_Bound, self)
                        )
                    eq_Sense, rhs, range_bound_value = constraint_lb_ub_to_rhs_range_val(self.Lower_Bound, Var_Value)
                if self.Check_Linear:
                    self.LP_Problem.LP_Problem.linear_constraints.set_rhs(self.name, rhs)
                    self.LP_Problem.LP_Problem.linear_constraints.set_senses(self.name, eq_Sense)
                    self.LP_Problem.LP_Problem.linear_constraints.set_range_values(self.name, range_bound_value)

            elif name == 'LP_Express':
                pass

    def __iadd__(self, Other_Val):
        # if self.LP_Problem is not None:
        #     self.LP_Problem._add_to_constraint(self.index, Other_Val)
        if self.LP_Problem is not None:
            problem_reference = self.LP_Problem
            self.LP_Problem.Constraint_Remove_funct(self)
            super(Prob_Constraint, self).__iadd__(Other_Val)
            problem_reference.Constraint_Adder(self, sloppy=False)
        else:
            super(Prob_Constraint, self).__iadd__(Other_Val)
        return self


class Prob_Objective(interface.Prob_Objective):
    def __init__(self, *args, **kwargs):
        super(Prob_Objective, self).__init__(*args, **kwargs)

    @property
    def Var_Value(self):
        return self.LP_Problem.LP_Problem.solution.get_objective_value()

    def __setattr__(self, name, Var_Value):

        if getattr(self, 'LP_Problem', None):
            if name == 'Max_Or_Min_type':
                self.LP_Problem.LP_Problem.Objective_Obj.set_sense(
                    {'min': self.LP_Problem.LP_Problem.Objective_Obj.eq_Sense.minimize, 'max': self.LP_Problem.LP_Problem.Objective_Obj.eq_Sense.maximize}[Var_Value])
            super(Prob_Objective, self).__setattr__(name, Var_Value)
        else:
            super(Prob_Objective, self).__setattr__(name, Var_Value)


class Prob_Configure(interface.MathProgConfiguration):

    def __init__(self, lp_method='Primal_Prop', Tolerance_Val=1e-9, Presolve_Prop=False, Verbosity_Level=0, Times_Up=None,
                 Solutions_Target_Prop="auto", Method_QP="Primal_Prop", *args, **kwargs):
        super(Prob_Configure, self).__init__(*args, **kwargs)
        self.lp_method = lp_method
        self.Tolerance_Val = Tolerance_Val
        self.Presolve_Prop = Presolve_Prop
        self.Verbosity_Level = Verbosity_Level
        self.Times_Up = Times_Up
        self.Solutions_Target_Prop = Solutions_Target_Prop
        self.Method_QP = Method_QP

    @property
    def lp_method(self):
        lpmethod = self.LP_Problem.LP_Problem.parameters.lpmethod
        try:
            Var_Value = lpmethod.get()
        except ReferenceError:
            Var_Value = lpmethod.default()
        return lpmethod.values[Var_Value]

    @lp_method.setter
    def lp_method(self, lp_method):
        if lp_method not in Linear_Programming_Methods:
            raise ValueError("LP Method %s is not valid (choose one of: %s)" % (lp_method, ", ".join(Linear_Programming_Methods)))
        lp_method = getattr(self.LP_Problem.LP_Problem.parameters.lpmethod.values, lp_method)
        self.LP_Problem.LP_Problem.parameters.lpmethod.set(lp_method)

    @property
    def Tolerance_Val(self):
        return self._tolerance

    @Tolerance_Val.setter
    def Tolerance_Val(self, Var_Value):
        self.LP_Problem.LP_Problem.parameters.simplex.tolerances.feasibility.set(Var_Value)
        self.LP_Problem.LP_Problem.parameters.simplex.tolerances.optimality.set(Var_Value)
        self.LP_Problem.LP_Problem.parameters.mip.tolerances.integrality.set(Var_Value)
        self.LP_Problem.LP_Problem.parameters.mip.tolerances.absmipgap.set(Var_Value)
        self.LP_Problem.LP_Problem.parameters.mip.tolerances.mipgap.set(Var_Value)
        self._tolerance = Var_Value

    @property
    def Presolve_Prop(self):
        return self._presolve

    @Presolve_Prop.setter
    def Presolve_Prop(self, Var_Value):
        if self.LP_Problem is not None:
            Presolve_Prop = self.LP_Problem.LP_Problem.parameters.preprocessing.Presolve_Prop
            if Var_Value == True:
                Presolve_Prop.set(Presolve_Prop.values.on)
            elif Var_Value == False:
                Presolve_Prop.set(Presolve_Prop.values.off)
            else:
                raise ValueError('%s this is not bool value for Presolve_Prop property.')
        self._presolve = Var_Value

    @property
    def Verbosity_Level(self):
        return self._verbosity

    @Verbosity_Level.setter
    def Verbosity_Level(self, Var_Value):

        class StreamHandler(StringIO):

            def __init__(self, logger, *args, **kwargs):
                StringIO.__init__(self, *args, **kwargs)
                self.logger = logger

        class ErrorStreamHandler(StreamHandler):

            def flush(self):
                self.logger.error(self.getvalue())

        class WarningStreamHandler(StreamHandler):

            def flush(self):
                self.logger.warn(self.getvalue())

        class LogStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        class ResultsStreamHandler(StreamHandler):

            def flush(self):
                self.logger.debug(self.getvalue())

        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        error_stream_handler = ErrorStreamHandler(logger)
        warning_stream_handler = WarningStreamHandler(logger)
        log_stream_handler = LogStreamHandler(logger)
        results_stream_handler = LogStreamHandler(logger)
        if self.LP_Problem is not None:
            LP_Problem = self.LP_Problem.LP_Problem
            if Var_Value == 0:
                LP_Problem.set_error_stream(error_stream_handler)
                LP_Problem.set_warning_stream(warning_stream_handler)
                LP_Problem.set_log_stream(log_stream_handler)
                LP_Problem.set_results_stream(results_stream_handler)
            elif Var_Value == 1:
                LP_Problem.set_error_stream(sys.stderr)
                LP_Problem.set_warning_stream(warning_stream_handler)
                LP_Problem.set_log_stream(log_stream_handler)
                LP_Problem.set_results_stream(results_stream_handler)
            elif Var_Value == 2:
                LP_Problem.set_error_stream(sys.stderr)
                LP_Problem.set_warning_stream(sys.stderr)
                LP_Problem.set_log_stream(log_stream_handler)
                LP_Problem.set_results_stream(results_stream_handler)
            elif Var_Value == 3:
                LP_Problem.set_error_stream(sys.stderr)
                LP_Problem.set_warning_stream(sys.stderr)
                LP_Problem.set_log_stream(sys.stdout)
                LP_Problem.set_results_stream(sys.stdout)
            else:
                raise Exception(
                    "%s valid Verbosity level is between 0 and 3."
                    % Var_Value
                )
        self._verbosity = Var_Value

    @property
    def Times_Up(self):
        return self._timeout

    @Times_Up.setter
    def Times_Up(self, Var_Value):
        if self.LP_Problem is not None:
            if Var_Value is None:
                self.LP_Problem.LP_Problem.parameters.timelimit.reset()
            else:
                self.LP_Problem.LP_Problem.parameters.timelimit.set(Var_Value)
        self._timeout = Var_Value

    @property
    def Solutions_Target_Prop(self):
        if self.LP_Problem is not None:
            return Solution_type_target[self.LP_Problem.LP_Problem.parameters.solutiontarget.get()]
        else:
            return None

    @Solutions_Target_Prop.setter
    def Solutions_Target_Prop(self, Var_Value):
        if self.LP_Problem is not None:
            if Var_Value is None:
                self.LP_Problem.LP_Problem.parameters.solutiontarget.reset()
            else:
                try:
                    Solutions_Target_Prop = Solution_type_target.index(Var_Value)
                except ValueError:
                    raise ValueError("%s is not a valid solution target. Choose between valid solution target %s" % (Var_Value, str(Solution_type_target)))
                self.LP_Problem.LP_Problem.parameters.solutiontarget.set(Solutions_Target_Prop)
        self._solution_target = self.Solutions_Target_Prop

    @property
    def Method_QP(self):
        Var_Value = self.LP_Problem.LP_Problem.parameters.qpmethod.get()
        return self.LP_Problem.LP_Problem.parameters.qpmethod.values[Var_Value]

    @Method_QP.setter
    def Method_QP(self, Var_Value):
        if Var_Value not in Methods_QP:
            raise ValueError("%s is not a valid Method_QP. Choose between  valid method %s" % (Var_Value, str(Methods_QP)))
        method = getattr(self.LP_Problem.LP_Problem.parameters.qpmethod.values, Var_Value)
        self.LP_Problem.LP_Problem.parameters.qpmethod.set(method)
        self._qp_method = Var_Value


class Prob_Model(interface.Prob_Model):
    def __init__(self, LP_Problem=None, *args, **kwargs):

        super(Prob_Model, self).__init__(*args, **kwargs)

        if LP_Problem is None:
            self.LP_Problem = cplex.Cplex()

        elif isinstance(LP_Problem, cplex.Cplex):
            self.LP_Problem = LP_Problem
            zipped_var_args = zip(self.LP_Problem.LP_Vars.get_names(),
                                  self.LP_Problem.LP_Vars.get_lower_bounds(),
                                  self.LP_Problem.LP_Vars.get_upper_bounds()
            )
            for name, Lower_Bound, Upper_Bound in zipped_var_args:
                var = Prob_Variable(name, Lower_Bound=Lower_Bound, Upper_Bound=Upper_Bound, LP_Problem=self)
                super(Prob_Model, self).Add_Variable_Prob(var)  # To addtion of the variable to the glpk LP_Problem
            zipped_constr_args = zip(self.LP_Problem.linear_constraints.get_names(),
                                     self.LP_Problem.linear_constraints.get_rows(),
                                     self.LP_Problem.linear_constraints.get_senses(),
                                     self.LP_Problem.linear_constraints.get_rhs()

            )
            LP_Vars = self.LP_Vars
            for name, row, eq_Sense, rhs in zipped_constr_args:
                constraint_variables = [LP_Vars[i - 1] for i in row.ind]
                lhs = _unevaluated_Add(*[val * LP_Vars[i - 1] for i, val in zip(row.ind, row.val)])
                if isinstance(lhs, int):
                    lhs = sympy.Integer(lhs)
                elif isinstance(lhs, float):
                    lhs = sympy.RealNumber(lhs)
                if eq_Sense == 'E':
                    constr = Prob_Constraint(lhs, Lower_Bound=rhs, Upper_Bound=rhs, name=name, LP_Problem=self)
                elif eq_Sense == 'G':
                    constr = Prob_Constraint(lhs, Lower_Bound=rhs, name=name, LP_Problem=self)
                elif eq_Sense == 'L':
                    constr = Prob_Constraint(lhs, Upper_Bound=rhs, name=name, LP_Problem=self)
                elif eq_Sense == 'R':
                    range_val = self.LP_Problem.linear_constraints.get_rhs(name)
                    if range_val > 0:
                        constr = Prob_Constraint(lhs, Lower_Bound=rhs, Upper_Bound=rhs + range_val, name=name, LP_Problem=self)
                    else:
                        constr = Prob_Constraint(lhs, Lower_Bound=rhs + range_val, Upper_Bound=rhs, name=name, LP_Problem=self)
                else:
                    raise Exception('%s is not a known constraint eq_Sense.' % eq_Sense)

                for variable in constraint_variables:
                    try:
                        self.Vars_To_Constr_Map[variable.name].add(name)
                    except KeyError:
                        self.Vars_To_Constr_Map[variable.name] = set([name])

                super(Prob_Model, self).Constraint_Adder(
                    constr,
                    sloppy=True
                )
            try:
                objective_name = self.LP_Problem.Objective_Obj.get_name()
            except cplex.exceptions.CplexSolverError as e:
                if 'CPLEX Error  1219:' not in str(e):
                    raise e
            else:
                linear_expression = _unevaluated_Add(*[_unevaluated_Mul(sympy.RealNumber(coeff), LP_Vars[index]) for index, coeff in
                                       enumerate(self.LP_Problem.Objective_Obj.get_linear()) if coeff != 0.])

                try:
                    quadratic = self.LP_Problem.Objective_Obj.get_quadratic()
                except IndexError:
                    quadratic_expression = Zero
                else:
                    quadratic_expression = self.quad_expression_getter(quadratic)

                self.objective_var = Prob_Objective(
                    linear_expression + quadratic_expression,
                    LP_Problem=self,
                    Max_Or_Min_type={self.LP_Problem.Objective_Obj.eq_Sense.minimize: 'min', self.LP_Problem.Objective_Obj.eq_Sense.maximize: 'max'}[
                        self.LP_Problem.Objective_Obj.get_sense()],
                    name=objective_name
                )
        else:
            raise Exception("the given Problem is not CPLEX model in nature.")
        self.configuration = Prob_Configure(LP_Problem=self, Verbosity_Level=0)

    def __getstate__(self):
        Temporary_File = tempfile.mktemp(suffix=".sav")
        self.LP_Problem.write(Temporary_File)
        cplex_binary = open(Temporary_File, 'rb').read()
        repr_dict = {'cplex_binary': cplex_binary, 'Lp_Status': self.Lp_Status, '__configs': self.configuration}
        return repr_dict

    def __setstate__(self, repr_dict):
        Temporary_File = tempfile.mktemp(suffix=".sav")
        open(Temporary_File, 'wb').write(repr_dict['cplex_binary'])
        LP_Problem = cplex.Cplex(Temporary_File)
        if repr_dict['Lp_Status'] == 'optimal':
            # this will turn off the logging 
            LP_Problem.set_error_stream(None)
            LP_Problem.set_warning_stream(None)
            LP_Problem.set_log_stream(None)
            LP_Problem.set_results_stream(None)
            LP_Problem.solve()  # optimal start so nothing to do
        self.__init__(LP_Problem=LP_Problem)
        self.configuration = Prob_Configure.Funct_Cloning(repr_dict['__config'], LP_Problem=self)

    @property
    def Objective_Obj(self):
        return self.objective_var

    @Objective_Obj.setter
    def Objective_Obj(self, Var_Value):
        if self.objective_var is not None:
            for variable in self.Objective_Obj.LP_Vars:
                try:
                    self.LP_Problem.Objective_Obj.set_linear(variable.name, 0.)
                except cplex.exceptions.CplexSolverError as e:
                    if " 1210:" not in str(e):  # 1210 = variable not found in the model error
                        raise e
            if self.Objective_Obj.Check_Quadratic:
                if self.objective_var.LP_Express.is_Mul:
                    args = (self.objective_var.LP_Express, )
                else:
                    args = self.objective_var.LP_Express.args
                for arg in args:
                    vars = tuple(arg.atoms(sympy.Symbol))
                    assert len(vars) <= 2
                    try:
                        if len(vars) == 1:
                            self.LP_Problem.Objective_Obj.set_quadratic_coefficients(vars[0].name, vars[0].name, 0)
                        else:
                            self.LP_Problem.Objective_Obj.set_quadratic_coefficients(vars[0].name, vars[1].name, 0)
                    except cplex.exceptions.CplexSolverError as e:
                        if " 1210:" not in str(e):  # 1210 = variable not found in the model error
                            raise e

        super(Prob_Model, self.__class__).Objective_Obj.fset(self, Var_Value)
        LP_Express = self.objective_var.LP_Express
        if isinstance(LP_Express, float) or isinstance(LP_Express, int) or LP_Express.is_Number:
            pass
        else:
            if LP_Express.is_Symbol:
                self.LP_Problem.Objective_Obj.set_linear(LP_Express.name, 1.)
            if LP_Express.is_Mul:
                express_terms = (LP_Express, )
            elif LP_Express.is_Add:
                express_terms = LP_Express.args
            else:
                raise ValueError(
                    "Provided Objective_Obj %s doesn't seem to be appropriate to be solved. please make correction" %
                    self.objective_var)

            for term_val in express_terms:
                factors = term_val.args
                coeff = factors[0]
                vars = factors[1:]
                assert len(vars) <= 2
                if len(vars) == 2:
                    if vars[0].name == vars[1].name:
                        self.LP_Problem.Objective_Obj.set_quadratic_coefficients(vars[0].name, vars[1].name, 2*float(coeff))
                    else:
                        self.LP_Problem.Objective_Obj.set_quadratic_coefficients(vars[0].name, vars[1].name, float(coeff))
                else:
                    if vars[0].is_Symbol:
                        self.LP_Problem.Objective_Obj.set_linear(vars[0].name, float(coeff))
                    elif vars[0].is_Pow:
                        var = vars[0].args[0]
                        self.LP_Problem.Objective_Obj.set_quadratic_coefficients(var.name, var.name, 2*float(coeff))  # Multiply by 2 because it's on diagonal

            self.LP_Problem.Objective_Obj.set_sense(
                {'min': self.LP_Problem.Objective_Obj.eq_Sense.minimize, 'max': self.LP_Problem.Objective_Obj.eq_Sense.maximize}[
                    Var_Value.Max_Or_Min_type])
        self.LP_Problem.Objective_Obj.set_name(Var_Value.name)
        Var_Value.LP_Problem = self

    @property
    def Primal_Val(self):
        if self.LP_Problem:
            Primal_Val = collections.OrderedDict()
            for variable, Primal_Prop in zip(self.LP_Vars, self.LP_Problem.solution.get_values()):
                Primal_Val[variable.name] = variable.Make_Primal_Bound_Rounded(Primal_Prop)
            return Primal_Val
        else:
            return None

    @property
    def Cost_Reducer(self):
        if self.LP_Problem:
            return collections.OrderedDict(
                zip([variable.name for variable in self.LP_Vars], self.LP_Problem.solution.get_reduced_costs()))
        else:
            return None

    @property
    def Dual_Val(self):
        if self.LP_Problem:
            return collections.OrderedDict(
                zip([constraint.name for constraint in self._Constraints_], self.LP_Problem.solution.get_activity_levels()))
        else:
            return None

    @property
    def Shadow_Pricing(self):
        if self.LP_Problem:
            return collections.OrderedDict(
                zip([constraint.name for constraint in self._Constraints_], self.LP_Problem.solution.get_dual_values()))
        else:
            return None


    def __str__(self):
        Temporary_File = tempfile.mktemp(suffix=".lp")
        self.LP_Problem.write(Temporary_File)
        cplex_form = open(Temporary_File).read()
        return cplex_form

    def optimize_funct(self):
        self.LP_Problem.solve()
        cplex_status = self.LP_Problem.solution.get_status()
        self._status = Cplex_to_status[cplex_status]
        return self.Lp_Status

    @staticmethod
    def cplex_sensetosympy(eq_Sense, Translate=None):
        if not Translate: Translate = {'E': '==', 'L': '<', 'G': '>'}
        try:
            return Translate[eq_Sense]
        except KeyError as e:
            raise Exception(' '.join(('Sense', eq_Sense, 'is not a proper relational operator, e.g. >, <, == etc.')))

    def Add_Variable_Prob(self, variable):
        super(Prob_Model, self).Add_Variable_Prob(variable)
        if variable.Lower_Bound is None:
            Lower_Bound = -cplex.infinity
        else:
            Lower_Bound = variable.Lower_Bound
        if variable.Upper_Bound is None:
            Upper_Bound = cplex.infinity
        else:
            Upper_Bound = variable.Upper_Bound
        vtype = dICT_CPLEX_LPTYPE[variable.Prob_Type]
        if vtype == 'C':  # because CPLEX by default set the problemtype to MILP 
            self.LP_Problem.LP_Vars.add([0.], Lower_Bound=[Lower_Bound], Upper_Bound=[Upper_Bound], names=[variable.name])
        else:
            self.LP_Problem.LP_Vars.add([0.], Lower_Bound=[Lower_Bound], Upper_Bound=[Upper_Bound], types=[vtype], names=[variable.name])
        variable.LP_Problem = self
        return variable

    def Remove_Variables_Prob(self, LP_Vars):
        # do not call the parent method call to avoid hard variable removal from sympy expressions
        self.LP_Problem.LP_Vars.delete([variable.name for variable in LP_Vars])
        for variable in LP_Vars:
            del self.Vars_To_Constr_Map[variable.name]
            variable.LP_Problem = None
            del self.LP_Vars[variable.name]

    def Constraint_Adder(self, constraint, sloppy=False):
        super(Prob_Model, self).Constraint_Adder(constraint, sloppy=sloppy)
        constraint._LP_Problem = None
        if constraint.Check_Linear:
            if constraint.LP_Express.is_Add:
                _dict_coeff = constraint.LP_Express.as_coefficients_dict()
                indices = [var.name for var in _dict_coeff.keys()]
                values = [float(val) for val in _dict_coeff.values()]
            elif constraint.LP_Express.is_Mul:
                variable = list(constraint.LP_Express.atoms(sympy.Symbol))[0]
                indices = [variable.name]
                values = [float(constraint.LP_Express.coeff(variable))]
            elif constraint.LP_Express.is_Atom and constraint.LP_Express.is_Symbol:
                indices = [constraint.LP_Express.name]
                values = [1.]
            elif constraint.LP_Express.is_Number:
                indices = []
                values = []
            else:
                raise ValueError('Something is wrong with constraint %s' % constraint)

            eq_Sense, rhs, range_bound_value = constraint_lb_ub_to_rhs_range_val(constraint.Lower_Bound, constraint.Upper_Bound)
            if constraint.Var_Indicator is None:
                self.LP_Problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=indices, val=values)], senses=[eq_Sense], rhs=[rhs],
                    range_values=[range_bound_value], names=[constraint.name])
            else:
                if eq_Sense == 'R':
                    raise ValueError('CPLEX does not support indicator constraint having upper and lower bound.')
                else:
                    self.LP_Problem.indicator_constraints.add(
                        lin_expr=cplex.SparsePair(ind=indices, val=values), eq_Sense=eq_Sense, rhs=rhs, name=constraint.name,
                        indvar=constraint.Var_Indicator.name, complemented=abs(constraint.Var_Active)-1)
        # TODO: used to implement the quadratics constraints 
        elif constraint.Check_Quadratic:
            raise NotImplementedError('Quadratic _Constraints_ (like %s) are not supported yet.' % constraint)
        else:
            raise ValueError("CPLEX only supports linear or the quadratic constraint. %s this is not the required constraint " % constraint)
        constraint.LP_Problem = self
        return constraint

    def Constraints_Remover(self, _Constraints_):
        super(Prob_Model, self).Constraints_Remover(_Constraints_)
        for constraint in _Constraints_:
            if constraint.Check_Linear:
                self.LP_Problem.linear_constraints.delete(constraint.name)
            elif constraint.Check_Quadratic:
                self.LP_Problem.quadratic_constraints.delete(constraint.name)

    def _set_linear_objective_term(self, variable, coefficient):
        self.LP_Problem.Objective_Obj.set_linear(variable.name, float(coefficient))

    def quad_expression_getter(self, quadratic=None):
        if quadratic is None:
            try:
                quadratic = self.LP_Problem.Objective_Obj.get_quadratic()
            except IndexError:
                return Zero
        express_terms = []
        for i, SparsePair_maker in enumerate(quadratic):
            for j, val in zip(SparsePair_maker.ind, SparsePair_maker.val):
                if i < j:
                    express_terms.append(val*self.LP_Vars[i]*self.LP_Vars[j])
                elif i == j:
                    express_terms.append(0.5*val*self.LP_Vars[i]**2)
                else:
                    pass  # Only look at upper triangle for solving
        return _unevaluated_Add(*express_terms)


if __name__ == '__main__':

    from optimizelp.cplex_interface import Prob_Model, Prob_Variable, Prob_Constraint, Prob_Objective

    x1 = Prob_Variable('x1', Lower_Bound=0)
    x2 = Prob_Variable('x2', Lower_Bound=0)
    x3 = Prob_Variable('x3', Lower_Bound=0)
    c1 = Prob_Constraint(x1 + x2 + x3, Upper_Bound=100, name='c1')
    c2 = Prob_Constraint(10 * x1 + 4 * x2 + 5 * x3, Upper_Bound=600, name='c2')
    c3 = Prob_Constraint(2 * x1 + 2 * x2 + 6 * x3, Upper_Bound=300, name='c3')
    obj = Prob_Objective(10 * x1 + 6 * x2 + 4 * x3, Max_Or_Min_type='max')
    lp_model = Prob_Model(name='Simple lp_model')
    lp_model.Objective_Obj = obj
    lp_model.add([c1, c2, c3])
    print(lp_model)
    Lp_Status = lp_model.optimize_funct()
    print("Lp_Status:", lp_model.Lp_Status)
    print("Objective_Obj Var_Value:", lp_model.Objective_Obj.Var_Value)

    for var_name, var in lp_model.LP_Vars.items():
        print(var_name, "=", var.Primal_Prop)


        # from cplex import Cplex
        # LP_Problem = Cplex()
        # LP_Problem.read("../tests/data/lp_model.lp")

        # solver = Prob_Model(LP_Problem=LP_Problem)
        # print solver
        # solver.optimize_funct()
        # print solver.Objective_Obj.Var_Value
        # solver.add(z)
        # solver.add(constr)
        # # print solver
        # print solver.optimize_funct()
        # print solver.Objective_Obj