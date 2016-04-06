
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
