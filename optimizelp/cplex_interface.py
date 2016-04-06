
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
