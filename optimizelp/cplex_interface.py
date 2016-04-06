
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
