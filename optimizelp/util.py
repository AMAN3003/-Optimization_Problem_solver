"""Utility functions for optimizelp."""

import logging

import os


log = logging.getLogger(__name__)
import tempfile
from subprocess import check_output


def Funct_solver_glpsol(glp_prob):
    """Solve glpk LP_Problem with glpsol commandline solver. Mainly for testing purposes.

    # Examples
    # --------

    # >>> LP_Problem = glp_create_prob()
    # ... glp_read_lp(LP_Problem, None, "../tests/data/lp_model.lp")
    # ... solution = Funct_solver_glpsol(LP_Problem)
    # ... print 'abcgdhh'
    # 'abcgdhh'
    # >>> print solution
    # 0.839784

    # Returns
    # -------
    # dict
    #     A dictionary containing the objective value (key ='objval')
    #     and variable primals.
    """
    from swiglpk import glp_get_row_name, glp_get_col_name, glp_write_lp, glp_get_num_rows, glp_get_num_cols

    Row__Ids = [glp_get_row_name(glp_prob, i) for i in range(1, glp_get_num_rows(glp_prob) + 1)]

    Column__Ids = [glp_get_col_name(glp_prob, i) for i in range(1, glp_get_num_cols(glp_prob) + 1)]

    Temporary_File = tempfile.mktemp(suffix='.lp')
    
    glp_write_lp(glp_prob, None, Temporary_File)
    
    cmd = ['glpsol', '--lp', Temporary_File, '-w', Temporary_File + '.sol', '--log', '/dev/null']
    TermOut = check_output(cmd)
    log.info(TermOut)

    with open(Temporary_File + '.sol') as solution_handle:
        # print solution_handle.read()
        solution = dict()
        for i, line in enumerate(solution_handle.readlines()):
            if i <= 1 or line == '\n':
                pass
            elif i <= len(Row__Ids):
                solution[Row__Ids[i - 2]] = line.strip().split(' ')
            elif i <= len(Row__Ids) + len(Column__Ids) + 1:
                solution[Column__Ids[i - 2 - len(Row__Ids)]] = line.strip().split(' ')
            else:
                print(i)
                print(line)
                raise Exception("exceptons ohhh..")
    return solution
