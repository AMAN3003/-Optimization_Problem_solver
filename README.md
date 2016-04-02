# README #
# optimizelp #
This is python package which provides generic interface to a series of optimization tools. Currently supported solvers will be:          
**Glpk AND Cplex**

**optimizelp** will use sympy for problem formulation (problem constraints, problems objectives, problem variables, etc.). Adding interface to other optimization solvers is just simple sub-classing of the abstract interface and implementing the solver techniques for new solvers.

**Abstract Interface** for the solver is Present in this Repository and Interface of cplex and glpk are being updated.

#### Use of the Repository Requires the Following Package to Function 
* *sympy* 
* *swiglpk*
* *glpk*


#### example
    y1 = Prob_Variable('y1', Lower_Bound=0)
    y2 = Prob_Variable('y2', Lower_Bound=0)
    y3 = Prob_Variable('y3', Lower_Bound=0)
    c1 = Prob_Constraint(y1 + y2 + y3, Upper_Bound=300)
    c2 = Prob_Constraint(15 * y1 + 6 * y2 + 2 * y3, Upper_Bound=500)
    c3 = Prob_Constraint(20 * y1 + 4 * y2 + 5 * y3, Upper_Bound=200)
    obj = Prob_Objective(4 * x1 + 2 * y2 + 8 * y3, Max_Or_Min_type='max')
    lp_model = Prob_Model(name='Simple lp_model')
    lp_model.Objective_Obj = obj
    lp_model.add([c1, c2, c3])
    Lp_Status=lp_model.optimize_funct()
    print "status:", lp_model.Lp_Status