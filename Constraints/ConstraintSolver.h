
/*
 *  ConstraintSolver.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Constraint_Solver_h
#define SCI_project_Constraint_Solver_h 1

#include <Constraints/BaseConstraint.h>
#include <Classlib/Stack.h>

struct StackItem;

class ConstraintSolver {
public:
   ConstraintSolver();
   ConstraintSolver( const Real epsilon );
   ~ConstraintSolver();
   
   void SetEpsilon( const Real epsilon );
   Real GetEpsilon() const;
   
   void SetMaxDepth( const Index max );
   Index GetMaxDepth() const;
   
   int VariablesChanged() const {return changed;}
   void ResetChanged() {changed = 0;}
   
   friend class BaseVariable;
private:
   void AddVariable( BaseVariable* v );
   void RemoveVariable( BaseVariable* v );
   void Solve( BaseVariable* var, const VarCore& newValue, const Scheme scheme );
   
   Real Epsilon;
   Index MaxDepth;
   int changed;
   
   Stack<StackItem> stack;
   
   Index NumVariables;
   Array1<BaseVariable*> variables;
};


#endif
