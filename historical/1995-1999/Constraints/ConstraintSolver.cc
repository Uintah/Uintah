
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

#include <Constraints/ConstraintSolver.h>
#include <Classlib/Debug.h>
#include <stdio.h>
#include <string.h>

static DebugSwitch cs_debug("ConstraintSolver", "Print");
static DebugSwitch cs2_debug("ConstraintSolver", "Stack");


ostream& operator<<( ostream& os, StackItem& i ) {
   os << i.var->GetName() << ":  ";
   switch (i.rtype) {
   case UnInit:
      os << "UnInit";
      break;
   case RecurseInitial:
      os << "RecurseInitial";
      break;
   case RecurseNormal:
      os << "RecurseNormal";
      break;
   case RecurseMax:
      os << "RecurseMax";
      break;
   }
   os << " " << (int)i.iter << "/" << i.var->GetNumConstraints();

   return os;
}


ConstraintSolver::ConstraintSolver()
: Epsilon(1E-6), stack(200), NumVariables(0), variables(0), MaxDepth(25),
  changed(0)
{
}


ConstraintSolver::ConstraintSolver( const Real epsilon )
: Epsilon(epsilon), stack(200), NumVariables(0), variables(0), MaxDepth(25),
  changed(0)
{
}


ConstraintSolver::~ConstraintSolver()
{
}


void
ConstraintSolver::SetEpsilon( const Real epsilon )
{
   Epsilon = epsilon;
}


Real
ConstraintSolver::GetEpsilon() const
{
   return Epsilon;
}


void
ConstraintSolver::SetMaxDepth( const Index max )
{
   MaxDepth = max;
}


Index
ConstraintSolver::GetMaxDepth() const
{
   return MaxDepth;
}


void
ConstraintSolver::AddVariable( BaseVariable* v )
{
   variables.add(v);
   NumVariables++;
}


void
ConstraintSolver::RemoveVariable( BaseVariable* v )
{
   for (Index i=0; i<variables.size(); i++)
      if (variables[i] == v) {
	 variables.remove(i);
	 NumVariables--;
	 return;
      }
}


void
ConstraintSolver::Solve( BaseVariable* var, const VarCore& newValue, const Scheme scheme )
{
   changed = !(var->data.epsilonequal(Epsilon, newValue));
   
   Index index, index2;
   int abort(0);
   VarCore newval(newValue);

   for (index=0; index<NumVariables; index++)
      variables[index]->level = variables[index]->levellevel = 0;
   stack.push(StackItem(var));

   while (!stack.empty() && !abort) {
      StackItem& item = stack.top();
      BaseVariable* v(item.var);

      if (cs2_debug) {
	 cout << "Stack top: (" << stack.size() << ")" << endl;
	 for (int i=stack.size()-1;i>=0;i--)
	    cout << stack[i] << "    " << newval << endl;
	 cout << "Stack bottom." << endl;
      }

      switch (item.rtype) {
      case RecurseInitial:
	  {
	      int reallynew = !(v->data.epsilonequal(Epsilon, newval));
	      if (!reallynew) {
		  stack.pop();
		  break;
	      }
	 
	      if (cs_debug) {
		  cout << "Recursion level = " << stack.size() << endl;
	    
		  cout << v->name << " S(" << v->levellevel << ")*";
		  for (index=0; index<v->level; index++)
		      cout << " ";
		  cout << "*" << endl;
	    
		  cout << "Old value (" << v->data << ") " << (reallynew?"!=":"==")
		      << " newval (" << newval << ").  Using Epsilon of ("
			  << Epsilon << ")." << endl;
	    
		  cout << "LevelLevel is " << v->levellevel
		      << " and Level is " << v->level << "." << endl;
	      }
	 
	      v->data = newval;

	      if (v->level++ == MaxDepth) {
		  v->level = 0;
		  if (++(v->levellevel) < v->numconstraints) {
		      if (cs_debug)
			  cerr << "Maximum recursion level reached..." << endl;
		      item.rtype = RecurseMax;
		  } else {
		      if (cs_debug) {
			  cout << v->name << " E(" << v->levellevel << ")*";
			  for (index=0; index<v->level; index++)
			      cout << " ";
			  cout << "*" << endl;
		  
			  cout << "Recursion level = " << stack.size()-1 << endl;
		      }
	       
		      cerr << "Maximum level reached for all constraints!" << endl;
		      cout << "Accepting current approximation." << endl;
		      cout << "Recursion level = " << stack.size()-1 << endl;
	       
		      abort = 1;	       
		  }
	      } else {
		  item.rtype = RecurseNormal;
	      }
	  }
	 break;
      case RecurseNormal:
	 if (item.iter < v->numconstraints) {
	    if (v->constraints[v->constraint_order[item.iter]]
		->Satisfy(v->constraint_indexs[v->constraint_order[item.iter]],
			  scheme, Epsilon, v, newval)) {
	       stack.push(StackItem(v));
	    }
	    item.iter++;
	 } else {
	    if (v->level == 0) {
	       v->level = MaxDepth-1;
	       v->levellevel--;
	    }
	    else
	       v->level--;
	    
	    if (cs_debug) {
	       cout << v->name << " E(" << v->levellevel << ")*";
	       for (index=0; index<v->level; index++)
		  cout << " ";
	       cout << "*" << endl;
	       cout << "Recursion level = " << stack.size()-1 << endl;
	    }
	    stack.pop();
	 }
	 break;
      case RecurseMax:
	 if (item.iter < v->numconstraints) {
	    index2 = (item.iter + v->levellevel) % v->numconstraints;
	    if (v->constraints[v->constraint_order[index2]]
		->Satisfy(v->constraint_indexs[v->constraint_order[index2]],
			  scheme, Epsilon, v, newval)) {
	       stack.push(StackItem(v));
	    }
	    item.iter++;
	 } else {
	    if (v->level == 0) {
	       v->level = MaxDepth-1;
	       v->levellevel--;
	    }
	    else
	       v->level--;
	    
	    if (cs_debug) {
	       cout << v->name << " E(" << v->levellevel << ")*";
	       for (index=0; index<v->level; index++)
		  cout << " ";
	       cout << "*" << endl;
	       cout << "Recursion level = " << stack.size()-1 << endl;
	    }
	    stack.pop();
	 }
	 break;
      default:
	 cerr << "ConstraintSolver::Solve something wrong." << endl;
	 break;
      }
   }

   if (!stack.empty())
      stack.remove_all();
}

