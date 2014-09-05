//static char *id="@(#) $Id$";

/*
 *  ConstraintSolver.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Constraints/ConstraintSolver.h>
#include <SCICore/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
#include <stdio.h>
#include <string.h>

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;

static DebugSwitch cs_debug("ConstraintSolver", "Print");
static DebugSwitch cs2_debug("ConstraintSolver", "Stack");

ostream& operator<<( ostream& os, const StackItem& i ) {
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
      BaseVariable (*v)(item.var);      // without the () around *v, visualC++ gets confused

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

void StackItem::print( ostream& os )
{
    os<<"StackItem:  "<<var->GetName()<<","<<rtype<<","<<iter<<endl;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.5  1999/10/07 02:07:15  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/09/08 02:26:38  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/18 21:45:25  sparker
// Array1 const correctness, and subsequent fixes
// Array1 bug fix courtesy Tom Thompson
//
// Revision 1.2  1999/08/17 06:38:16  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:54  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
