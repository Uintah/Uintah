/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/Constraints/ConstraintSolver.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <list>

using std::list;

namespace SCIRun {


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
  : Epsilon(1E-6),
    MaxDepth(25),
    changed(false),
    variables(0)
{
}


ConstraintSolver::ConstraintSolver( const Real epsilon )
  : Epsilon(epsilon),
    MaxDepth(25),
    changed(false),
    variables(0)
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
   variables.push_back(v);
}


void
ConstraintSolver::RemoveVariable( BaseVariable* v )
{
  std::remove(variables.begin(), variables.end(), v);
}


void
ConstraintSolver::Solve(BaseVariable* var,
			const VarCore& newValue, const Scheme scheme )
{
  changed |= !(var->data.epsilonequal(Epsilon, newValue));
   
  Index index, index2;
  int abort(0);
  VarCore newval(newValue);

  list<StackItem> itemstack;

  for (index=0; index< variables.size(); index++)
  {
    variables[index]->level = variables[index]->levellevel = 0;
  }
  itemstack.push_front(StackItem(var));

  while (!itemstack.empty() && !abort)
  {
    StackItem& item = itemstack.front();
    BaseVariable *v = item.var;

    //cout << "Stack top: (" << itemstack.front() << ")" << endl;

    switch (item.rtype)
    {
    case RecurseInitial:
      {
	const bool reallynew = !(v->data.epsilonequal(Epsilon, newval));
	if (!reallynew)
	{
	  itemstack.pop_front();
	  break;
	}
	 
	if (cs_debug)
	{
	  cout << "Recursion level = " << itemstack.size() << endl;
	    
	  cout << v->name << " S(" << v->levellevel << ")*";
	  for (index=0; index<(Index)(v->level); index++)
	    cout << " ";
	  cout << "*" << endl;
	    
	  cout << "Old value (" << v->data << ") " << (reallynew?"!=":"==")
	       << " newval (" << newval << ").  Using Epsilon of ("
	       << Epsilon << ")." << endl;
	    
	  cout << "LevelLevel is " << v->levellevel
	       << " and Level is " << v->level << "." << endl;
	}
	 
	v->data = newval;

	if (v->level++ == MaxDepth)
	{
	  v->level = 0;
	  if (++(v->levellevel) < v->numconstraints)
	  {
	    if (cs_debug)
	    {
	      cerr << "Maximum recursion level reached..." << endl;
	    }
	    item.rtype = RecurseMax;
	  }
	  else
	  {
	    if (cs_debug)
	    {
	      cout << v->name << " E(" << v->levellevel << ")*";
	      for (index=0; index< (Index)(v->level); index++)
	      {
		cout << " ";
	      }
	      cout << "*" << endl;
		  
	      cout << "Recursion level = " << itemstack.size()-1 << endl;
	    }
	       
	    cerr << "Maximum level reached for all constraints!" << endl;
	    cout << "Accepting current approximation." << endl;
	    cout << "Recursion level = " << itemstack.size()-1 << endl;
	       
	    abort = 1;	       
	  }
	}
	else
	{
	  item.rtype = RecurseNormal;
	}
      }
      break;

    case RecurseNormal:
      if (item.iter < v->numconstraints)
      {
	if (v->constraints[v->constraint_order[item.iter]]
	    ->Satisfy(v->constraint_indexs[v->constraint_order[item.iter]],
		      scheme, Epsilon, v, newval))
	{
	  StackItem si(v);
	  itemstack.push_front(si);
	}
	item.iter++;
      }
      else
      {
	if (v->level == 0)
	{
	  v->level = MaxDepth-1;
	  v->levellevel--;
	}
	else
	{
	  v->level--;
	}
	    
	if (cs_debug)
	{
	  cout << v->name << " E(" << v->levellevel << ")*";
	  for (index=0; index<(Index)(v->level); index++)
	    cout << " ";
	  cout << "*" << endl;
	  cout << "Recursion level = " << itemstack.size()-1 << endl;
	}
	itemstack.pop_front();
      }
      break;

    case RecurseMax:
      if (item.iter < v->numconstraints)
      {
	index2 = (item.iter + v->levellevel) % v->numconstraints;
	if (v->constraints[v->constraint_order[index2]]
	    ->Satisfy(v->constraint_indexs[v->constraint_order[index2]],
		      scheme, Epsilon, v, newval))
	{
	  itemstack.push_front(StackItem(v));
	}
	item.iter++;
      }
      else
      {
	if (v->level == 0)
	{
	  v->level = MaxDepth-1;
	  v->levellevel--;
	}
	else
	{
	  v->level--;
	}
	    
	if (cs_debug)
	{
	  cout << v->name << " E(" << v->levellevel << ")*";
	  for (index=0; index<(Index)(v->level); index++)
	  {
	    cout << " ";
	  }
	  cout << "*" << endl;
	  cout << "Recursion level = " << itemstack.size()-1 << endl;
	}
	itemstack.pop_front();
      }
      break;

    default:
      cerr << "ConstraintSolver::Solve something wrong." << endl;
      break;
    }
  }
}


void StackItem::print( ostream& os )
{
  os<<"StackItem:  "<<var->GetName()<<","<<rtype<<","<<iter<<endl;
}

} // End namespace SCIRun

