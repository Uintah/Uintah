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


ConstraintSolver::ConstraintSolver( const double epsilon )
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
ConstraintSolver::SetEpsilon( const double epsilon )
{
   Epsilon = epsilon;
}


double
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
   
  VarCore newval(newValue);
  list<StackItem> itemstack;
  bool abort_flag = false;
  unsigned int i;

  for (i=0; i < variables.size(); i++)
  {
    variables[i]->level = 0;
  }
  itemstack.push_front(StackItem(var));

  while (!itemstack.empty() && !abort_flag)
  {
    StackItem& item = itemstack.front();
    BaseVariable *v = item.var;

    switch (item.rtype)
    {
    case RecurseInitial:
      {
	const bool notnew = v->data.epsilonequal(Epsilon, newval);

	if (cs_debug)
	{
	  cout << "Recursion level = " << itemstack.size() << endl;
	    
	  cout << v->name << " S(" << v->level / v->numconstraints << ")*";
	  for (i=0; i < (v->level % v->numconstraints); i++)
	    cout << " ";
	  cout << "*" << endl;
	    
	  cout << "Old value (" << v->data << ") " << (notnew?"==":"!=")
	       << " newval (" << newval << ").  Using Epsilon of ("
	       << Epsilon << ")." << endl;
	    
	  cout << "LevelLevel is " << v->level / v->numconstraints
	       << " and Level is " << v->level % v->numconstraints 
	       << "." << endl;
	}
	 
	if (notnew)
	{
	  itemstack.pop_front();
	  break;
	}
	 
	v->data = newval;
	v->level++;
	if (v->level > v->numconstraints * MaxDepth)
	{
	  if (cs_debug)
	  {
	    cout << v->name << " E(" << v->level / v->numconstraints << ")*";
	    for (i=0; i < v->level; i++)
	    {
	      cout << " ";
	    }
	    cout << "*" << endl;
	    
	    cout << "Recursion level = " << itemstack.size()-1 << endl;
	  }
	       
	  cerr << "Maximum level reached for all constraints!" << endl;
	  cout << "Accepting current approximation." << endl;
	  cout << "Recursion level = " << itemstack.size()-1 << endl;
	       
	  abort_flag = true;	       
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
	const unsigned int offset = v->level / v->numconstraints;
	const unsigned int index = (item.iter + offset) % v->numconstraints;
	const unsigned int order = v->constraint_order[index];
	BaseVariable *nextv;
	if (v->constraints[order]->Satisfy(v->constraint_indices[order],
					   scheme, Epsilon, nextv, newval))
	{
	  itemstack.push_front(StackItem(nextv));
	}
	item.iter++;
      }
      else
      {
	v->level--;
	
	if (cs_debug)
	{
	  cout << v->name << " E(" << v->level / v->numconstraints << ")*";
	  for (i = 0; i <v->level % v->numconstraints; i++)
	    cout << " ";
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

