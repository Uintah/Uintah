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
   os << " " << (int)i.iter << "/" << i.var->GetNumConstraints();

   return os;
}


void StackItem::print( ostream& os )
{
  os << *this << endl;
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
  unsigned int i;

  for (i=0; i < variables.size(); i++)
  {
    variables[i]->level = 0;
  }
  itemstack.push_front(StackItem(var));

  while (!itemstack.empty())
  {
    StackItem& item = itemstack.front();
    BaseVariable *v = item.var;

    if (item.iter == 0)
    {
      // Skip this item if there is no real change to be made.
      if (v->data.epsilonequal(Epsilon, newval))
      {
	itemstack.pop_front();
	continue;
      }

      v->data = newval;
      v->level++;
      if (v->level > v->numconstraints * MaxDepth)
      {
	cerr << "Maximum level reached for all constraints!" << endl;
	cout << "Accepting current approximation." << endl;
	cout << "Recursion level = " << itemstack.size()-1 << endl;
	break;
      }
    }
    
    if (item.iter < v->numconstraints)
    {
      // Ordered depth first search on the current constraints.
      // If level is getting too high, try a different order.
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
      itemstack.pop_front();
    }
  }
}


} // End namespace SCIRun

