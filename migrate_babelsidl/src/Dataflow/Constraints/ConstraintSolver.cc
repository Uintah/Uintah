/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <stdio.h>
#include <string.h>

#include <iostream>
#include <algorithm>
#include <list>

using std::cerr;
using std::cout;
using std::endl;
using std::ostream;
using std::list;

namespace SCIRun {

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
  variables.erase(std::remove(variables.begin(), variables.end(), v),
		  variables.end());
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

