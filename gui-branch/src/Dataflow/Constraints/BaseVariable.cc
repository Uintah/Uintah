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
 *  BaseVariable.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Constraints/BaseVariable.h>
#include <Dataflow/Constraints/ConstraintSolver.h>
#include <iostream>
using std::ostream;

#include <stdio.h>

namespace SCIRun {

BaseVariable::BaseVariable( const string& name,
			    ConstraintSolver* s, const Scheme scheme,
			    const Point& value )
: name(name), data(value, VarCore::Rigid), solver(s),
  level(0), numconstraints(0), scheme(scheme)
{
   solver->AddVariable(this);
}


BaseVariable::BaseVariable( const string& name,
			    ConstraintSolver* s, const Scheme scheme,
			    const double value )
: name(name), data(value, VarCore::Rigid), solver(s),
  level(0), numconstraints(0), scheme(scheme)
{
   solver->AddVariable(this);
}


BaseVariable::~BaseVariable()
{
   solver->RemoveVariable(this);
}


void
BaseVariable::Order()
{
   if (numconstraints == 0)
      return;
   
   int index; // Goes negative at end of loop.
   Index index2, highindex;

   for (index=0; (Index)index< numconstraints; index++)
      constraint_order[index] = index;

   // Bubble sort.
   for (index=numconstraints-2; index>=0; index--)
      for (index2=index; index2<numconstraints-1; index2++)
	 if (HigherPriority(constraint_priorities[constraint_order[index+1]],
			    constraint_priorities[constraint_order[index]])) {
	    highindex = constraint_order[index];
	    constraint_order[index] = constraint_order[index+1];
	    constraint_order[index+1] = highindex;
	 }  
}


void
BaseVariable::Set( const Point& newValue, const Scheme s )
{
   if (s == DefaultScheme)
      solver->Solve(this, VarCore(newValue), scheme);
   else
      solver->Solve(this, VarCore(newValue), s);
}


void
BaseVariable::Set( const double newValue, const Scheme s )
{
   if (s == DefaultScheme)
      solver->Solve(this, VarCore(newValue), scheme);
   else
      solver->Solve(this, VarCore(newValue), s);
}


void
BaseVariable::SetDelta( const Vector& deltaValue, const Scheme s )
{
   if (s == DefaultScheme)
      solver->Solve(this, VarCore(data.point() + deltaValue), scheme);
   else
      solver->Solve(this, VarCore(data.point() + deltaValue), s);
}


void
BaseVariable::SetDelta( const double deltaValue, const Scheme s )
{
   if (s == DefaultScheme)
      solver->Solve(this, VarCore(data.real() + deltaValue), scheme);
   else
      solver->Solve(this, VarCore(data.real() + deltaValue), s);
}


void
BaseVariable::Move( const Point& newValue )
{
   data = newValue;
}


void
BaseVariable::Move( const double newValue )
{
   data = newValue;
}


void
BaseVariable::MoveDelta( const Vector& deltaValue )
{
   data += deltaValue;
}


void
BaseVariable::MoveDelta( const double deltaValue )
{
   data += deltaValue;
}


Index
BaseVariable::Register( BaseConstraint* constraint, const Index index )
{
   constraints.push_back(constraint);
   constraint_indices.push_back(index);
   constraint_priorities.push_back(P_Default);
   constraint_order.push_back(0);
   numconstraints++;

   return numconstraints-1;
}


void
BaseVariable::RegisterPriority( const Index index, const VPriority p )
{
   ASSERT(index < numconstraints);

   constraint_priorities[index] = p;
}


void
BaseVariable::print( ostream& os )
{
   os << name << " " << data;
}


void
BaseVariable::printc( ostream& os, const Index c )
{
   os << name << " " << data
      << " (Index " << constraint_indices[c] << ") "
      << " (Constraint " << PriorityString(constraint_priorities[c]) << ")";
}


/******* Miscellaneous Functions *******/

char*
PriorityString( const VPriority p )
{
   static char temp[20];
   
   switch (p) {
   case P_Lowest:	        return "Lowest";
   case P_LowMedium:		return "LowMedium";
   case P_Default:	        return "Default";
   case P_HighMedium:		return "HighMedium";
   case P_Highest:		return "Highest";
   default:
      sprintf(temp, "priority[%d]", p);
      return temp;
   }
}


char*
SchemeString( const Scheme s )
{
   static char temp[20];
   
   switch (s) {
   case Scheme1:		return "Scheme 1";
   case Scheme2:		return "Scheme 2";
   case Scheme3:		return "Scheme 3";
   case Scheme4:		return "Scheme 4";
   case Scheme5:		return "Scheme 5";
   case Scheme6:		return "Scheme 6";
   case Scheme7:		return "Scheme 7";
   case Scheme8:		return "Scheme 8";
   default:
      sprintf(temp, "scheme[%d]", s);
      return temp;
   }
}

} // End namespace SCIRun

