
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


#include <stdio.h>
#include <string.h>
#include <Constraints/BaseVariable.h>
#include <Constraints/ConstraintSolver.h>


BaseVariable::BaseVariable( const clString& name,
			    ConstraintSolver* s, const Scheme scheme,
			    const Point& value )
: name(name), scheme(scheme), data(value, VarCore::Rigid),
  numconstraints(0), levellevel(0), level(0), solver(s)
{
   solver->AddVariable(this);
}


BaseVariable::BaseVariable( const clString& name,
			    ConstraintSolver* s, const Scheme scheme,
			    const Real value )
: name(name), scheme(scheme), data(value, VarCore::Rigid),
  numconstraints(0), levellevel(0), level(0), solver(s)
{
   solver->AddVariable(this);
}


BaseVariable::~BaseVariable()
{
}


void
BaseVariable::Order()
{
   if (numconstraints == 0)
      return;
   
   int index; // Goes negative at end of loop.
   Index index2, highindex;

   for (index=0; index<numconstraints; index++)
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
BaseVariable::Set( const Real newValue, const Scheme s )
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
BaseVariable::SetDelta( const Real deltaValue, const Scheme s )
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
BaseVariable::Move( const Real newValue )
{
   data = newValue;
}


void
BaseVariable::MoveDelta( const Vector& deltaValue )
{
   data += deltaValue;
}


void
BaseVariable::MoveDelta( const Real deltaValue )
{
   data += deltaValue;
}


Index
BaseVariable::Register( BaseConstraint* constraint, const Index index )
{
   constraints.grow(1);
   constraints[numconstraints] = constraint;
   constraint_indexs.grow(1);
   constraint_indexs[numconstraints] = index;
   constraint_priorities.grow(1);
   constraint_priorities[numconstraints] = P_Default;
   constraint_order.grow(1);
   constraint_order[numconstraints] = 0;
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
      << " (Index " << constraint_indexs[c] << ") "
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

