
/*
 *  BaseConstraint.h
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
#include <Constraints/BaseConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch bc_debug("BaseConstraint", "Print");

static int recursion=0;

/******* Variables *******/

Variable::Variable( const clString& name, const Scheme scheme,
		    const Point& value )
: name(name), scheme(scheme), value(value)
{
   numconstraints = 0;
   levellevel = level = 0;
}

Variable::~Variable()
{
}

void
Variable::Order()
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
Variable::Resolve()
{
   Index index;

   for (index = 0; index < numconstraints; index++)
      constraints[constraint_order[index]]->Satisfy(constraint_indexs[constraint_order[index]], Scheme1);
}


void
Variable::Set( const Point& newValue )
{
   level = levellevel = 0;
   Assign(newValue, scheme);
}


void
Variable::SetDelta( const Vector& deltaValue )
{
   recursion = 0;
   Assign(value + deltaValue, scheme);
}


inline int
epsilonequal( const Real Epsilon, const Point& p1, const Point& p2 )
{
   return ((RealAbs(p1.x()-p2.x()) < Epsilon)
	   && (RealAbs(p1.y()-p2.y()) < Epsilon)
	   && (RealAbs(p1.z()-p2.z()) < Epsilon));
}


const Index MaxDepth = 25;
void
Variable::Assign( const Point& newValue, const Scheme scheme )
{
   if (level <= -1234) return; // Backing out
   
   Index index, index2;
   int reallynew = !(epsilonequal(Epsilon, value, newValue));

   if (bc_debug) {
      cout << "Recursion level = " << recursion++ << endl;
      
      cout << name << " S(" << levellevel << ")*";
      for (index=0; index<level; index++)
	 cout << " ";
      cout << "*" << endl;
      
      cout << "Old value (" << (Point)value << ") " << (reallynew?"!=":"==")
	   << " newValue (" << (Point)newValue << ").  Using Epsilon of ("
	   << Epsilon << ")." << endl;

      cout << "LevelLevel is " << levellevel << " and Level is " << level << "." << endl;
   }
   
   value = newValue;

   if (++level == MaxDepth) {
      level = 0;
      if (++levellevel < numconstraints) {
	 cerr << "Maximum recursion level reached...\n";
	 for (index = levellevel; index < numconstraints+levellevel; index++) {
	    index2 = index % numconstraints;
	    constraints[constraint_order[index2]]
	       ->Satisfy(constraint_indexs[constraint_order[index2]], scheme);
	 }
      }
      else {
	 if (bc_debug) {
	    cout << name << " E(" << levellevel << ")*";
	    for (index=0; index<level; index++)
	       cout << " ";
	    cout << "*" << endl;
	    
	    cout << "Recursion level = " << --recursion << endl;
	 }

	 levellevel = level = -1234;

	 cerr << "Maximum level reached for all constraints!" << endl;
	 cout << "Accepting current approximation." << endl;
	 cout << "Recursion level = " << recursion << endl;

	 return;
      }
   }
   else if (reallynew)
      for (index = 0; index < numconstraints; index++)
	 constraints[constraint_order[index]]
	    ->Satisfy(constraint_indexs[constraint_order[index]], scheme);
   if (level == 0) {
      level = MaxDepth-1;
      levellevel--;
   }
   else
      level--;

   if (bc_debug) {
      cout << name << " E(" << levellevel << ")*";
      for (index=0; index<level; index++)
	 cout << " ";
      cout << "*" << endl;
      
      cout << "Recursion level = " << --recursion << endl;
   }
}

void
Variable::print( ostream& os )
{
   os << name << " " << value;
}

void
Variable::printc( ostream& os, const Index c )
{
   os << name << " " << value
      << " (Index " << constraint_indexs[c] << ") "
      << " (Constraint " << PriorityString(constraint_priorities[c]) << ")";
}


/******* BaseConstraints *******/

BaseConstraint::BaseConstraint( const clString& name, const Index nschemes,
				const Index varCount )
: name(name), nschemes(nschemes), varCount(varCount),
  vars(varCount), var_indexs(varCount), var_choices(nschemes, varCount)
{
   whichMethod = 0;
}

BaseConstraint::~BaseConstraint()
{
}

void
BaseConstraint::Priorities( const VPriority p1,
			    const VPriority p2,
			    const VPriority p3,
			    const VPriority p4,
			    const VPriority p5,
			    const VPriority p6,
			    const VPriority p7 )
{
   Index p=0;
   
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p1);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p2);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p3);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p4);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p5);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p6);
   if (p == varCount) return;
   vars[p]->RegisterPriority(var_indexs[p++], p7);
}

void
BaseConstraint::VarChoices( const Scheme scheme,
			    const Index i1,
			    const Index i2,
			    const Index i3,
			    const Index i4,
			    const Index i5,
			    const Index i6,
			    const Index i7 )
{
   ASSERT(scheme<nschemes);
   Index p=0;
   
   if (p == varCount) return;
   var_choices(scheme, p++) = i1;
   if (p == varCount) return;
   var_choices(scheme, p++) = i2;
   if (p == varCount) return;
   var_choices(scheme, p++) = i3;
   if (p == varCount) return;
   var_choices(scheme, p++) = i4;
   if (p == varCount) return;
   var_choices(scheme, p++) = i5;
   if (p == varCount) return;
   var_choices(scheme, p++) = i6;
   if (p == varCount) return;
   var_choices(scheme, p++) = i7;
}

void
BaseConstraint::Satisfy( const Index, const Scheme )
{
   Error("BaseConstraint: Can't satisfy!");
}

Index
BaseConstraint::ChooseChange( const Index index, const Scheme scheme )
{
   callingMethod = index;
   
   return whichMethod = var_choices(scheme, index);
}

void
BaseConstraint::print( ostream& os )
{
    int i;

    for (Index j=0; j < nschemes; j++) {
       os << name << " (" << SchemeString((Scheme)j) << ") (";
       for (i = 0; i < varCount; i++) {
	  if (i != whichMethod) {
	     os << "\t";
	     vars[i]->printc(os, var_indexs[i]);
	     os << " (->" << var_choices(j, i) << ")";
	     os << endl;
	  }
       }
       os << "\t-> ";
       vars[whichMethod]->printc(os, var_indexs[whichMethod]);
       os << " (->" << var_choices(j, whichMethod) << ")";
       os << ")" << endl;
    }
}

void
BaseConstraint::printc( ostream& os, const Scheme scheme )
{
    int i;

    os << name << " (" << SchemeString(scheme) << ") (";
    os << "Called by " << callingMethod << ") (" << endl;
    for (i = 0; i < varCount; i++) {
       if (i != whichMethod) {
	  os << "\t";
	  vars[i]->printc(os, var_indexs[i]);
	  os << " (->" << var_choices(scheme, i) << ")";
	  os << endl;
       }
    }
    os << "\t-> ";
    vars[whichMethod]->printc(os, var_indexs[whichMethod]);
    os << " (->" << var_choices(scheme, whichMethod) << ")";
    os << ")" << endl;
}

void
BaseConstraint::Register()
{
   Index index;

   for (index = 0; index < varCount; index++)
      var_indexs[index] = vars[index]->Register(this, index);
}

/******* Miscellaneous Functions *******/

char*
PriorityString( const VPriority p )
{
   static char temp[20];
   
   switch (p) {
   case P_Constant:		return "Constant";
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

