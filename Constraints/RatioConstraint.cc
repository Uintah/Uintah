
/*
 *  RatioConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Constraints/RatioConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch rc_debug("Constraints", "Ratio");

RatioConstraint::RatioConstraint( const clString& name,
				  const Index numSchemes,
				  RealVariable* numer, RealVariable* denom,
				  RealVariable* ratio )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = numer;
   vars[1] = denom;
   vars[2] = ratio;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

RatioConstraint::~RatioConstraint()
{
}


int
RatioConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   RealVariable& numer = *vars[0];
   RealVariable& denom = *vars[1];
   RealVariable& ratio = *vars[2];

   if (rc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   Real temp;
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      var = vars[0];
      c = denom * ratio;
      return 1;
   case 1:
      if (ratio < Epsilon)
	 temp = denom; // Don't change denom since 0/any == 0
      else
	 temp = numer / ratio;
      var = vars[1];
      c = temp;
      return 1;
   case 2:
      if (denom < Epsilon)
	 temp = ratio; // Don't change denom since 0/any == 0
      else
	 temp = numer / denom;
      var = vars[2];
      c = temp;
      return 1;
   default:
      cerr << "Unknown variable in Ratio Constraint!" << endl;
      break;
   }
   return 0;
}

