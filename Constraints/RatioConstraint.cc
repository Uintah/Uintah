
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

static DebugSwitch rc_debug("BaseConstraint", "Ratio");

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


void
RatioConstraint::Satisfy( const Index index, const Scheme scheme )
{
   RealVariable& v0 = *vars[0];
   RealVariable& v1 = *vars[1];
   RealVariable& v2 = *vars[2];

   if (rc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   Real temp;
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v0.Assign(v1.GetReal() * v2.GetReal(), scheme);
      break;
   case 1:
      if (v2.GetReal() < v1.GetEpsilon())
	 temp = v1.GetReal(); // Don't change v1 since 0/any == 0
      else
	 temp = v0.GetReal() / v2.GetReal();
      v1.Assign(temp, scheme);
      break;
   case 2:
      if (v1.GetReal() < v2.GetEpsilon())
	 temp = v2.GetReal(); // Don't change v1 since 0/any == 0
      else
	 temp = v0.GetReal() / v1.GetReal();
      v2.Assign(temp, scheme);
      break;
   default:
      cerr << "Unknown variable in Ratio Constraint!" << endl;
      break;
   }
}

