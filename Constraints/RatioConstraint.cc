
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
				  Variable* numerInX, Variable* denomInX,
				  Variable* ratioInX )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = numerInX;
   vars[1] = denomInX;
   vars[2] = ratioInX;
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
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];

   if (rc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   Point temp(0,0,0);
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      temp.x(v1.Get().x() * v2.Get().x());
      v0.Assign(temp, scheme);
      break;
   case 1:
      if (v2.Get().x() < v1.GetEpsilon())
	 temp.x(v1.Get().x()); // Don't change v1 since 0/any == 0
      else
	 temp.x(v0.Get().x() / v2.Get().x());
      v1.Assign(temp, scheme);
      break;
   case 2:
      if (v1.Get().x() < v2.GetEpsilon())
	 temp.x(v2.Get().x()); // Don't change v1 since 0/any == 0
      else
	 temp.x(v0.Get().x() / v1.Get().x());
      v2.Assign(temp, scheme);
      break;
   default:
      cerr << "Unknown variable in Ratio Constraint!" << endl;
      break;
   }
}

