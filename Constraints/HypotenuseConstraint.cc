
/*
 *  HypotenousConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/HypotenuseConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch hc_debug("BaseConstraint", "Hypotenous");

HypotenuseConstraint::HypotenuseConstraint( const clString& name,
					    const Index numSchemes,
					    Variable* distInX, Variable* hypoInX )
:BaseConstraint(name, numSchemes, 2)
{
   vars[0] = distInX;
   vars[1] = hypoInX;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

HypotenuseConstraint::~HypotenuseConstraint()
{
}

void
HypotenuseConstraint::Satisfy( const Index index, const Scheme scheme )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];

   if (hc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   
   /* 2 * dist^2 = hypo^2 */
   switch (ChooseChange(index, scheme)) {
   case 0:
      v0.Assign(Point(sqrt(v1.Get().x() * v1.Get().x() / 2.0), 0, 0),
		scheme);
      break;
   case 1:
      v1.Assign(Point(sqrt(2.0 * v0.Get().x() * v0.Get().x()), 0, 0),
		scheme);
      break;
   default:
      cerr << "Unknown variable in Hypotenous Constraint!" << endl;
      break;
   }
}

