
/*
 *  PythagorasConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/PythagorasConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch pc_debug("BaseConstraint", "Pythagoras");

PythagorasConstraint::PythagorasConstraint( const clString& name,
					    const Index numSchemes,
					    RealVariable* dist1, RealVariable* dist2,
					    RealVariable* hypo )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = dist1;
   vars[1] = dist2;
   vars[2] = hypo;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

PythagorasConstraint::~PythagorasConstraint()
{
}


void
PythagorasConstraint::Satisfy( const Index index, const Scheme scheme )
{
   RealVariable& v0 = *vars[0];
   RealVariable& v1 = *vars[1];
   RealVariable& v2 = *vars[2];
   Real t;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   /* A^2 + B^2 = C^2 */
   switch (ChooseChange(index, scheme)) {
   case 0:
      if ((t = v2.GetReal() * v2.GetReal() - v1.GetReal() * v1.GetReal()) >= 0.0)
	 v0.Assign(sqrt(t), scheme);
      break;
   case 1:
      if ((t = v2.GetReal() * v2.GetReal() - v0.GetReal() * v0.GetReal()) >= 0.0)
	 v1.Assign(sqrt(t), scheme);
      break;
   case 2:
      if ((t = v0.GetReal() * v0.GetReal() + v1.GetReal() * v1.GetReal()) >= 0.0)
	 v2.Assign(sqrt(t), scheme);
      break;
   default:
      cerr << "Unknown variable in Pythagoras Constraint!" << endl;
      break;
   }
}

