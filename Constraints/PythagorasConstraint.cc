
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

static DebugSwitch pc_debug("Constraints", "Pythagoras");

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


int
PythagorasConstraint::Satisfy( const Index index, const Scheme scheme, const Real,
			       BaseVariable*& var, VarCore& c )
{
   RealVariable& dist1 = *vars[0];
   RealVariable& dist2 = *vars[1];
   RealVariable& hypo = *vars[2];
   Real t;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      if ((t = hypo * hypo - dist2 * dist2) >= 0.0) {
	 var = vars[0];
	 c = sqrt(t);
	 return 1;
      }
      break;
   case 1:
      if ((t = hypo * hypo - dist1 * dist1) >= 0.0) {
	 var = vars[1];
	 c = sqrt(t);
	 return 1;
      }
      break;
   case 2:
      if ((t = dist1 * dist1 + dist2 * dist2) >= 0.0) {
	 var = vars[2];
	 c = sqrt(t);
	 return 1;
      }
      break;
   default:
      cerr << "Unknown variable in Pythagoras Constraint!" << endl;
      break;
   }
   return 0;
}

