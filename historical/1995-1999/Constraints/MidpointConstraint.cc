
/*
 *  MidpointConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/MidpointConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch mc_debug("Constraints", "Midpoint");

MidpointConstraint::MidpointConstraint( const clString& name,
					const Index numSchemes,
					PointVariable* end1, PointVariable* end2,
					PointVariable* p )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = end1;
   vars[1] = end2;
   vars[2] = p;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

MidpointConstraint::~MidpointConstraint()
{
}


int
MidpointConstraint::Satisfy( const Index index, const Scheme scheme, const Real,
			     BaseVariable*& var, VarCore& c )
{
   PointVariable& end1 = *vars[0];
   PointVariable& end2 = *vars[1];
   PointVariable& p = *vars[2];

   if (mc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      var = vars[0];
      c = (Point)end2 + ((Point)p - end2) * 2.0;
      return 1;
   case 1:
      var = vars[1];
      c = (Point)end1 + ((Point)p - end1) * 2.0;
      return 1;
   case 2:
      var = vars[2];
      c = (Point)end1 + ((Point)end2 - end1) / 2.0;
      return 1;
   default:
      cerr << "Unknown variable in Midpoint Constraint!" << endl;
      break;
   }
   return 0;
}

