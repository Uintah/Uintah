
/*
 *  LineConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/LineConstraint.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch lc_debug("Constraints", "Line");

LineConstraint::LineConstraint( const clString& name,
				const Index numSchemes,
				PointVariable* p1, PointVariable* p2,
				PointVariable* p3 )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = p3;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

LineConstraint::~LineConstraint()
{
}


int
LineConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			 BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[0];
   PointVariable& p2 = *vars[1];
   PointVariable& p3 = *vars[2];
   Vector norm;

   if (lc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = (Point)p3 - p2;
      if (norm.length2() < Epsilon) {
	 c = (Point)p3;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p1 - p2, norm));
	 c = (Point)p2 + (norm * t);
      }
      var = vars[0];
      return 1;
   case 1:
      norm = (Point)p3 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p3;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p2 - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[1];
      return 1;
   case 2:
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p2;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p3 - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[2];
      return 1;
   default:
      cerr << "Unknown variable in Line Constraint!" << endl;
      break;
   }
   return 0;
}

