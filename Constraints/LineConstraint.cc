
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

static DebugSwitch lc_debug("BaseConstraint", "Line");

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


void
LineConstraint::Satisfy( const Index index, const Scheme scheme )
{
   PointVariable& v0 = *vars[0];
   PointVariable& v1 = *vars[1];
   PointVariable& v2 = *vars[2];
   Vector norm;

   if (lc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = v2.GetPoint() - v1.GetPoint();
      if (norm.length2() < v2.GetEpsilon()) {
	 v0.Assign(v2.GetPoint(), scheme);
      } else {
	 norm.normalize();
	 Real t = Dot(v0.GetPoint() - v1.GetPoint(), norm);
	 v0.Assign(v1.GetPoint() + (norm * t), scheme);
      }
      break;
   case 1:
      norm = v2.GetPoint() - v0.GetPoint();
      if (norm.length2() < v2.GetEpsilon()) {
	 v1.Assign(v2.GetPoint(), scheme);
      } else {
	 norm.normalize();
	 Real t = Dot(v1.GetPoint() - v0.GetPoint(), norm);
	 v1.Assign(v0.GetPoint() + (norm * t), scheme);
      }
      break;
   case 2:
      norm = v1.GetPoint() - v0.GetPoint();
      if (norm.length2() < v2.GetEpsilon()) {
	 v2.Assign(v1.GetPoint(), scheme);
      } else {
	 norm.normalize();
	 Real t = Dot(v2.GetPoint() - v0.GetPoint(), norm);
	 v2.Assign(v0.GetPoint() + (norm * t), scheme);
      }
      break;
   default:
      cerr << "Unknown variable in Line Constraint!" << endl;
      break;
   }
}

