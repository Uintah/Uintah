
/*
 *  AngleConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Constraints/AngleConstraint.h>
#include <Geometry/Plane.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch ac_debug("Constraints", "Angle");

AngleConstraint::AngleConstraint( const clString& name,
				  const Index numSchemes,
				  PointVariable* center, PointVariable* end1,
				  PointVariable* end2, PointVariable* p,
				  RealVariable* angle )
:BaseConstraint(name, numSchemes, 5)
{
   vars[0] = center;
   vars[1] = end1;
   vars[2] = end2;
   vars[3] = p;
   vars[4] = angle;

   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

AngleConstraint::~AngleConstraint()
{
}


int
AngleConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   PointVariable& center = *vars[0];
   PointVariable& end1 = *vars[1];
   PointVariable& end2 = *vars[2];
   PointVariable& p = *vars[3];
   RealVariable& angle = *vars[4];
   Vector v;

   if (ac_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      NOT_FINISHED("Line Constraint:  center");
      break;
   case 1:
      NOT_FINISHED("Line Constraint:  end1");
      break;
   case 2:
      NOT_FINISHED("Line Constraint:  end2");
      break;
   case 3:
      v = (((Point)end1 - center) * cos(angle) + ((Point)end2 - center) * sin(angle));

      if (v.length2() < Epsilon) {
	 c = (Point)end1;
      } else {
	 v.normalize();
	 Real t = Dot((Point)p - center, v);
	 c = (Point)center + (v * t);
      }
      var = vars[3];
      return 1;
   case 4:
      v = (Point)p - center;
      Real x(Dot((Point)end1 - center,v));
      Real y(Dot((Point)end2 - center,v));
      
      if ((fabs(x) > Epsilon) || (fabs(y) > Epsilon)) {
	 var = vars[4];
	 c = atan2(y,x);
	 return 1;
      }
      break;
   default:
      cerr << "Unknown variable in Angle Constraint!" << endl;
      break;
   }
   return 0;
}
