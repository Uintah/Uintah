
/*
 *  AngleConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/AngleConstraint.h>
#include <Geometry/Plane.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch ac_debug("BaseConstraint", "Angle");

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


void
AngleConstraint::Satisfy( const Index index, const Scheme scheme )
{
   PointVariable& v0 = *vars[0];
   PointVariable& v1 = *vars[1];
   PointVariable& v2 = *vars[2];
   PointVariable& v3 = *vars[3];
   RealVariable& v4 = *vars[4];
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
      v = ((v1.GetPoint() - v0.GetPoint()) * cos(v4.GetReal())
	   + (v2.GetPoint() - v0.GetPoint()) * sin(v4.GetReal()));

      if (v.length2() < v3.GetEpsilon()) {
	 v3.Assign(v1.GetPoint(), scheme);
      } else {
	 v.normalize();
	 Real t = Dot(v3.GetPoint() - v0.GetPoint(), v);
	 v3.Assign(v0.GetPoint() + (v * t), scheme);
      }
      break;
   case 4:
      v = v3.GetPoint() - v0.GetPoint();
      Real x(Dot(v1.GetPoint() - v0.GetPoint(),v));
      Real y(Dot(v2.GetPoint() - v0.GetPoint(),v));
      
      if ((fabs(x) > v4.GetEpsilon()) || (fabs(y) > v4.GetEpsilon()))
	 v4.Assign(atan2(y,x), scheme);
      break;
   default:
      cerr << "Unknown variable in Angle Constraint!" << endl;
      break;
   }
}

