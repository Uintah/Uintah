
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
				  Variable* center, Variable* end1,
				  Variable* end2, Variable* p,
				  Variable* angleInX )
:BaseConstraint(name, numSchemes, 4)
{
   vars[0] = center;
   vars[1] = end1;
   vars[2] = end2;
   vars[3] = p;
   vars[4] = angleInX;

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
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Variable& v3 = *vars[3];
   Variable& v4 = *vars[4];
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
      v = ((v1.Get() - v0.Get()) * cos(v4.Get().x())
	   + (v2.Get() - v0.Get()) * sin(v4.Get().x()));

      if (v.length2() < v3.GetEpsilon()) {
	 v3.Assign(v0.Get(), scheme);
      } else {
	 v.normalize();
	 Real t = Dot(v3.Get() - v0.Get(), v);
	 v3.Assign(v0.Get() + (v * t), scheme);
      }
      break;
   case 4:
      v = v3.Get() - v0.Get();
      Real x(Dot(v1.Get() - v0.Get(),v));
      Real y(Dot(v2.Get() - v0.Get(),v));
      
      if ((abs(x) > v4.GetEpsilon()) || (abs(y) > v4.GetEpsilon()))
	 v4.Assign(Point(atan2(y,x), 0, 0), scheme);
      break;
   default:
      cerr << "Unknown variable in Angle Constraint!" << endl;
      break;
   }
}

