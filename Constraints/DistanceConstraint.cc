
/*
 *  DistanceConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/DistanceConstraint.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch dc_debug("BaseConstraint", "Distance");

DistanceConstraint::DistanceConstraint( const clString& name,
					const Index numSchemes,
					PointVariable* p1, PointVariable* p2,
					RealVariable* dist )
:BaseConstraint(name, numSchemes, 3),
 guess(1, 0, 0), minimum(0.0)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = dist;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

DistanceConstraint::~DistanceConstraint()
{
}


void
DistanceConstraint::Satisfy( const Index index, const Scheme scheme )
{
   PointVariable& v0 = *vars[0];
   PointVariable& v1 = *vars[1];
   RealVariable& v2 = *vars[2];
   Vector v;
   Real t;

   if (dc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v = (v0.GetPoint() - v1.GetPoint());
      if (v.length2() < v0.GetEpsilon())
	 v = guess;
      else
	 v.normalize();
      if (v2.GetReal() < minimum) {
	 t = minimum;
	 v2.Assign(t, scheme);
      } else
	 t = v2.GetReal();
      v0.Assign(v1.GetPoint() + (v * t), scheme);
      break;
   case 1:
      v = (v1.GetPoint() - v0.GetPoint());
      if (v.length2() < v1.GetEpsilon())
	 v = guess;
      else
	 v.normalize();
      if (v2.GetReal() < minimum) {
	 t = minimum;
	 v2.Assign(t, scheme);
      } else
	 t = v2.GetReal();
      v1.Assign(v0.GetPoint() + (v * t), scheme);
      break;
   case 2:
      t = (v1.GetPoint() - v0.GetPoint()).length();
      if (t < minimum) {
	 t = minimum;
	 if (index == 1) {
	    v = (v1.GetPoint() - v0.GetPoint());
	    if (v.length2() < v1.GetEpsilon())
	       v = guess;
	    else
	       v.normalize();
	    v0.Assign(v0.GetPoint() + (v*t), scheme);
	 } else {
	    v = (v0.GetPoint() - v1.GetPoint());
	    if (v.length2() < v0.GetEpsilon())
	       v = guess;
	    else
	       v.normalize();
	    v1.Assign(v1.GetPoint() + (v*t), scheme);
	 }
      }
      v2.Assign(t, scheme);
      break;
   default:
      cerr << "Unknown variable in Distance Constraint!" << endl;
      break;
   }
}

