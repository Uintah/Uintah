
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
					Variable* p1, Variable* p2,
					Variable* distInX )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = distInX;
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
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Point temp;

   if (dc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   
   /* Q <- Sc + Sr * Normalize(P-Sc) */
   switch (ChooseChange(index, scheme)) {
   case 0:
      temp = v1.Get()
		+ ((v0.Get() - v1.Get()).normal()
		   * v2.Get().x());
      temp.z(0.0);
      v0.Assign(temp,
		scheme);
      break;
   case 1:
      temp = v0.Get()
		+ ((v1.Get() - v0.Get()).normal()
		   * v2.Get().x());
      temp.z(0.0);
      v1.Assign(temp,
		scheme);
      break;
   case 2:
      temp.x((v1.Get() - v0.Get()).length());
      v2.Assign(temp, scheme);
      break;
   default:
      cerr << "Unknown variable in Distance Constraint!" << endl;
      break;
   }
}

