
/*
 *  Plane4Constraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/PlaneConstraint.h>
#include <Geometry/Plane.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch p4c_debug("BaseConstraint", "Plane");

PlaneConstraint::PlaneConstraint( const clString& name,
				    const Index numSchemes,
				    Variable* p1, Variable* p2,
				    Variable* p3, Variable* p4)
:BaseConstraint(name, numSchemes, 4)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = p3;
   vars[3] = p4;

   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

PlaneConstraint::~PlaneConstraint()
{
}


void
PlaneConstraint::Satisfy( const Index index, const Scheme scheme )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Variable& v3 = *vars[3];

   if (p4c_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      {
	 Plane plane(v1.Get(), v2.Get(), v3.Get());
	 v0.Assign(plane.project(v0.Get()), scheme);
      }
      break;
   case 1:
      {
	 Plane plane(v0.Get(), v2.Get(), v3.Get());
	 v1.Assign(plane.project(v1.Get()), scheme);
      }
      break;
   case 2:
      {
	 Plane plane(v0.Get(), v1.Get(), v3.Get());
	 v2.Assign(plane.project(v2.Get()), scheme);
      }
      break;
   case 3:
      
      {
	 Plane plane(v0.Get(), v1.Get(), v2.Get());
	 v3.Assign(plane.project(v3.Get()), scheme);
      }
      break;
   default:
      cerr << "Unknown variable in Plane Constraint!" << endl;
      break;
   }
}

