
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

static DebugSwitch pc_debug("BaseConstraint", "Plane");

PlaneConstraint::PlaneConstraint( const clString& name,
				  const Index numSchemes,
				  PointVariable* p1, PointVariable* p2,
				  PointVariable* p3, PointVariable* p4)
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
   PointVariable& v0 = *vars[0];
   PointVariable& v1 = *vars[1];
   PointVariable& v2 = *vars[2];
   PointVariable& v3 = *vars[3];
   Vector vec1, vec2;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      vec1 = (v1.GetPoint() - v2.GetPoint());
      vec2 = (v3.GetPoint() - v2.GetPoint());
      if (Cross(vec1, vec2).length2() < v0.GetEpsilon()) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(v1.GetPoint(), v2.GetPoint(), v3.GetPoint());
	 v0.Assign(plane.project(v0.GetPoint()), scheme);
      }
      break;
   case 1:
      vec1 = (v0.GetPoint() - v2.GetPoint());
      vec2 = (v3.GetPoint() - v2.GetPoint());
      if (Cross(vec1, vec2).length2() < v1.GetEpsilon()) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(v0.GetPoint(), v2.GetPoint(), v3.GetPoint());
	 v1.Assign(plane.project(v1.GetPoint()), scheme);
      }
      break;
   case 2:
      vec1 = (v0.GetPoint() - v1.GetPoint());
      vec2 = (v3.GetPoint() - v1.GetPoint());
      if (Cross(vec1, vec2).length2() < v2.GetEpsilon()) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(v0.GetPoint(), v1.GetPoint(), v3.GetPoint());
	 v2.Assign(plane.project(v2.GetPoint()), scheme);
      }
      break;
   case 3:
      vec1 = (v0.GetPoint() - v1.GetPoint());
      vec2 = (v2.GetPoint() - v1.GetPoint());
      if (Cross(vec1, vec2).length2() < v3.GetEpsilon()) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(v0.GetPoint(), v1.GetPoint(), v2.GetPoint());
	 v3.Assign(plane.project(v3.GetPoint()), scheme);
      }
      break;
   default:
      cerr << "Unknown variable in Plane Constraint!" << endl;
      break;
   }
}

