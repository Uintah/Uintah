
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
   Vector vec1, vec2;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      vec1 = (v1.Get() - v2.Get());
      vec2 = (v3.Get() - v2.Get());
      if (Cross(vec1, vec2).length2() < v0.GetEpsilon()) {
	 if (vec1.length2() < v0.GetEpsilon()) {
	    if (vec2.length2() < v0.GetEpsilon()) {
	       v0.Assign(v2.Get(), scheme);
	    } else {
	       vec2.normalize();
	       Real t = Dot(v0.Get() - v2.Get(), vec2);
	       v0.Assign(v2.Get() + (vec2 * t), scheme);
	    }
	 } else {
	    vec1.normalize();
	    Real t = Dot(v0.Get() - v2.Get(), vec1);
	    v0.Assign(v2.Get() + (vec1 * t), scheme);
	 }
      } else {
	 Plane plane(v1.Get(), v2.Get(), v3.Get());
	 v0.Assign(plane.project(v0.Get()), scheme);
      }
      break;
   case 1:
      vec1 = (v0.Get() - v2.Get());
      vec2 = (v3.Get() - v2.Get());
      if (Cross(vec1, vec2).length2() < v1.GetEpsilon()) {
	 if (vec1.length2() < v1.GetEpsilon()) {
	    if (vec2.length2() < v1.GetEpsilon()) {
	       v1.Assign(v2.Get(), scheme);
	    } else {
	       vec2.normalize();
	       Real t = Dot(v1.Get() - v2.Get(), vec2);
	       v1.Assign(v2.Get() + (vec2 * t), scheme);
	    }
	 } else {
	    vec1.normalize();
	    Real t = Dot(v1.Get() - v2.Get(), vec1);
	    v1.Assign(v2.Get() + (vec1 * t), scheme);
	 }
      } else {
	 Plane plane(v0.Get(), v2.Get(), v3.Get());
	 v1.Assign(plane.project(v1.Get()), scheme);
      }
      break;
   case 2:
      vec1 = (v0.Get() - v1.Get());
      vec2 = (v3.Get() - v1.Get());
      if (Cross(vec1, vec2).length2() < v2.GetEpsilon()) {
	 if (vec1.length2() < v2.GetEpsilon()) {
	    if (vec2.length2() < v2.GetEpsilon()) {
	       v2.Assign(v1.Get(), scheme);
	    } else {
	       vec2.normalize();
	       Real t = Dot(v2.Get() - v1.Get(), vec2);
	       v2.Assign(v1.Get() + (vec2 * t), scheme);
	    }
	 } else {
	    vec1.normalize();
	    Real t = Dot(v2.Get() - v1.Get(), vec1);
	    v2.Assign(v1.Get() + (vec1 * t), scheme);
	 }
      } else {
	 Plane plane(v0.Get(), v1.Get(), v3.Get());
	 v2.Assign(plane.project(v2.Get()), scheme);
      }
      break;
   case 3:
      vec1 = (v0.Get() - v1.Get());
      vec2 = (v2.Get() - v1.Get());
      if (Cross(vec1, vec2).length2() < v3.GetEpsilon()) {
	 if (vec1.length2() < v3.GetEpsilon()) {
	    if (vec2.length2() < v3.GetEpsilon()) {
	       v3.Assign(v1.Get(), scheme);
	    } else {
	       vec2.normalize();
	       Real t = Dot(v3.Get() - v1.Get(), vec2);
	       v3.Assign(v1.Get() + (vec2 * t), scheme);
	    }
	 } else {
	    vec1.normalize();
	    Real t = Dot(v3.Get() - v1.Get(), vec1);
	    v3.Assign(v1.Get() + (vec1 * t), scheme);
	 }
      } else {
	 Plane plane(v0.Get(), v1.Get(), v2.Get());
	 v3.Assign(plane.project(v3.Get()), scheme);
      }
      break;
   default:
      cerr << "Unknown variable in Plane Constraint!" << endl;
      break;
   }
}

