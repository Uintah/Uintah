
/*
 *  ProjectConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/ProjectConstraint.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch pc_debug("Constraints", "Project");

ProjectConstraint::ProjectConstraint( const clString& name,
				      const Index numSchemes,
				      PointVariable* projection, PointVariable* point,
				      PointVariable* p1, PointVariable* p2 )
:BaseConstraint(name, numSchemes, 4)
{
   vars[0] = projection;
   vars[1] = point;
   vars[2] = p1;
   vars[3] = p2;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

ProjectConstraint::~ProjectConstraint()
{
}


int
ProjectConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& projection = *vars[0];
   PointVariable& point = *vars[1];
   PointVariable& p1 = *vars[2];
   PointVariable& p2 = *vars[3];
   Vector norm;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p2;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)point - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[0];
      return 1;
   case 1:
      Point proj;
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 proj = (Point)p2;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)point - p1, norm));
	 proj = (Point)p1 + (norm * t);
      }
      c = (Point)projection + ((Point)point-proj);
      var = vars[1];
      return 1;
   case 2:
      Error("ProjectConstraint:  Can only satisfy projection");
      break;
   default:
      cerr << "Unknown variable in Project Constraint!" << endl;
      break;
   }
   return 0;
}

