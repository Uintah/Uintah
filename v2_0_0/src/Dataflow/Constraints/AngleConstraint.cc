/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  AngleConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Dataflow/Constraints/AngleConstraint.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Debug.h>
#include <Core/Util/NotFinished.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <math.h>

namespace SCIRun {


static DebugSwitch ac_debug("Constraints", "Angle");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
AngleConstraint::AngleConstraint( const string& name,
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
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
AngleConstraint::~AngleConstraint()
{
}


/***************************************************************************
 * The Satisfy method is where the constraint is maintained.
 * The BaseConstraint ChooseChange method is used to select which variable
 *      should be altered to maintain the constraint.
 * Reference variables are frequently used to speed up accesses of the
 *      constraint's variables and to make the Satisfy method more legible.
 * Satisfy should return 1 if it is able to satisfy the constraint, and
 *      0 otherwise.
 */
bool
AngleConstraint::Satisfy( const Index index, const Scheme scheme,
			  const double Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   PointVariable& center = *vars[0];
   PointVariable& end1 = *vars[1];
   PointVariable& end2 = *vars[2];
   PointVariable& p = *vars[3];
   RealVariable& angle = *vars[4];
   Vector v, v1, v2;

   if (ac_debug) {
      ChooseChange(index, scheme);
      print(cout);
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
      v1 = (Point)end1 - center;
      v2 = (Point)end2 - center;
      if ((v1.length2() >= Epsilon) && (v2.length2() >= Epsilon)) {
	 v2 = Cross(v2.normal(), v1.normal());
	 if (v2.length2() >= Epsilon) {
	    v2 = Cross(v1, v2.normal()); // Find orthogonal v2.
	    v = (v1 * cos(angle) + v2 * sin(angle));
	    
	    if (v.length2() < Epsilon) {
	       c = (Point)end1;
	    } else {
	       v.normalize();
	       double t(Dot((Point)p - center, v));
	       c = (Point)center + (v * t);
	    }
	    var = vars[3];
	    return true;
	 }
      }
   case 4:
      v = (Point)p - center;
      v1 = (Point)end1 - center;
      v2 = (Point)end2 - center;
      if ((v.length2() >= Epsilon)
	  && (v1.length2() >= Epsilon) && (v2.length2() >= Epsilon)) {
	 v2 = Cross(v2.normal(), v1.normal());
	 if (v2.length2() >= Epsilon) {
	    v2 = Cross(v1, v2.normal()); // Find orthogonal v2.
	    
	    const double x(Dot(v1, v)), y(Dot(v2, v));

	    if ((fabs(x) > Epsilon) || (fabs(y) > Epsilon)) {
	       var = vars[4];
	       c = atan2(y,x);
	       return true;
	    }
	 }
      }
      break;
   default:
      cerr << "Unknown variable in Angle Constraint!" << endl;
      break;
   }
   return false;
}

} // End namespace SCIRun

