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


#include <Dataflow/Constraints/ProjectConstraint.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch pc_debug("Constraints", "Project");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
ProjectConstraint::ProjectConstraint( const string& name,
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
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
ProjectConstraint::~ProjectConstraint()
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
ProjectConstraint::Satisfy( const Index index, const Scheme scheme,
			    const double Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& projection = *vars[0];
   PointVariable& point = *vars[1];
   PointVariable& p1 = *vars[2];
   PointVariable& p2 = *vars[3];
   Vector norm;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p2;
      } else {
	 norm.normalize();
	 const double t = Dot((Point)point - p1, norm);
	 c = (Point)p1 + (norm * t);
      }
      var = vars[0];
      return true;

   case 1:
      {
	  Point proj;
	  norm = (Point)p2 - p1;
	  if (norm.length2() < Epsilon) {
	      proj = (Point)p2;
	  } else {
	      norm.normalize();
	      const double t = Dot((Point)point - p1, norm);
	      proj = (Point)p1 + (norm * t);
	  }
	  c = (Point)projection + ((Point)point-proj);
	  var = vars[1];
      }
      return true;

   case 2:
      ASSERTFAIL("ProjectConstraint:  Can only satisfy projection");
      //break;

   default:
      cerr << "Unknown variable in Project Constraint!" << endl;
      break;
   }
   return false;
}

} // End namespace SCIRun
