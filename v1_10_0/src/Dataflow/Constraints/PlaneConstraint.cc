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
 *  PlaneConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Constraints/PlaneConstraint.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch pc_debug("Constraints", "Plane");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
PlaneConstraint::PlaneConstraint( const string& name,
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
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
PlaneConstraint::~PlaneConstraint()
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
PlaneConstraint::Satisfy( const Index index, const Scheme scheme,
			  const double Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[0];
   PointVariable& p2 = *vars[1];
   PointVariable& p3 = *vars[2];
   PointVariable& p4 = *vars[3];
   Vector vec1, vec2;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      vec1 = ((Point)p2 - p3);
      vec2 = ((Point)p4 - p3);
      if (Cross(vec1, vec2).length2() < Epsilon) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(p2, p3, p4);
	 var = vars[0];
	 c = plane.project(p1);
	 return true;
      }
      break;
   case 1:
      vec1 = ((Point)p1 - p3);
      vec2 = ((Point)p4 - p3);
      if (Cross(vec1, vec2).length2() < Epsilon) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(p1, p3, p4);
	 var = vars[1];
	 c = plane.project(p2);
	 return true;
      }
      break;
   case 2:
      vec1 = ((Point)p1 - p2);
      vec2 = ((Point)p4 - p2);
      if (Cross(vec1, vec2).length2() < Epsilon) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(p1, p2, p4);
	 var = vars[2];
	 c = plane.project(p3);
	 return true;
      }
      break;
   case 3:
      vec1 = ((Point)p1 - p2);
      vec2 = ((Point)p3 - p2);
      if (Cross(vec1, vec2).length2() < Epsilon) {
	 if (pc_debug) cerr << "No Plane." << endl;
      } else {
	 Plane plane(p1, p2, p3);
	 var = vars[3];
	 c = plane.project(p4);
	 return true;
      }
      break;
   default:
      cerr << "Unknown variable in Plane Constraint!" << endl;
      break;
   }
   return false;
}

} // End namespace SCIRun

