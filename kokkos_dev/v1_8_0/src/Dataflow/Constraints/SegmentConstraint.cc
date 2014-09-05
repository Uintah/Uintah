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
 *  SegmentConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Constraints/SegmentConstraint.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch sc_debug("Constraints", "Segment");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
SegmentConstraint::SegmentConstraint( const string& name,
				      const Index numSchemes,
				      PointVariable* end1, PointVariable* end2,
				      PointVariable* p )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = end1;
   vars[1] = end2;
   vars[2] = p;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
SegmentConstraint::~SegmentConstraint()
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
SegmentConstraint::Satisfy( const Index index, const Scheme scheme,
			    const double Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& end1 = *vars[0];
   PointVariable& end2 = *vars[1];
   PointVariable& p = *vars[2];
   Vector v1, v2;

   if (sc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v1 = (Point)end2 - end1;
      v2 = (Point)p - end1;
      if (v1.length2() < v2.length2()) {
	 c = (Point)p;
	 var = vars[0];
	 return true;
      }
      break;

   case 1:
      v1 = (Point)end1 - end2;
      v2 = (Point)p - end2;
      if (v1.length2() < v2.length2()) {
	 c = (Point)p;
	 var = vars[1];
	 return true;
      }
      break;

   case 2:
      {
	  Vector norm((Point)end2 - end1);
	  if (norm.length2() < Epsilon) {
	      c = (Point)end2;
	  } else {
	      const double length = norm.normalize();
	      double t = Dot((Point)p - end1, norm);
	      // Check if new point is outside segment.
	      if (t < 0)
		  t = 0;
	      else if (t > length)
		  t = length;
	      c = (Point)end1 + (norm * t);
	  }
	  var = vars[2];
      }
      return true;

   default:
      cerr << "Unknown variable in Segment Constraint!" << endl;
      break;
   }

   return false;
}

} // End namespace SCIRun

