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


#include <Dataflow/Constraints/DistanceConstraint.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch dc_debug("Constraints", "Distance");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
DistanceConstraint::DistanceConstraint( const string& name,
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

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
DistanceConstraint::~DistanceConstraint()
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
DistanceConstraint::Satisfy( const Index index, const Scheme scheme,
			     const double Epsilon,
			     BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[0];
   PointVariable& p2 = *vars[1];
   RealVariable& dist = *vars[2];
   Vector v;
   double t;

   if (dc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v = ((Point)p1 - p2);
      if (v.length2() < Epsilon)
	 v = guess;
      else
	 v.normalize();
      if (dist < minimum) {
	 t = minimum;
      } else
	 t = dist;
      var = vars[0];
      c = (Point)p2 + (v * t);
      return true;

   case 1:
      v = ((Point)p2 - p1);
      if (v.length2() < Epsilon)
	 v = guess;
      else
	 v.normalize();
      if (dist < minimum) {
	 t = minimum;
      } else
	 t = dist;
      var = vars[1];
      c = (Point)p1 + (v * t);
      return true;

   case 2:
      t = ((Point)p2 - p1).length();
      if (t < minimum) {
	 t = minimum;
      }
      var = vars[2];
      c = t;
      return true;

   default:
      cerr << "Unknown variable in Distance Constraint!" << endl;
      break;
   }
   return false;
}


void
DistanceConstraint::SetDefault( const Vector& v )
{
   guess = v;
}


void
DistanceConstraint::SetMinimum( const double min )
{
   ASSERT(min>0.0);

   minimum = min;
}

} // End namespace SCIRun

