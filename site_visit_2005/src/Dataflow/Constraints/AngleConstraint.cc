/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Util/NotFinished.h>

#include <math.h>

#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
AngleConstraint::AngleConstraint( const string& name,
				  const Index numSchemes,
				  PointVariable* center, PointVariable* end1,
				  PointVariable* end2, PointVariable* p,
				  RealVariable* angle ) :
  BaseConstraint(name, numSchemes, 5)
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

