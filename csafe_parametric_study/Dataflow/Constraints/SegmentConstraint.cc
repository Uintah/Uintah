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

