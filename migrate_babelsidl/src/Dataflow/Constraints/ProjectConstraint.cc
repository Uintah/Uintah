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
