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
 *  PythagorasConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Dataflow/Constraints/PythagorasConstraint.h>
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
PythagorasConstraint::PythagorasConstraint( const string& name,
					    const Index numSchemes,
					    RealVariable* dist1, RealVariable* dist2,
					    RealVariable* hypo )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = dist1;
   vars[1] = dist2;
   vars[2] = hypo;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
PythagorasConstraint::~PythagorasConstraint()
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
PythagorasConstraint::Satisfy( const Index index, const Scheme scheme,
			       const double,
			       BaseVariable*& var, VarCore& c )
{
   RealVariable& dist1 = *vars[0];
   RealVariable& dist2 = *vars[1];
   RealVariable& hypo = *vars[2];
   double t;

   switch (ChooseChange(index, scheme)) {
   case 0:
      if ((t = hypo * hypo - dist2 * dist2) >= 0.0) {
	 var = vars[0];
	 c = sqrt(t);
	 return true;
      }
      break;

   case 1:
      if ((t = hypo * hypo - dist1 * dist1) >= 0.0) {
	 var = vars[1];
	 c = sqrt(t);
	 return true;
      }
      break;

   case 2:
      if ((t = dist1 * dist1 + dist2 * dist2) >= 0.0) {
	 var = vars[2];
	 c = sqrt(t);
	 return true;
      }
      break;

   default:
      cerr << "Unknown variable in Pythagoras Constraint!" << endl;
      break;
   }

   return false;
}

} // End namespace SCIRun

