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
 *  RatioConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Constraints/RatioConstraint.h>
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
RatioConstraint::RatioConstraint( const string& name,
				  const Index numSchemes,
				  RealVariable* numer, RealVariable* denom,
				  RealVariable* ratio )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = numer;
   vars[1] = denom;
   vars[2] = ratio;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
RatioConstraint::~RatioConstraint()
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
RatioConstraint::Satisfy( const Index index, const Scheme scheme,
			  const double Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   RealVariable& numer = *vars[0];
   RealVariable& denom = *vars[1];
   RealVariable& ratio = *vars[2];

   double temp;
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      var = vars[0];
      c = denom * ratio;
      return true;
   case 1:
      if (ratio < Epsilon)
	 temp = denom; // Don't change denom since 0/any == 0
      else
	 temp = numer / ratio;
      var = vars[1];
      c = temp;
      return true;
   case 2:
      if (denom < Epsilon)
	 temp = ratio; // Don't change denom since 0/any == 0
      else
	 temp = numer / denom;
      var = vars[2];
      c = temp;
      return true;
   default:
      cerr << "Unknown variable in Ratio Constraint!" << endl;
      break;
   }
   return false;
}

} // End namespace SCIRun

