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
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch rc_debug("Constraints", "Ratio");

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

   if (rc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
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

