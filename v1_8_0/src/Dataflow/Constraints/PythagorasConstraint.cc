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
#include <Core/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace SCIRun {


static DebugSwitch pc_debug("Constraints", "Pythagoras");

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

   if (pc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
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

