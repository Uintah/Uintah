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
 *  CenterConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Constraints/CenterConstraint.h>
#include <Core/Util/Debug.h>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

namespace SCIRun {


static DebugSwitch cc_debug("Constraints", "Center");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 * This constructor centers between two PointVariables.
 */
CenterConstraint::CenterConstraint( const string& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2 )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}


/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 * This constructor centers between three PointVariables.
 */
CenterConstraint::CenterConstraint( const string& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2,
				    PointVariable* p3 )
:BaseConstraint(name, numSchemes, 4)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   vars[3] = p3;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}


/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 * This constructor centers between four PointVariables.
 */
CenterConstraint::CenterConstraint( const string& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2,
				    PointVariable* p3, PointVariable* p4 )
:BaseConstraint(name, numSchemes, 5)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   vars[3] = p3;
   vars[4] = p4;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}


/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
CenterConstraint::~CenterConstraint()
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
CenterConstraint::Satisfy( const Index index, const Scheme scheme,
			   const double,
			   BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[1];
   PointVariable& p2 = *vars[2];

   if (cc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      switch(varCount) {
      case 3:
	 var = vars[0];
	 c = AffineCombination(p1, 0.5, p2, 0.5);
	 break;
      case 4:
	 {
	     PointVariable& p3 = *vars[3];
	     var = vars[0];
	     c = AffineCombination(p1, 1.0/3.0, p2, 1.0/3.0,
				   p3, 1.0/3.0);
	 }
	 break;
      case 5:
	 {
	     PointVariable& pthree = *vars[3];
	     PointVariable& p4 = *vars[4];
	     var = vars[0];
	     c = AffineCombination(p1, 0.25, p2, 0.25,
				   pthree, 0.25, p4, 0.25);
	 }
	 break;
      default:
	 break;
      }
      return true;
   case 1:
   case 2:
      ASSERTFAIL("CenterConstraint:  Can only satisfy center");
      //break;
   case 3:
      ASSERT(varCount >= 4);
      ASSERTFAIL("CenterConstraint:  Can only satisfy center");
      //break;
   case 4:
      ASSERT(varCount >= 5);
      ASSERTFAIL("CenterConstraint:  Can only satisfy center");
      //break;
   default:
      cerr << "Unknown variable in Center Constraint!" << endl;
      break;
   }
   return false;
}

} // End namespace SCIRun

