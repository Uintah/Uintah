//static char *id="@(#) $Id$";

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


#include <PSECore/Constraints/RatioConstraint.h>
#include <SCICore/Util/Debug.h>
#include <iostream.h>

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;

static DebugSwitch rc_debug("Constraints", "Ratio");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
RatioConstraint::RatioConstraint( const clString& name,
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
int
RatioConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   RealVariable& numer = *vars[0];
   RealVariable& denom = *vars[1];
   RealVariable& ratio = *vars[2];

   if (rc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   Real temp;
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      var = vars[0];
      c = denom * ratio;
      return 1;
   case 1:
      if (ratio < Epsilon)
	 temp = denom; // Don't change denom since 0/any == 0
      else
	 temp = numer / ratio;
      var = vars[1];
      c = temp;
      return 1;
   case 2:
      if (denom < Epsilon)
	 temp = ratio; // Don't change denom since 0/any == 0
      else
	 temp = numer / denom;
      var = vars[2];
      c = temp;
      return 1;
   default:
      cerr << "Unknown variable in Ratio Constraint!" << endl;
      break;
   }
   return 0;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/09/08 02:26:39  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:38:19  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
