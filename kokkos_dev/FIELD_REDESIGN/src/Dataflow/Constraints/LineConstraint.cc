//static char *id="@(#) $Id$";

/*
 *  LineConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Constraints/LineConstraint.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;
using namespace SCICore::Geometry;

static DebugSwitch lc_debug("Constraints", "Line");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
LineConstraint::LineConstraint( const clString& name,
				const Index numSchemes,
				PointVariable* p1, PointVariable* p2,
				PointVariable* p3 )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = p3;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
LineConstraint::~LineConstraint()
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
LineConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			 BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[0];
   PointVariable& p2 = *vars[1];
   PointVariable& p3 = *vars[2];
   Vector norm;

   if (lc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = (Point)p3 - p2;
      if (norm.length2() < Epsilon) {
	 c = (Point)p3;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p1 - p2, norm));
	 c = (Point)p2 + (norm * t);
      }
      var = vars[0];
      return 1;
   case 1:
      norm = (Point)p3 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p3;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p2 - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[1];
      return 1;
   case 2:
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p2;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)p3 - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[2];
      return 1;
   default:
      cerr << "Unknown variable in Line Constraint!" << endl;
      break;
   }
   return 0;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/10/07 02:07:16  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:38  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:38:17  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:54  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
