//static char *id="@(#) $Id$";

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


#include <PSECore/Constraints/ProjectConstraint.h>
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

static DebugSwitch pc_debug("Constraints", "Project");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
ProjectConstraint::ProjectConstraint( const clString& name,
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
int
ProjectConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& projection = *vars[0];
   PointVariable& point = *vars[1];
   PointVariable& p1 = *vars[2];
   PointVariable& p2 = *vars[3];
   Vector norm;

   if (pc_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      norm = (Point)p2 - p1;
      if (norm.length2() < Epsilon) {
	 c = (Point)p2;
      } else {
	 norm.normalize();
	 Real t(Dot((Point)point - p1, norm));
	 c = (Point)p1 + (norm * t);
      }
      var = vars[0];
      return 1;
   case 1:
      {
	  Point proj;
	  norm = (Point)p2 - p1;
	  if (norm.length2() < Epsilon) {
	      proj = (Point)p2;
	  } else {
	      norm.normalize();
	      Real t(Dot((Point)point - p1, norm));
	      proj = (Point)p1 + (norm * t);
	  }
	  c = (Point)projection + ((Point)point-proj);
	  var = vars[1];
      }
      return 1;
   case 2:
      ASSERTFAIL("ProjectConstraint:  Can only satisfy projection");
      //break;
   default:
      cerr << "Unknown variable in Project Constraint!" << endl;
      break;
   }
   return 0;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.5.2.3  2000/10/26 14:16:45  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.6  2000/06/15 19:50:56  sparker
// Fixed warnings
//
// Revision 1.5  1999/10/07 02:07:16  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/09/08 02:26:39  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/18 20:20:17  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:18  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:55  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
