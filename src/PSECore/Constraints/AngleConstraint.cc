//static char *id="@(#) $Id$";

/*
 *  AngleConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Constraints/AngleConstraint.h>
#include <SCICore/Geometry/Plane.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Util/Debug.h>
#include <SCICore/Util/NotFinished.h>
#include <math.h>

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;
using namespace SCICore::Geometry;

static DebugSwitch ac_debug("Constraints", "Angle");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
AngleConstraint::AngleConstraint( const clString& name,
				  const Index numSchemes,
				  PointVariable* center, PointVariable* end1,
				  PointVariable* end2, PointVariable* p,
				  RealVariable* angle )
:BaseConstraint(name, numSchemes, 5)
{
   vars[0] = center;
   vars[1] = end1;
   vars[2] = end2;
   vars[3] = p;
   vars[4] = angle;

   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
AngleConstraint::~AngleConstraint()
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
AngleConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			  BaseVariable*& var, VarCore& c )
{
   PointVariable& center = *vars[0];
   PointVariable& end1 = *vars[1];
   PointVariable& end2 = *vars[2];
   PointVariable& p = *vars[3];
   RealVariable& angle = *vars[4];
   Vector v, v1, v2;

   if (ac_debug) {
      ChooseChange(index, scheme);
      print(cout);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      NOT_FINISHED("Line Constraint:  center");
      break;
   case 1:
      NOT_FINISHED("Line Constraint:  end1");
      break;
   case 2:
      NOT_FINISHED("Line Constraint:  end2");
      break;
   case 3:
      v1 = (Point)end1 - center;
      v2 = (Point)end2 - center;
      if ((v1.length2() >= Epsilon) && (v2.length2() >= Epsilon)) {
	 v2 = Cross(v2.normal(), v1.normal());
	 if (v2.length2() >= Epsilon) {
	    v2 = Cross(v1, v2.normal()); // Find orthogonal v2.
	    v = (v1 * cos(angle) + v2 * sin(angle));
	    
	    if (v.length2() < Epsilon) {
	       c = (Point)end1;
	    } else {
	       v.normalize();
	       Real t(Dot((Point)p - center, v));
	       c = (Point)center + (v * t);
	    }
	    var = vars[3];
	    return 1;
	 }
      }
   case 4:
      v = (Point)p - center;
      v1 = (Point)end1 - center;
      v2 = (Point)end2 - center;
      if ((v.length2() >= Epsilon)
	  && (v1.length2() >= Epsilon) && (v2.length2() >= Epsilon)) {
	 v2 = Cross(v2.normal(), v1.normal());
	 if (v2.length2() >= Epsilon) {
	    v2 = Cross(v1, v2.normal()); // Find orthogonal v2.
	    
	    Real x(Dot(v1, v)), y(Dot(v2, v));

	    using namespace SCICore::Math;
	    if ((Abs(x) > Epsilon) || (Abs(y) > Epsilon)) {
	       var = vars[4];
	       c = atan2(y,x);
	       return 1;
	    }
	 }
      }
      break;
   default:
      cerr << "Unknown variable in Angle Constraint!" << endl;
      break;
   }
   return 0;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:37  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/19 23:18:03  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.2  1999/08/17 06:38:15  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:52  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//
