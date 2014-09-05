//static char *id="@(#) $Id$";

/*
 *  DistanceConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <PSECore/Constraints/DistanceConstraint.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Util/Debug.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;

static DebugSwitch dc_debug("Constraints", "Distance");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
DistanceConstraint::DistanceConstraint( const clString& name,
					const Index numSchemes,
					PointVariable* p1, PointVariable* p2,
					RealVariable* dist )
:BaseConstraint(name, numSchemes, 3),
 guess(1, 0, 0), minimum(0.0)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = dist;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
DistanceConstraint::~DistanceConstraint()
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
DistanceConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			     BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[0];
   PointVariable& p2 = *vars[1];
   RealVariable& dist = *vars[2];
   Vector v;
   Real t;

   if (dc_debug) {
      ChooseChange(index, scheme);
      printc(cout, scheme);
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v = ((Point)p1 - p2);
      if (v.length2() < Epsilon)
	 v = guess;
      else
	 v.normalize();
      if (dist < minimum) {
	 t = minimum;
      } else
	 t = dist;
      var = vars[0];
      c = (Point)p2 + (v * t);
      return 1;
   case 1:
      v = ((Point)p2 - p1);
      if (v.length2() < Epsilon)
	 v = guess;
      else
	 v.normalize();
      if (dist < minimum) {
	 t = minimum;
      } else
	 t = dist;
      var = vars[1];
      c = (Point)p1 + (v * t);
      return 1;
   case 2:
      t = ((Point)p2 - p1).length();
      if (t < minimum) {
	 t = minimum;
      }
      var = vars[2];
      c = t;
      return 1;
   default:
      cerr << "Unknown variable in Distance Constraint!" << endl;
      break;
   }
   return 0;
}


void
DistanceConstraint::SetDefault( const Vector& v )
{
   guess = v;
}


void
DistanceConstraint::SetMinimum( const Real min )
{
   ASSERT(min>0.0);

   minimum = min;
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
// Revision 1.1.1.1  1999/04/24 23:12:53  dav
// Import sources
//
//
