//static char *id="@(#) $Id$";

/*
 *  SegmentConstraint.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <PSECore/Constraints/SegmentConstraint.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Util/Debug.h>

namespace PSECore {
namespace Constraints {

using SCICore::Util::DebugSwitch;
using namespace SCICore::Geometry;

static DebugSwitch sc_debug("Constraints", "Segment");

/***************************************************************************
 * The constructor initializes the constraint's variables.
 * The last line should call the BaseConstraint Register method, which
 *      registers the constraint with its variables.
 */
SegmentConstraint::SegmentConstraint( const clString& name,
				      const Index numSchemes,
				      PointVariable* end1, PointVariable* end2,
				      PointVariable* p )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = end1;
   vars[1] = end2;
   vars[2] = p;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
}

/***************************************************************************
 * The destructor frees the constraint's allocated structures.
 * The BaseConstraint's destructor frees all the standard structures.
 * Therefore, most constraints' destructors will not need to do anything.
 */
SegmentConstraint::~SegmentConstraint()
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
SegmentConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& end1 = *vars[0];
   PointVariable& end2 = *vars[1];
   PointVariable& p = *vars[2];
   Vector v1, v2;

   if (sc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v1 = (Point)end2 - end1;
      v2 = (Point)p - end1;
      if (v1.length2() < v2.length2()) {
	 c = (Point)p;
	 var = vars[0];
	 return 1;
      }
      break;
   case 1:
      v1 = (Point)end1 - end2;
      v2 = (Point)p - end2;
      if (v1.length2() < v2.length2()) {
	 c = (Point)p;
	 var = vars[1];
	 return 1;
      }
      break;
   case 2:
      {
	  Vector norm((Point)end2 - end1);
	  if (norm.length2() < Epsilon) {
	      c = (Point)end2;
	  } else {
	      Real length = norm.normalize();
	      Real t = Dot((Point)p - end1, norm);
	      // Check if new point is outside segment.
	      if (t < 0)
		  t = 0;
	      else if (t > length)
		  t = length;
	      c = (Point)end1 + (norm * t);
	  }
	  var = vars[2];
      }
      return 1;
   default:
      cerr << "Unknown variable in Segment Constraint!" << endl;
      break;
   }
   return 0;
}

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:20  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:53  dav
// Import sources
//
//
