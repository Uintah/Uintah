
/*
 *  SegmentConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/SegmentConstraint.h>
#include <Geometry/Vector.h>
#include <Classlib/Debug.h>

static DebugSwitch sc_debug("Constraints", "Segment");

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
};

SegmentConstraint::~SegmentConstraint()
{
}


int
SegmentConstraint::Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			    BaseVariable*& var, VarCore& c )
{
   PointVariable& end1 = *vars[0];
   PointVariable& end2 = *vars[1];
   PointVariable& p = *vars[2];

   if (sc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      NOT_FINISHED("Segment Constraint:  segment_p1");
      break;
   case 1:
      NOT_FINISHED("Segment Constraint:  segment_p2");
      break;
   case 2:
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
      return 1;
   default:
      cerr << "Unknown variable in Segment Constraint!" << endl;
      break;
   }
   return 0;
}

