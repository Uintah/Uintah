
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

static DebugSwitch sc_debug("BaseConstraint", "Segment");

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


void
SegmentConstraint::Satisfy( const Index index, const Scheme scheme )
{
   PointVariable& v0 = *vars[0];
   PointVariable& v1 = *vars[1];
   PointVariable& v2 = *vars[2];

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
      Vector norm(v1.GetPoint() - v0.GetPoint());
      if (norm.length2() < v2.GetEpsilon()) {
	 v2.Assign(v1.GetPoint(), scheme);
      } else {
	 Real length = norm.normalize();
	 Real t = Dot(v2.GetPoint() - v0.GetPoint(), norm);
	 // Check if new point is outside segment.
	 if (t < 0)
	    t = 0;
	 else if (t > length)
	    t = length;
	 v2.Assign(v0.GetPoint() + (norm * t), scheme);
      }
      break;
   default:
      cerr << "Unknown variable in Segment Constraint!" << endl;
      break;
   }
}

