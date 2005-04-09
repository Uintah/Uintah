
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


#ifndef SCI_project_Segment_Constraint_h
#define SCI_project_Segment_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class SegmentConstraint : public BaseConstraint {
public:
   SegmentConstraint( const clString& name,
		      const Index numSchemes,
		      PointVariable* end1, PointVariable* end2,
		      PointVariable* p );
   virtual ~SegmentConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

#endif
