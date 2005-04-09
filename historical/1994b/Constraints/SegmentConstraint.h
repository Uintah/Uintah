
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
		      Variable* segment_p1, Variable* segment_p2,
		      Variable* p );
   virtual ~SegmentConstraint();

protected:
   virtual void Satisfy( const Index index, const Scheme scheme );
};

#endif
