
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

namespace PSECommon {
namespace Constraints {

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

} // End namespace Constraints
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:56  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:07  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif
