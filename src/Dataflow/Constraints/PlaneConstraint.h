
/*
 *  PlaneConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Plane_Constraint_h
#define SCI_project_Plane_Constraint_h 1

#include <Constraints/BaseConstraint.h>

namespace PSECommon {
namespace Constraints {

class PlaneConstraint : public BaseConstraint {
public:
   PlaneConstraint( const clString& name,
		    const Index numSchemes,
		    PointVariable* p1, PointVariable* p2,
		    PointVariable* p3, PointVariable* p4);
   virtual ~PlaneConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

} // End namespace Constraints
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:55  mcq
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
