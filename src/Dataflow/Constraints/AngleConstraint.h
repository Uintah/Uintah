
/*
 *  AngleConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Angle_Constraint_h
#define SCI_project_Angle_Constraint_h 1

#include <Constraints/BaseConstraint.h>

namespace PSECommon {
namespace Constraints {

class AngleConstraint : public BaseConstraint {
public:
   AngleConstraint( const clString& name,
		    const Index numSchemes,
		    PointVariable* center, PointVariable* end1,
		    PointVariable* end2, PointVariable* p,
		    RealVariable* angle );
   virtual ~AngleConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& v, VarCore& c );
};

} // End namespace Constraints
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:52  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:05  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif
