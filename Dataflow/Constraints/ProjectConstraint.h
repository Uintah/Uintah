
/*
 *  ProjectConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Project_Constraint_h
#define SCI_project_Project_Constraint_h 1

#include <Constraints/BaseConstraint.h>

namespace PSECommon {
namespace Constraints {

// This constraint only finds the projection or the point.

class ProjectConstraint : public BaseConstraint {
public:
   ProjectConstraint( const clString& name,
		      const Index numSchemes,
		      PointVariable* projection, PointVariable* point,
		      PointVariable* p1, PointVariable* p2 );
   virtual ~ProjectConstraint();

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
// Revision 1.1.1.1  1999/04/24 23:12:53  dav
// Import sources
//
//

#endif
