
/*
 *  CenterConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Center_Constraint_h
#define SCI_project_Center_Constraint_h 1

#include <Constraints/BaseConstraint.h>

namespace PSECommon {
namespace Constraints {

// This constraint only finds the center (i.e. one-way constraint).

class CenterConstraint : public BaseConstraint {
public:
   CenterConstraint( const clString& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2 );
   CenterConstraint( const clString& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2,
		     PointVariable* p3 );
   CenterConstraint( const clString& name,
		     const Index numSchemes,
		     PointVariable* center,
		     PointVariable* p1, PointVariable* p2,
		     PointVariable* p3, PointVariable* p4 );
   virtual ~CenterConstraint();
   
protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

} // End namespace Constraints
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:06  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:53  dav
// Import sources
//
//

#endif
