
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

#include <SCICore/share/share.h>
#include <PSECore/Constraints/BaseConstraint.h>

namespace PSECore {
namespace Constraints {

// This constraint only finds the center (i.e. one-way constraint).

class SCICORESHARE CenterConstraint : public BaseConstraint {
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
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:16  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:53  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:06  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:53  dav
// Import sources
//
//

#endif
