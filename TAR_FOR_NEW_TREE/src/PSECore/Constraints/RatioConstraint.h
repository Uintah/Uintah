
/*
 *  RatioConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Ratio_Constraint_h
#define SCI_project_Ratio_Constraint_h 1

#include <PSECore/share/share.h>
#include <PSECore/Constraints/BaseConstraint.h>

namespace PSECore {
namespace Constraints {

class PSECORESHARE RatioConstraint : public BaseConstraint {
public:
   RatioConstraint( const clString& name,
		    const Index numSchemes,
		    RealVariable* numer, RealVariable* denom,
		    RealVariable* ratio );
   virtual ~RatioConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

} // End namespace Constraints
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/26 23:57:03  moulding
// changed SCICORESHARE to PSECORESHARE
//
// Revision 1.2  1999/08/17 06:38:19  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:56  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:07  dav
// added back PSECore .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:52  dav
// Import sources
//
//

#endif
