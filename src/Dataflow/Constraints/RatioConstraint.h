
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

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

namespace SCIRun {

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

} // End namespace SCIRun


#endif
