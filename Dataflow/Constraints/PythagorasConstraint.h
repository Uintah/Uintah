
/*
 *  PythagorasConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Pythagoras_Constraint_h
#define SCI_project_Pythagoras_Constraint_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

namespace SCIRun {

class PSECORESHARE PythagorasConstraint : public BaseConstraint {
public:
   PythagorasConstraint( const clString& name,
			 const Index numSchemes,
			 RealVariable* dist1, RealVariable* dist2,
			 RealVariable* hypo );
   virtual ~PythagorasConstraint();
   
protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

} // End namespace SCIRun


#endif
