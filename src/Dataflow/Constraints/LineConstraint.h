
/*
 *  LineConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Line_Constraint_h
#define SCI_project_Line_Constraint_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Constraints/BaseConstraint.h>

namespace SCIRun {

class PSECORESHARE LineConstraint : public BaseConstraint {
public:
   LineConstraint( const clString& name,
		   const Index numSchemes,
		   PointVariable* p1, PointVariable* p2,
		   PointVariable* p3 );
   virtual ~LineConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

} // End namespace SCIRun


#endif
