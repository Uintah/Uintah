
/*
 *  MidpointConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Midpoint_Constraint_h
#define SCI_project_Midpoint_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class MidpointConstraint : public BaseConstraint {
public:
   MidpointConstraint( const clString& name,
		       const Index numSchemes,
		       PointVariable* end1, PointVariable* end2,
		       PointVariable* p );
   virtual ~MidpointConstraint();
   
protected:
   virtual void Satisfy( const Index index, const Scheme scheme );
};

#endif
