
/*
 *  DistanceConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Distance_Constraint_h
#define SCI_project_Distance_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class DistanceConstraint : public BaseConstraint {
public:
   DistanceConstraint( const clString& name,
		       const Index numSchemes,
		       Variable* p1, Variable* p2,
		       Variable* distInX );
   ~DistanceConstraint();

protected:
   void Satisfy( const Index index );
};

#endif
