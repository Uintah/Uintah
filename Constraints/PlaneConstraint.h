
/*
 *  PlaneConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Plane_Constraint_h
#define SCI_project_Plane_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class PlaneConstraint : public BaseConstraint {
public:
   PlaneConstraint( const clString& name,
		    const Index numSchemes,
		    Variable* p,
		    Variable* norm, Variable* offsetInX );
   ~PlaneConstraint();

protected:
   virtual void Satisfy( const Index index );
};

#endif
