
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
		    PointVariable* p1, PointVariable* p2,
		    PointVariable* p3, PointVariable* p4);
   virtual ~PlaneConstraint();

protected:
   virtual void Satisfy( const Index index, const Scheme scheme );
};

#endif
