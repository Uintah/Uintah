
/*
 *  ProjectConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Project_Constraint_h
#define SCI_project_Project_Constraint_h 1

#include <Constraints/BaseConstraint.h>


// This constraint only finds the projection or the point.

class ProjectConstraint : public BaseConstraint {
public:
   ProjectConstraint( const clString& name,
		      const Index numSchemes,
		      PointVariable* projection, PointVariable* point,
		      PointVariable* p1, PointVariable* p2 );
   virtual ~ProjectConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );
};

#endif
