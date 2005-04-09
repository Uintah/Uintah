
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

#include <Constraints/BaseConstraint.h>


// This constraint only finds the center (i.e. one-way constraint).

class CenterConstraint : public BaseConstraint {
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

#endif
