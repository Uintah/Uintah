
/*
 *  AngleConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_project_Angle_Constraint_h
#define SCI_project_Angle_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class AngleConstraint : public BaseConstraint {
public:
   AngleConstraint( const clString& name,
		    const Index numSchemes,
		    PointVariable* center, PointVariable* end1,
		    PointVariable* end2, PointVariable* p,
		    RealVariable* angle );
   virtual ~AngleConstraint();

protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& v, VarCore& c );
};

#endif
