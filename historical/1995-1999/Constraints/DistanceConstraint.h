
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
		       PointVariable* p1, PointVariable* p2,
		       RealVariable* dist );
   virtual ~DistanceConstraint();

   // Use this to set the default direction used when p1==p2.
   // Defaults to (1,0,0).
   void SetDefault( const Vector& v );
   void SetMinimum( const Real min );
   
protected:
   virtual int Satisfy( const Index index, const Scheme scheme, const Real Epsilon,
			BaseVariable*& var, VarCore& c );

private:
   Vector guess;
   Real minimum;
};


#endif
