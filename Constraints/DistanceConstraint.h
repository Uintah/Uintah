
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
   virtual ~DistanceConstraint();

   // Use this to set the default direction used when p1==p2.
   // Defaults to (1,0,0).
   inline void SetDefault( const Vector& v );
   inline void SetMinimum( const Real min );
   
protected:
   virtual void Satisfy( const Index index, const Scheme scheme );

private:
   Vector guess;
   Real minimum;
};


inline void
DistanceConstraint::SetDefault( const Vector& v )
{
   guess = v;
}


inline void
DistanceConstraint::SetMinimum( const Real min )
{
   ASSERT(min>0.0);

   minimum = min;
}


#endif
