
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


#include <Constraints/DistanceConstraint.h>
#include <Geometry/Vector.h>

DistanceConstraint::DistanceConstraint( const char* name,
					const Index numSchemes,
					Variable* p1, Variable* p2,
					Variable* distInX )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = p1;
   vars[1] = p2;
   vars[2] = distInX;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


void
DistanceConstraint::Satisfy( const Index index )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Point temp;

   ChooseChange(index);
   print();
   
   /* Q <- Sc + Sr * Normalize(P-Sc) */
   switch (ChooseChange(index)) {
   case 0:
      v0.Assign(v1.Get()
		+ ((v0.Get() - v1.Get()).normal()
		   * v2.Get().x()));
      break;
   case 1:
      v1.Assign(v0.Get()
		+ ((v1.Get() - v0.Get()).normal()
		   * v2.Get().x()));
      break;
   case 2:
      temp.x((v1.Get() - v0.Get()).length());
      v2.Assign(temp);
      break;
   default:
      cerr << "Unknown variable in distance!" << endl;
      break;
   }
}

