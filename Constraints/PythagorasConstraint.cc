
/*
 *  PythagorasConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/PythagorasConstraint.h>


PythagorasConstraint::PythagorasConstraint( const clString& name,
					    const Index numSchemes,
					    Variable* dist1InX, Variable* dist2InX,
					    Variable* hypoInX )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = dist1InX;
   vars[1] = dist2InX;
   vars[2] = hypoInX;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


void
PythagorasConstraint::Satisfy( const Index index )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Point temp;
   
   ChooseChange(index);
   print();
   
   /* A^2 + B^2 = C^2 */
   switch (ChooseChange(index)) {
   case 0:
      temp.x(sqrt(v2.Get().x() * v2.Get().x() - v1.Get().x() * v1.Get().x()));
      v0.Assign(temp);
      break;
   case 1:
      temp.x(sqrt(v2.Get().x() * v2.Get().x() - v0.Get().x() * v0.Get().x()));
      v1.Assign(temp);
      break;
   case 2:
      temp.x(sqrt(v0.Get().x() * v0.Get().x() + v1.Get().x() * v1.Get().x()));
      v2.Assign(temp);
      break;
   default:
      cerr << "Unknown variable in Pythagoras Constraint!" << endl;
      break;
   }
}

