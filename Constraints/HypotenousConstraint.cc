
/*
 *  HypotenousConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/HypotenousConstraint.h>


HypotenousConstraint::HypotenousConstraint( const clString& name,
					    const Index numSchemes,
					    Variable* HypoInX, Variable* distInX )
:BaseConstraint(name, numSchemes, 2)
{
   vars[0] = HypoInX;
   vars[1] = distInX;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

HypotenousConstraint::~HypotenousConstraint()
{
}

void
HypotenousConstraint::Satisfy( const Index index )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Point temp;
   
   ChooseChange(index);
   print();
   
   /* 2A^2 = C^2 */
   switch (ChooseChange(index)) {
   case 0:
      temp.x(sqrt(2.0 * v1.Get().x() * v1.Get().x()));
      v0.Assign(temp);
      break;
   case 1:
      temp.x(sqrt(v0.Get().x() * v0.Get().x() / 2.0));
      v1.Assign(temp);
      break;
   default:
      cerr << "Unknown variable in Hypotenous!" << endl;
      break;
   }
}

