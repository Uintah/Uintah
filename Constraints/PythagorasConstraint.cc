
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
#include <Classlib/Debug.h>

static DebugSwitch pc_debug("BaseConstraint", "Pythagoras");

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

PythagorasConstraint::~PythagorasConstraint()
{
}


void
PythagorasConstraint::Satisfy( const Index index, const Scheme scheme )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];

   if (pc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   /* A^2 + B^2 = C^2 */
   switch (ChooseChange(index, scheme)) {
   case 0:
      v0.Assign(Point(sqrt(v2.Get().x() * v2.Get().x() - v1.Get().x() * v1.Get().x()),
		      0, 0),
		scheme);
      break;
   case 1:
      v1.Assign(Point(sqrt(v2.Get().x() * v2.Get().x() - v0.Get().x() * v0.Get().x()),
		      0, 0),
		scheme);
      break;
   case 2:
      v2.Assign(Point(sqrt(v0.Get().x() * v0.Get().x() + v1.Get().x() * v1.Get().x()),
		      0, 0),
		scheme);
      break;
   default:
      cerr << "Unknown variable in Pythagoras Constraint!" << endl;
      break;
   }
}

