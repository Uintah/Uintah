
/*
 *  LineConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/LineConstraint.h>
#include <Geometry/Vector.h>

LineConstraint::LineConstraint( const clString& name,
				const Index numSchemes,
				Variable* line_p1, Variable* line_p2,
				Variable* p )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = line_p1;
   vars[1] = line_p2;
   vars[2] = p;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


void
LineConstraint::Satisfy( const Index index )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];
   Vector norm;
   double t;

   ChooseChange(index);
   print();
   
   switch (ChooseChange(index)) {
   case 0:
      NOT_FINISHED("Line Constraint:  line_p1");
      break;
   case 1:
      NOT_FINISHED("Line Constraint:  line_p2");
      break;
   case 2:
      norm = (v1.Get() - v0.Get());
      t = -((Dot(v0.Get(), norm) - Dot(v2.Get(), norm))
	    / Dot(v1.Get(), norm));
      v2.Assign(v0.Get() + (v1.Get().vector() * t));
      break;
   default:
      cerr << "Unknown variable in Line Constraint!" << endl;
      break;
   }
}

