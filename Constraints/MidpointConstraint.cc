
/*
 *  MidpointConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Constraints/MidpointConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch mc_debug("BaseConstraint", "Midpoint");

MidpointConstraint::MidpointConstraint( const clString& name,
					const Index numSchemes,
					Variable* end1, Variable* end2,
					Variable* p )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = end1;
   vars[1] = end2;
   vars[2] = p;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};

MidpointConstraint::~MidpointConstraint()
{
}


void
MidpointConstraint::Satisfy( const Index index, const Scheme scheme )
{
   Variable& v0 = *vars[0];
   Variable& v1 = *vars[1];
   Variable& v2 = *vars[2];

   if (mc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      v0.Assign(v1.Get() + (v2.Get() - v1.Get()) * 2.0, scheme);
      break;
   case 1:
      v1.Assign(v0.Get() + (v2.Get() - v0.Get()) * 2.0, scheme);
      break;
   case 2:
      v2.Assign(v0.Get() + ((v1.Get() - v0.Get()) / 2.0), scheme);
      break;
   default:
      cerr << "Unknown variable in Midpoint Constraint!" << endl;
      break;
   }
}

