
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


#include <Constraints/CenterConstraint.h>
#include <Classlib/Debug.h>

static DebugSwitch cc_debug("Constraints", "Center");

CenterConstraint::CenterConstraint( const clString& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2 )
:BaseConstraint(name, numSchemes, 3)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


CenterConstraint::CenterConstraint( const clString& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2,
				    PointVariable* p3 )
:BaseConstraint(name, numSchemes, 4)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   vars[3] = p3;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


CenterConstraint::CenterConstraint( const clString& name,
				    const Index numSchemes,
				    PointVariable* center,
				    PointVariable* p1, PointVariable* p2,
				    PointVariable* p3, PointVariable* p4 )
:BaseConstraint(name, numSchemes, 5)
{
   vars[0] = center;
   vars[1] = p1;
   vars[2] = p2;
   vars[3] = p3;
   vars[4] = p4;
   whichMethod = 0;

   // Tell the variables about ourself.
   Register();
};


CenterConstraint::~CenterConstraint()
{
}


int
CenterConstraint::Satisfy( const Index index, const Scheme scheme, const Real,
			     BaseVariable*& var, VarCore& c )
{
   PointVariable& p1 = *vars[1];
   PointVariable& p2 = *vars[2];

   if (cc_debug) {
      ChooseChange(index, scheme);
      print();
   }
   
   switch (ChooseChange(index, scheme)) {
   case 0:
      switch(varCount) {
      case 3:
	 var = vars[0];
	 c = AffineCombination(p1, 0.5, p2, 0.5);
	 break;
      case 4:
	 PointVariable& p3 = *vars[3];
	 var = vars[0];
	 c = AffineCombination(p1, 1.0/3.0, p2, 1.0/3.0,
			       p3, 1.0/3.0);
	 break;
      case 5:
	 PointVariable& pthree = *vars[3];
	 PointVariable& p4 = *vars[4];
	 var = vars[0];
	 c = AffineCombination(p1, 0.25, p2, 0.25,
			       pthree, 0.25, p4, 0.25);
	 break;
      default:
	 break;
      }
      return 1;
   case 1:
   case 2:
      ASSERT(!"CenterConstraint:  Can only satisfy center");
      break;
   case 3:
      ASSERT(varCount >= 4);
      ASSERT(!"CenterConstraint:  Can only satisfy center");
      break;
   case 4:
      ASSERT(varCount >= 5);
      ASSERT(!"CenterConstraint:  Can only satisfy center");
      break;
   default:
      cerr << "Unknown variable in Center Constraint!" << endl;
      break;
   }
   return 0;
}

