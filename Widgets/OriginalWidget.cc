
/*
 *  InterfaceWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include "InterfaceWidget.h"
#include "DistanceConstraint.h"
#include "HypotenousConstraint.h"


const Index NumCons = 7;
const Index NumVars = 6;
const Index NumPts = 4;
const Index NumPrims = 4;
const Index NumSchemes = 2;

InterfaceWidget::InterfaceWidget()
: BaseWidget(NumVars, NumCons, NumPrims, NumPts)
{
   Index vindex, cindex, pindex;
   
   vindex = 0;
   variables[vindex++] = new Variable("P1", Scheme1, Point(100,100,0));
   variables[vindex++] = new Variable("P2", Scheme2, Point(200,100,0));
   variables[vindex++] = new Variable("P3", Scheme1, Point(200,200,0));
   variables[vindex++] = new Variable("P4", Scheme2, Point(100,200,0));
   variables[vindex++] = new Variable("DIST", Scheme1, Point(100,100,100));
   variables[vindex++] = new Variable("HYPO", Scheme1, Point(sqrt(2*100*100),
							     sqrt(2*100*100),
							     sqrt(2*100*100)));

   pindex = 0;
   primitives[pindex++] = new Primitive(variables[0]->GetRef(),
					variables[1]->GetRef());
   primitives[pindex++] = new Primitive(variables[1]->GetRef(),
					variables[2]->GetRef());
   primitives[pindex++] = new Primitive(variables[2]->GetRef(),
					variables[3]->GetRef());
   primitives[pindex++] = new Primitive(variables[3]->GetRef(),
					variables[0]->GetRef());

   cindex = 0;
   constraints[cindex] = new DistanceConstraint("H13",
						NumSchemes,
						variables[0],
						variables[2],
						variables[5]);
   constraints[cindex]->VarChoices(Scheme1, 2, 2, 1);
   constraints[cindex]->VarChoices(Scheme2, 1, 0, 1);
   constraints[cindex]->Priorities(P_Highest, P_Highest, P_Default);
   cindex++;
   constraints[cindex] = new DistanceConstraint("H24",
						NumSchemes,
						variables[1],
						variables[3],
						variables[5]);
   constraints[cindex]->VarChoices(Scheme1, 1, 0, 1);
   constraints[cindex]->VarChoices(Scheme2, 2, 2, 1);
   constraints[cindex]->Priorities(P_Highest, P_Highest, P_Default);
   cindex++;
   constraints[cindex] = new HypotenousConstraint("H",
						  NumSchemes,
						  variables[5],
						  variables[4]);
   constraints[cindex]->VarChoices(Scheme1, 1, 0);
   constraints[cindex]->VarChoices(Scheme2, 1, 0);
   constraints[cindex]->Priorities(P_Highest, P_Default);
   cindex++;
   constraints[cindex] = new DistanceConstraint("S12",
						NumSchemes,
						variables[0],
						variables[1],
						variables[4]);
   constraints[cindex]->VarChoices(Scheme1, 1, 1, 1);
   constraints[cindex]->VarChoices(Scheme2, 0, 0, 0);
   constraints[cindex]->Priorities(P_Default, P_Default, P_LowMedium);
   cindex++;
   constraints[cindex] = new DistanceConstraint("S14",
						NumSchemes,
						variables[0],
						variables[3],
						variables[4]);
   constraints[cindex]->VarChoices(Scheme1, 1, 1, 1);
   constraints[cindex]->VarChoices(Scheme2, 0, 0, 0);
   constraints[cindex]->Priorities(P_Default, P_Default, P_LowMedium);
   cindex++;
   constraints[cindex] = new DistanceConstraint("S32",
						NumSchemes,
						variables[2],
						variables[1],
						variables[4]);
   constraints[cindex]->VarChoices(Scheme1, 1, 1, 1);
   constraints[cindex]->VarChoices(Scheme2, 0, 0, 0);
   constraints[cindex]->Priorities(P_Default, P_Default, P_LowMedium);
   cindex++;
   constraints[cindex] = new DistanceConstraint("S34",
						NumSchemes,
						variables[2],
						variables[3],
						variables[4]);
   constraints[cindex]->VarChoices(Scheme1, 1, 1, 1);
   constraints[cindex]->VarChoices(Scheme2, 0, 0, 0);
   constraints[cindex]->Priorities(P_Default, P_Default, P_LowMedium);
   cindex++;

   // Verification.
   ASSERT(vindex==NumVariables && cindex==NumConstraints && pindex==NumPrimitives);
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
}


InterfaceWidget::~InterfaceWidget()
{
}


