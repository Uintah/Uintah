#include <Uintah/Grid/BoundCondFactory.h>
#include <Uintah/Grid/NoneBoundCond.h>
#include <Uintah/Grid/SymmetryBoundCond.h>
#include <Uintah/Grid/NeighBoundCond.h>
#include <Uintah/Grid/VelocityBoundCond.h>
#include <Uintah/Grid/TemperatureBoundCond.h>
#include <Uintah/Grid/PressureBoundCond.h>
#include <Uintah/Grid/DensityBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;
using namespace Uintah;

void BoundCondFactory::create(const ProblemSpecP& ps,
				   std::vector<BoundCondBase*>& objs)

{
   for(ProblemSpecP child = ps->findBlock("BCType"); child != 0;
       child = child->findNextBlock("BCType")){
     
     map<string,string> bc_attr;
     child->getAttributes(bc_attr);
     
     if (bc_attr["var"] == "None")
       objs.push_back(new NoneBoundCond(child));
     
     else if (bc_attr["label"] == "Symmetric")
       objs.push_back(new SymmetryBoundCond(child));
     
     else if (bc_attr["var"] ==  "Neighbor")
       objs.push_back(new NeighBoundCond(child));
     
     else if (bc_attr["label"] == "Velocity")
       objs.push_back(new VelocityBoundCond(child,bc_attr["var"]));
     
     else if (bc_attr["label"] == "Temperature") 
       objs.push_back(new TemperatureBoundCond(child,bc_attr["var"]));
     
     else if (bc_attr["label"] == "Pressure")
       objs.push_back(new PressureBoundCond(child,bc_attr["var"]));
     
     else if (bc_attr["label"] == "Density")
       objs.push_back(new DensityBoundCond(child,bc_attr["var"]));

     else {
       cerr << "Unknown Boundary Condition Type " << "(" << bc_attr["var"] 
	    << ")" << endl;
       //	exit(1);
     }
   }
}

// $Log$
// Revision 1.6  2000/12/09 01:22:03  guilkey
// Fix to the creation of Symmetric Boundary conditions.
//
// Revision 1.5  2000/12/08 02:36:56  jas
// Fixed some bugs for the names associated with NoneBC, NeighborBC and
// SymmetryBC.
//
// Revision 1.4  2000/11/02 21:25:55  jas
// Rearranged the boundary conditions so there is consistency between ICE
// and MPM.  Added fillFaceFlux for the Neumann BC condition.  BCs are now
// declared differently in the *.ups file.
//
// Revision 1.3  2000/10/26 23:27:20  jas
// Added Density Boundary Conditions needed for ICE.
//
// Revision 1.2  2000/10/18 03:46:46  jas
// Added pressure boundary conditions.
//
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//
