#include <Uintah/Grid/BoundCondFactory.h>
#include <Uintah/Grid/NoneBoundCond.h>
#include <Uintah/Grid/SymmetryBoundCond.h>
#include <Uintah/Grid/NeighBoundCond.h>
#include <Uintah/Grid/KinematicBoundCond.h>
#include <Uintah/Grid/TempThermalBoundCond.h>
#include <Uintah/Grid/FluxThermalBoundCond.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;
using namespace Uintah;

void BoundCondFactory::create(const ProblemSpecP& ps,
				   std::vector<BoundCond*>& objs)

{
   for(ProblemSpecP child = ps->findBlock("BCType"); child != 0;
       child = child->findNextBlock("BCType")){
     
     map<string,string> bc_attr;
     child->getAttributes(bc_attr);
     
     if (bc_attr["var"] == "none")
       objs.push_back(new NoneBoundCond(child));
     
     else if (bc_attr["var"] == "symmetry")
       objs.push_back(new SymmetryBoundCond(child));
     
     else if (bc_attr["var"] ==  "neigh")
       objs.push_back(new NeighBoundCond(child));
     
     else if (bc_attr["var"] == "velocity")
       objs.push_back(new KinematicBoundCond(child));
     
     else if (bc_attr["var"] == "temperature")
       objs.push_back(new TempThermalBoundCond(child));
     
     else if (bc_attr["var"] == "heat_flux")
       objs.push_back(new FluxThermalBoundCond(child));
     
     else {
       cerr << "Unknown Boundary Condition Type " << "(" << bc_attr["var"] 
	    << ")" << endl;
       //	exit(1);
     }
   }
}

// $Log$
// Revision 1.1  2000/06/27 22:31:50  jas
// Grid boundary conditions that are stored at the patch level.
//
