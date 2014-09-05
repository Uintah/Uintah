#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/NoneBoundCond.h>
#include <Packages/Uintah/Core/Grid/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/NeighBoundCond.h>
#include <Packages/Uintah/Core/Grid/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/PressureBoundCond.h>
#include <Packages/Uintah/Core/Grid/DensityBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
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

