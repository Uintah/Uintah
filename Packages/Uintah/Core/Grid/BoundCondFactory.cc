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
#include <cstdlib>

using namespace std;
using namespace Uintah;

void BoundCondFactory::create(const ProblemSpecP& ps,BCData& objs)

{
   for(ProblemSpecP child = ps->findBlock("BCType"); child != 0;
       child = child->findNextBlock("BCType")){
     
     map<string,string> bc_attr;
     child->getAttributes(bc_attr);
     int mat_id;
     if (bc_attr["id"] != "all")
       mat_id = atoi(bc_attr["id"].c_str());
     else
       mat_id = 0;
     
     if (bc_attr["var"] == "None") {
       BoundCondBase* bc = new NoneBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Symmetric") {
       BoundCondBase* bc = new SymmetryBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["var"] ==  "Neighbor") {
       BoundCondBase* bc = new NeighBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Velocity") {
       BoundCondBase* bc = new VelocityBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Temperature") {
       BoundCondBase* bc = new TemperatureBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Pressure") {
       BoundCondBase* bc = new PressureBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Density") {
       BoundCondBase* bc = new DensityBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }

     else {
       cerr << "Unknown Boundary Condition Type " << "(" << bc_attr["var"] 
	    << ")" << endl;
       //	exit(1);
     }
   }
}

