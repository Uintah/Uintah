#include <Core/Malloc/Allocator.h>
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
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <stdlib.h>

using namespace std;
using namespace Uintah;

void BoundCondFactory::create(const ProblemSpecP& ps,BCData& objs)

{
   for(ProblemSpecP child = ps->findBlock("BCType"); child != 0;
       child = child->findNextBlock("BCType")){
     
     map<string,string> bc_attr;
     child->getAttributes(bc_attr);
     int mat_id;
     // Check to see if "id" is defined
     if (bc_attr.find("id") == bc_attr.end()) 
       SCI_THROW(ProblemSetupException("id is not specified in the BCType tag"));
     
     if (bc_attr["id"] != "all")
       mat_id = atoi(bc_attr["id"].c_str());
     else
       mat_id = 0;
     
     if (bc_attr["var"] == "None") {
       BoundCondBase* bc = scinew NoneBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Symmetric") {
       BoundCondBase* bc = scinew SymmetryBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["var"] ==  "Neighbor") {
       BoundCondBase* bc = scinew NeighBoundCond(child);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Velocity" && 
             (bc_attr["var"]   == "Neumann"  ||
              bc_attr["var"]   == "NegInterior"  ||
              bc_attr["var"]   == "LODI" ||
              bc_attr["var"]   == "Neumann_CkValve"  ||
              bc_attr["var"]   == "Dirichlet") ) {
       BoundCondBase* bc = scinew VelocityBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Temperature" &&
             (bc_attr["var"]   == "Neumann"  ||
              bc_attr["var"]   == "LODI" ||
              bc_attr["var"]   == "Dirichlet") ) {
       BoundCondBase* bc = scinew TemperatureBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Pressure" &&
             (bc_attr["var"]   == "Neumann"  ||
              bc_attr["var"]   == "Dirichlet") ) {
       BoundCondBase* bc = scinew PressureBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }
     
     else if (bc_attr["label"] == "Density" &&
             (bc_attr["var"]   == "Neumann"  ||
              bc_attr["var"]   == "LODI" ||
              bc_attr["var"]   == "Dirichlet") ) {
       BoundCondBase* bc = scinew DensityBoundCond(child,bc_attr["var"]);
       objs.setBCValues(mat_id,bc);
     }

     else {
       cerr << "Unknown Boundary Condition Type " << "(" << bc_attr["var"] 
	    << ")  " << bc_attr["label"]<<endl;
       exit(1);
     }
   }
}

