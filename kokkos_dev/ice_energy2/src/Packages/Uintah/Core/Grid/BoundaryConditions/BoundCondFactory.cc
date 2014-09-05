#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/NoneBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SymmetryBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/NeighBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/VelocityBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/PressureBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/DensityBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SpecificVolBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/MassFracBoundCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <stdlib.h>

using namespace std;
using namespace Uintah;

void BoundCondFactory::create(ProblemSpecP& child,
                              BoundCondBase* &bc, int& mat_id)

{
  map<string,string> bc_attr;
  child->getAttributes(bc_attr);
  
  //__________________________________
  // add Models transport variables labels here
  // hard coding ...yuck
  bool ModelsBC = false;   //
  string::size_type pos1 = bc_attr["label"].find ("massFraction");
  string::size_type pos2 = bc_attr["label"].find ("scalar");
  string::size_type pos3 = bc_attr["label"].find ("cumulativeEnergyReleased");
  string::size_type pos4 = bc_attr["label"].find ("mixtureFraction");
  
  if ( pos1 != std::string::npos || pos2 != std::string::npos 
    || pos3 != std::string::npos || pos4 != std::string::npos){
    ModelsBC = true;
  }
  
  // Check to see if "id" is defined
  if (bc_attr.find("id") == bc_attr.end()) 
    SCI_THROW(ProblemSetupException("id is not specified in the BCType tag", __FILE__, __LINE__));
  
  if (bc_attr["id"] != "all")
    mat_id = atoi(bc_attr["id"].c_str());
  else
    mat_id = -1;  // Old setting was 0.

  if (bc_attr["var"] == "None") {
    bc = scinew NoneBoundCond(child);
  }
  
  else if (bc_attr["label"] == "Symmetric") {
    bc = scinew SymmetryBoundCond(child);
  }
  
  else if (bc_attr["var"] ==  "Neighbor") {
    bc = scinew NeighBoundCond(child);
  }
  
  else if (bc_attr["label"] == "Velocity" && 
           (bc_attr["var"]   == "Neumann"  ||
            bc_attr["var"]   == "LODI" ||
            bc_attr["var"]   == "Custom" ||
            bc_attr["var"]   == "creep" ||
            bc_attr["var"]   == "slip" ||
            bc_attr["var"]   == "MMS_1" ||
            bc_attr["var"]   == "Dirichlet") ) {
    bc = scinew VelocityBoundCond(child,bc_attr["var"]);
  }
  
  else if (bc_attr["label"] == "Temperature" &&
           (bc_attr["var"]   == "Neumann"  ||
            bc_attr["var"]   == "LODI" ||
            bc_attr["var"]   == "Custom" ||
            bc_attr["var"]   == "slip" ||
            bc_attr["var"]   == "MMS_1" ||
            bc_attr["var"]   == "Dirichlet") ) {
    bc = scinew TemperatureBoundCond(child,bc_attr["var"]);
  }
  
  else if (bc_attr["label"] == "Pressure" &&
           (bc_attr["var"]   == "Neumann"  ||
            bc_attr["var"]   == "LODI" ||
            bc_attr["var"]   == "Custom" ||
            bc_attr["var"]   == "MMS_1" ||
            bc_attr["var"]   == "Dirichlet") ) {
    bc = scinew PressureBoundCond(child,bc_attr["var"]);
  }
  
  else if (bc_attr["label"] == "Density" &&
           (bc_attr["var"]   == "Neumann"  ||
            bc_attr["var"]   == "Dirichlet_perturbed"  ||
            bc_attr["var"]   == "LODI" ||
            bc_attr["var"]   == "Custom" ||
            bc_attr["var"]   == "Dirichlet") ) {
    bc = scinew DensityBoundCond(child,bc_attr["var"]);
  } 
  else if (bc_attr["label"] == "SpecificVol" &&
           (bc_attr["var"]   == "Neumann" ||
            bc_attr["var"]   == "Dirichlet" ||
            bc_attr["var"]   == "computeFromEOS" ||
            bc_attr["var"]   == "computeFromDensity") ) {
    bc = scinew SpecificVolBoundCond(child,bc_attr["var"]);
  }
  else if (ModelsBC &&
           (bc_attr["var"]   == "Neumann"  ||
            bc_attr["var"]   == "Dirichlet") ) {  
    bc = scinew MassFractionBoundCond(child,bc_attr["var"],bc_attr["label"]);
  }
  else {
    ostringstream warn;
    warn << "BoundCondFactory: Unknown Boundary Condition: "<< bc_attr["label"]
         << " Type " << "(" << bc_attr["var"] << ")  " <<endl;
    SCI_THROW(ProblemSetupException(warn.str(), __FILE__, __LINE__));
  }
}

