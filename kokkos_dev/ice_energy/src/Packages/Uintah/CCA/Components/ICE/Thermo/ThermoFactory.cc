#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ConstantThermo.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraDetailed.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraMixtureFraction.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraSingleMixture.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <string>

using namespace Uintah;

ThermoInterface* ThermoFactory::create(ProblemSpecP& ps, ModelSetup* setup, ICEMaterial* ice_matl)
{
  ProblemSpecP child = ps->findBlock("Thermo");
  if(!child)
    return 0; // The tag might not be there in old input files
  std::string type;
  if(!child->getAttribute("type",type))
    throw ProblemSetupException("No type for Thermo", __FILE__, __LINE__); 
  
  if (type == "constant") 
    return scinew ConstantThermo(child, setup, ice_matl);
  else if (type == "cantera detailed")
    return scinew CanteraDetailed(child, setup, ice_matl);
  else if (type == "cantera mixture fraction")
    return scinew CanteraMixtureFraction(child, setup, ice_matl);
  else if (type == "cantera single mixture") 
    return scinew CanteraSingleMixture(child, setup, ice_matl);
  else
    throw ProblemSetupException("Unknown Thermo Type ("+type+")", __FILE__, __LINE__);
}
