
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Material.h>

using namespace Uintah;

SimulationState::SimulationState()
{
   delt_label = new VarLabel("delt", ReductionVariable<double>::getTypeDescription());
}

void SimulationState::registerMaterial(Material* matl)
{
   matl->setDWIndex((int)matls.size());
   matl->setVFIndex((int)matls.size());
   matls.push_back(matl);
}

//
// $Log$
// Revision 1.4  2000/04/28 08:11:33  sparker
// ConstitutiveModelFactory should return null on failure
// MPMMaterial checks for failed constitutive model creation
// DWindex and VFindex are now initialized
// Fixed input file to match ConstitutiveModelFactory
//
// Revision 1.3  2000/04/26 06:48:54  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/04/24 21:04:38  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.1  2000/04/20 18:56:30  sparker
// Updates to MPM
//
//

