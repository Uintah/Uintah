
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Grid/Reductions.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   delt_label = new VarLabel("delt", ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());

  // Get the physical constants that are shared between codes.
  // For now it is just gravity.

  ProblemSpecP phys_cons_ps = ps->findBlock("PhysicalConstants");
  phys_cons_ps->require("gravity",d_gravity);

}

void SimulationState::registerMaterial(Material* matl)
{
   matl->setDWIndex((int)matls.size());
   matls.push_back(matl);
}

//
// $Log$
// Revision 1.7  2000/05/18 18:48:30  jas
// Added gravity.  It is read in from the input file.
//
// Revision 1.6  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.5  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
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

