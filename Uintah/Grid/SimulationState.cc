
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Grid/Reductions.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   delt_label = new VarLabel("delT",
    ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());

   strain_energy_label = new VarLabel("StrainEnergy",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());

   kinetic_energy_label = new VarLabel("KineticEnergy",
    ReductionVariable<double, Reductions::Sum<double> >::getTypeDescription());

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
// Revision 1.10  2000/05/31 22:27:52  guilkey
// Added stuff for integrated quanities.
//
// Revision 1.9  2000/05/31 20:25:32  guilkey
// Added the beginnings of a Sum reduction, which would take data from
// multiple patches, materials, etc. and add them together.  The immediate
// application is for computing the strain energy and storing it.  I'm
// going to need some help with this.
//
// Revision 1.8  2000/05/30 18:15:10  dav
// Changed delt to delT.  Should the MPM code use this delT, or the delT it creates in MPMLabel?
//
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

