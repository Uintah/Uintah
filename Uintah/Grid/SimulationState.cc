
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/Reductions.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah;

SimulationState::SimulationState(ProblemSpecP &ps)
{
   VarLabel* nonconstDelt = scinew VarLabel("delT",
    ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   nonconstDelt->allowMultipleComputes();
   delt_label = nonconstDelt;
   
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

void SimulationState::registerMPMMaterial(MPMMaterial* matl)
{
   mpm_matls.push_back(matl);
   registerMaterial(matl);
}

void SimulationState::registerICEMaterial(ICEMaterial* matl)
{
   ice_matls.push_back(matl);
   registerMaterial(matl);
}

SimulationState::~SimulationState()
{
  delete delt_label;

  for (int i = 0; i < (int)matls.size(); i++) 
    delete matls[i];
  for (int i = 0; i < (int)mpm_matls.size(); i++) 
    delete mpm_matls[i];
}

//
// $Log$
// Revision 1.19  2001/01/09 22:34:56  jas
// Moved registerMaterial to private:.  This is called when you either
// register a MPM or ICE material.  There is no need to call registerMaterial
// inside the application.
//
// Revision 1.18  2001/01/05 18:57:12  witzel
// allow delT to have multiple computes in the task graph
//
// Revision 1.17  2000/11/13 21:39:57  guilkey
// Added stuff for ICEMaterial analogous to the MPMMaterial stuff added
// last week.
//
// Revision 1.16  2000/11/07 22:42:40  guilkey
// Added a vector of MPMMaterial* so that we no longer need to do the
// dynamic cast of a Material* to an MPMMaterial*.  The point here is to
// make coupling with either Arches or ICE more straigtforward.
//
// Revision 1.15  2000/09/28 23:22:01  jas
// Added (int) to remove g++ warnings for STL size().  Reordered initialization
// to coincide with *.h declarations.
//
// Revision 1.14  2000/09/25 20:37:43  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.13  2000/08/09 03:18:05  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.12  2000/08/08 01:32:47  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.11  2000/06/27 20:14:09  guilkey
// Removed Kinetic and Strain energy labels and associated stuff.
//
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

