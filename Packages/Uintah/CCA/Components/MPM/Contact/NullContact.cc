
// NullContact.cc
// One of the derived Contact classes.  This particular
// class is used when no contact is desired.  This would
// be used for example when a single velocity field is
// present in the problem, so doing contact wouldn't make
// sense.

#include "NullContact.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
using namespace Uintah;

NullContact::NullContact(ProblemSpecP& ps, SimulationStateP& d_sS)
{
  // Constructor
 
  IntVector v_f;
  ps->require("vel_fields",v_f);
  std::cout << "vel_fields = " << v_f << endl;

  d_sharedState = d_sS;

}

NullContact::~NullContact()
{
  // Destructor

}

void NullContact::initializeContact(const Patch* /*patch*/,
                                    int /*vfindex*/,
                                    DataWarehouseP& /*new_dw*/)
{

}

void NullContact::exMomInterpolated(const ProcessorGroup*,
				    const Patch* patch,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& new_dw)
{

  //  All this does is carry forward the array from gVelocityLabel
  //  to gMomExedVelocityLabel

  int numMatls = d_sharedState->getNumMPMMatls();

  // Retrieve necessary data from DataWarehouse
  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
    int vfindex = mpm_matl->getVFIndex();
    new_dw->get(gvelocity[vfindex], lb->gVelocityLabel, vfindex, patch,
                  Ghost::None, 0);

    new_dw->put(gvelocity[vfindex], lb->gMomExedVelocityLabel, vfindex, patch);
  }

}

void NullContact::exMomIntegrated(const ProcessorGroup*,
				  const Patch* patch,
                                  DataWarehouseP& /*old_dw*/,
                                  DataWarehouseP& new_dw)
{

  //  All this does is carry forward the array from gVelocityStarLabel
  //  and gAccelerationLabel to gMomExedVelocityStarLabel and 
  //  gMomExedAccelerationLabel respectively

  int numMatls = d_sharedState->getNumMPMMatls();

  vector<NCVariable<double> > gmass(numMatls);
  vector<NCVariable<Vector> > gvelocity_star(numMatls);
  vector<NCVariable<Vector> > gacceleration(numMatls);
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
   int vfindex = mpm_matl->getVFIndex();
   new_dw->get(gvelocity_star[vfindex], lb->gVelocityStarLabel, vfindex,
                  patch, Ghost::None, 0);
   new_dw->get(gacceleration[vfindex], lb->gAccelerationLabel, vfindex,
                  patch, Ghost::None, 0);

    new_dw->put(gvelocity_star[vfindex], lb->gMomExedVelocityStarLabel,
							 vfindex, patch);
    new_dw->put(gacceleration[vfindex], lb->gMomExedAccelerationLabel,
							 vfindex, patch);
  }

}

void NullContact::addComputesAndRequiresInterpolated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& /*old_dw*/,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires( new_dw, lb->gVelocityLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityLabel, idx, patch );

}

void NullContact::addComputesAndRequiresIntegrated( Task* t,
                                             const MPMMaterial* matl,
                                             const Patch* patch,
                                             DataWarehouseP& /*old_dw*/,
                                             DataWarehouseP& new_dw) const
{
  int idx = matl->getDWIndex();
  t->requires(new_dw, lb->gVelocityStarLabel, idx, patch, Ghost::None);
  t->requires(new_dw, lb->gAccelerationLabel, idx, patch, Ghost::None);

  t->computes( new_dw, lb->gMomExedVelocityStarLabel, idx, patch);
  t->computes( new_dw, lb->gMomExedAccelerationLabel, idx, patch);

}


