
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
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
using namespace Uintah;

NullContact::NullContact(ProblemSpecP& ps, SimulationStateP& d_sS,MPMLabel* Mlb)
{
  // Constructor
 
  IntVector v_f;
  ps->require("vel_fields",v_f);

  d_sharedState = d_sS;
  lb = Mlb;

}

NullContact::~NullContact()
{
  // Destructor

}

void NullContact::initializeContact(const Patch* /*patch*/,
                                    int /*vfindex*/,
                                    DataWarehouse* /*new_dw*/)
{

}

void NullContact::exMomInterpolated(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){

      //  All this does is carry forward the array from gVelocityLabel
      //  to gMomExedVelocityLabel

      // Retrieve necessary data from DataWarehouse
      NCVariable<Vector> gvelocity;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity, lb->gVelocityLabel, dwi, patch, Ghost::None, 0);
      new_dw->put(gvelocity, lb->gMomExedVelocityLabel, dwi, patch);
    }
  }
}

void NullContact::exMomIntegrated(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){

      //  All this does is carry forward the array from gVelocityStarLabel
      //  and gAccelerationLabel to gMomExedVelocityStarLabel and 
      //  gMomExedAccelerationLabel respectively

      NCVariable<Vector> gv_star;
      NCVariable<Vector> gacc;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gv_star, lb->gVelocityStarLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(gacc,    lb->gAccelerationLabel, dwi, patch, Ghost::None, 0);

      new_dw->put(gv_star, lb->gMomExedVelocityStarLabel, dwi, patch);
      new_dw->put(gacc,    lb->gMomExedAccelerationLabel, dwi, patch);
    }
  }
}

void NullContact::addComputesAndRequiresInterpolated( Task* t,
						const PatchSet*,
						const MaterialSet* ) const
{
  t->requires( Task::NewDW, lb->gVelocityLabel,Ghost::None);
  t->computes( lb->gMomExedVelocityLabel);
}

void NullContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* ,
					     const MaterialSet*) const
{
  t->requires(Task::NewDW, lb->gVelocityStarLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gAccelerationLabel, Ghost::None);

  t->computes( lb->gMomExedVelocityStarLabel);
  t->computes( lb->gMomExedAccelerationLabel);
}
