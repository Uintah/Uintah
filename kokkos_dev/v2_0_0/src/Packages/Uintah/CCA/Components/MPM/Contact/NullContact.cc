
// NullContact.cc
// One of the derived Contact classes.  This particular
// class is used when no contact is desired.  This would
// be used for example when a single velocity field is
// present in the problem, so doing contact wouldn't make
// sense.

#include <Packages/Uintah/CCA/Components/MPM/Contact/NullContact.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
using namespace Uintah;

#define FRACTURE
#undef FRACTURE

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

void NullContact::exMomInterpolated(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      NCVariable<Vector> gvelocity;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->getModifiable(   gvelocity, lb->gVelocityLabel, dwi, patch);
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
      NCVariable<Vector> gv_star;
      NCVariable<Vector> gacc;
      NCVariable<double> frictionalWork;

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->getModifiable(gv_star, lb->gVelocityStarLabel,        dwi, patch);
      new_dw->getModifiable(gacc,    lb->gAccelerationLabel,        dwi, patch);
#ifdef FRACTURE
      new_dw->getModifiable(frictionalWork,lb->frictionalWorkLabel, dwi, patch);
#else	    
      new_dw->allocateAndPut(frictionalWork,lb->frictionalWorkLabel,dwi, patch);
      frictionalWork.initialize(0.);
#endif      
    }
  }
}

void NullContact::addComputesAndRequiresInterpolated( Task* t,
						const PatchSet*,
						const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->modifies(lb->gVelocityLabel, mss);
}

void NullContact::addComputesAndRequiresIntegrated( Task* t,
					     const PatchSet* ,
					     const MaterialSet* ms) const
{
  const MaterialSubset* mss = ms->getUnion();
  t->modifies(lb->gVelocityStarLabel, mss);
  t->modifies(lb->gAccelerationLabel, mss);
#ifdef FRACTURE
  t->modifies(lb->frictionalWorkLabel, mss);
#else    
  t->computes(lb->frictionalWorkLabel);
#endif
}
