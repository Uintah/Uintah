// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "NullHEBurn.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

NullHEBurn::NullHEBurn(ProblemSpecP& /*ps*/)
{
  // Constructor
 
  d_burnable = false;

}

NullHEBurn::~NullHEBurn()
{
  // Destructor

}

void NullHEBurn::initializeBurnModelData(const Patch* patch,
                                         const MPMMaterial* matl,
                                         DataWarehouse* new_dw)
{
  // Nothing to be done

}

bool NullHEBurn::getBurns() const
{
  return d_burnable;
}

void NullHEBurn::addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const
{
  task->requires(Task::OldDW, lb->pMassLabel, matl->thisMaterial(),
		 Ghost::None);
  task->requires(Task::NewDW, lb->pVolumeDeformedLabel, matl->thisMaterial(),
		 Ghost::None);
  
  task->computes(lb->pMassLabel_preReloc,matl->thisMaterial());
  task->computes(lb->pVolumeLabel_preReloc, matl->thisMaterial());
}

void NullHEBurn::computeMassRate(const PatchSubset* patches,
				 const MPMMaterial* matl,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    //int matlindex = matl->getDWIndex();
    //  const MPMLabel* lb = MPMLabel::getLabels();

    // Carry the mass and volume forward
    ParticleSubset* pset = old_dw->getParticleSubset(matl->getDWIndex(), patch);
    ParticleVariable<double> pmass;
    old_dw->get(pmass, lb->pMassLabel, pset);
    ParticleVariable<double> pvolume;
    new_dw->get(pvolume, lb->pVolumeDeformedLabel, pset);

    new_dw->put(pmass,lb->pMassLabel_preReloc);
    new_dw->put(pvolume,lb->pVolumeLabel_preReloc);
  }
}
