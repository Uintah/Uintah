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
                                         DataWarehouseP& new_dw)
{
  // Nothing to be done

}

bool NullHEBurn::getBurns() const
{
  return d_burnable;
}

void NullHEBurn::addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const Patch* patch,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw) const
{
  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
                                patch, Ghost::None);
  
  task->requires(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(),
                                patch, Ghost::None);
  
  task->computes(new_dw, lb->pMassLabel_preReloc,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->pVolumeLabel_preReloc,matl->getDWIndex(),patch);
}

void NullHEBurn::computeMassRate(const Patch* patch,
				 const MPMMaterial* matl,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw)
{
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

