/* REFERENCED */
static char *id="@(#) $Id$";

// NullHEBurn.cc
//
// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "NullHEBurn.h"
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah::MPM;

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

void NullHEBurn::addCheckIfComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const Patch* /*patch*/,
                                               DataWarehouseP& /*old_dw*/,
                                               DataWarehouseP& /*new_dw*/) const
{
  // Nothing is done so no dependencies

}

void NullHEBurn::addMassRateComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const Patch* patch,
                                               DataWarehouseP& old_dw,
                                               DataWarehouseP& new_dw) const
{
  const MPMLabel* lb = MPMLabel::getLabels();

  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
                                patch, Ghost::None);
  
  task->requires(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(),
                                patch, Ghost::None);
  
  task->computes(new_dw, lb->pMassLabel_preReloc,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->pVolumeLabel_preReloc,matl->getDWIndex(),patch);
}

void NullHEBurn::checkIfIgnited(const Patch* patch,
				const MPMMaterial* matl,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw)
{
  // For the NullHEBurn model, nothing needs to be done here

}

void NullHEBurn::computeMassRate(const Patch* patch,
				 const MPMMaterial* matl,
				 DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw)
{
  int matlindex = matl->getDWIndex();
  const MPMLabel* lb = MPMLabel::getLabels();

  // Carry the mass and volume forward
  ParticleSubset* pset = old_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeDeformedLabel, pset);

  new_dw->put(pmass,lb->pMassLabel_preReloc);
  new_dw->put(pvolume,lb->pVolumeLabel_preReloc);

}

// $Log$
// Revision 1.6  2000/06/16 23:23:38  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.5  2000/06/15 21:57:02  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.4  2000/06/08 16:49:44  guilkey
// Added more stuff to the burn models.  Most infrastructure is now
// in place to change the mass and volume, we just need a little bit of science.
//
// Revision 1.3  2000/06/06 18:04:02  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.2  2000/06/03 05:22:06  sparker
// Added .cvsignore
//
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//
