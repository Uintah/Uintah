/* REFERENCED */
static char *id="@(#) $Id$";

// SimpleHEBurn.cc
//
// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "SimpleHEBurn.h"
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah::MPM;

SimpleHEBurn::SimpleHEBurn(ProblemSpecP& ps)
{
  // Constructor

  ps->require("a",a);
  ps->require("b",b); 
  d_burnable = true;
  
  std::cerr << "a = " << a << " b = " << b << std::endl;

}

SimpleHEBurn::~SimpleHEBurn()
{
  // Destructor

}

void SimpleHEBurn::initializeBurnModelData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouseP& new_dw)
{
   const MPMLabel* lb = MPMLabel::getLabels();

   ParticleVariable<int> pIsIgnited;
   new_dw->allocate(pIsIgnited, lb->pIsIgnitedLabel, matl->getDWIndex(), patch);
   CCVariable<double> burnedMass;
   new_dw->allocate(burnedMass, lb->cBurnedMassLabel, matl->getDWIndex(),patch);

   ParticleSubset* pset = pIsIgnited.getParticleSubset();
   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
        particleIndex idx = *iter;
	pIsIgnited[idx]=0;
   }
   new_dw->put(pIsIgnited, lb->pIsIgnitedLabel, matl->getDWIndex(), patch);
   new_dw->put(burnedMass, lb->cBurnedMassLabel, matl->getDWIndex(), patch);
}

void SimpleHEBurn::addCheckIfComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const Patch* patch,
                                                 DataWarehouseP& old_dw,
                                                 DataWarehouseP& new_dw) const
{

  const MPMLabel* lb = MPMLabel::getLabels();

  task->requires(old_dw, lb->pIsIgnitedLabel, matl->getDWIndex(),
				patch, Ghost::None);

//  task->requires(new_dw, lb->pTemperatureRateLabel, matl->getDWIndex(),
//				patch, Ghost::None);

  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(old_dw, lb->cBurnedMassLabel,matl->getDWIndex(), 
				patch, Ghost::None);

  task->computes(new_dw, lb->pIsIgnitedLabel,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->cBurnedMassLabel,matl->getDWIndex(),patch);

  task->requires(old_dw, lb->delTLabel);

}

void SimpleHEBurn::addMassRateComputesAndRequires(Task* task,
                                                  const MPMMaterial* matl,
                                                  const Patch* patch,
                                                  DataWarehouseP& old_dw,
                                                  DataWarehouseP& new_dw) const
{
  const MPMLabel* lb = MPMLabel::getLabels();

  task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(new_dw, lb->pIsIgnitedLabel, matl->getDWIndex(),
				patch, Ghost::None);

  task->requires(old_dw, lb->delTLabel);

  task->requires(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(),
                                patch, Ghost::None);

  task->computes(new_dw, lb->pMassLabel,matl->getDWIndex(),patch);

  task->computes(new_dw, lb->pVolumeLabel,matl->getDWIndex(),patch);
}

void SimpleHEBurn::checkIfIgnited(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{

  double heatFluxToParticle;

  int matlindex = matl->getDWIndex();

  const MPMLabel* lb = MPMLabel::getLabels();
  // Create array for the particle's "IsIgnited" flag
  ParticleVariable<int> pIsIgnited;
  old_dw->get(pIsIgnited, lb->pIsIgnitedLabel,
                                matlindex, patch,Ghost::None,0);


  ParticleVariable<int> pissurf;
  old_dw->get(pissurf, lb->pSurfLabel,
                                matlindex, patch,Ghost::None,0);


//  ParticleVariable<double> pTemperatureRate;
//  new_dw->get(pTemperatureRate, lb->pTemperatureRateLabel,
//                                matlindex, patch,Ghost::None,0);

  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, matlindex, patch,Ghost::None,0);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  double specificHeat = matl->getSpecificHeat();

  ParticleSubset* pset = pmass.getParticleSubset();
//  ASSERT(pset == pTemperatureRate.getParticleSubset());
  ASSERT(pset == pIsIgnited.getParticleSubset());

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

//     heatFluxToParticle = pmass[idx]*pTemperatureRate[idx]*delT
//							*specificHeat;

     //Insert some conditional on heatFluxToParticle here
     if(pissurf[idx] == 1){
	pIsIgnited[idx] = 1;
     }
     else {
	pIsIgnited[idx] = 0;
     }
  }

  new_dw->put(pIsIgnited,lb->pIsIgnitedLabel, matlindex, patch);
  new_dw->put(pissurf,lb->pSurfLabel, matlindex, patch);
}
 
void SimpleHEBurn::computeMassRate(const Patch* patch,
				   const MPMMaterial* matl,
				   DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw)
{

  int matlindex = matl->getDWIndex();
  const MPMLabel* lb = MPMLabel::getLabels();

  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, matlindex, patch,Ghost::None,0);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume,lb->pVolumeDeformedLabel, matlindex, patch,Ghost::None,0);

  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, matlindex, patch, Ghost::None, 0);

  ParticleVariable<int> pIsIgnited;
  new_dw->get(pIsIgnited, lb->pIsIgnitedLabel,
                                matlindex, patch,Ghost::None,0);

  CCVariable<double> burnedMass;
  old_dw->get(burnedMass, lb->cBurnedMassLabel,
			matlindex, patch,Ghost::None,0);

  IntVector ci;

  ParticleSubset* pset = pmass.getParticleSubset();
  ASSERT(pset == pvolume.getParticleSubset());
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     if(pIsIgnited[idx]==1){
	pmass[idx]   = pmass[idx] - .5*a;
	pvolume[idx] = pmass[idx]/1000.0;
        ci =  patch->findCell(px[idx]);
	burnedMass[ci] += .5*a;
	
	if(pmass[idx]<0.0){
	  pmass[idx]=0.0;
	  burnedMass[ci] -= .5*a;
	}
     }
     else {
	// Do nothing to the particle mass and volume
     }
  }

  new_dw->put(pmass,lb->pMassLabel, matlindex, patch);
  new_dw->put(pvolume,lb->pVolumeLabel, matlindex, patch);
  new_dw->put(burnedMass,lb->cBurnedMassLabel, matlindex, patch);
}
 
// $Log$
// Revision 1.6  2000/06/13 23:05:35  guilkey
// Added some stuff to SimpleBurn model.
//
// Revision 1.5  2000/06/08 17:37:07  guilkey
// Fixed small error.
//
// Revision 1.4  2000/06/08 17:09:11  guilkey
// Changed an old_dw to a new_dw.
//
// Revision 1.3  2000/06/08 16:49:45  guilkey
// Added more stuff to the burn models.  Most infrastructure is now
// in place to change the mass and volume, we just need a little bit of science.
//
// Revision 1.2  2000/06/06 18:04:02  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//
