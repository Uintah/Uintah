// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "SimpleHEBurn.h"
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;

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
					   DataWarehouse* new_dw)
{
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
  ParticleVariable<int> pIsIgnited;
  new_dw->allocate(pIsIgnited, lb->pIsIgnitedLabel, pset);
  CCVariable<double> burnedMass;
  new_dw->allocate(burnedMass,lb->cBurnedMassLabel, matl->getDWIndex(), patch);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++) {
    particleIndex idx = *iter;
    pIsIgnited[idx]=0;
  }

  new_dw->put(pIsIgnited, lb->pIsIgnitedLabel);
  new_dw->put(burnedMass, lb->cBurnedMassLabel, matl->getDWIndex(), patch);
}

bool SimpleHEBurn::getBurns() const
{
  return d_burnable;
}

void SimpleHEBurn::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet* patches) const
{
//  task->requires(new_dw, lb->pTemperatureRateLabel, matl->getDWIndex(),
//				patch, Ghost::None);

  task->requires(Task::NewDW, lb->pTemperatureLabel_preReloc,
		 matl->thisMaterial(), Ghost::None);
  
  task->requires(Task::OldDW, lb->pIsIgnitedLabel,
		 matl->thisMaterial(), Ghost::None);

//  task->requires(old_dw, lb->pSurfLabel, matl->getDWIndex(),
//				patch, Ghost::None);

  task->requires(Task::OldDW, lb->pMassLabel,
		 matl->thisMaterial(), Ghost::None);

  task->requires(Task::NewDW, lb->pVolumeDeformedLabel,
		 matl->thisMaterial(), Ghost::None);

  task->requires(Task::OldDW, lb->cBurnedMassLabel,
		 matl->thisMaterial(), Ghost::None);

  task->requires(Task::OldDW, lb->delTLabel);

  task->computes(lb->pIsIgnitedLabel_preReloc,matl->thisMaterial());
  task->computes(lb->pMassLabel_preReloc,matl->thisMaterial());
  task->computes(lb->cBurnedMassLabel,matl->thisMaterial());
//  task->computes(lb->pSurfLabel_preReloc,matl->getDWIndex(),patch);
  task->computes(lb->pVolumeLabel_preReloc, matl->thisMaterial());
}

void SimpleHEBurn::computeMassRate(const PatchSubset* patches,
				   const MPMMaterial* matl,
				   DataWarehouse* old_dw,
				   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int matlindex = matl->getDWIndex();

    ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);

    ParticleVariable<double> pmass;
    old_dw->get(pmass, lb->pMassLabel, pset);
    ParticleVariable<double> pvolume;
    new_dw->get(pvolume,lb->pVolumeDeformedLabel, pset);
    ParticleVariable<Point> px;
    old_dw->get(px, lb->pXLabel, pset);
    ParticleVariable<int> pIsIgnitedOld;
    old_dw->get(pIsIgnitedOld, lb->pIsIgnitedLabel, pset);
    //  ParticleVariable<int> pissurf;
    //  old_dw->get(pissurf, lb->pSurfLabel, pset);
    ParticleVariable<double> pTemperature;
    old_dw->get(pTemperature, lb->pTemperatureLabel, pset);
    CCVariable<double> burnedMass;
    old_dw->get(burnedMass, lb->cBurnedMassLabel,
		matlindex, patch,Ghost::None,0);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel);

    ParticleVariable<int> pIsIgnitedNew;
    new_dw->allocate(pIsIgnitedNew, lb->pIsIgnitedLabel_preReloc, pset);

    ParticleSubset* remove_subset =
      new ParticleSubset(pset->getParticleSet(), false, matlindex, patch);

    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
      particleIndex idx = *iter;

      if(pTemperature[idx] > b || pIsIgnitedOld[idx]==1){
        pIsIgnitedNew[idx]=1;
	IntVector ci;
	if(patch->findCell(px[idx],ci)){
          if(pmass[idx]>0.0){
	    double rho = pmass[idx]/pvolume[idx];
	    pmass[idx] -= a;
	    pvolume[idx] = pmass[idx]/rho;
	    burnedMass[ci] += a;
	
	    if(pmass[idx]<=0.0){
	      burnedMass[ci] -= pmass[idx];
	      pmass[idx]=0.0;
	      pvolume[idx]=0.0;
	      remove_subset->addParticle(idx);
	      cout << px[idx] << " " <<  pmass[idx] << endl;
	      //Find neighboring particle, ignite it
	      double npd=9999.9;
	      double cpd;
	      particleIndex newburn_idx = idx;
	      for(ParticleSubset::iterator iter2 = pset->begin();
		  iter2 != pset->end(); iter2++){
		particleIndex idx2 = *iter2;

		cpd = (px[idx] - px[idx2]).length();
		if(cpd < npd && idx !=idx2 && pIsIgnitedOld[idx2]==0){
		  npd = cpd;
		  newburn_idx=idx2;
		}
	      }
	      cout << "NewParticle " << px[newburn_idx] << endl;
	      pIsIgnitedNew[newburn_idx]=1;
	      pIsIgnitedNew[idx]=2;
	    }
          }
	}
      }
      else if(pIsIgnitedOld[idx]==2){
	pIsIgnitedNew[idx]=2;
      }
    }

    new_dw->put(pmass,lb->pMassLabel_preReloc);
    new_dw->put(pvolume,lb->pVolumeLabel_preReloc);
    //  new_dw->put(pissurf,lb->pSurfLabel_preReloc);
    new_dw->put(burnedMass,lb->cBurnedMassLabel, matlindex, patch);
    new_dw->put(pIsIgnitedNew,lb->pIsIgnitedLabel_preReloc);

    new_dw->deleteParticles(remove_subset);
  }
}
 
