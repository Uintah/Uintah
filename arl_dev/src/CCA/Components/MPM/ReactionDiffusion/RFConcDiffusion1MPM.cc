/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/ReactionDiffusion/RFConcDiffusion1MPM.h>
#include <CCA/Components/MPM/ReactionDiffusion/ReactionDiffusionLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Task.h>

#include <iostream>
using namespace std;
using namespace Uintah;


RFConcDiffusion1MPM::RFConcDiffusion1MPM(ProblemSpecP& ps, SimulationStateP& sS, MPMFlags* Mflag, string diff_type):
  ScalarDiffusionModel(ps, sS, Mflag, diff_type) {
	
  ps->require("initial_chemical_potential", init_potential);
}

RFConcDiffusion1MPM::~RFConcDiffusion1MPM() {

}

void RFConcDiffusion1MPM::scheduleComputeFlux(Task* task, const MPMMaterial* matl, 
                                              const PatchSet* patch) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  task->requires(Task::OldDW, d_lb->pXLabel,                   matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pSizeLabel,                matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pMassLabel,                matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pVolumeLabel,              matlset, gan, NGP);
  task->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,  matlset, gan, NGP);
  task->requires(Task::NewDW, d_lb->gMassLabel,                matlset, gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,     matlset, gan, 2*NGN);

  task->requires(Task::OldDW, d_rdlb->pConcentrationLabel,     matlset, gan, NGP);
  task->requires(Task::NewDW, d_rdlb->gHydrostaticStressLabel, matlset, gan, 2*NGN);

  task->computes(d_rdlb->gdCdtLabel,  matlset);
}

void RFConcDiffusion1MPM::computeFlux(const Patch* patch, const MPMMaterial* matl,
                                                DataWarehouse* old_dw, DataWarehouse* new_dw)
{

  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;


  ParticleInterpolator* interpolator = d_Mflag->d_interpolator->clone(patch);
  vector<IntVector> ni(interpolator->size());
  vector<Vector> d_S(interpolator->size());

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();
           
  int dwi = matl->getDWIndex();
  constParticleVariable<Point>   px;
  constParticleVariable<double>  pvol,pMass;
  constParticleVariable<double>  pConcentration;
  constParticleVariable<Matrix3> psize;
  constParticleVariable<Matrix3> deformationGradient;
  constNCVariable<double>        gConcentration,gMass;
  constNCVariable<double>        gHydrostaticStress;

  ParticleVariable<Vector>       pConcentrationGradient;
  ParticleVariable<Vector>       pHydroStressGradient;
  ParticleVariable<Vector>       pPotentialFlux;
  NCVariable<double>             gdCdt;

  ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, NGP, d_lb->pXLabel);

  old_dw->get(px,                  d_lb->pXLabel,                  pset);
  old_dw->get(pvol,                d_lb->pVolumeLabel,             pset);
  old_dw->get(pMass,               d_lb->pMassLabel,               pset);
  old_dw->get(psize,               d_lb->pSizeLabel,               pset);
  old_dw->get(pConcentration,      d_rdlb->pConcentrationLabel,    pset);
  old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);

  new_dw->get(gConcentration,     d_rdlb->gConcentrationLabel,     dwi, patch, gac,2*NGN);
  new_dw->get(gHydrostaticStress, d_rdlb->gHydrostaticStressLabel, dwi, patch, gac,2*NGN);
  new_dw->get(gMass,              d_lb->gMassLabel,                dwi, patch, gnone, 0);
  new_dw->allocateAndPut(gdCdt,   d_rdlb->gdCdtLabel,    dwi, patch);

  new_dw->allocateTemporary(pConcentrationGradient, pset);
  new_dw->allocateTemporary(pHydroStressGradient,   pset);
  new_dw->allocateTemporary(pPotentialFlux,         pset);
  
  gdCdt.initialize(0.);

  double chem_potential;
  double mech_potential; 
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;

    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    pConcentrationGradient[idx] = Vector(0.0,0.0,0.0);
    pHydroStressGradient[idx]   = Vector(0.0,0.0,0.0);
    for (int k = 0; k < d_Mflag->d_8or27; k++){
      for (int j = 0; j<3; j++) {
          pConcentrationGradient[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
          pHydroStressGradient[idx][j] += gHydrostaticStress[ni[k]] * d_S[k][j] * oodx[j];
      }
	  }

    chem_potential = -diffusivity;
    mech_potential = diffusivity * (1 - pConcentration[idx]/max_concentration)
                     * pConcentration[idx]*init_potential;

    pPotentialFlux[idx] = chem_potential*pConcentrationGradient[idx]
                          + mech_potential*pHydroStressGradient[idx];
    //cout << "id: " << idx << " CG: " << pConcentrationGradient[idx] << ", PF: " << pPotentialFlux[idx] << endl;
  } //End of Particle Loop

  for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
    particleIndex idx = *iter;
  
    // Get the node indices that surround the cell
    interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

    Vector dU_dx = pPotentialFlux[idx];
    double Cdot_cond = 0.0;
    IntVector node(0,0,0);

    for (int k = 0; k < d_Mflag->d_8or27; k++){
      node = ni[k];
      if(patch->containsNode(node)){
        Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
        Cdot_cond = Dot(div, dU_dx);
        gdCdt[node] -= Cdot_cond;
      }
    }
  } // End of Particle Loop 

	delete interpolator;
}

void RFConcDiffusion1MPM::scheduleComputeDivergence(Task* task, const MPMMaterial* matl, 
		                                                const PatchSet* patch) const
{
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::OldDW, d_sharedState->get_delt_label());

  task->requires(Task::NewDW, d_rdlb->gConcentrationLabel,     gnone);
  task->requires(Task::NewDW, d_rdlb->gConcentrationNoBCLabel, gnone);
  task->requires(Task::NewDW, d_rdlb->gdCdtLabel,              gnone);
  task->computes(d_rdlb->gConcentrationRateLabel, matlset);
  task->computes(d_rdlb->gConcentrationStarLabel, matlset);
}

void RFConcDiffusion1MPM::computeDivergence(const Patch* patch, const MPMMaterial* matl,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;
  int dwi = matl->getDWIndex();

  constNCVariable<double> gConc_Old;
  constNCVariable<double> gConc_OldNoBC;
  constNCVariable<double> gdCdt;

  NCVariable<double> gConcRate;
  NCVariable<double> gConcStar;

  delt_vartype delT;
  old_dw->get(delT, d_sharedState->get_delt_label(), patch->getLevel() );
 
  new_dw->get(gConc_Old,     d_rdlb->gConcentrationLabel,     dwi, patch,gnone,0);
  new_dw->get(gConc_OldNoBC, d_rdlb->gConcentrationNoBCLabel, dwi, patch,gnone,0);
  new_dw->get(gdCdt,         d_rdlb->gdCdtLabel,              dwi, patch,gnone,0);

  new_dw->allocateAndPut(gConcRate,  d_rdlb->gConcentrationRateLabel, dwi,patch);
  new_dw->allocateAndPut(gConcStar, d_rdlb->gConcentrationStarLabel, dwi,patch);

  gConcStar.initialize(0.0);

  MPMBoundCond bc;

  for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        gConcStar[c] = gConc_Old[c] + gdCdt[c] * delT;
  }

  bc.setBoundaryCondition(patch, dwi, "SD-Type", gConcStar, d_Mflag->d_interpolator_type);

  for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
    IntVector c = *iter;
    gConcRate[c] = (gConcStar[c] - gConc_OldNoBC[c]) / delT;
  }
}
