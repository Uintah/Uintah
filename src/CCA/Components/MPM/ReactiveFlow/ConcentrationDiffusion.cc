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

#include <CCA/Components/MPM/ReactiveFlow/ConcentrationDiffusion.h>
#include <CCA/Components/MPM/ReactiveFlow/PotentialField.h>
#include <Core/Math/Short27.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <CCA/Components/MPM/MPMBoundCond.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Util/DebugStream.h>

using namespace std;
using namespace Uintah;

#define EROSION
#undef EROSION

#define StartCompInDiff
#undef StartCompInDiff
#define CompIntDiffPCon //pConcentrationGradient
#undef CompIntDiffPCon
#define EndCompInDiff
#undef EndCompInDiff
#define StartSolveDiffEq
#undef StartSolveDiffEq
#define BeforeConRateSolveDiffEq
#undef BeforeConRateSolveDiffEq
#define ExtDiffRateSolveDiffEq
#undef ExtDiffRateSolveDiffEq
#define AfterConRateSolveDiffEq
#undef AfterConRateSolveDiffEq
#define IntDiffRateGCL
#undef IntDiffRateGCL
#define IntDiffRateNOBC
#undef IntDiffRateNOBC
#define BeforeIntDiffRateConStar
#undef BeforeIntDiffRateConStar
#define AfterIntDiffRateConStar
#undef AfterIntDiffRateConStar
#define IntDiffRateConRate
#undef IntDiffRateConRate

static DebugStream cout_doing("ConcentrationDiffusion", false);
static DebugStream cout_concentration("MPMConcentration", false);

ConcentrationDiffusion::ConcentrationDiffusion(SimulationStateP& sS,MPMLabel* labels,
                               MPMFlags* flags)
{
  d_lb = labels;
  d_flag = flags;
  d_sharedState = sS;

  if(d_flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else {
    NGP=2;
    NGN=2;
  }
}

ConcentrationDiffusion::~ConcentrationDiffusion()
{
}

void ConcentrationDiffusion::scheduleComputeInternalDiffusionRate(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{  
  Task* t = scinew Task("MPM::computeInternalDiffusionRate",
                        this, &ConcentrationDiffusion::computeInternalDiffusionRate);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pVolumeLabel,                    gan, NGP);
  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gan, NGP);
  t->requires(Task::NewDW, d_lb->gConcentrationLabel,             gan, 2*NGN);
  t->requires(Task::NewDW, d_lb->gMeanStressLabel,                gan, 2*NGN);
  t->requires(Task::NewDW, d_lb->gDetDeformationGradLabel,        gan, 2*NGN);
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gdCdtLabel);
  
  if(d_flag->d_fracture) { // for FractureMPM
      t->requires(Task::NewDW, d_lb->pgCodeLabel,                   gan, NGP);
      t->requires(Task::NewDW, d_lb->GConcentrationLabel,             gac, 2*NGN);
      t->requires(Task::NewDW, d_lb->GMassLabel,                    gnone);
      t->computes(d_lb->GdCdtLabel);
    }

  sched->addTask(t, patches, matls);
}
//__________________________________
//
void ConcentrationDiffusion::scheduleComputeNodalConcentrationFlux(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{ 
  if(d_flag->d_computeNodalConcentrationFlux == false)
    return;

  // This task only exists to compute the diagnostic gHeatFluxLabel
  // which is not used in any of the subsequent calculations
    
  Task* t = scinew Task("MPM::computeNodalConcentrationFlux",
                        this, &ConcentrationDiffusion::computeNodalConcentrationFlux);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,             gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,          gan, NGP);
  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,          gan, NGP);
  t->requires(Task::NewDW, d_lb->gConcentrationLabel,   gac, 2*NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,          gnone);
  t->computes(d_lb->gConcentrationFluxLabel);
  
  sched->addTask(t, patches, matls);
}

void ConcentrationDiffusion::scheduleSolveDiffusionEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  /* solveDiffusionEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  Task* t = scinew Task("MPM::solveDiffusionEquations",
                        this, &ConcentrationDiffusion::solveDiffusionEquations);

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, d_lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, d_lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, d_lb->gExternalDiffusionRateLabel,          gnone);
  t->requires(Task::NewDW, d_lb->gdCdtLabel,                           gnone);
  //t->requires(Task::NewDW, d_lb->gConcentrationContactDiffusionRateLabel,  gnone);
  t->modifies(d_lb->gConcentrationRateLabel);

  if(d_flag->d_fracture) { // for FractureMPM
      t->requires(Task::NewDW, d_lb->GMassLabel,                         gnone);
      t->requires(Task::NewDW, d_lb->GVolumeLabel,                       gnone);
      t->requires(Task::NewDW, d_lb->GExternalDiffusionRateLabel,        gnone);
      t->requires(Task::NewDW, d_lb->GdTdtLabel,                         gnone);
      //t->requires(Task::NewDW, d_lb->GConcentrationContactDiffusionRateLabel,gnone);
      t->computes(d_lb->GTemperatureRateLabel);
    }
  sched->addTask(t, patches, matls);
}

void ConcentrationDiffusion::scheduleIntegrateDiffusionRate(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* matls)
{
  /* integrateDiffusionRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  Task* t = scinew Task("MPM::integrateDiffusionRate",
                        this, &ConcentrationDiffusion::integrateDiffusionRate);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, d_lb->gConcentrationLabel,     Ghost::None);
  t->requires(Task::NewDW, d_lb->gConcentrationNoBCLabel, Ghost::None);
  t->modifies(             d_lb->gConcentrationRateLabel, mss);
  t->computes(d_lb->gConcentrationStarLabel);

  if(d_flag->d_fracture) { // for FractureMPM
      t->requires(Task::NewDW, d_lb->GConcentrationLabel,     Ghost::None);
      t->requires(Task::NewDW, d_lb->GConcentrationNoBCLabel, Ghost::None);
      t->modifies(             d_lb->GConcentrationRateLabel, mss);
      t->computes(d_lb->GTemperatureStarLabel);
    }
  sched->addTask(t, patches, matls);
}

void ConcentrationDiffusion::computeInternalDiffusionRate(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeInternalDiffusionRate on patch " << patch->getID()<<"\t\t MPM"<< endl;
    if (cout_concentration.active())
      cout_concentration << " Patch = " << patch->getID() << endl;

    ParticleInterpolator* interpolator = d_flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

      if (cout_concentration.active())
        cout_concentration << "  Material = " << m << endl;

      // compute diffusion variables!!!!!!!
      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
      double Cv = mpm_matl->getSpecificHeat();
      double diffusivity = mpm_matl->getDiffusivity();
      double bolt = mpm_matl->getBoltzmann();
      double satM = mpm_matl->getSaturationMax();
      //cout << "diffusivity: " << diffusivity << endl;
      //cout << "h: " << dx.x() << endl;
      double potential = 0;
      double omega = mpm_matl->getOmega();

      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol,pMass;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      ParticleVariable<Vector>      pConcentrationGradient;
      constNCVariable<double>       gConcentration,gMass;
      NCVariable<double>            gdCdt;
      constNCVariable<double>       detF;
      constNCVariable<double>       mStress;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      old_dw->get(px,           d_lb->pXLabel,                         pset);
      old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
      old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
      old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
      old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
      new_dw->get(gConcentration, d_lb->gConcentrationLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->get(detF,         d_lb->gDetDeformationGradLabel, dwi, patch, gnone, 0);
      new_dw->get(mStress,   d_lb->gMeanStressLabel, dwi, patch, gnone, 0);
      new_dw->allocateAndPut(gdCdt, d_lb->gdCdtLabel,    dwi, patch);
      new_dw->allocateTemporary(pConcentrationGradient, pset);
  
#ifdef StartCompInDiff
      cout << "Start Compute Internal Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " gConcentration: " << gConcentration[n] << endl;
      }
#endif
      gdCdt.initialize(0.);

      // for FractureMPM
      /*constParticleVariable<Short27> pgCode;
      constNCVariable<double> GConcentration;
      constNCVariable<double> GMass;
      NCVariable<double> GdCdt;
      if(d_flag->d_fracture) {
    	  new_dw->get(pgCode,       d_lb->pgCodeLabel, pset);
          new_dw->get(GConcentration, d_lb->GConcentrationLabel, dwi,patch,gac,2*NGN);
          new_dw->get(GMass,        d_lb->GMassLabel,        dwi,patch,gnone, 0);
          new_dw->allocateAndPut(GdCdt, d_lb->GdCdtLabel,    dwi,patch);
          GdCdt.initialize(0.);
      }
	  */

      // Compute the concentration gradient at each particle and project
      // the particle plastic work temperature rate to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell

        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        pConcentrationGradient[idx] = Vector(0.0,0.0,0.0);
        for (int k = 0; k < d_flag->d_8or27; k++){
        	//cout << ni[k] << " gConc: " << gConcentration[ni[k]] << endl;
          for (int j = 0; j<3; j++) {
        	potential = PotentialField::gaoPotential(gConcentration[ni[k]], satM, bolt,
        										detF[ni[k]], omega, mStress[ni[k]], 0.0);
            //pConcentrationGradient[idx][j] +=
                  //gConcentration[ni[k]] * d_S[k][j] * oodx[j];
        	pConcentrationGradient[idx][j] += potential * d_S[k][j] * oodx[j];
    
            if (cout_concentration.active()) {
              cout_concentration << "   node = " << ni[k]
                                 << " gConcentration = " << gConcentration[ni[k]]
                                 << " idx = " << idx
                                 << " pConcentrationGrad = " << pConcentrationGradient[idx][j]
                                 << endl;
            }
          }
          // Project the mass weighted particle plastic work temperature
          // rate to the grid
        } // Loop over local nodes
        pConcentrationGradient[idx] *= diffusivity;
        //cout << idx << " pConcentrationGradient: " << pConcentrationGradient[idx] << endl;
      } // Loop over particles
#ifdef CompIntDiffPCon
      cout << "List of computed pConcGradients in Compute Internal Diff Rate" << endl;
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
           particleIndex idx = *iter;
           cout << idx << " pConcGrad1: " << pConcentrationGradient[idx] << endl;
      }
#endif
      // Compute rate of temperature change at the grid due to conduction
      // and plastic work
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector dC_dx = pConcentrationGradient[idx]*pMass[idx];

        double Cdot_cond = 0.0;
        IntVector node(0,0,0);

        //cout << idx << " pConcGrad: " << pConcentrationGradient[idx] << endl;

        for (int k = 0; k < d_flag->d_8or27; k++){
          node = ni[k];
          if(patch->containsNode(node)){
        	  //cout << node << " conc_gMass: " << gMass[node] << endl;
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
           Cdot_cond = Dot(div, dC_dx);
           gdCdt[node] -= Cdot_cond;

           if (cout_concentration.active()) {
              cout_concentration << "   node = " << node << " div = " << div
                        		 << " dC_dx = " << dC_dx << " alpha = " << alpha*Cv
                        		 << " Cdot_cond = " << Cdot_cond*Cv*gMass[node]
                        		 << " gdCdt = " << gdCdt[node]
                        		 << endl;
           } // cout_concentration
          } // if patch contains node
        } // Loop over local nodes
      } // Loop over particles
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          gdCdt[n] /= gMass[n];
      }
#ifdef EndCompInDiff
      cout << "End Compute Internal Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " gdCdt: " << gdCdt[n] << endl;
      }
#endif
    }  // End of loop over materials
    delete interpolator;
  }  // End of loop over patches
}
//______________________________________________________________________
//
void ConcentrationDiffusion::computeNodalConcentrationFlux(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
	cout << "entering concflux" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // This task only exists to compute the diagnostic gConcentrationFluxLabel
    // which is not used in any of the subsequent calculations

    if (cout_doing.active())
      cout_doing <<"Doing computeNodalConcentrationFlux on patch " << patch->getID()<<"\t\t MPM"<< endl;
    if (cout_concentration.active())
      cout_concentration << " Patch = " << patch->getID() << endl;
      
    ParticleInterpolator* interpolator = d_flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gnone = Ghost::None;
    
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

      if (cout_concentration.active())
        cout_concentration << "  Material = " << m << endl;

      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
            
      NCVariable<Vector> gConcentrationFlux;
      constNCVariable<double> gConcentration, gMass;
      constParticleVariable<Point>  px;
      constParticleVariable<double> pMass;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      new_dw->get(gConcentration, d_lb->gConcentrationLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      old_dw->get(px,           d_lb->pXLabel,           pset);
      old_dw->get(pMass,        d_lb->pMassLabel,        pset);
      old_dw->get(psize,        d_lb->pSizeLabel,        pset);
      old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
      
      new_dw->allocateAndPut(gConcentrationFlux, d_lb->gConcentrationFluxLabel,  dwi, patch);
      gConcentrationFlux.initialize(Vector(0.0));

      //__________________________________
      // Create a temporary variables for the mass weighted nodal
      // concentration gradient
      NCVariable<Vector> gpdCdx;
      ParticleVariable<Vector> pdCdx;
      new_dw->allocateTemporary(gpdCdx, patch, gnone, 0);
      new_dw->allocateTemporary(pdCdx, pset);
      
      gpdCdx.initialize(Vector(0.,0.,0.));

      // Compute the concentration gradient at each particle
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pdCdx[idx] = Vector(0,0,0);
        
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        for (int k = 0; k < d_flag->d_8or27; k++){
          for (int j = 0; j<3; j++) {
            pdCdx[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
          } 
        }
      }  // particles
      
      // project the mass weighted particle temperature gradient to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],deformationGradient[idx]);
                                                            
        Vector pdCdx_massWt = pdCdx[idx] * pMass[idx];
        
        for (int k = 0; k < d_flag->d_8or27; k++){
          if(patch->containsNode(ni[k])){
            gpdCdx[ni[k]] +=  (pdCdx_massWt*S[k]);
          } 
        }
      }  // particles

      // compute the nodal concentration gradient by dividing
      // gpdTdx by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        gConcentrationFlux[n] = -kappa * gpdCdx[n]/gMass[n];
        cout << n << " gConcentrationFlux: " << gConcentrationFlux[n] << endl;
      }
    }  // End of loop over materials
    delete interpolator;
  }  // End of loop over patches
}


void ConcentrationDiffusion::solveDiffusionEquations(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* /*old_dw*/,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing solveDiffusionEquations on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


    string interp_type = d_flag->d_interpolator_type;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double Cv = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalDiffusionRate,gvolume;
      constNCVariable<double> concentrationContactDiffusionRate,gdCdt;
            
      new_dw->get(mass,    d_lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, d_lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalDiffusionRate, d_lb->gExternalDiffusionRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->get(gdCdt,   d_lb->gdCdtLabel,      dwi, patch, Ghost::None, 0);
      //new_dw->get(concentrationContactDiffusionRate,d_lb->gConcentrationContactDiffusionRateLabel,
                                                  //dwi, patch, Ghost::None, 0);

#ifdef StartSolveDiffEq
      cout << "Start Solve Diffusion Equation" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " gdCdt: " << gdCdt[n] << endl;
      }
#endif
      // Create variables for the results
      NCVariable<double> concentrationRate, GconcentrationRate;
      new_dw->getModifiable(concentrationRate, d_lb->gConcentrationRateLabel,dwi,patch);

#ifdef BeforeConRateSolveDiffEq
      cout << "Before concRate Solve Diffusion Equation" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " concentrationRate: " << concentrationRate[n] << endl;
      }
#endif

#ifdef ExtConRateSolveDiffEq
      cout << "ExtDiffusionRate Solve Diffusion Equation" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " externalDiffusionRate: " << externalDiffusionRate[n] << endl;
      }
#endif

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        //cout << c << " conmass " << mass[c] << endl;
        //cout << c << " gdCdt: " << gdCdt[c] << endl;
        //cout << c << " extDiffRate: " << externalDiffusionRate[c] << endl;
        concentrationRate[c] = gdCdt[c]*((mass[c]-1.e-200)/mass[c]); //+
        //cout << "masscalc: " << (mass[c]-1.e-200)/mass[c] << endl;
           //(externalDiffusionRate[c])/(mass[c]*Cv);
      } // End of loop over iter
#ifdef AfterConRateSolveDiffEq
      cout << "After concRate Solve Diffusion Equation" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " concentrationRate: " << concentrationRate[n] << endl;
      }
#endif
    }
  }
}


void ConcentrationDiffusion::integrateDiffusionRate(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset*,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing integrateDiffusionRate on patch " << patch->getID()<< "\t\t MPM"<< endl;


    Ghost::GhostType  gnone = Ghost::None;
    string interp_type = d_flag->d_interpolator_type;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> conc_old,conc_oldNoBC;
      NCVariable<double> conc_rate,concStar;
      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
 
      new_dw->get(conc_old,    d_lb->gConcentrationLabel,     dwi,patch,gnone,0);
      new_dw->get(conc_oldNoBC,d_lb->gConcentrationNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(conc_rate, d_lb->gConcentrationRateLabel, dwi,patch);
      new_dw->allocateAndPut(concStar, d_lb->gConcentrationStarLabel, dwi,patch);
      concStar.initialize(0.0);
#ifdef IntDiffRateGCL
      cout << "Integrate Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " conc_old: " << conc_old[n] << endl;
      }
#endif

#ifdef IntDiffRateNOBC
      cout << "Integrate Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " conc_oldNoBC: " << conc_oldNoBC[n] << endl;
      }
#endif



      MPMBoundCond bc;

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        //cout << "concentration: " << c << conc_old[c] << endl;
        concStar[c] = conc_old[c] + conc_rate[c] * delT;
        //cout << c << " concrate: " << conc_rate[c] << endl;
        //cout << c << " conStar: " << concStar[c] << endl;
      }

#ifdef BeforeIntDiffRateConStar
      cout << "Integrate Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " Before concStar: " << concStar[n] << endl;
      }
#endif
      // Apply grid boundary conditions to the concentration
      bc.setBoundaryCondition(  patch,dwi,"Concentration",concStar,interp_type);
#ifdef AfterIntDiffRateConStar
      cout << "Integrate Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " After concStar: " << concStar[n] << endl;
      }
#endif
      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        //conc_rate[c] = (concStar[c] - conc_oldNoBC[c]) / delT;
        conc_rate[c] = concStar[c];
      }
#ifdef IntDiffRateConRate
      cout << "Integrate Diffusion Rate" << endl;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
          IntVector n = *iter;
          cout << n << " conc_rate: " << conc_rate[n] << endl;
      }
#endif
    } // matls
  } // patches
}
