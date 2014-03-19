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
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gdTdtLabel);
  
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
  t->requires(Task::NewDW, d_lb->gdTdtLabel,                           gnone);
  t->requires(Task::NewDW, d_lb->gConcentrationContactDiffusionRateLabel,  gnone);
  t->modifies(d_lb->gConcentrationRateLabel);

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
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol,pMass;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      ParticleVariable<Vector>      pConcentrationGradient;
      constNCVariable<double>       gConcentration,gMass;
      NCVariable<double>            gdTdt;

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
      new_dw->allocateAndPut(gdTdt, d_lb->gdTdtLabel,    dwi, patch);
      new_dw->allocateTemporary(pConcentrationGradient, pset);
  
      gdTdt.initialize(0.);

      // Compute the concentration gradient at each particle and project
      // the particle plastic work temperature rate to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        pConcentrationGradient[idx] = Vector(0.0,0.0,0.0);
        for (int k = 0; k < d_flag->d_8or27; k++){
          for (int j = 0; j<3; j++) {
            pConcentrationGradient[idx][j] +=
                  gConcentration[ni[k]] * d_S[k][j] * oodx[j];
    
            if (cout_concentration.active()) {
              cout_concentration << "   node = " << ni[k]
                                 << " gConcentration = " << gConcentration[ni[k]]
                                 << " idx = " << idx
                                 << " pTempGrad = " << pConcentrationGradient[idx][j]
                                 << endl;
            }
          }
          // Project the mass weighted particle plastic work temperature
          // rate to the grid
        } // Loop over local nodes
      } // Loop over particles

      // Compute rate of temperature change at the grid due to conduction
      // and plastic work
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector dT_dx = pConcentrationGradient[idx];
        double Tdot_cond = 0.0;
        IntVector node(0,0,0);

        for (int k = 0; k < d_flag->d_8or27; k++){
          node = ni[k];
          if(patch->containsNode(node)){
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
           Tdot_cond = Dot(div, dT_dx)*(alpha/gMass[node]);
           gdTdt[node] -= Tdot_cond;

           if (cout_concentration.active()) {
              cout_concentration << "   node = " << node << " div = " << div
                        		 << " dT_dx = " << dT_dx << " alpha = " << alpha*Cv
                        		 << " Tdot_cond = " << Tdot_cond*Cv*gMass[node]
                        		 << " gdTdt = " << gdTdt[node]
                        		 << endl;
           } // cout_concentration
          } // if patch contains node
        } // Loop over local nodes

      } // Loop over particles 
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
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // This task only exists to compute the diagnostic gHeatFluxLabel
    // which is not used in any of the subsequent calculations

    if (cout_doing.active())
      cout_doing <<"Doing computeNodalHeatFlux on patch " << patch->getID()<<"\t\t MPM"<< endl;
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
      // temperature gradient
      NCVariable<Vector> gpdTdx;
      ParticleVariable<Vector> pdTdx;
      new_dw->allocateTemporary(gpdTdx, patch, gnone, 0);
      new_dw->allocateTemporary(pdTdx, pset);
      
      gpdTdx.initialize(Vector(0.,0.,0.));

      // Compute the concentration gradient at each particle
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pdTdx[idx] = Vector(0,0,0);
        
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        for (int k = 0; k < d_flag->d_8or27; k++){
          for (int j = 0; j<3; j++) {
            pdTdx[idx][j] += gConcentration[ni[k]] * d_S[k][j] * oodx[j];
          } 
        }
      }  // particles
      
      // project the mass weighted particle temperature gradient to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],deformationGradient[idx]);
                                                            
        Vector pdTdx_massWt = pdTdx[idx] * pMass[idx];
        
        for (int k = 0; k < d_flag->d_8or27; k++){
          if(patch->containsNode(ni[k])){
            gpdTdx[ni[k]] +=  (pdTdx_massWt*S[k]);        
          } 
        }
      }  // particles

      // compute the nodal concentration gradient by dividing
      // gpdTdx by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        gConcentrationFlux[n] = -kappa * gpdTdx[n]/gMass[n];
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
      constNCVariable<double> mass,externalConcentrationRate,gvolume;
      constNCVariable<double> concentrationContactDiffusionRate,gdTdt;
            
      new_dw->get(mass,    d_lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, d_lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalConcentrationRate, d_lb->gExternalDiffusionRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->get(gdTdt,   d_lb->gdTdtLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(concentrationContactDiffusionRate,
                  d_lb->gConcentrationContactDiffusionRateLabel,
                                                  dwi, patch, Ghost::None, 0);

      // Create variables for the results
      NCVariable<double> tempRate, GtempRate;
      new_dw->getModifiable(tempRate, d_lb->gConcentrationRateLabel,dwi,patch);

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        tempRate[c] = gdTdt[c]*((mass[c]-1.e-200)/mass[c]) +
           (externalConcentrationRate[c])/(mass[c]*Cv)+concentrationContactDiffusionRate[c];
      } // End of loop over iter
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
      
      MPMBoundCond bc;

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        concStar[c] = conc_old[c] + conc_rate[c] * delT;
      }
      // Apply grid boundary conditions to the temperature 
      bc.setBoundaryCondition(  patch,dwi,"Concentration",concStar,interp_type);


      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        conc_rate[c] = (concStar[c] - conc_oldNoBC[c]) / delT;
      }
    } // matls
  } // patches
}
