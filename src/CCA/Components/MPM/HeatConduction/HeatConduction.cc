/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Grid/Task.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Util/DebugStream.h>

using namespace std;
using namespace Uintah;

#define EROSION
#undef EROSION

static DebugStream cout_doing("HeatConduction", false);
static DebugStream cout_heat("MPMHeat", false);

HeatConduction::HeatConduction(MaterialManagerP& sS,MPMLabel* labels, 
                               MPMFlags* flags)
{
  d_lb = labels;
  d_flag = flags;
  d_materialManager = sS;

  if(d_flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else {
    NGP=2;
    NGN=2;
  }
}

HeatConduction::~HeatConduction()
{
}

void HeatConduction::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{  
  Task* t = scinew Task("MPM::computeInternalHeatRate",
                        this, &HeatConduction::computeInternalHeatRate);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pVolumeLabel,                    gan, NGP);
  t->requires(Task::OldDW, d_lb->pTemperatureGradientLabel,       gan, NGP);
  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gan, NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gdTdtLabel);

  sched->addTask(t, patches, matls);
}
//__________________________________
//
void HeatConduction::scheduleComputeNodalHeatFlux(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{ 
  if(d_flag->d_computeNodalHeatFlux == false)
    return;

  // This task only exists to compute the diagnostic gHeatFluxLabel
  // which is not used in any of the subsequent calculations
    
  Task* t = scinew Task("MPM::computeNodalHeatFlux",
                        this, &HeatConduction::computeNodalHeatFlux);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,             gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,          gan, NGP);
  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel, gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,          gan, NGP);
  t->requires(Task::NewDW, d_lb->gTemperatureLabel,   gac, 2*NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,          gnone);
  t->computes(d_lb->gHeatFluxLabel);
  
  sched->addTask(t, patches, matls);
}

void HeatConduction::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  /* solveHeatEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  Task* t = scinew Task("MPM::solveHeatEquations",
                        this, &HeatConduction::solveHeatEquations);

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, d_lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, d_lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, d_lb->gExternalHeatRateLabel,               gnone);
  t->requires(Task::NewDW, d_lb->gdTdtLabel,                           gnone);
  t->requires(Task::NewDW, d_lb->gThermalContactTemperatureRateLabel,  gnone);
  t->modifies(d_lb->gTemperatureRateLabel);

  sched->addTask(t, patches, matls);
}

void HeatConduction::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* matls)
{
  /* integrateTemperatureRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  Task* t = scinew Task("MPM::integrateTemperatureRate",
                        this, &HeatConduction::integrateTemperatureRate);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_lb->delTLabel );

  t->requires(Task::NewDW, d_lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, d_lb->gTemperatureNoBCLabel, Ghost::None);
  t->modifies(             d_lb->gTemperatureRateLabel, mss);
  t->computes(d_lb->gTemperatureStarLabel);

  sched->addTask(t, patches, matls);
}

void HeatConduction::computeInternalHeatRate(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeInternalHeatRate on patch " << patch->getID()<<"\t\t MPM"<< endl;
    if (cout_heat.active())
      cout_heat << " Patch = " << patch->getID() << endl;

    ParticleInterpolator* interpolator = d_flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gnone = Ghost::None;
    for(unsigned int m = 0; m < d_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );

      if (cout_heat.active())
        cout_heat << "  Material = " << m << endl;

      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
      double Cv = mpm_matl->getSpecificHeat();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol;
      constParticleVariable<Matrix3> psize, deformationGradient;
      constParticleVariable<Vector>  pTempGrad;
      constNCVariable<double>       gTemperature,gMass;
      NCVariable<double>            gdTdt;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      old_dw->get(px,           d_lb->pXLabel,                         pset);
      old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
      old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
      old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
      old_dw->get(pTempGrad,    d_lb->pTemperatureGradientLabel,       pset);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->allocateAndPut(gdTdt, d_lb->gdTdtLabel,    dwi, patch);
  
      gdTdt.initialize(0.);

      // Compute rate of temperature change at the grid due to conduction
      // and plastic work
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,
                                          psize[idx], deformationGradient[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector dT_dx = pTempGrad[idx];
        double Tdot_cond = 0.0;
        IntVector node(0,0,0);

        // TODO:  get this division by mass OUT OF HERE!  This is
        // creating a lot more divisions than are necessary
        for (int k = 0; k < NN; k++){
          node = ni[k];
          if(patch->containsNode(node)){
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
           Tdot_cond = Dot(div, dT_dx)*(alpha/gMass[node]);
           gdTdt[node] -= Tdot_cond;

           if (cout_heat.active()) {
              cout_heat << "   node = " << node << " div = " << div 
                        << " dT_dx = " << dT_dx << " alpha = " << alpha*Cv 
                        << " Tdot_cond = " << Tdot_cond*Cv*gMass[node]
                        << " gdTdt = " << gdTdt[node] 
                        << endl;
           } // cout_heat
          } // if patch contains node
        } // Loop over local nodes
      } // Loop over particles 
    }  // End of loop over materials
    delete interpolator;
  }  // End of loop over patches
}
//______________________________________________________________________
//
void HeatConduction::computeNodalHeatFlux(const ProcessorGroup*,
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
    if (cout_heat.active())
      cout_heat << " Patch = " << patch->getID() << endl;
      
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
    
    for(unsigned int m = 0; m < d_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );

      if (cout_heat.active())
        cout_heat << "  Material = " << m << endl;

      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
            
      NCVariable<Vector> gHeatFlux;
      constNCVariable<double> gTemperature, gMass;
      constParticleVariable<Point>  px;
      constParticleVariable<double> pMass;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      new_dw->get(gTemperature, d_lb->gTemperatureLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      old_dw->get(px,           d_lb->pXLabel,           pset);
      old_dw->get(pMass,        d_lb->pMassLabel,        pset);
      old_dw->get(psize,        d_lb->pSizeLabel,        pset);
      old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
      
      new_dw->allocateAndPut(gHeatFlux, d_lb->gHeatFluxLabel,  dwi, patch);  
      gHeatFlux.initialize(Vector(0.0));

      //__________________________________
      // Create a temporary variables for the mass weighted nodal
      // temperature gradient
      NCVariable<Vector> gpdTdx;
      ParticleVariable<Vector> pdTdx;
      new_dw->allocateTemporary(gpdTdx, patch, gnone, 0);
      new_dw->allocateTemporary(pdTdx, pset);
      
      gpdTdx.initialize(Vector(0.,0.,0.));

      // Compute the temperature gradient at each particle 
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;
        pdTdx[idx] = Vector(0,0,0);
        
        int NN = interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,
                                           psize[idx],deformationGradient[idx]);

        for (int k = 0; k < NN; k++){
          for (int j = 0; j<3; j++) {
            pdTdx[idx][j] += gTemperature[ni[k]] * d_S[k][j] * oodx[j];
          } 
        }
      }  // particles
      
      // project the mass weighted particle temperature gradient to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],
                                         deformationGradient[idx]);
                                                            
        Vector pdTdx_massWt = pdTdx[idx] * pMass[idx];
        
        for (int k = 0; k < NN; k++){
          if(patch->containsNode(ni[k])){
            gpdTdx[ni[k]] +=  (pdTdx_massWt*S[k]);        
          } 
        }
      }  // particles

      // compute the nodal temperature gradient by dividing
      // gpdTdx by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector n = *iter;
        gHeatFlux[n] = -kappa * gpdTdx[n]/gMass[n];
      }
    }  // End of loop over materials
    delete interpolator;
  }  // End of loop over patches
}


void HeatConduction::solveHeatEquations(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* /*old_dw*/,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing solveHeatEquations on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


    string interp_type = d_flag->d_interpolator_type;
    for(unsigned int m = 0; m < d_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      double Cv = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactTemperatureRate,gdTdt;
            
      new_dw->get(mass,    d_lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, d_lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, d_lb->gExternalHeatRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->get(gdTdt,   d_lb->gdTdtLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(thermalContactTemperatureRate,
                  d_lb->gThermalContactTemperatureRateLabel,
                                                  dwi, patch, Ghost::None, 0);

      // Create variables for the results
      NCVariable<double> tempRate, GtempRate;
      new_dw->getModifiable(tempRate, d_lb->gTemperatureRateLabel,dwi,patch);

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        tempRate[c] = gdTdt[c]*((mass[c]-1.e-200)/mass[c]) +
           (externalHeatRate[c])/(mass[c]*Cv)+thermalContactTemperatureRate[c];
      } // End of loop over iter
    }
  }
}


void HeatConduction::integrateTemperatureRate(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset*,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing integrateTemperatureRate on patch " << patch->getID()<< "\t\t MPM"<< endl;


    Ghost::GhostType  gnone = Ghost::None;
    string interp_type = d_flag->d_interpolator_type;
    for(unsigned int m = 0; m < d_materialManager->getNumMatls( "MPM" ); m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temp_old,temp_oldNoBC;
      NCVariable<double> temp_rate,tempStar;
      delt_vartype delT;
      old_dw->get(delT, d_lb->delTLabel, getLevel(patches) );
 
      new_dw->get(temp_old,    d_lb->gTemperatureLabel,     dwi,patch,gnone,0);
      new_dw->get(temp_oldNoBC,d_lb->gTemperatureNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(temp_rate, d_lb->gTemperatureRateLabel, dwi,patch);
      new_dw->allocateAndPut(tempStar, d_lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);

      MPMBoundCond bc;

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        tempStar[c] = temp_old[c] + temp_rate[c] * delT;
      }
      // Apply grid boundary conditions to the temperature 
      bc.setBoundaryCondition(  patch,dwi,"Temperature",tempStar,interp_type);

      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
      }
    } // matls
  } // patches
}
