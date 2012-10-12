/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/HeatConduction/HeatConduction.h>
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

using namespace Uintah;

#define EROSION
#undef EROSION

static DebugStream cout_doing("HeatConduction", false);
static DebugStream cout_heat("MPMHeat", false);

HeatConduction::HeatConduction(SimulationStateP& sS,MPMLabel* labels, 
                               MPMFlags* flags)
{
  d_lb = labels;
  d_flag = flags;
  d_sharedState = sS;

  if(d_flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(d_flag->d_8or27==27 || d_flag->d_8or27==64){
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
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pVolumeLabel,                    gan, NGP);
  t->requires(Task::OldDW, d_lb->pDeformationMeasureLabel,        gan, NGP);
  t->requires(Task::NewDW, d_lb->gTemperatureLabel,               gan, 2*NGN);
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gdTdtLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->pgCodeLabel,                   gan, NGP);
    t->requires(Task::NewDW, d_lb->GTemperatureLabel,             gac, 2*NGN);
    t->requires(Task::NewDW, d_lb->GMassLabel,                    gnone);
    t->computes(d_lb->GdTdtLabel);
  }
  
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

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->GMassLabel,                         gnone);
    t->requires(Task::NewDW, d_lb->GVolumeLabel,                       gnone);
    t->requires(Task::NewDW, d_lb->GExternalHeatRateLabel,             gnone);
    t->requires(Task::NewDW, d_lb->GdTdtLabel,                         gnone);
    t->requires(Task::NewDW, d_lb->GThermalContactTemperatureRateLabel,gnone);
    t->computes(d_lb->GTemperatureRateLabel);
  }

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

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, d_lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, d_lb->gTemperatureNoBCLabel, Ghost::None);
  t->modifies(             d_lb->gTemperatureRateLabel, mss);
  t->computes(d_lb->gTemperatureStarLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->GTemperatureLabel,     Ghost::None);
    t->requires(Task::NewDW, d_lb->GTemperatureNoBCLabel, Ghost::None);
    t->modifies(             d_lb->GTemperatureRateLabel, mss);
    t->computes(d_lb->GTemperatureStarLabel);
  }
                     
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

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

      if (cout_heat.active())
        cout_heat << "  Material = " << m << endl;

      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
      double Cv = mpm_matl->getSpecificHeat();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol,pMass;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> deformationGradient;
      ParticleVariable<Vector>      pTemperatureGradient;
      constNCVariable<double>       gTemperature,gMass;
      NCVariable<double>            gdTdt;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      old_dw->get(px,           d_lb->pXLabel,                         pset);
      old_dw->get(pvol,         d_lb->pVolumeLabel,                    pset);
      old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
      old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
      old_dw->get(deformationGradient, d_lb->pDeformationMeasureLabel, pset);
      new_dw->get(gTemperature, d_lb->gTemperatureLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->allocateAndPut(gdTdt, d_lb->gdTdtLabel,    dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      gdTdt.initialize(0.);

      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      constNCVariable<double> GTemperature;
      constNCVariable<double> GMass;
      NCVariable<double> GdTdt;
      if(d_flag->d_fracture) { 
        new_dw->get(pgCode,       d_lb->pgCodeLabel, pset);
        new_dw->get(GTemperature, d_lb->GTemperatureLabel, dwi,patch,gac,2*NGN);
        new_dw->get(GMass,        d_lb->GMassLabel,        dwi,patch,gnone, 0);
        new_dw->allocateAndPut(GdTdt, d_lb->GdTdtLabel,    dwi,patch);     
        GdTdt.initialize(0.);
      }

      // Compute the temperature gradient at each particle and project
      // the particle plastic work temperature rate to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        for (int k = 0; k < d_flag->d_8or27; k++){
          for (int j = 0; j<3; j++) {
            pTemperatureGradient[idx][j] += 
                  gTemperature[ni[k]] * d_S[k][j] * oodx[j];
    
            if (cout_heat.active()) {
              cout_heat << "   node = " << ni[k]
                        << " gTemp = " << gTemperature[ni[k]]
                        << " idx = " << idx
                        << " pTempGrad = " << pTemperatureGradient[idx][j]
                        << endl;
            }
          }
          // Project the mass weighted particle plastic work temperature
          // rate to the grid
        } // Loop over local nodes
      } // Loop over particles

      if(d_flag->d_fracture) { // for FractureMPM
      // Compute the temperature gradient at each particle and project
      // the particle plastic work temperature rate to the grid
        for (ParticleSubset::iterator iter = pset->begin();
             iter != pset->end(); iter++){
          particleIndex idx = *iter;

          // Get the node indices that surround the cell
          interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

          pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
          for (int k = 0; k < d_flag->d_8or27; k++){
            for (int j = 0; j<3; j++) {
              if(pgCode[idx][k]==1) { // above crack
                pTemperatureGradient[idx][j] +=
                    gTemperature[ni[k]] * d_S[k][j] * oodx[j];
              }
              else if(pgCode[idx][k]==2) { // below crack
                pTemperatureGradient[idx][j] +=
                    GTemperature[ni[k]] * d_S[k][j] * oodx[j];
              }
            }
          } // Loop over local nodes
        } // Loop over particles
      }  // if fracture

      // Compute rate of temperature change at the grid due to conduction
      // and plastic work
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector dT_dx = pTemperatureGradient[idx];
        double Tdot_cond = 0.0;
        IntVector node(0,0,0);

        for (int k = 0; k < d_flag->d_8or27; k++){
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

        if(d_flag->d_fracture) { // for FractureMPM
          for (int k = 0; k < d_flag->d_8or27; k++){
            node = ni[k];
            if(patch->containsNode(node)){
              Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],
                                            d_S[k].z()*oodx[2]);
              if(pgCode[idx][k]==1) { // above crack    
                Tdot_cond = Dot(div, dT_dx)*(alpha/gMass[node]);
                gdTdt[node] -= Tdot_cond;
              }
              else if(pgCode[idx][k]==2) { // below crack
                Tdot_cond = Dot(div, dT_dx)*(alpha/GMass[node]);
                GdTdt[node] -= Tdot_cond;
              }
            } // if patch contains node
          } // Loop over local nodes
        }

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
    
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

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
        
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],deformationGradient[idx]);

        for (int k = 0; k < d_flag->d_8or27; k++){
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
        interpolator->findCellAndWeights(px[idx],ni,S,psize[idx],deformationGradient[idx]);
                                                            
        Vector pdTdx_massWt = pdTdx[idx] * pMass[idx];
        
        for (int k = 0; k < d_flag->d_8or27; k++){
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
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
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

      // for FractureMPM
      constNCVariable<double> Gmass,GexternalHeatRate,Gvolume;
      constNCVariable<double> GthermalContactTemperatureRate,GdTdt;
      if(d_flag->d_fracture) {
        new_dw->get(Gmass,   d_lb->GMassLabel,      dwi, patch, Ghost::None, 0);
        new_dw->get(Gvolume, d_lb->GVolumeLabel,    dwi, patch, Ghost::None, 0);
        new_dw->get(GexternalHeatRate, d_lb->GExternalHeatRateLabel,
                    dwi, patch, Ghost::None, 0);
        new_dw->get(GdTdt,   d_lb->GdTdtLabel,      dwi, patch, Ghost::None, 0);
        new_dw->get(GthermalContactTemperatureRate,
                    d_lb->GThermalContactTemperatureRateLabel,
                    dwi, patch, Ghost::None, 0);      
      }

      // Create variables for the results
      NCVariable<double> tempRate, GtempRate;
      new_dw->getModifiable(tempRate, d_lb->gTemperatureRateLabel,dwi,patch);

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        tempRate[c] = gdTdt[c]*((mass[c]-1.e-200)/mass[c]) +
           (externalHeatRate[c])/(mass[c]*Cv)+thermalContactTemperatureRate[c];
      } // End of loop over iter

      if(d_flag->d_fracture) { // for FractureMPM
        new_dw->allocateAndPut(GtempRate,d_lb->GTemperatureRateLabel,dwi,patch);
        GtempRate.initialize(0.0);
        for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
          GtempRate[c]=GdTdt[c]*((Gmass[c]-1.e-200)/Gmass[c]) +
           (GexternalHeatRate[c])/
                               (Gmass[c]*Cv)+GthermalContactTemperatureRate[c];
        } // End of loop over iter
      }  
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
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temp_old,temp_oldNoBC;
      NCVariable<double> temp_rate,tempStar;
      delt_vartype delT;
      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
 
      new_dw->get(temp_old,    d_lb->gTemperatureLabel,     dwi,patch,gnone,0);
      new_dw->get(temp_oldNoBC,d_lb->gTemperatureNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(temp_rate, d_lb->gTemperatureRateLabel, dwi,patch);
      new_dw->allocateAndPut(tempStar, d_lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);

      // for FractureMPM
      constNCVariable<double> Gtemp_old,Gtemp_oldNoBC;
      NCVariable<double> Gtemp_rate,GtempStar;
      if(d_flag->d_fracture) {
       new_dw->get(Gtemp_old,    d_lb->GTemperatureLabel,    dwi,patch,gnone,0);
       new_dw->get(Gtemp_oldNoBC,d_lb->GTemperatureNoBCLabel,dwi,patch,gnone,0);
       new_dw->getModifiable(Gtemp_rate,d_lb->GTemperatureRateLabel,dwi,patch);
       new_dw->allocateAndPut(GtempStar,d_lb->GTemperatureStarLabel,dwi,patch);
       GtempStar.initialize(0.0);
      }
      
      MPMBoundCond bc;

      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        tempStar[c] = temp_old[c] + temp_rate[c] * delT;
      }
      // Apply grid boundary conditions to the temperature 
      bc.setBoundaryCondition(  patch,dwi,"Temperature",tempStar,interp_type);

      if(d_flag->d_fracture) { // for FractureMPM
        for(NodeIterator iter=patch->getExtraNodeIterator();
                         !iter.done();iter++){
          IntVector c = *iter;
          GtempStar[c]=Gtemp_old[c] +Gtemp_rate[c] * delT;
        }
        // Apply grid boundary conditions to the temperature 
        bc.setBoundaryCondition(patch,dwi,"Temperature",GtempStar,interp_type);
      }

      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
      }

      if(d_flag->d_fracture) { // for FractureMPM
        for(NodeIterator iter=patch->getExtraNodeIterator();
                         !iter.done();iter++){
          IntVector c = *iter;
          Gtemp_rate[c]= (GtempStar[c]-Gtemp_oldNoBC[c]) / delT;
        }
      } // fracture
    } // matls
  } // patches
}
