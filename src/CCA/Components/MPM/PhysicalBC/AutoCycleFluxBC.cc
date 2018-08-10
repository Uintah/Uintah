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

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>
#include <CCA/Components/MPM/PhysicalBC/AutoCycleFluxBC.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <string>
#include <iostream>

using namespace Uintah;

static DebugStream cout_doing("AutoCycleFluxBC", false);

#define USE_FLUX_RESTRICTION

AutoCycleFluxBC::AutoCycleFluxBC(MaterialManagerP& materialManager, MPMFlags* mpm_flags) :
    FluxBCModel(materialManager, mpm_flags)
{
  d_flux_sign = 1.0;
  d_auto_cycle_min = mpm_flags->d_autoCycleMin;
  d_auto_cycle_max = mpm_flags->d_autoCycleMax;
}

AutoCycleFluxBC::~AutoCycleFluxBC()
{

}

void AutoCycleFluxBC::scheduleInitializeScalarFluxBCs(const LevelP& level, SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();

  d_load_curve_index = scinew MaterialSubset();
  d_load_curve_index->add(0);
  d_load_curve_index->addReference();

  int nofSFBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    std::string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "ScalarFlux"){
      d_load_curve_index->add(nofSFBCs++);
    }
  }
  if (nofSFBCs > 0) {
    printSchedule(patches,cout_doing,"AutoCycleFluxBC::countMaterialPointsPerFluxLoadCurve");
    printSchedule(patches,cout_doing,"AutoCycleFluxBC::scheduleInitializeScalarFluxBCs");
    // Create a task that calculates the total number of particles
    // associated with each load curve.
    Task* t = scinew Task("AutoCycleFluxBC::countMaterialPointsPerFluxLoadCurve", this,
                          &AutoCycleFluxBC::countMaterialPointsPerFluxLoadCurve);
    t->requires(Task::NewDW, d_mpm_lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(d_mpm_lb->materialPointsPerLoadCurveLabel, d_load_curve_index, Task::OutOfDomain);
    sched->addTask(t, patches, d_materialManager->allMaterials( "MPM" ));

#if 1
    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("AutoCycleFluxBC::initializeScalarFluxBC", this,
                    &AutoCycleFluxBC::initializeScalarFluxBC);
    t->requires(Task::NewDW, d_mpm_lb->materialPointsPerLoadCurveLabel,
                d_load_curve_index, Task::OutOfDomain, Ghost::None);
    sched->addTask(t, patches, d_materialManager->allMaterials( "MPM" ));
#endif
  }

  if(d_load_curve_index->removeReference())
      delete d_load_curve_index;
}

void AutoCycleFluxBC::initializeScalarFluxBC(const ProcessorGroup*, const PatchSubset* patches,
                                         const MaterialSubset*, DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  double time = 0.0;
  printTask(patches,patches->get(0),cout_doing,"Doing initialize ScalarFluxBC");
  if (cout_doing.active())
    cout_doing << "Current Time (Initialize ScalarFlux BC) = " << time << std::endl;

  // Calculate the scalar flux at each particle
  for(int p=0;p<patches->size();p++){
    int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
    for(int m = 0; m < numMPMMatls; m++){
      int nofSFBCs = 0;
      for(int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
        std::string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
        if (bcs_type == "ScalarFlux") {

          // Get the material points per load curve
          sumlong_vartype numPart = 0;
          new_dw->get(numPart,d_mpm_lb->materialPointsPerLoadCurveLabel,0,nofSFBCs++);

          // Save the material points per load curve in the ScalarFluxBC object
          ScalarFluxBC* pbc = dynamic_cast<ScalarFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
          pbc->numMaterialPoints(numPart);

          if (cout_doing.active()){
            cout_doing << "    Load Curve = "
                       << nofSFBCs << " Num Particles = " << numPart << std::endl;
          }
        }   // if pressure loop
      }    // loop over all Physical BCs
    }     // matl loop
  }      // patch loop
}

void AutoCycleFluxBC::scheduleApplyExternalScalarFlux(SchedulerP& sched, const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Ghost::GhostType gnone = Ghost::None;
  if (!d_mpm_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"AutoCycleFluxBC::scheduleApplyExternalScalarFlux");

  Task* t=scinew Task("AutoCycleFluxBC::applyExternalScalarFlux", this,
                      &AutoCycleFluxBC::applyExternalScalarFlux);

  t->requires(Task::OldDW, d_mpm_lb->simulationTimeLabel);

  t->requires(Task::OldDW, d_mpm_lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, d_mpm_lb->pSizeLabel,              Ghost::None);
  if(d_mpm_flags->d_doScalarDiffusion){
    // JBH -- Fixme -- TODO -- Fold into diffusion sublabel?
    t->requires(Task::OldDW, d_mpm_lb->diffusion->pArea,            Ghost::None);
  }
  t->requires(Task::OldDW, d_mpm_lb->pVolumeLabel,            Ghost::None);
  t->requires(Task::OldDW, d_mpm_lb->pDeformationMeasureLabel,Ghost::None);
  if(d_mpm_flags->d_autoCycleUseMinMax){
    t->requires(Task::OldDW, d_mpm_lb->diffusion->rMaxConcentration,    gnone);
    t->requires(Task::OldDW, d_mpm_lb->diffusion->rMinConcentration,    gnone);
  }else{
    t->requires(Task::OldDW, d_mpm_lb->diffusion->rTotalConcentration,  gnone);
    // JBH -- FIXME -- TODO  Fold into diffusion sublabel?
    t->requires(Task::OldDW, d_mpm_lb->partCountLabel,        Ghost::None);
  }

#if defined USE_FLUX_RESTRICTION
  if(d_mpm_flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, d_mpm_lb->diffusion->pConcentration,       gnone);
  }
#endif

  t->computes(d_mpm_lb->diffusion->pExternalScalarFlux_preReloc);

  if (d_mpm_flags->d_useLoadCurves) {
    t->requires(Task::OldDW, d_mpm_lb->pLoadCurveIDLabel,     Ghost::None);
  }

  sched->addTask(t, patches, matls);
}

void AutoCycleFluxBC::applyExternalScalarFlux(const ProcessorGroup* , const PatchSubset* patches,
                                          const MaterialSubset*, DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  // Get the current simulation time
  // double simTime = d_materialManager->getElapsedSimTime();

  simTime_vartype simTime;
  old_dw->get(simTime, d_mpm_lb->simulationTimeLabel);
  
  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalScalarFlux) = " << simTime << std::endl;

  // Calculate the flux at each particle for each flux bc
  std::vector<double> fluxPerPart;
  std::vector<ScalarFluxBC*> pbcP;
  if (d_mpm_flags->d_useLoadCurves) {
    for (int ii = 0;ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size();ii++){
      std::string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
      if (bcs_type == "ScalarFlux") {

        ScalarFluxBC* pbc =  dynamic_cast<ScalarFluxBC*>(MPMPhysicalBCFactory::mpmPhysicalBCs[ii]);
        pbcP.push_back(pbc);

        // Calculate the force per particle at current time
        //fluxPerPart.push_back(pbc->fluxPerParticle(time));
      }
    }
  }

  sumlong_vartype totalparts;
  sum_vartype totalconc;
  max_vartype maxconc;
  min_vartype minconc;
  double avgconc;

  if(d_mpm_flags->d_autoCycleUseMinMax){
    old_dw->get(maxconc, d_mpm_lb->diffusion->rMaxConcentration);
    old_dw->get(minconc, d_mpm_lb->diffusion->rMinConcentration);
    if(d_flux_sign > 0){
      if(minconc > d_auto_cycle_max && minconc < 4e11){
        d_flux_sign = -1.0;
      }
    }else{
      if(maxconc < d_auto_cycle_min && maxconc > -4e11){
        d_flux_sign = 1.0;
      }
    }
  }else{
    old_dw->get(totalparts, d_mpm_lb->partCountLabel);
    old_dw->get(totalconc,  d_mpm_lb->diffusion->rTotalConcentration);

    avgconc = totalconc/(double)totalparts;

    if(d_flux_sign > 0){
      if(avgconc > d_auto_cycle_max)
        d_flux_sign = -1.0;
    }else{
      if(avgconc < d_auto_cycle_min)
        d_flux_sign = 1.0;
    }
  }

  // Loop thru patches to update scalar flux
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing applyExternalScalarFlux");

    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.

    int numMPMMatls=d_materialManager->getNumMatls( "MPM" );

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the particle data
      constParticleVariable<Point>   px;
      constParticleVariable<Vector>  parea;
      constParticleVariable<double>  pvol;
      constParticleVariable<Matrix3> psize;
      constParticleVariable<Matrix3> pDeformationMeasure;
      ParticleVariable<double> pExternalScalarFlux;
      ParticleVariable<double> pExternalScalarFlux_pR;
      ParticleVariable<double> pAvgConc;

      old_dw->get(px,    d_mpm_lb->pXLabel,    pset);
      if(d_mpm_flags->d_doScalarDiffusion){
        // JBH -- Fixme -- TODO -- Fold into diffusion sublabel?
        old_dw->get(parea, d_mpm_lb->diffusion->pArea, pset);
      }
      old_dw->get(pvol,  d_mpm_lb->pVolumeLabel, pset);
      old_dw->get(psize, d_mpm_lb->pSizeLabel, pset);
      old_dw->get(pDeformationMeasure, d_mpm_lb->pDeformationMeasureLabel, pset);
      // JBH -- FIXME -- Why are we doing this if we're not doing scalar diffusion in the first place (see above fixme)
      new_dw->allocateAndPut(pExternalScalarFlux,
                                       d_mpm_lb->diffusion->pExternalScalarFlux_preReloc,  pset);
#if defined USE_FLUX_RESTRICTION
      constParticleVariable<double> pConcentration;
      if(d_mpm_flags->d_doScalarDiffusion){
        old_dw->get(pConcentration, d_mpm_lb->diffusion->pConcentration, pset);
      }
#endif

      if (d_mpm_flags->d_useLoadCurves) {
        constParticleVariable<IntVector> pLoadCurveID;
        old_dw->get(pLoadCurveID, d_mpm_lb->pLoadCurveIDLabel, pset);
        bool do_FluxBCs=false;
        for (int ii = 0; ii < (int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++) {
          std::string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
          if (bcs_type == "ScalarFlux") {
            do_FluxBCs=true;
          }
        }

        // Get the load curve data
        if(do_FluxBCs){
          // Iterate over the particles
          for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
            particleIndex idx = *iter;
            pExternalScalarFlux[idx] = 0.0;
            for(int k=0;k<3;k++){
            int loadCurveID = pLoadCurveID[idx](k)-1;
              if (loadCurveID >= 0) {
#if 0
                pExternalScalarFlux[idx] = d_flux_sign * fluxPerPart[loadCurveID];
#else
                ScalarFluxBC* pbc = pbcP[loadCurveID];
                double area = parea[idx].length();
                pExternalScalarFlux[idx] += d_flux_sign * pbc->fluxPerParticle(simTime, area) / pvol[idx];
#endif
#if defined USE_FLUX_RESTRICTION
                if(d_mpm_flags->d_doScalarDiffusion){
                  double flux_restriction = (4 + log(1-pConcentration[idx]))/4;
                  if (flux_restriction < 0.0){
                    flux_restriction = 0.0;
                  }
                  pExternalScalarFlux[idx] *= flux_restriction;
                }
#endif
              } // endif loadCurveID >=0
            } // for k
          }
        } else {
          for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
            pExternalScalarFlux[*iter] = 0.;
          }
        }
      } else { // if use load curves
        for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
          pExternalScalarFlux[*iter] = 0.;
        }
      }
    } // matl loop
  }  // patch loop
}

void AutoCycleFluxBC::countMaterialPointsPerFluxLoadCurve(const ProcessorGroup*,
                                                      const PatchSubset* patches,
                                                      const MaterialSubset*,
                                                      DataWarehouse* old_dw,
                                                      DataWarehouse* new_dw)
{
  printTask(patches, patches->get(0), cout_doing,
                       "countMaterialPointsPerLoadCurve");
  // Find the number of pressure BCs in the problem
  int nofSFBCs = 0;
  for (int ii = 0; ii<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); ii++){
    std::string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[ii]->getType();
    if (bcs_type == "ScalarFlux") {
      nofSFBCs++;

      // Loop through the patches and count
      for(int p=0;p<patches->size();p++){
        const Patch* patch = patches->get(p);
        int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
        int numPts = 0;
        for(int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<IntVector> pLoadCurveID;
          new_dw->get(pLoadCurveID, d_mpm_lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            for(int k = 0;k<3;k++){
              if (pLoadCurveID[idx](k) == (nofSFBCs)){
                 ++numPts;
              }
            }
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts),
                    d_mpm_lb->materialPointsPerLoadCurveLabel, 0, nofSFBCs-1);
      }  // patch loop
    }
  }
}
