/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/PhysicalBC/ScalarFluxBC.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <string>
#include <iostream>

using namespace Uintah;

static DebugStream cout_doing("FluxBCModel", false);

#define USE_FLUX_RESTRICTION

FluxBCModel::FluxBCModel(SimulationStateP& shared_state, MPMFlags* mpm_flags)
{
  d_load_curve_index = 0;
  d_shared_state = shared_state;
  d_mpm_lb = scinew MPMLabel();
  d_mpm_flags = mpm_flags;

}

FluxBCModel::~FluxBCModel()
{
  delete d_mpm_lb;

}

void FluxBCModel::scheduleInitializeScalarFluxBCs(const LevelP& level, SchedulerP& sched)
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
    printSchedule(patches,cout_doing,"FluxBCModel::countMaterialPointsPerFluxLoadCurve");
    printSchedule(patches,cout_doing,"FluxBCModel::scheduleInitializeScalarFluxBCs");
    // Create a task that calculates the total number of particles
    // associated with each load curve.
    Task* t = scinew Task("FluxBCModel::countMaterialPointsPerFluxLoadCurve", this,
                          &FluxBCModel::countMaterialPointsPerFluxLoadCurve);
    t->requires(Task::NewDW, d_mpm_lb->pLoadCurveIDLabel, Ghost::None);
    t->computes(d_mpm_lb->materialPointsPerLoadCurveLabel, d_load_curve_index, Task::OutOfDomain);
    sched->addTask(t, patches, d_shared_state->allMPMMaterials());

#if 1
    // Create a task that calculates the force to be associated with
    // each particle based on the pressure BCs
    t = scinew Task("FluxBCModel::initializeScalarFluxBC", this,
                    &FluxBCModel::initializeScalarFluxBC);
    t->requires(Task::NewDW, d_mpm_lb->materialPointsPerLoadCurveLabel,
                d_load_curve_index, Task::OutOfDomain, Ghost::None);
    sched->addTask(t, patches, d_shared_state->allMPMMaterials());
#endif
  }

  if(d_load_curve_index->removeReference())
      delete d_load_curve_index;
}

void FluxBCModel::initializeScalarFluxBC(const ProcessorGroup*, const PatchSubset* patches,
                                         const MaterialSubset*, DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  std::cout << "doing Initialize flux bc" << std::endl;
  double time = 0.0;
  printTask(patches,patches->get(0),cout_doing,"Doing initialize ScalarFluxBC");
  if (cout_doing.active())
    cout_doing << "Current Time (Initialize ScalarFlux BC) = " << time << std::endl;

  // Calculate the scalar flux at each particle
  for(int p=0;p<patches->size();p++){
    int numMPMMatls=d_shared_state->getNumMPMMatls();
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

void FluxBCModel::scheduleApplyExternalScalarFlux(SchedulerP& sched, const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  Ghost::GhostType  gnone = Ghost::None;
  if (!d_mpm_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"FluxBCModel::scheduleApplyExternalScalarFlux");

  Task* t=scinew Task("FluxBCModel::applyExternalScalarFlux", this,
                      &FluxBCModel::applyExternalScalarFlux);

  t->requires(Task::OldDW, d_mpm_lb->pXLabel,                 Ghost::None);
  t->requires(Task::OldDW, d_mpm_lb->pSizeLabel,              Ghost::None);
  if(d_mpm_flags->d_doScalarDiffusion){
    // JBH -- Fixme -- Todo -- Move to diffusion sublabel?
    t->requires(Task::OldDW, d_mpm_lb->diffusion->pArea,            Ghost::None);
  }
  t->requires(Task::OldDW, d_mpm_lb->pVolumeLabel,            Ghost::None);
  t->requires(Task::OldDW, d_mpm_lb->pDeformationMeasureLabel,Ghost::None);
#if defined USE_FLUX_RESTRICTION
  if(d_mpm_flags->d_doScalarDiffusion){
    t->requires(Task::OldDW, d_mpm_lb->diffusion->pConcentration,     gnone);
  }
#endif
  t->computes(             d_mpm_lb->diffusion->pExternalScalarFlux_preReloc);
  if (d_mpm_flags->d_useLoadCurves) {
    t->requires(Task::OldDW, d_mpm_lb->pLoadCurveIDLabel,     Ghost::None);
  }

  sched->addTask(t, patches, matls);
}

void FluxBCModel::applyExternalScalarFlux(const ProcessorGroup* , const PatchSubset* patches,
                                          const MaterialSubset*, DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  // Get the current simulation time
  double time = d_shared_state->getElapsedSimTime();

  if (cout_doing.active())
    cout_doing << "Current Time (applyExternalScalarFlux) = " << time << std::endl;

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

  // Loop thru patches to update scalar flux
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing applyExternalScalarFlux");

    // Place for user defined loading scenarios to be defined,
    // otherwise pExternalForce is just carried forward.

    int numMPMMatls=d_shared_state->getNumMPMMatls();

    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_shared_state->getMPMMaterial( m );
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

      old_dw->get(px,    d_mpm_lb->pXLabel,    pset);
      if(d_mpm_flags->d_doScalarDiffusion){
        // JBH -- Fixme -- todo -- move to MPMDiffusion sublabel?
        old_dw->get(parea, d_mpm_lb->diffusion->pArea, pset);
      }
      old_dw->get(pvol,  d_mpm_lb->pVolumeLabel, pset);
      old_dw->get(psize, d_mpm_lb->pSizeLabel, pset);
      old_dw->get(pDeformationMeasure, d_mpm_lb->pDeformationMeasureLabel, pset);
      new_dw->allocateAndPut(pExternalScalarFlux,
                                       d_mpm_lb->diffusion->pExternalScalarFlux_preReloc,  pset);

#if defined USE_FLUX_RESTRICTION
      constParticleVariable<double> pConcentration;
      if(d_mpm_flags->d_doScalarDiffusion){
        old_dw->get(pConcentration, d_mpm_lb->diffusion->pConcentration, pset);
      }
#endif

      if (d_mpm_flags->d_useLoadCurves) {
        constParticleVariable<int> pLoadCurveID;
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
            int loadCurveID = pLoadCurveID[idx]-1;
            if (loadCurveID < 0) {
              pExternalScalarFlux[idx] = 0.0;
            } else {
#if 0
              pExternalScalarFlux[idx] = fluxPerPart[loadCurveID];
#else
              ScalarFluxBC* pbc = pbcP[loadCurveID];
              double area = parea[idx].length();
              pExternalScalarFlux[idx] = pbc->fluxPerParticle(time, area) / pvol[idx];
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
            }
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

void FluxBCModel::countMaterialPointsPerFluxLoadCurve(const ProcessorGroup*,
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
        int numMPMMatls=d_shared_state->getNumMPMMatls();
        int numPts = 0;
        for(int m = 0; m < numMPMMatls; m++){
          MPMMaterial* mpm_matl = d_shared_state->getMPMMaterial( m );
          int dwi = mpm_matl->getDWIndex();

          ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
          constParticleVariable<int> pLoadCurveID;
          new_dw->get(pLoadCurveID, d_mpm_lb->pLoadCurveIDLabel, pset);

          ParticleSubset::iterator iter = pset->begin();
          for(;iter != pset->end(); iter++){
            particleIndex idx = *iter;
            if (pLoadCurveID[idx] == (nofSFBCs)) ++numPts;
          }
        } // matl loop
        new_dw->put(sumlong_vartype(numPts),
                    d_mpm_lb->materialPointsPerLoadCurveLabel, 0, nofSFBCs-1);
      }  // patch loop
    }
  }
}
