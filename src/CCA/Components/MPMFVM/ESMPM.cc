/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/MPMFVM/ESMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <string>

using namespace Uintah;

static DebugStream cout_doing("ESMPM_DOING_COUT", false);

//#define DEBUG_VEL
#undef CBDI_FLUXBCS

ESMPM::ESMPM(const ProcessorGroup* myworld) : UintahParallelComponent(myworld)
{
  d_amrmpm = scinew AMRMPM(myworld);
  d_esfvm = scinew ElectrostaticSolve(myworld);

  d_mpm_lb = scinew MPMLabel();
  d_fvm_lb = scinew FVMLabel();

  d_mpm_flags = 0;
  d_data_archiver = 0;
  d_switch_criteria = 0;

  d_TINY_RHO  = 1.e-12;

  d_one_matl  = d_esfvm->d_es_matl;

  d_one_matlset  = d_esfvm->d_es_matlset;

  d_conductivity_model = 0;
}

ESMPM::~ESMPM()
{
  delete d_amrmpm;
  delete d_esfvm;
  delete d_mpm_lb;
  delete d_fvm_lb;
  if(!d_conductivity_model)
    delete d_conductivity_model;
}

void ESMPM::problemSetup(const ProblemSpecP& prob_spec, const ProblemSpecP& restart_prob_spec,
                         GridP& grid, SimulationStateP& shared_state)
{
  d_shared_state = shared_state;
  d_data_archiver = dynamic_cast<Output*>(getPort("output"));
  Scheduler* sched = dynamic_cast<Scheduler*>(getPort("scheduler"));

  //**** Start MPM Section *****
  d_amrmpm->attachPort("output", d_data_archiver);
  d_amrmpm->attachPort("scheduler", sched);
  d_amrmpm->problemSetup(prob_spec, restart_prob_spec, grid, d_shared_state);

  //**** Start FVM Section *****
  d_esfvm->attachPort("output", d_data_archiver);
  d_esfvm->attachPort("scheduler", sched);
  SolverInterface* solver = dynamic_cast<SolverInterface*>(getPort("solver"));
  if(!solver){
    throw InternalError("ElectrostaticSolve needs a solver component to work", __FILE__, __LINE__);
  }
  d_esfvm->attachPort("solver", solver);

  d_esfvm->setWithMPM(true);

  d_esfvm->problemSetup(prob_spec, restart_prob_spec, grid, d_shared_state);

  d_switch_criteria = dynamic_cast<SwitchingCriteria*>(getPort("switch_criteria"));

  if(d_switch_criteria){
    d_switch_criteria->problemSetup(prob_spec, restart_prob_spec, d_shared_state);
  }

  ProblemSpecP mpm_ps = 0;
  mpm_ps = prob_spec->findBlock("MPM");

  if(!mpm_ps){
    mpm_ps = restart_prob_spec->findBlock("MPM");
  }

  d_mpm_flags = d_amrmpm->flags;

  d_conductivity_model = scinew ESConductivityModel(d_shared_state,
                                                    d_mpm_flags, d_mpm_lb, d_fvm_lb);
}

void ESMPM::outputProblemSpec(ProblemSpecP& prob_spec)
{
  d_amrmpm->outputProblemSpec(prob_spec);
}

void ESMPM::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,cout_doing,"ESMPM::scheduleInitialize");
  d_amrmpm->scheduleInitialize(level, sched);
}

void ESMPM::scheduleRestartInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level, cout_doing, "ESMPM::scheduleRestartInitialize");
  d_amrmpm->scheduleRestartInitialize(level, sched);
}

void ESMPM::restartInitialize()
{
  if(cout_doing.active())
    cout_doing << "Doing restartInitialize \t\t\t ESMPM" << std::endl;

  d_amrmpm->restartInitialize();
}

void ESMPM::scheduleComputeStableTimestep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleComputeStableTimestep(level, sched);
}

void ESMPM::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  // Only schedule once
  if(level->getIndex() > 0)
    return;

  const MaterialSet* mpm_matls = d_shared_state->allMPMMaterials();
  const MaterialSet* all_matls = d_shared_state->allMaterials();

  int maxLevels = level->getGrid()->numLevels();
  GridP grid = level->getGrid();

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->schedulePartitionOfUnity(        sched, patches, mpm_matls);
    d_amrmpm->scheduleComputeZoneOfInfluence(  sched, patches, mpm_matls);
    d_amrmpm->scheduleApplyExternalLoads(      sched, patches, mpm_matls);
    d_amrmpm->d_fluxbc->scheduleApplyExternalScalarFlux( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleInterpolateParticlesToGrid_CFI( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleCoarsenNodalData_CFI( sched, patches, mpm_matls, AMRMPM::coarsenData);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleNormalizeNodalVelTempConc(sched, patches, mpm_matls);
    d_amrmpm->scheduleExMomInterpolated(        sched, patches, mpm_matls);
    //scheduleInterpolateParticlesToCellFC(sched, patches, mpm_matls, all_matls);
    d_conductivity_model->scheduleComputeConductivity(sched, patches, all_matls, d_one_matl);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_esfvm->scheduleBuildMatrixAndRhs(     sched, level, d_one_matlset);
    d_amrmpm->scheduleComputeInternalForce( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_esfvm->scheduleSolve(sched, level, d_one_matlset);
    d_amrmpm->scheduleComputeInternalForce_CFI( sched, patches, mpm_matls);
  }

  if(d_mpm_flags->d_doScalarDiffusion){
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_esfvm->scheduleUpdateESPotential(  sched, level, d_one_matlset);
      d_amrmpm->scheduleComputeFlux(       sched, patches, mpm_matls);
      d_amrmpm->scheduleComputeDivergence( sched, patches, mpm_matls);
    }

    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_amrmpm->scheduleComputeDivergence_CFI( sched, patches, mpm_matls);
    }
  }

  for (int l = 0; l < maxLevels-1; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleCoarsenNodalData_CFI2( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleComputeAndIntegrateAcceleration(sched, patches, mpm_matls);
    d_amrmpm->scheduleExMomIntegrated(                sched, patches, mpm_matls);
    d_amrmpm->scheduleSetGridBoundaryConditions(      sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleComputeLAndF( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleInterpolateToParticlesAndUpdate(sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleComputeStressTensor( sched, patches, mpm_matls);
  }

  if(d_mpm_flags->d_computeScaleFactor){
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
      d_amrmpm->scheduleComputeParticleScaleFactor( sched, patches, mpm_matls);
    }
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleFinalParticleUpdate( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    if(d_mpm_flags->d_refineParticles){
      d_amrmpm->scheduleAddParticles( sched, patches, mpm_matls);
    }
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleReduceFlagsExtents( sched, patches, mpm_matls);
  }
}

void ESMPM::scheduleFinalizeTimestep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleFinalizeTimestep(level, sched);
}

void ESMPM::scheduleInterpolateParticlesToCellFC(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* mpm_matls,
                                                 const MaterialSet* all_matls)
{
  const Level* level = getLevel(patches);
  if(!d_mpm_flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches, cout_doing, "ESMPM::scheduleInterpolateParticlesToCellFC");

  Task* t = scinew Task("ESMPM::interpolateParticlesToCellFC",
                        this, &ESMPM::interpolateParticlesToCellFC);

  t->requires(Task::NewDW, d_mpm_lb->gConcentrationLabel, Ghost::AroundCells, 1);
  t->requires(Task::NewDW, d_mpm_lb->gMassLabel,          Ghost::AroundCells, 1);

  t->computes(d_fvm_lb->fcxConductivity, d_one_matl, Task::OutOfDomain);
  t->computes(d_fvm_lb->fcyConductivity, d_one_matl, Task::OutOfDomain);
  t->computes(d_fvm_lb->fczConductivity, d_one_matl, Task::OutOfDomain);

  sched->addTask(t, level->eachPatch(), all_matls);
}

void ESMPM::interpolateParticlesToCellFC(const ProcessorGroup* pg,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  IntVector i100(1,0,0);
  IntVector i110(1,1,0);
  IntVector i111(1,1,1);
  IntVector i011(0,1,1);
  IntVector i001(0,0,1);
  IntVector i101(1,0,1);
  IntVector i010(0,1,0);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, cout_doing,
              "Doing ESMPM::interpolateParticlesToCellFC");

    int num_matls = d_shared_state->getNumMPMMatls();
    Vector cell_dim = patch->getLevel()->dCell();
    double cell_vol = cell_dim.x() * cell_dim.y() * cell_dim.z();

    SFCXVariable<double> fcx_conductivity;
    SFCYVariable<double> fcy_conductivity;
    SFCZVariable<double> fcz_conductivity;

    SFCXVariable<double> fcx_mass;
    SFCYVariable<double> fcy_mass;
    SFCZVariable<double> fcz_mass;

    new_dw->allocateAndPut(fcx_conductivity,  d_fvm_lb->fcxConductivity,    0, patch);
    new_dw->allocateAndPut(fcy_conductivity,  d_fvm_lb->fcyConductivity,    0, patch);
    new_dw->allocateAndPut(fcz_conductivity,  d_fvm_lb->fczConductivity,    0, patch);

    new_dw->allocateTemporary(fcx_mass, patch, Ghost::None, 0);
    new_dw->allocateTemporary(fcy_mass, patch, Ghost::None, 0);
    new_dw->allocateTemporary(fcz_mass, patch, Ghost::None, 0);

    fcx_conductivity.initialize(0.0);
    fcy_conductivity.initialize(0.0);
    fcz_conductivity.initialize(0.0);

    fcx_mass.initialize(d_TINY_RHO * cell_vol);
    fcy_mass.initialize(d_TINY_RHO * cell_vol);
    fcz_mass.initialize(d_TINY_RHO * cell_vol);

    IntVector lowidx = patch->getCellLowIndex();
    IntVector highidx = patch->getExtraCellHighIndex();
    for(int m = 0; m < num_matls; m++){
      MPMMaterial* mpm_matl = d_shared_state->getMPMMaterial(m);
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> gconc;
      constNCVariable<double> gmass;

      new_dw->get(gconc, d_mpm_lb->gConcentrationLabel, dwi, patch, Ghost::AroundCells, 1);
      new_dw->get(gmass, d_mpm_lb->gMassLabel,          dwi, patch, Ghost::AroundCells, 1);


      for(CellIterator iter=CellIterator(lowidx, highidx); !iter.done(); iter++){
        IntVector c = *iter;

        fcx_conductivity[c] += .25 * gmass[c]        * gconc[c];
        fcx_conductivity[c] += .25 * gmass[c + i010] * gconc[c + i010];
        fcx_conductivity[c] += .25 * gmass[c + i011] * gconc[c + i011];
        fcx_conductivity[c] += .25 * gmass[c + i001] * gconc[c + i001];
        fcx_mass[c] += .25 * gmass[c];
        fcx_mass[c] += .25 * gmass[c + i010];
        fcx_mass[c] += .25 * gmass[c + i011];
        fcx_mass[c] += .25 * gmass[c + i001];

        fcy_conductivity[c] += .25 * gmass[c]        * gconc[c];
        fcy_conductivity[c] += .25 * gmass[c + i100] * gconc[c + i100];
        fcy_conductivity[c] += .25 * gmass[c + i101] * gconc[c + i101];
        fcy_conductivity[c] += .25 * gmass[c + i001] * gconc[c + i001];
        fcy_mass[c] += .25 * gmass[c];
        fcy_mass[c] += .25 * gmass[c + i100];
        fcy_mass[c] += .25 * gmass[c + i101];
        fcy_mass[c] += .25 * gmass[c + i001];

        fcz_conductivity[c] += .25 * gmass[c] * gconc[c];
        fcz_conductivity[c] += .25 * gmass[c + i100] * gconc[c + i100];
        fcz_conductivity[c] += .25 * gmass[c + i110] * gconc[c + i110];
        fcz_conductivity[c] += .25 * gmass[c + i010] * gconc[c + i010];
        fcz_mass[c] += .25 * gmass[c];
        fcz_mass[c] += .25 * gmass[c + i100];
        fcz_mass[c] += .25 * gmass[c + i110];
        fcz_mass[c] += .25 * gmass[c + i010];
      }
    } // End material loop

    for(CellIterator iter=CellIterator(lowidx, highidx); !iter.done(); iter++){
      IntVector c = *iter;
      fcx_conductivity[c] = fcx_conductivity[c] / fcx_mass[c];
      fcy_conductivity[c] = fcy_conductivity[c] / fcy_mass[c];
      fcz_conductivity[c] = fcz_conductivity[c] / fcz_mass[c];
    }

  } // End patch loop
}

void ESMPM::fcLinearInterpolator(const Patch* patch, const Point& pos,
                                 std::vector<IntVector>& ni, std::vector<double>& S)
{
  Vector cell_dim = patch->getLevel()->dCell();
  Point anchor = patch->getLevel()->getAnchor();
  Point norm_pos = Point((pos - anchor)/cell_dim);

  IntVector cell_idx(Floor(norm_pos.x()), Floor(norm_pos.y()),
                             Floor(norm_pos.z()));

  ni[0] = cell_idx;                       // face center x-
  ni[1] = cell_idx + IntVector(1, 0, 0);  // face center x+
  ni[2] = cell_idx;                       // face center y-
  ni[3] = cell_idx + IntVector(0, 1, 0);  // face center y+
  ni[4] = cell_idx;                       // face center z-
  ni[5] = cell_idx + IntVector(0, 0, 1);  // face center z+

}
