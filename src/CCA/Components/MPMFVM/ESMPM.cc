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
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <string>

using namespace Uintah;

static DebugStream cout_doing("ESMPM_DOING_COUT", true);

ESMPM::ESMPM(const ProcessorGroup* myworld) : UintahParallelComponent(myworld)
{
  d_amrmpm = scinew AMRMPM(myworld);
  d_esfvm = scinew ElectrostaticSolve(myworld);

  d_mpm_lb = scinew MPMLabel();
  d_fvm_lb = scinew FVMLabel();

  d_mpm_flags = 0;
  d_data_archiver = 0;
  d_switch_criteria = 0;

}

ESMPM::~ESMPM()
{
  delete d_amrmpm;
  delete d_esfvm;
  delete d_mpm_lb;
  delete d_fvm_lb;

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
  //const MaterialSet* all_matls = d_shared_state->allMaterials();

  int maxLevels = level->getGrid()->numLevels();
  GridP grid = level->getGrid();

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->schedulePartitionOfUnity(        sched, patches, mpm_matls);
    d_amrmpm->scheduleComputeZoneOfInfluence(  sched, patches, mpm_matls);
    d_amrmpm->scheduleApplyExternalLoads(      sched, patches, mpm_matls);
    //d_amrmpm->scheduleApplyExternalScalarFlux( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleInterpolateParticlesToGrid(     sched, patches, mpm_matls);
    scheduleInterpolateParticlesToCellFC(sched, patches, mpm_matls);
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
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleComputeInternalForce( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleComputeInternalForce_CFI( sched, patches, mpm_matls);
  }

  if(d_mpm_flags->d_doScalarDiffusion){
    for (int l = 0; l < maxLevels; l++) {
      const LevelP& level = grid->getLevel(l);
      const PatchSet* patches = level->eachPatch();
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
                                                 const MaterialSet* matls)
{
  const Level* level = getLevel(patches);
  if(!d_mpm_flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches, cout_doing, "ESMPM::scheduleInterpolateConcToGrid");

  Task* t = scinew Task("ESMPM::interpolateParticlesToCellFC",
                        this, &ESMPM::interpolateParticlesToCellFC);

  t->requires(Task::OldDW, d_mpm_lb->pXLabel,             Ghost::AroundCells, 1);
  t->requires(Task::OldDW, d_mpm_lb->pConcentrationLabel, Ghost::AroundCells, 1);
}

void ESMPM::interpolateParticlesToCellFC(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{

}
