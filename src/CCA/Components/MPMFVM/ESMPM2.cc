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

#include <CCA/Components/MPMFVM/ESMPM2.h>

#include <CCA/Components/FVM/FVMLabel.h>
#include <CCA/Components/FVM/GaussSolve.h>

#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SwitchingCriteria.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

#include <vector>

using namespace Uintah;

static DebugStream cout_doing("ESMPM2_DOING_COUT", false);

//#define DEBUG_VEL
#undef CBDI_FLUXBCS

ESMPM2::ESMPM2(const ProcessorGroup* myworld,
	       const MaterialManagerP materialManager) :
  ApplicationCommon(myworld, materialManager)
{
  d_amrmpm = scinew AMRMPM(myworld, m_materialManager);
  d_gaufvm = scinew GaussSolve(myworld, m_materialManager);

  d_mpm_lb = scinew MPMLabel();
  d_fvm_lb = scinew FVMLabel();

  d_mpm_flags = 0;
  d_switch_criteria = 0;

  d_TINY_RHO  = 1.e-12;

  d_gac = Ghost::AroundCells;

  d_es_matl  = d_gaufvm->d_es_matl;

  d_es_matlset  = d_gaufvm->d_es_matlset;
}

ESMPM2::~ESMPM2()
{
  d_amrmpm->releaseComponents();
  d_gaufvm->releaseComponents();

  delete d_amrmpm;
  delete d_gaufvm;
  delete d_mpm_lb;
  delete d_fvm_lb;
}

void ESMPM2::problemSetup(const ProblemSpecP& prob_spec,
			  const ProblemSpecP& restart_prob_spec,
                          GridP& grid)
{
  //**** Start MPM Section *****
  d_amrmpm->setComponents( this );
  dynamic_cast<ApplicationCommon*>(d_amrmpm)->problemSetup( prob_spec );
  
  d_amrmpm->problemSetup(prob_spec, restart_prob_spec, grid);

  //**** Start FVM Section *****
  d_gaufvm->setComponents( this );
  dynamic_cast<ApplicationCommon*>(d_gaufvm)->problemSetup( prob_spec );
 
  d_gaufvm->setWithMPM(true);
  d_gaufvm->problemSetup(prob_spec, restart_prob_spec, grid);

  d_switch_criteria =
    dynamic_cast<SwitchingCriteria*>(getPort("switch_criteria"));

  if(d_switch_criteria){
    d_switch_criteria->problemSetup(prob_spec, restart_prob_spec, m_materialManager);
  }

  ProblemSpecP mpm_ps = 0;
  mpm_ps = prob_spec->findBlock("MPM");

  if(!mpm_ps){
    mpm_ps = restart_prob_spec->findBlock("MPM");
  }

  d_mpm_flags = d_amrmpm->flags;

  //ProblemSpecP esmpm_ps = prob_spec->findBlock("ESMPM");
}

void ESMPM2::outputProblemSpec(ProblemSpecP& prob_spec)
{
  d_amrmpm->outputProblemSpec(prob_spec);
}

void ESMPM2::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,cout_doing,"ESMPM2::scheduleInitialize");
  Task* task = scinew Task("ESMPM2::initialize", this, &ESMPM2::initialize);
  task->computes(d_fvm_lb->ccESPotential);
  sched->addTask(task, level->eachPatch(), d_es_matlset);

  d_amrmpm->scheduleInitialize(level, sched);
}

void ESMPM2::initialize(const ProcessorGroup*, const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    CCVariable<double>   es_potential;
    new_dw->allocateAndPut(es_potential, d_fvm_lb->ccESPotential, 0, patch);
    es_potential.initialize(0.0);
  }
}

void ESMPM2::scheduleRestartInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level, cout_doing, "ESMPM2::scheduleRestartInitialize");
  d_amrmpm->scheduleRestartInitialize(level, sched);
}

void ESMPM2::restartInitialize()
{
  if(cout_doing.active())
    cout_doing << "Doing restartInitialize \t\t\t ESMPM2" << std::endl;

  d_amrmpm->restartInitialize();
}

void ESMPM2::scheduleComputeStableTimeStep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleComputeStableTimeStep(level, sched);
}

void ESMPM2::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
{
  // Only schedule once
  if(level->getIndex() > 0)
    return;

  const MaterialSet* mpm_matls = m_materialManager->allMaterials( "MPM" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const MaterialSubset* mpm_matlsub = mpm_matls->getUnion();

  int maxLevels = level->getGrid()->numLevels();
  GridP grid = level->getGrid();

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->schedulePartitionOfUnity(       sched, patches, mpm_matls);
    d_amrmpm->scheduleComputeZoneOfInfluence( sched, patches, mpm_matls);
    d_amrmpm->scheduleApplyExternalLoads(     sched, patches, mpm_matls);
    d_amrmpm->d_fluxbc->scheduleApplyExternalScalarFlux( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_amrmpm->scheduleInterpolateParticlesToGrid(sched, patches, mpm_matls);
    scheduleComputeCCChargeMass(sched, patches, mpm_matlsub, d_es_matl, all_matls);
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
    d_amrmpm->scheduleConcInterpolated(sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_gaufvm->scheduleBuildMatrixAndRhs(    sched, level, d_es_matlset);
    d_amrmpm->scheduleComputeInternalForce( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_gaufvm->scheduleSolve(sched, level, d_es_matlset);
    d_amrmpm->scheduleComputeInternalForce_CFI( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    d_gaufvm->scheduleUpdateESPotential(sched, level, d_es_matlset);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    scheduleInterpESPotentialToPart(sched, patches, mpm_matlsub, d_es_matl, all_matls);
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

void ESMPM2::scheduleFinalizeTimestep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleFinalizeTimestep(level, sched);
}

void ESMPM2::scheduleComputeCCChargeMass(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSubset* mpm_matls,
                                         const MaterialSubset* es_matls,
                                         const MaterialSet* all_matls)
{
  printSchedule(patches,cout_doing,"ESMPM2::scheduleComputeCCChargeMass");

  Task* task = scinew Task("ESMPM2::computeCCChargeMass", this,
                           &ESMPM2::computeCCChargeMass);

  task->requires(Task::OldDW, d_mpm_lb->pPosChargeLabel,    mpm_matls, d_gac, 0);
  task->requires(Task::OldDW, d_mpm_lb->pNegChargeLabel,    mpm_matls, d_gac, 0);
  task->requires(Task::OldDW, d_mpm_lb->pPermittivityLabel, mpm_matls, d_gac, 0);
  task->requires(Task::OldDW, d_mpm_lb->pXLabel,            mpm_matls, d_gac, 0);
  task->computes(d_fvm_lb->ccPosCharge,    es_matls);
  task->computes(d_fvm_lb->ccNegCharge,    es_matls);
  task->computes(d_fvm_lb->ccPermittivity, es_matls);

  sched->addTask(task, patches, all_matls);

}

void ESMPM2::computeCCChargeMass(const ProcessorGroup*, const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  std::vector<IntVector> ni(8);

  for(int p=0;p<patches->size();p++){
      const Patch* patch = patches->get(p);

      Vector cell_dim = patch->getLevel()->dCell();
      Point anchor = patch->getLevel()->getAnchor();

      IntVector low_idx  = patch->getCellLowIndex();
      IntVector high_idx = patch->getCellHighIndex();

      CCVariable<double> cc_poscharge;
      CCVariable<double> cc_negcharge;
      CCVariable<double> cc_permittivity;
      CCVariable<int> cc_part_count;

      new_dw->allocateAndPut(cc_poscharge,    d_fvm_lb->ccPosCharge,    0, patch, d_gac, 0);
      new_dw->allocateAndPut(cc_negcharge,    d_fvm_lb->ccNegCharge,    0, patch, d_gac, 0);
      new_dw->allocateAndPut(cc_permittivity, d_fvm_lb->ccPermittivity, 0, patch, d_gac, 0);
      new_dw->allocateTemporary(cc_part_count, patch);

      cc_poscharge.initialize(0.0);
      cc_negcharge.initialize(0.0);
      cc_permittivity.initialize(0.0);
      cc_part_count.initialize(0);

      int numMatls = m_materialManager->getNumMatls( "MPM" );
      for(int m = 0; m < numMatls; m++){
        MPMMaterial* mpm_matl = (MPMMaterial* ) m_materialManager->getMaterial( "MPM", m );
        int dwi = mpm_matl->getDWIndex();

        constParticleVariable<double> p_poscharge;
        constParticleVariable<double> p_negcharge;
        constParticleVariable<double> p_permittivity;
        constParticleVariable<Point> p_position;

        ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, d_gac,
                                                           0, d_mpm_lb->pXLabel);

        old_dw->get(p_position,     d_mpm_lb->pXLabel,            pset);
        old_dw->get(p_poscharge,    d_mpm_lb->pPosChargeLabel,    pset);
        old_dw->get(p_negcharge,    d_mpm_lb->pNegChargeLabel,    pset);
        old_dw->get(p_permittivity, d_mpm_lb->pPermittivityLabel, pset);

        ParticleSubset::iterator iter;

        for(iter  = pset->begin(); iter != pset->end(); iter++){
          particleIndex idx = *iter;

          Point norm_pos = Point((p_position[idx] - anchor)/cell_dim);

          IntVector cell_idx(Floor(norm_pos.x()), Floor(norm_pos.y()),
                             Floor(norm_pos.z()));

          cc_poscharge[cell_idx] += p_poscharge[idx];
          cc_negcharge[cell_idx] += p_negcharge[idx];
          cc_permittivity[cell_idx] += p_permittivity[idx];
          cc_part_count[cell_idx] += 1;
        }
      }// End of materials loop
      for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
        IntVector c = *iter;
        if(cc_permittivity[c] > 0){
          cc_permittivity[c] = cc_permittivity[c]/((double)cc_part_count[c]);
        }
      }
    }// End of patch loop
}

void ESMPM2::scheduleInterpESPotentialToPart(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* es_matls,
                                            const MaterialSet* all_matls)
{
  printSchedule(patches,cout_doing,"ESMPM2::scheduleInterpESPotentialToPart");

  Task* task = scinew Task("ESMPM2::interpESPotentialToPart", this,
                           &ESMPM2::interpESPotentialToPart);

  task->requires(Task::NewDW, d_fvm_lb->ccESPotential, es_matls,  d_gac, 1);
  task->requires(Task::OldDW, d_mpm_lb->pXLabel,       mpm_matls, d_gac, 0);
  task->computes(d_mpm_lb->pESPotential,     mpm_matls);
  task->computes(d_mpm_lb->pESGradPotential, mpm_matls);

  sched->addTask(task, patches, all_matls);

}

void ESMPM2::interpESPotentialToPart(const ProcessorGroup*, const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  std::vector<IntVector> ni(8);
  std::vector<double> S(8);
  std::vector<Vector> dS(8);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Vector cell_dim = patch->getLevel()->dCell();

    // Normally the anchor is associated with a node position but
    // since interpolation is based on cell centers the anchor needs
    // to be shifted so that it associated with the cell center of the
    // anchor cell.
    Point anchor = patch->getLevel()->getAnchor();
    anchor.x(anchor.x() + cell_dim.x()/2);
    anchor.y(anchor.y() + cell_dim.y()/2);
    anchor.z(anchor.z() + cell_dim.z()/2);

    // double cell_vol = cell_dim.x() * cell_dim.y() * cell_dim.z();
    double dxinv[3] = {1/cell_dim.x(), 1/cell_dim.y(), 1/cell_dim.z()};

    IntVector low_idx  = patch->getCellLowIndex();
    IntVector high_idx = patch->getExtraCellHighIndex();

    constCCVariable<double> cc_espotential;
    new_dw->get(cc_espotential, d_fvm_lb->ccESPotential, 0, patch, d_gac, 1);

    int numMatls = m_materialManager->getNumMatls( "MPM" );
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial* ) m_materialManager->getMaterial( "MPM", m );
      int dwi = mpm_matl->getDWIndex();

      constParticleVariable<Point> p_position;
      ParticleVariable<double> p_espotential;
      ParticleVariable<Vector> p_esgradpotential;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, d_gac,
                                                             0, d_mpm_lb->pXLabel);

      old_dw->get(p_position, d_mpm_lb->pXLabel, pset);

      new_dw->allocateAndPut(p_espotential,     d_mpm_lb->pESPotential,     pset);
      new_dw->allocateAndPut(p_esgradpotential, d_mpm_lb->pESGradPotential, pset);

      ParticleSubset::iterator iter;

      for(iter  = pset->begin(); iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Point norm_pos = Point((p_position[idx] - anchor)/cell_dim);

        IntVector cell_idx(Floor(norm_pos.x()), Floor(norm_pos.y()),
                           Floor(norm_pos.z()));

        // Point cell_center = Point(cell_idx.x() + .5, cell_idx.y() + .5,
        //                           cell_idx.z() + .5);

        double fx = norm_pos.x() - cell_idx.x();
        double fy = norm_pos.y() - cell_idx.y();
        double fz = norm_pos.z() - cell_idx.z();
        double fx1 = 1 - fx;
        double fy1 = 1 - fy;
        double fz1 = 1 - fz;

        ni[0] = cell_idx;
        ni[1] = cell_idx + IntVector(0, 0, 1);
        ni[2] = cell_idx + IntVector(0, 1, 0);
        ni[3] = cell_idx + IntVector(0, 1, 1);
        ni[4] = cell_idx + IntVector(1, 0, 0);
        ni[5] = cell_idx + IntVector(1, 0, 1);
        ni[6] = cell_idx + IntVector(1, 1, 0);
        ni[7] = cell_idx + IntVector(1, 1, 1);

        S[0] = fx1 * fy1 * fz1;
        S[1] = fx1 * fy1 * fz;
        S[2] = fx1 * fy * fz1;
        S[3] = fx1 * fy * fz;
        S[4] = fx * fy1 * fz1;
        S[5] = fx * fy1 * fz;
        S[6] = fx * fy * fz1;
        S[7] = fx * fy * fz;

        dS[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
        dS[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
        dS[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
        dS[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
        dS[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
        dS[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
        dS[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
        dS[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);

        double espotential = 0;
        Vector esgradpotential(0.0, 0.0, 0.0);

        for(int i = 0; i < 8; i++){
          IntVector node = ni[i];
          espotential += cc_espotential[node] * S[i];
          for(int j = 0; j < 3; j++){
            esgradpotential[j] += cc_espotential[node] * dS[i][j] * dxinv[j];
          }
        }

        p_espotential[idx] = espotential;
        p_esgradpotential[idx] = esgradpotential;

      }
    }// End of materials loop
  }// End of patch loop

}
