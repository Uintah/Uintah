/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <CCA/Components/FVM/FVMLabel.h>
#include <CCA/Components/FVM/ElectrostaticSolve.h>

#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/PhysicalBC/FluxBCModel.h>

#include <CCA/Components/MPMFVM/ESConductivityModel.h>
#include <CCA/Components/MPMFVM/ESConductivityModelFactory.h>

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

static DebugStream cout_doing("ESMPM_DOING_COUT", false);

//#define DEBUG_VEL
#undef CBDI_FLUXBCS

ESMPM::ESMPM(const ProcessorGroup* myworld,
	     const MaterialManagerP materialManager) :
  ApplicationCommon(myworld, materialManager)
{
  d_amrmpm = scinew AMRMPM(myworld, m_materialManager);
  d_esfvm = scinew ElectrostaticSolve(myworld, m_materialManager);

  d_mpm_lb = scinew MPMLabel();
  d_fvm_lb = scinew FVMLabel();

  d_mpm_flags = 0;
  d_switch_criteria = 0;

  d_TINY_RHO  = 1.e-12;

  d_gac = Ghost::AroundCells;

  d_es_matl  = d_esfvm->d_es_matl;

  d_es_matlset  = d_esfvm->d_es_matlset;

  d_conductivity_model = 0;
}

ESMPM::~ESMPM()
{
  d_amrmpm->releaseComponents();
  d_esfvm->releaseComponents();

  delete d_amrmpm;
  delete d_esfvm;
  delete d_mpm_lb;
  delete d_fvm_lb;
  if(!d_conductivity_model)
    delete d_conductivity_model;
}

void ESMPM::problemSetup(const ProblemSpecP& prob_spec,
			 const ProblemSpecP& restart_prob_spec,
                         GridP& grid)
{
  //**** Start MPM Section *****
  d_amrmpm->setComponents( this );
  dynamic_cast<ApplicationCommon*>(d_amrmpm)->problemSetup( prob_spec );
  
  d_amrmpm->problemSetup(prob_spec, restart_prob_spec, grid);

  //**** Start FVM Section *****
  d_esfvm->setComponents( this );
  dynamic_cast<ApplicationCommon*>(d_esfvm)->problemSetup( prob_spec );

  d_esfvm->setWithMPM(true);
  d_esfvm->problemSetup(prob_spec, restart_prob_spec, grid);


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

  ProblemSpecP esmpm_ps = prob_spec->findBlock("ESMPM");
  esmpm_ps->require("conductivity_model", d_cd_model_name);

  d_conductivity_model =
    ESConductivityModelFactory::create(prob_spec, m_materialManager,
				       d_mpm_flags, d_mpm_lb, d_fvm_lb);
}

void ESMPM::outputProblemSpec(ProblemSpecP& prob_spec)
{
  d_amrmpm->outputProblemSpec(prob_spec);
}

void ESMPM::scheduleInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level,cout_doing,"ESMPM::scheduleInitialize");
  Task* task = scinew Task("ESMPM::initialize", this, &ESMPM::initialize);
  task->computesVar(d_fvm_lb->ccESPotential);
  sched->addTask(task, level->eachPatch(), d_es_matlset);

  d_amrmpm->scheduleInitialize(level, sched);
}

void ESMPM::initialize(const ProcessorGroup*, const PatchSubset* patches,
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

void ESMPM::scheduleRestartInitialize(const LevelP& level, SchedulerP& sched)
{
  printSchedule(level, cout_doing, "ESMPM::scheduleRestartInitialize");
  d_amrmpm->scheduleRestartInitialize(level, sched);
}

void ESMPM::scheduleComputeStableTimeStep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleComputeStableTimeStep(level, sched);
}

void ESMPM::scheduleTimeAdvance( const LevelP& level, SchedulerP& sched)
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
    d_conductivity_model->scheduleComputeConductivity(sched, patches, all_matls, d_es_matl);
    d_amrmpm->scheduleConcInterpolated(sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_esfvm->scheduleBuildMatrixAndRhs(     sched, level, d_es_matlset);
    d_amrmpm->scheduleComputeInternalForce( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    const PatchSet* patches = level->eachPatch();
    d_esfvm->scheduleSolve(sched, level, d_es_matlset);
    d_amrmpm->scheduleComputeInternalForce_CFI( sched, patches, mpm_matls);
  }

  for (int l = 0; l < maxLevels; l++) {
    const LevelP& level = grid->getLevel(l);
    d_esfvm->scheduleUpdateESPotential(sched, level, d_es_matlset);
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

void ESMPM::scheduleFinalizeTimestep(const LevelP& level, SchedulerP& sched)
{
  d_amrmpm->scheduleFinalizeTimestep(level, sched);
}

void ESMPM::scheduleInterpESPotentialToPart(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* es_matls,
                                            const MaterialSet* all_matls)
{
  printSchedule(patches,cout_doing,"ESMPM::scheduleInterpESPotentialToPart");

  Task* task = scinew Task("ESMPM::interpESPotentialToPart", this,
                           &ESMPM::interpESPotentialToPart);

  task->requiresVar(Task::NewDW, d_fvm_lb->ccESPotential, es_matls,  d_gac, 1);
  task->requiresVar(Task::OldDW, d_mpm_lb->pXLabel,       mpm_matls, d_gac, 0);
  task->computesVar(d_mpm_lb->pESPotential,     mpm_matls);
  task->computesVar(d_mpm_lb->pESGradPotential, mpm_matls);

  sched->addTask(task, patches, all_matls);

}

void ESMPM::interpESPotentialToPart(const ProcessorGroup*, const PatchSubset* patches,
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
      MPMMaterial* mpm_matl = (MPMMaterial* ) m_materialManager->getMaterial( "MPM",  m );
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
