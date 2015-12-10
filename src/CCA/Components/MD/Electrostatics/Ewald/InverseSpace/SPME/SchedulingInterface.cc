/*
 *
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
 *
 * ----------------------------------------------------------
 * SPME_Scheduling.cc
 *
 *  Created on: May 15, 2014
 *      Author: jbhooper
 */

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>

#include <CCA/Ports/Scheduler.h>

#include <CCA/Components/MD/Electrostatics/Ewald/InverseSpace/SPME/SPME.h>


using namespace Uintah;

// Scheduling related helpers for the base SPME class
void SPME::addInitializeRequirements(Task* task, MDLabel* label) const
{
  // Empty
}

void SPME::addInitializeComputes(Task* task, MDLabel* label) const
{

  task->computes(label->electrostatic->sForwardTransformPlan);
  task->computes(label->electrostatic->sBackwardTransformPlan);

// Universal reduction variables
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->electrostatic->rElectrostaticRealEnergy);
  task->computes(label->electrostatic->rElectrostaticRealStress);

// Universal particle variables
  task->computes(label->electrostatic->pF_electroInverse);
  task->computes(label->electrostatic->pF_electroReal);

  if (f_polarizable)
  {
// Polarizable specific reduction variables
    task->computes(label->electrostatic->rElectrostaticInverseStressDipole);
// Polarizable specific particle variables
    task->computes(label->electrostatic->pMu);
    task->computes(label->electrostatic->pE_electroReal);
    task->computes(label->electrostatic->pE_electroInverse);
  }
}

void SPME::addSetupRequirements(Task* task, MDLabel* d_label) const
{
// Empty
}

void SPME::addSetupComputes(Task* task, MDLabel* d_label) const
{
  task->computes(d_label->electrostatic->dElectrostaticDependency);
}

void SPME::addCalculateRequirements(       Task        * task
                                   ,       MDLabel     * labels
                                   , const PatchSet    * patches
                                   , const MaterialSet * matls
                                   , const LevelP      & level ) const
{

  task->requires(Task::OldDW, labels->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW, labels->global->pID, Ghost::None, 0);
  // Ensures that SPME::Setup runs first
  task->requires(Task::NewDW, labels->electrostatic->dElectrostaticDependency);

  if (f_polarizable) {
    task->requires(Task::OldDW, labels->electrostatic->pMu, Ghost::None, 0);
  }
}

void SPME::addCalculateComputes(       Task        * task
                               ,       MDLabel     * labels
                               , const PatchSet    * patches
                               , const MaterialSet * matls
                               , const LevelP      & level ) const
{
// These are the variables provided at the end of the entire SPME routine,
// and have nothing to do with computes/requires from the subscheduler.

// Universal reduction variables
  const MaterialSubset* matl_subset = matls->getUnion();

  task->computes(labels->electrostatic->rElectrostaticInverseEnergy, level.get_rep(), matl_subset, Task::NormalDomain);
  task->computes(labels->electrostatic->rElectrostaticRealEnergy, level.get_rep(), matl_subset, Task::NormalDomain);
  task->computes(labels->electrostatic->rElectrostaticInverseStress, level.get_rep(), matl_subset, Task::NormalDomain);
  task->computes(labels->electrostatic->rElectrostaticRealStress, level.get_rep(), matl_subset, Task::NormalDomain);
// We should probably actually concatenate the forces into a single
// pF_electrostatic for gating to the integrator.
// Universal



  task->computes(labels->electrostatic->pF_electroInverse_preReloc, level.get_rep(), matl_subset, Task::NormalDomain);
  task->computes(labels->electrostatic->pF_electroReal_preReloc, level.get_rep(), matl_subset, Task::NormalDomain);

  if (f_polarizable) {
    task->computes(labels->electrostatic->pMu_preReloc, level.get_rep(), matl_subset, Task::NormalDomain);
    task->computes(labels->electrostatic->rElectrostaticInverseStressDipole, level.get_rep(), matl_subset, Task::NormalDomain);
//    task->computes(labels->electrostatic->pE_electroInverse_preReloc, level.get_rep(), matl_subset, Task::NormalDomain);
//    task->computes(labels->electrostatic->pE_electroReal_preReloc, level.get_rep(), matl_subset, Task::NormalDomain);
  }


}

void SPME::addFinalizeRequirements(Task* task, MDLabel* d_label) const
{
  // Empty
}

void SPME::addFinalizeComputes(Task* task, MDLabel* d_label) const
{
  // Empty
}

void SPME::registerRequiredParticleStates(varLabelArray& particleState,
                                          varLabelArray& particleState_preReloc,
                                          MDLabel* d_label) const {

  // We absolutely need per-particle information to implement polarizable SPME
  if (f_polarizable) {
    particleState.push_back(d_label->electrostatic->pMu);
    particleState_preReloc.push_back(d_label->electrostatic->pMu_preReloc);
    
//    particleState.push_back(d_label->electrostatic->pE_electroReal);
//    particleState_preReloc.push_back(
//                          d_label->electrostatic->pE_electroReal_preReloc);
    
//    particleState.push_back(d_label->electrostatic->pE_electroInverse);
//    particleState_preReloc.push_back(
//                          d_label->electrostatic->pE_electroInverse_preReloc);
  }

  // We -probably- don't need relocatable Force information, however it may be
  // the easiest way to implement the required per-particle Force information.
  particleState.push_back(d_label->electrostatic->pF_electroInverse);
  particleState_preReloc.push_back(
                        d_label->electrostatic->pF_electroInverse_preReloc);
  particleState.push_back(d_label->electrostatic->pF_electroReal);
  particleState_preReloc.push_back(
                        d_label->electrostatic->pF_electroReal_preReloc);

  // Note:  Per particle charges may be required in some FF implementations
  //        (i.e. ReaxFF), however we will let the FF themselves register these
  //        variables if these are present and needed.

}

// Scheduling routines for the SPME subscheduler
void SPME::scheduleInitializeLocalStorage(const ProcessorGroup* pg,
                                          const PatchSet*       patches,
                                          const MaterialSet*    materials,
                                          DataWarehouse*        subOldDW,
                                          DataWarehouse*        subNewDW,
                                          const MDLabel*        label,
                                          const LevelP&         level,
                                          SchedulerP&           sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleInitializeLocalStorage");

  Task* task = scinew Task("SPME::initializeLocalStorage",
                           this,
                           &SPME::initializeLocalStorage,
                           label);

  task->setType(Task::OncePerProc);
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches = loadBal->getPerProcessorPatchSet(level);

  task->computes(label->SPME_dep->dInitializeQ);

  sched->addTask(task, perproc_patches, materials);

}

void SPME::scheduleCalculateRealspace(const ProcessorGroup*     pg,
                                      const PatchSet*           patches,
                                      const MaterialSet*        materials,
                                            DataWarehouse*      subOldDW,
                                            DataWarehouse*      subNewDW,
                                      const SimulationStateP*   sharedState,
                                      const MDLabel*            label,
                                            CoordinateSystem*   coordSys,
                                            SchedulerP&         sched,
                                            DataWarehouse*      parentOldDW)
{
    printSchedule(patches, spme_cout, "SPME::scheduleCalculateRealspace");

    /* if (ff->getPolarizableScreening() == THOLE ) {
     * task = scinew TASK("SPME::calculateRealspaceTholeDipole",
     *                    this,
     *                    &SPME::calculateRealspaceTholeDipole,
     *                    sharedState,
     *                    label,
     *                    coordSys,
     *                    parentOldDW,
     *
           ff->getPolarizableScreening() == GAUSSIAN )
    */
    Task* task;
    bool do_thole = false;
    if (f_polarizable) {
      if (!do_thole)
      {
        task = scinew Task("SPME::realspacePointDipole",
                           this,
                           &SPME::calculateRealspacePointDipole,
                           sharedState,
                           label,
                           coordSys,
                           parentOldDW
                          );

      }
      else
      {
        task = scinew Task("SPME::realspaceTholeDipole",
                           this,
                           &SPME::calculateRealspaceTholeDipole,
                           sharedState,
                           label,
                           coordSys,
                           parentOldDW
                          );
      }
      // Also requires the last iteration's dipole guess, which does change
      // for the polarizability iteration
      task->requires(Task::OldDW,
                     label->electrostatic->pMu,
                     Ghost::AroundCells,
                     d_electrostaticGhostCells);

      // And provides the field
      task->computes(label->electrostatic->pE_electroReal_preReloc);
    }
    else {
      task = scinew Task("SPME::calculateRealspace",
                         this,
                         &SPME::calculateRealspace,
                         sharedState,
                         label,
                         coordSys,
                         parentOldDW
                        );
    }

    task->requires(Task::ParentOldDW,
                   label->global->pX,
                   Ghost::AroundCells,
                   d_electrostaticGhostCells);

    task->requires(Task::ParentOldDW,
                   label->global->pID,
                   Ghost::AroundCells,
                   d_electrostaticGhostCells);

    // Computes the realspace contribution to the electrostatic field,
    // force, and stress tensor.
    task->computes(label->electrostatic->pF_electroReal_preReloc);  //Force
    task->computes(label->electrostatic->rElectrostaticRealEnergy); //Energy
    task->computes(label->electrostatic->rElectrostaticRealStress); //Stress

    sched->addTask(task, patches, materials);

}

void SPME::scheduleCalculatePretransform(const ProcessorGroup*      pg,
                                         const PatchSet*            patches,
                                         const MaterialSet*         materials,
                                               DataWarehouse*       subOldDW,
                                               DataWarehouse*       subNewDW,
                                         const SimulationStateP*    simState,
                                         const MDLabel*             label,
                                               CoordinateSystem*    coordSys,
                                               SchedulerP&          sched,
                                               DataWarehouse*       parentOldDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCalculatePreTransform");

  Task* task;
  if (f_polarizable) {
    task = scinew Task("SPME::calculatePreTransformDipole",
                       this,
                       &SPME::calculatePreTransformDipole,
                       simState,
                       label,
                       coordSys,
                       parentOldDW);

    // We need per particle dipoles to map the Charge grid
    task->requires(Task::OldDW, label->electrostatic->pMu, Ghost::None, 0);

  }
  else {
    task = scinew Task("SPME::calculatePreTransform",
                       this,
                       &SPME::calculatePreTransform,
                       simState,
                       label,
                       coordSys,
                       parentOldDW);

  }

  task->requires(Task::NewDW, label->SPME_dep->dInitializeQ);

  // Dummy dependency necessary for non-dipole case
  task->computes(label->SPME_dep->dPreTransform);

//  // Setup requires the position and ID arrays from the parent process
//  task->requires(Task::ParentOldDW, label->global->pX, Ghost::AroundNodes, d_electrostaticGhostCells);
//  task->requires(Task::ParentOldDW, label->global->pID, Ghost::AroundNodes, d_electrostaticGhostCells);

//  task->requires(Task::ParentNewDW, d_label->pXLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_lb->pChargeLabel, Ghost::AroundNodes, CUTOFF_RADIUS);
//  task->requires(Task::OldDW, d_label->pParticleIDLabel, Ghost::AroundNodes, CUTOFF_RADIUS);

//  // Computes (copies from parent to local) position and ID, sets dependency flag
//  task->computes(label->electrostatic->dSubschedulerDependency);
//  task->computes(label->global->pX);
//  task->computes(label->global->pID);



  sched->addTask(task, patches, materials);
}

void SPME::scheduleReduceNodeLocalQ(const ProcessorGroup*   pg,
                                    const PatchSet*         patches,
                                    const MaterialSet*      materials,
                                    DataWarehouse*          subOldDW,
                                    DataWarehouse*          subNewDW,
                                    const MDLabel*          label,
                                    SchedulerP&             sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleReduceNodeLocalQ");

  Task* task = scinew Task("SPME::reduceNodeLocalQ",
                           this,
                           &SPME::reduceNodeLocalQ,
                           label);

  // A lot of the subscheduled tasks work with their own memory management
  // for intermediate, node-local calculations and/or reductions.

  // As such, we need to place an artificial dependency chain to get the task
  // graph laid out correctly.
  task->requires(Task::NewDW, label->SPME_dep->dPreTransform, Ghost::None, 0);
  task->computes(label->SPME_dep->dReduceNodeLocalQ);
//  // FIXME!  Is this redundant with the following "modifies"?
//  task->requires(Task::NewDW, label->electrostatic->dSubschedulerDependency, Ghost::None, 0);
////  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
//  task->modifies(label->electrostatic->dSubschedulerDependency);
////  task->modifies(d_label->subSchedulerDependencyLabel);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleTransformRealToFourier(const ProcessorGroup* pg,
                                          const PatchSet*       patches,
                                          const MaterialSet*    materials,
                                          DataWarehouse*        subOldDW,
                                          DataWarehouse*        subNewDW,
                                          const MDLabel*        label,
                                          const LevelP&         level,
                                          SchedulerP&           sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleTransformRealToFourier");

  Task* task = scinew Task("SPME::transformRealToFourier",
                           this,
                           &SPME::transformRealToFourier,
                           label);

  // Note the different setup here:

  // Each proc has a chunk of data which is internally coordinated for the FFT
  // transformation.  Since we use FFTW with an MPI/threaded interface, we
  // manage our own data transfer, making task->usesMPI(true) necessary.

  // Finally, we need to build a patch set to pass to it which is per proc
  // instead of per patch.
  task->setType(Task::OncePerProc);
  task->usesMPI(true);
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches = loadBal->getPerProcessorPatchSet(level);

  // Ensure all nodes have populated their nodeLocalQ variable
  task->requires(Task::NewDW,
                 label->SPME_dep->dReduceNodeLocalQ,
                 Ghost::AroundNodes,
                 SHRT_MAX);
  // FIXME!  JBH - 2/2015  (Double check to make sure this is the right amount
  // of info to pull in, and the right way (nodes vs. cells) to do it.

  task->computes(label->SPME_dep->dTransformRealToFourier);
//  task->requires(Task::NewDW, label->electrostatic->dSubschedulerDependency, Ghost::None, 0);
//
//  task->modifies(label->electrostatic->dSubschedulerDependency);


  sched->addTask(task, perproc_patches, materials);
}

void SPME::scheduleCalculateInFourierSpace(const ProcessorGroup* pg,
                                           const PatchSet* patches,
                                           const MaterialSet* materials,
                                           DataWarehouse* subOldDW,
                                           DataWarehouse* subNewDW,
                                           const MDLabel* label,
                                           SchedulerP& sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCalculateInFourierSpace");

  Task* task = scinew Task("SPME::calculateInFourierSpace",
                           this,
                           &SPME::calculateInFourierSpace,
                           label);

  task->requires(Task::NewDW, label->SPME_dep->dTransformRealToFourier);
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
  task->computes(label->SPME_dep->dCalculateInFourierSpace);

  sched->addTask(task, patches, materials);

//  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
/*
  task->modifies(label->electrostatic->dSubschedulerDependency);
  task->computes(label->electrostatic->rElectrostaticInverseEnergy);
  task->computes(label->electrostatic->rElectrostaticInverseStress);
*/
//  task->modifies(d_label->subSchedulerDependencyLabel);
//  task->computes(d_label->electrostaticReciprocalEnergyLabel);
//  task->computes(d_label->electrostaticReciprocalStressLabel);

}

void SPME::scheduleTransformFourierToReal(const ProcessorGroup* pg,
                                          const PatchSet*       patches,
                                          const MaterialSet*    materials,
                                          DataWarehouse*        subOldDW,
                                          DataWarehouse*        subNewDW,
                                          const MDLabel*        label,
                                          const LevelP&         level,
                                          SchedulerP&           sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleTransformFourierToReal");

  Task* task = scinew Task("SPME::transformFourierToReal",
                           this,
                           &SPME::transformFourierToReal,
                           label);

  // Note the different setup here:

  // Each proc has a chunk of data which is internally coordinated for the FFT
  // transformation.  Since we use FFTW with an MPI/threaded interface, we
  // manage our own data transfer, making task->usesMPI(true) necessary.

  // Finally, we need to build a patch set to pass to it which is per proc
  // instead of per patch.
  task->setType(Task::OncePerProc);
  task->usesMPI(true);
  LoadBalancer* loadBal = sched->getLoadBalancer();
  const PatchSet* perproc_patches =  loadBal->getPerProcessorPatchSet(level);

  task->requires(Task::NewDW,
                 label->SPME_dep->dCalculateInFourierSpace,
                 Ghost::None,
                 0);
  task->computes(label->SPME_dep->dTransformFourierToReal);

//  task->requires(Task::NewDW, label->electrostatic->dSubschedulerDependency, Ghost::None, 0);
////  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
//  task->modifies(label->electrostatic->dSubschedulerDependency);
////  task->modifies(d_label->subSchedulerDependencyLabel);


  sched->addTask(task, perproc_patches, materials);
}

void SPME::scheduleDistributeNodeLocalQ(const ProcessorGroup*   pg,
                                        const PatchSet*         patches,
                                        const MaterialSet*      materials,
                                        DataWarehouse*          subOldDW,
                                        DataWarehouse*          subNewDW,
                                        const MDLabel*          label,
                                        SchedulerP&             sched)
{
  printSchedule(patches, spme_cout, "SPME::scheduleDistributeNodeLocalQ");

  Task* task = scinew Task("SPME::distributeNodeLocalQ-force",
                           this,
                           &SPME::distributeNodeLocalQ,
                           label);

  task->requires(Task::NewDW, label->SPME_dep->dTransformFourierToReal);
  task->computes(label->SPME_dep->dDistributeNodeLocalQ);

//  task->requires(Task::NewDW,
//                 label->electrostatic->dSubschedulerDependency,
//                 Ghost::None,
//                 0);
//
//  task->modifies(label->electrostatic->dSubschedulerDependency);
  //  task->requires(Task::NewDW, d_label->subSchedulerDependencyLabel, Ghost:: Ghost::None, 0);
//  task->modifies(d_label->subSchedulerDependencyLabel);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleUpdateFieldAndStress(const ProcessorGroup*   pg,
                                        const PatchSet*         patches,
                                        const MaterialSet*      materials,
                                              DataWarehouse*    subOldDW,
                                              DataWarehouse*    subNewDW,
                                        const MDLabel*          label,
                                              CoordinateSystem* coordSystem,
                                              SchedulerP&       sched,
                                              DataWarehouse*    parentOldDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleUpdateFieldandStress");

  Task* task = scinew Task("SPME::updateFieldAndStress",
                           this,
                           &SPME::dipoleUpdateFieldAndStress,
                           label,
                           coordSystem,
                           parentOldDW);

  // Requires the dipoles from the last iteration
  task->requires(Task::OldDW, label->electrostatic->pMu, Ghost::None, 0);
  task->requires(Task::NewDW,
                 label->SPME_dep->dDistributeNodeLocalQ,
                 Ghost::None,
                 0);

  // Calculates the new inverse space field prediction and updates the inverse space stress tensor
  task->computes(label->electrostatic->pE_electroInverse_preReloc);
  task->computes(label->electrostatic->rElectrostaticInverseStressDipole);

  sched->addTask(task, patches, materials);

}

void SPME::scheduleCheckConvergence(const ProcessorGroup*       pg,
                                    const PatchSet*             patches,
                                    const MaterialSet*          materials,
                                          DataWarehouse*        subOldDW,
                                          DataWarehouse*        subNewDW,
                                    const MDLabel*              label,
                                          SchedulerP&           sched,
                                          DataWarehouse*        parentOldDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCheckConvergence");

  Task* task = scinew Task("SPME::checkConvergence",
                           this,
                           &SPME::checkConvergence,
                           label,
                           parentOldDW);

  task->requires(Task::ParentOldDW, label->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW, label->electrostatic->pMu, Ghost::None, 0);
  task->requires(Task::NewDW, label->electrostatic->pMu, Ghost::None, 0);

  task->computes(label->electrostatic->rPolarizationDeviation);

  sched->addTask(task, patches, materials);
}

void SPME::scheduleCalculateNewDipoles(const ProcessorGroup*    pg,
                                       const PatchSet*          patches,
                                       const MaterialSet*       materials,
                                             DataWarehouse*     subOldDW,
                                             DataWarehouse*     subNewDW,
                                       const SimulationStateP*  sharedState,
                                       const MDLabel*           label,
                                             SchedulerP&        sched,
                                             DataWarehouse*     parentOldDW)
{
  printSchedule(patches, spme_cout, "SPME::scheduleCalculateNewDipoles");

  Task* task = scinew Task("SPME::calculateNewDipoles",
                           this,
                           &SPME::calculateNewDipoles,
                           sharedState,
                           label,
                           parentOldDW);

  // Requires the updated field from both the realspace and reciprocal calculation
  // Also may want the dipoles from the previous iteration
  task->requires(Task::ParentOldDW, label->global->pX, Ghost::None, 0);
  task->requires(Task::OldDW, label->electrostatic->pMu, Ghost::None, 0);
  task->requires(Task::NewDW, label->electrostatic->pE_electroReal_preReloc,
                 Ghost::None, 0);
  task->requires(Task::NewDW, label->electrostatic->pE_electroInverse_preReloc,
                 Ghost::None, 0);

  // Overwrites each dipole array at iteration n with the full estimate of dipole array at iteration n+1
  task->computes(label->electrostatic->pMu);

  sched->addTask(task, patches, materials);
}
