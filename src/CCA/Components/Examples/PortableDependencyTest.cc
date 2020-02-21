/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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

#include <CCA/Components/Examples/PortableDependencyTest.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/Parallel/Portability.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/KokkosViews.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace std;
using namespace Uintah;

class DetailedTask;

PortableDependencyTest::PortableDependencyTest( const ProcessorGroup   * myworld
                                              , const MaterialManagerP   materialManager
                                              )
  : ApplicationCommon( myworld, materialManager )
{
  phi_label      = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

//______________________________________________________________________
//
PortableDependencyTest::~PortableDependencyTest()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}

//______________________________________________________________________
//
void PortableDependencyTest::problemSetup( const ProblemSpecP & params
                                         , const ProblemSpecP & restart_prob_spec
                                         ,       GridP        & /*grid*/
                                         )
{
  ProblemSpecP poisson = params->findBlock("Poisson");

  poisson->require("delt", delt_);

  mymat_ = scinew SimpleMaterial();

  m_materialManager->registerSimpleMaterial(mymat_);
}

//______________________________________________________________________
//
void PortableDependencyTest::scheduleInitialize( const LevelP     & level
                                               ,       SchedulerP & sched
                                               )
{
  Task* task = scinew Task("PortableDependencyTest::initialize", this, &PortableDependencyTest::initialize);

  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void PortableDependencyTest::scheduleRestartInitialize( const LevelP     & level
                                                      ,       SchedulerP & sched
                                                      )
{
}

//______________________________________________________________________
//
void PortableDependencyTest::scheduleComputeStableTimeStep( const LevelP     & level
                                                          ,       SchedulerP & sched
                                                          )
{
  Task* task = scinew Task("PortableDependencyTest::computeStableTimeStep", this, &PortableDependencyTest::computeStableTimeStep);

  task->requires(Task::NewDW, residual_label);
  task->computes(getDelTLabel(), level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void PortableDependencyTest::scheduleTimeAdvance( const LevelP     & level
                                                ,       SchedulerP & sched
                                                )
{
  scheduleTask1Computes( level, sched );
  scheduleTask2Modifies( level, sched );
  scheduleTask3Modifies( level, sched );
  scheduleTask4Requires( level, sched );
}

void PortableDependencyTest::scheduleTask1Computes( const LevelP     & level
                                                  ,       SchedulerP & sched
                                                  )
{
  auto TaskDependencies = [&](Task* task) {
	task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  };

  create_portable_tasks(TaskDependencies, this,
                        "PortableDependencyTest::task1Computes",
                        &PortableDependencyTest::task1Computes<UINTAH_CPU_TAG>,
                        &PortableDependencyTest::task1Computes<KOKKOS_OPENMP_TAG>,
                        &PortableDependencyTest::task1Computes<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

void PortableDependencyTest::scheduleTask2Modifies( const LevelP     & level
                                                  ,       SchedulerP & sched
                                                  )
{
  const Uintah::PatchSubset* const localPatches = level->allPatches()->getSubset( Uintah::Parallel::getMPIRank());

  auto TaskDependencies = [&](Task* task) {
    task->modifiesWithScratchGhost(phi_label, localPatches, Uintah::Task::ThisLevel, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  };

  create_portable_tasks(TaskDependencies, this,
                        "PortableDependencyTest::task2Modifies",
                        &PortableDependencyTest::task2Modifies<UINTAH_CPU_TAG>,
                        &PortableDependencyTest::task2Modifies<KOKKOS_OPENMP_TAG>,
                        &PortableDependencyTest::task2Modifies<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

void PortableDependencyTest::scheduleTask3Modifies( const LevelP     & level
                                                  ,       SchedulerP & sched
                                                  )
{
  const Uintah::PatchSubset* const localPatches = level->allPatches()->getSubset( Uintah::Parallel::getMPIRank());

  auto TaskDependencies = [&](Task* task) {
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->modifiesWithScratchGhost(phi_label, localPatches, Uintah::Task::ThisLevel, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
    task->computes(residual_label);
  };

  create_portable_tasks(TaskDependencies, this,
                        "PortableDependencyTest::task3Modifies",
                        &PortableDependencyTest::task3Modifies<UINTAH_CPU_TAG>,
                        &PortableDependencyTest::task3Modifies<KOKKOS_OPENMP_TAG>,
                        &PortableDependencyTest::task3Modifies<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

void PortableDependencyTest::scheduleTask4Requires( const LevelP     & level
                                                  ,       SchedulerP & sched
                                                  )
{
  auto Task4Dependencies = [&](Task* task4) {
    task4->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  };

  create_portable_tasks(Task4Dependencies, this,
                        "PortableDependencyTest::task4Requires",
                        &PortableDependencyTest::task4Requires<UINTAH_CPU_TAG>,
                        &PortableDependencyTest::task4Requires<KOKKOS_OPENMP_TAG>,
                        &PortableDependencyTest::task4Requires<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

//______________________________________________________________________
//
void PortableDependencyTest::computeStableTimeStep( const ProcessorGroup * pg
                                                  , const PatchSubset    * patches
                                                  , const MaterialSubset * /*matls*/
                                                  ,       DataWarehouse  *
                                                  ,       DataWarehouse  * new_dw
                                                  )
{
  if (pg->myRank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, residual_label);
  }
  new_dw->put(delt_vartype(delt_), getDelTLabel(), getLevel(patches));
}

//______________________________________________________________________
//
void PortableDependencyTest::initialize( const ProcessorGroup *
                                       , const PatchSubset    * patches
                                       , const MaterialSubset * matls
                                       ,       DataWarehouse  * /*old_dw*/
                                       ,       DataWarehouse  * new_dw
                                       )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face)) {

      if (patch->getBCType(face) == Patch::None) {
        int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < numChildren; child++) {
          Iterator nbound_ptr, nu;

          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);

          const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*>(bcb);
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++) {
            phi[*nbound_ptr] = value;
          }
          delete bcb;
        }
      }
    }
    new_dw->put(sum_vartype(-1), residual_label);
  }
}

//______________________________________________________________________
//
template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest::task1Computes( const PatchSubset                          * patches
                                          , const MaterialSubset                       * matls
                                          ,       OnDemandDataWarehouse                * old_dw
                                          ,       OnDemandDataWarehouse                * new_dw
                                          ,       UintahParams                         & uintahParams
                                          ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                          )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    // Prepare the ranges for both boundary conditions and main loop
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    Uintah::BlockRange range( l, h );
    auto phi = old_dw->getConstNCVariable<double, MemSpace> (phi_label, matl, patch, Ghost::AroundNodes, 1);
    auto newphi = new_dw->getNCVariable<double, MemSpace> (phi_label, matl, patch);
    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      newphi(i, j, k) = phi(i,j,k);
    });

    // Perform the main loop
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
      //newphi(i, j, k) = i + j * 0.17 + k * 0.42;
      newphi(i, j, k) = phi(i,j,k);
    });
  }
}

//______________________________________________________________________
//
template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest::task2Modifies( const PatchSubset                          * patches
                                          , const MaterialSubset                       * matls
                                          ,       OnDemandDataWarehouse                * old_dw
                                          ,       OnDemandDataWarehouse                * new_dw
                                          ,       UintahParams                         & uintahParams
                                          ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                          )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    // Prepare the ranges for both boundary conditions and main loop
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    Uintah::BlockRange range( l, h );
    auto newphi = new_dw->getNCVariable<double, MemSpace> (phi_label, matl, patch);
    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      //newphi(i, j, k) = newphi(i, j, k) - (i + j * 0.17 + k * 0.42);
    });

    // Perform the main loop
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
      newphi(i, j, k) = newphi(i, j, k) - (i + j * 0.17 + k * 0.42);
    });
  }
}

//______________________________________________________________________
//
template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest::task3Modifies( const PatchSubset                          * patches
                                          , const MaterialSubset                       * matls
                                          ,       OnDemandDataWarehouse                * old_dw
                                          ,       OnDemandDataWarehouse                * new_dw
                                          ,       UintahParams                         & uintahParams
                                          ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                          )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    // Prepare the ranges for both boundary conditions and main loop
    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    Uintah::BlockRange range( l, h );
    auto phi = old_dw->getConstNCVariable<double, MemSpace> (phi_label, matl, patch, Ghost::AroundNodes, 1);
    auto newphi = new_dw->getNCVariable<double, MemSpace> (phi_label, matl, patch);

    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    //Uintah::parallel_for<ExecSpace, LaunchBounds< 640,1 > >( execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      newphi(i, j, k) = phi(i,j,k);
    });

    // Perform the main loop
    Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA (int i, int j, int k, double& residual){
      newphi(i, j, k) = (1. / 6)
          * (phi(i + 1, j, k) + phi(i - 1, j, k) + phi(i, j + 1, k) +
              phi(i, j - 1, k) + phi(i, j, k + 1) + phi(i, j, k - 1));

//      printf("[task3Modifies] newphi[%d,%d,%d]: %f (%p) (%p) phi[%d,%d,%d] (%p): %f phi[%d,%d,%d] (%p): %f phi[%d,%d,%d] (%p): %f phi[%d,%d,%d] (%p): %f phi[%d,%d,%d]: %f phi[%d,%d,%d]: %f\n"
//            , i, j, k, newphi(i,j,k), &(newphi(i,j,k)), &(newphi(2,0,0))
//            , i + 1, j, k, &(phi(i + 1, j, k)), phi(i + 1, j, k)
//            , i - 1, j, k, &(phi(i - 1, j, k)), phi(i - 1, j, k)
//            , i, j + 1, k, &(phi(i, j + 1, k)), phi(i, j + 1, k)
//            , i, j - 1, k, &(phi(i, j - 1, k)),  phi(i, j - 1, k)
//            , i, j, k + 1, phi(i, j, k + 1)
//            , i, j, k - 1, phi(i, j, k - 1)
//            );

      //  printf("[task3Modifies] newphi[%d,%d,%d]: %g\n", i, j, k, newphi(i,j,k));
      double diff = newphi(i, j, k) - phi(i, j, k);
      residual += diff * diff;
    }, residual);
  }
}

//______________________________________________________________________
//
template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest::task4Requires( const PatchSubset                          * patches
                                          , const MaterialSubset                       * matls
                                          ,       OnDemandDataWarehouse                * old_dw
                                          ,       OnDemandDataWarehouse                * new_dw
                                          ,       UintahParams                         & uintahParams
                                          ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                          )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    // Prepare the ranges for both boundary conditions and main loop
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    Uintah::BlockRange range( l, h );
    auto phi = old_dw->getConstNCVariable<double, MemSpace> (phi_label, matl, patch, Ghost::AroundNodes, 1);

    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      //printf("[task4Requires] newphi[%d,%d,%d]: %f\n", i, j, k, newphi(i,j,k));
    });

    // Perform the main loop
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
      //if (i == 17 && j == 4 && k == 4) {
      //  printf("[task4Requires] phi[%d,%d,%d]: %g\n", i, j, k, phi(i,j,k));
      //}
    });
  }
}
