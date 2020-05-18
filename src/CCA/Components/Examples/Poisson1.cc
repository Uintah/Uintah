/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Examples/Poisson1.h>
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


//DS 05132020: Updated Poisson1 to use CC Variable or NCVariable. Comment/uncomment USE_CC_VARS to switch between two types.
#define USE_CC_VARS


#ifdef USE_CC_VARS
typedef CCVariable<double> vartype;
typedef constCCVariable<double> constvartype;
#define POISSON1_GHOST_TYPE Ghost::AroundCells
#define getLowIndex(patch) patch->getCellLowIndex()
#define getHighIndex(patch) patch->getCellHighIndex()
#else
typedef NCVariable<double> vartype;
typedef constNCVariable<double> constvartype;
#define POISSON1_GHOST_TYPE Ghost::AroundNodes
#define getLowIndex(patch) patch->getNodeLowIndex()
#define getHighIndex(patch) patch->getNodeHighIndex()
#endif



//______________________________________________________________________
// A sample implementation supporting three modes of execution:
//   &Poisson1::timeAdvance<UINTAH_CPU_TAG>    // Task supports non-Kokkos builds and is executed serially
//   &Poisson1::timeAdvance<KOKKOS_OPENMP_TAG> // Task supports Kokkos::OpenMP builds and is executed using OpenMP via Kokkos
//   &Poisson1::timeAdvance<KOKKOS_CUDA_TAG>   // Task supports Kokkos::Cuda builds and is executed using CUDA via Kokkos

Poisson1::Poisson1( const ProcessorGroup   * myworld
                  , const MaterialManagerP   materialManager
                  )
  : ApplicationCommon( myworld, materialManager )
{
  phi_label = VarLabel::create("phi", vartype::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

//______________________________________________________________________
//
Poisson1::~Poisson1()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(residual_label);
}

//______________________________________________________________________
//
void Poisson1::problemSetup( const ProblemSpecP & params
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
void Poisson1::scheduleInitialize( const LevelP     & level
                                 ,       SchedulerP & sched
                                 )
{
  Task* task = scinew Task("Poisson1::initialize", this, &Poisson1::initialize);

  task->computes(phi_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void Poisson1::scheduleRestartInitialize( const LevelP     & level
                                        ,       SchedulerP & sched
                                        )
{
}

//______________________________________________________________________
//
void Poisson1::scheduleComputeStableTimeStep( const LevelP     & level
                                            ,       SchedulerP & sched
                                            )
{
  Task* task = scinew Task("Poisson1::computeStableTimeStep", this, &Poisson1::computeStableTimeStep);

  task->requires(Task::NewDW, residual_label);
  task->computes(getDelTLabel(), level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void Poisson1::scheduleTimeAdvance( const LevelP     & level
                                  ,       SchedulerP & sched
                                  )
{
//______________________________________________________________________
// Legacy approach:

//  Task* task = scinew Task("Poisson1::timeAdvance", this, &Poisson1::timeAdvance);

//  task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
//  task->computes(phi_label);
//  task->computes(residual_label);
//  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());

//______________________________________________________________________
// Portable approach:

  auto TaskDependencies = [&](Task* task) {
    task->requires(Task::OldDW, phi_label, POISSON1_GHOST_TYPE, 1);
    task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, POISSON1_GHOST_TYPE, 1);
    task->computes(residual_label);
  };

  create_portable_tasks(TaskDependencies, this,
                        "Poisson1::timeAdvance",
                        &Poisson1::timeAdvance<UINTAH_CPU_TAG>,
                        &Poisson1::timeAdvance<KOKKOS_OPENMP_TAG>,
                        &Poisson1::timeAdvance<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

//______________________________________________________________________
//
void Poisson1::computeStableTimeStep( const ProcessorGroup * pg
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
void Poisson1::initialize( const ProcessorGroup *
                         , const PatchSubset    * patches
                         , const MaterialSubset * matls
                         ,       DataWarehouse  * /*old_dw*/
                         ,       DataWarehouse  * new_dw
                         )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    vartype phi;
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
void Poisson1::timeAdvance( const PatchSubset                          * patches
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

    double residual = 0;

    // Prepare the ranges for both boundary conditions and main loop
    IntVector l = getLowIndex(patch);
    IntVector h = getHighIndex(patch);

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector( patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1
                  );

    h -= IntVector( patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1
                  );

    Uintah::BlockRange range( l, h );

    auto phi = old_dw->getConstGridVariable<constvartype, double, MemSpace> (phi_label, matl, patch, POISSON1_GHOST_TYPE, 1);
    auto newphi = new_dw->getGridVariable<vartype, double, MemSpace> (phi_label, matl, patch);

    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    //Uintah::parallel_for<ExecSpace, LaunchBounds< 640,1 > >( execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      newphi(i, j, k) = phi(i,j,k);
    });

    // Perform the main loop
    Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA (int i, int j, int k, double& residual){
      newphi(i, j, k) = ( 1. / 6 ) *
                        ( phi(i + 1, j, k) + phi(i - 1, j, k) + phi(i, j + 1, k) +
                          phi(i, j - 1, k) + phi(i, j, k + 1) + phi(i, j, k - 1) );


//      printf("In lambda CUDA at %d,%d,%d), m_phi is at %p %p %g from %g, %g, %g, %g, %g, %g and m_newphi is %g\n", i, j, k,
//             phi.m_view.data(), &(phi(i,j,k)),
//             phi(i,j,k),
//             phi(i + 1, j, k), phi(i - 1, j, k), phi(i, j + 1, k),
//             phi(i, j - 1, k), phi(i, j, k + 1), phi(i, j, k - 1),
//             newphi(i,j,k));

      double diff = newphi(i, j, k) - phi(i, j, k);
      residual += diff * diff;
    }, residual);

  }
}
