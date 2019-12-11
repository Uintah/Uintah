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

//Poisson1 modified with dummy tasks to test resizing problem on GPU: tasks with different ghost cells
//density is a dummy label

#include <CCA/Components/Examples/GPUResizeTest1.h>
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

//test GPU resizing variable case

GPUResizeTest1::GPUResizeTest1( const ProcessorGroup   * myworld
                  , const MaterialManagerP   materialManager
                  )
  : ApplicationCommon( myworld, materialManager )
{
  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  density_label = VarLabel::create("density", NCVariable<double>::getTypeDescription());
  residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

//______________________________________________________________________
//
GPUResizeTest1::~GPUResizeTest1()
{
  VarLabel::destroy(phi_label);
  VarLabel::destroy(density_label);
  VarLabel::destroy(residual_label);
}

//______________________________________________________________________
//
void GPUResizeTest1::problemSetup( const ProblemSpecP & params
                           , const ProblemSpecP & restart_prob_spec
                           ,       GridP        & /*grid*/
                           )
{
  ProblemSpecP GPUResizeTest = params->findBlock("Poisson");	//not chaning ups file format

  GPUResizeTest->require("delt", delt_);

  mymat_ = scinew SimpleMaterial();

  m_materialManager->registerSimpleMaterial(mymat_);
}

//______________________________________________________________________
//
void GPUResizeTest1::scheduleInitialize( const LevelP     & level
                                 ,       SchedulerP & sched
                                 )
{
  Task* task = scinew Task("GPUResizeTest1::initialize", this, &GPUResizeTest1::initialize);

  task->computes(phi_label, nullptr, Uintah::Task::NormalDomain);
  task->computes(density_label);
  task->computes(residual_label);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void GPUResizeTest1::scheduleRestartInitialize( const LevelP     & level
                                        ,       SchedulerP & sched
                                        )
{
}

//______________________________________________________________________
//
void GPUResizeTest1::scheduleComputeStableTimeStep( const LevelP     & level
                                            ,       SchedulerP & sched
                                            )
{
  Task* task = scinew Task("GPUResizeTest1::computeStableTimeStep", this, &GPUResizeTest1::computeStableTimeStep);

  task->requires(Task::NewDW, residual_label);
  task->computes(getDelTLabel(), level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void GPUResizeTest1::scheduleTimeAdvance( const LevelP     & level
                                  ,       SchedulerP & sched
                                  )
{
  //part 1

  auto TaskDependencies = [&](Task* task) {
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->computes(phi_label, nullptr, Uintah::Task::NormalDomain);
    //task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 2);
    task->computes(residual_label);
  };

  create_portable_tasks(TaskDependencies, this,
                        "GPUResizeTest1::timeAdvance",
                        &GPUResizeTest1::timeAdvance<UINTAH_CPU_TAG>,
                        &GPUResizeTest1::timeAdvance<KOKKOS_OPENMP_TAG>,
                        &GPUResizeTest1::timeAdvance<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);



   //part 2
  auto TaskDependencies1 = [&](Task* task) {
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 2);
    task->computes(density_label, nullptr, Uintah::Task::NormalDomain);
  };

  create_portable_tasks(TaskDependencies1, this,
                        "GPUResizeTest1::timeRequires",
                        &GPUResizeTest1::timeAdvance1<UINTAH_CPU_TAG>,
                        &GPUResizeTest1::timeAdvance1<KOKKOS_OPENMP_TAG>,
                        &GPUResizeTest1::timeAdvance1<KOKKOS_CUDA_TAG>,
                        sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);


}

//______________________________________________________________________
//
void GPUResizeTest1::computeStableTimeStep( const ProcessorGroup * pg
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
void GPUResizeTest1::initialize( const ProcessorGroup *
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
    NCVariable<double> density;
    new_dw->allocateAndPut(phi, phi_label, matl, patch);
    new_dw->allocateAndPut(density, density_label, matl, patch);
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
void GPUResizeTest1::timeAdvance( const PatchSubset* patches,
                            const MaterialSubset* matls,
                            OnDemandDataWarehouse* old_dw,
                            OnDemandDataWarehouse* new_dw,
                            UintahParams& uintahParams,
                            ExecutionObject<ExecSpace, MemSpace>& execObj )
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

      double diff = newphi(i, j, k) - phi(i, j, k);
      residual += diff * diff;
    }, residual);
  }
}


//______________________________________________________________________
//
template <typename ExecSpace, typename MemSpace>
void GPUResizeTest1::timeAdvance1( const PatchSubset* patches,
                            const MaterialSubset* matls,
                            OnDemandDataWarehouse* old_dw,
                            OnDemandDataWarehouse* new_dw,
                            UintahParams& uintahParams,
                            ExecutionObject<ExecSpace, MemSpace>& execObj )
{

  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    // Prepare the ranges for both boundary conditions and main loop
    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 2,
                  patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 2,
                  patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 2);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 2,
                  patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 2,
                  patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 2);

    Uintah::BlockRange range( l, h );
    auto phi = old_dw->getConstNCVariable<double, MemSpace> (phi_label, matl, patch, Ghost::AroundNodes, 2);
    auto newphi = new_dw->getNCVariable<double, MemSpace> (density_label, matl, patch);
    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    //Uintah::parallel_for<ExecSpace, LaunchBounds< 640,1 > >( execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
        newphi(i, j, k) = phi(i,j,k);
    });

    // Perform the main loop
    Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA (int i, int j, int k, double& residual){
      newphi(i, j, k) = (1. / 6)
          * ( phi(i + 1, j, k) + phi(i - 1, j, k) + phi(i, j + 1, k) +
              phi(i, j - 1, k) + phi(i, j, k + 1) + phi(i, j, k - 1) +
			  phi(i + 2, j, k) + phi(i - 2, j, k) + phi(i, j + 2, k) +
			  phi(i, j - 2, k) + phi(i, j, k + 2) + phi(i, j, k - 2) );

      double diff = newphi(i, j, k) - phi(i, j, k);
      residual += diff * diff;
    }, residual);
  }
}
