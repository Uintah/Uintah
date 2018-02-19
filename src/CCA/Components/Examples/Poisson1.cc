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

#include <CCA/Components/Examples/Poisson1.h>
#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/KokkosViews.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

#include <sci_defs/kokkos_defs.h>


#define NUM_EXECUTION_SPACES 2
#define EXECUTION_SPACE_0 Kokkos::Cuda
#define EXECUTION_SPACE_1 Kokkos::OpenMP
#define MEMORY_SPACE_0    Kokkos::CudaSpace
#define MEMORY_SPACE_1    Kokkos::HostSpace
#define ALL_ARCHITECTURES

#define CALL_ASSIGN_PORTABLE_TASK(FUNCTION_NAME, TASK_DEPENDENCIES, PATCHES, MATERIALS) {      \
  Task* task = scinew Task("FUNCTION_NAME",                                                    \
                           this,                                                               \
                           &FUNCTION_NAME<EXECUTION_SPACE_1, MEMORY_SPACE_1>);                 \
  TASK_DEPENDENCIES(task);                                                                     \
  if (Uintah::Parallel::usingDevice()) {                                                       \
    task->usesDevice(true);                                                                    \
  }                                                                                            \
  sched->addTask(task, PATCHES, MATERIALS);                                                    \
}

using namespace std;
using namespace Uintah;

//A sample supporting three modes of execution:
//Kokkos CPU (UINTAH_ENABLE_KOKKOS is defined, but HAVE_CUDA is not defined)
//Kokkos GPU (UINTAH_ENABLE_KOKKOS is defined and HAVE_CUDA is defined)
//Legacy Uintah CPU (UINTAH_ENABLE_KOKKOS is not defined and HAVE_CUDA is not defined)
template <typename MEMORY_SPACE>
struct TimeAdvanceFunctor {

  typedef double value_type;

  //Declare the vars
#ifdef UINTAH_ENABLE_KOKKOS //Both CPU and GPU Kokkos runs need this.
  KokkosView3<const double> m_phi;
  KokkosView3<double> m_newphi;
#else  //Just do this the legacy CPU way.
  constNCVariable<double> & m_phi;
  NCVariable<double> & m_newphi;
#endif

//Retrieve the vars
#ifdef UINTAH_ENABLE_KOKKOS
  TimeAdvanceFunctor(KokkosView3<const double> & phi,
                     KokkosView3<double> & newphi)

      : m_phi( phi )
      , m_newphi( newphi ) {}
#else
  TimeAdvanceFunctor(constNCVariable<double> & phi,
                     NCVariable<double> & newphi)
      : m_phi(phi),
        m_newphi(newphi) {}
#endif

#ifdef UINTAH_ENABLE_KOKKOS
  KOKKOS_INLINE_FUNCTION
#endif
  void operator()(const int i,
                  const int j,
                  const int k,
                  double & residual) const
  {
    m_newphi(i, j, k) = (1. / 6)
        * (m_phi(i + 1, j, k) + m_phi(i - 1, j, k) + m_phi(i, j + 1, k) +
           m_phi(i, j - 1, k) + m_phi(i, j, k + 1) + m_phi(i, j, k - 1));
    //printf("At (%d,%d,%d), m_phi is %g and m_newphi is %g\n", i, j, k, m_phi(i,j,k), m_newphi(i,j,k));
    double diff = m_newphi(i, j, k) - m_phi(i, j, k);
    residual += diff * diff;
  }
};

Poisson1::Poisson1( const ProcessorGroup   * myworld
                  , const SimulationStateP   sharedState
                  )
  : ApplicationCommon( myworld, sharedState )
{
  phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
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

  m_sharedState->registerSimpleMaterial(mymat_);
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
  sched->addTask(task, level->eachPatch(), m_sharedState->allMaterials());
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
  sched->addTask(task, level->eachPatch(), m_sharedState->allMaterials());
}

//______________________________________________________________________
//
void Poisson1::scheduleTimeAdvance( const LevelP     & level
                                  ,       SchedulerP & sched
                                  )
{



  auto TaskDependencies = [&](Task* task) {
    task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
    task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
    task->computes(residual_label);
  };
  CALL_ASSIGN_PORTABLE_TASK(Poisson1::timeAdvance, TaskDependencies, level->eachPatch(), m_sharedState->allMaterials());

  //Task* task = scinew Task("Poisson1::timeAdvance", this, &Poisson1::timeAdvance);

  //#if defined(HAVE_CUDA) && defined(UINTAH_ENABLE_KOKKOS)
  //  if (Uintah::Parallel::usingDevice()) {
  //    task->usesDevice(true);
  //  }
  //#endif
  //task->requires(Task::OldDW, phi_label, Ghost::AroundNodes, 1);
  //task->computesWithScratchGhost(phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  //task->computes(residual_label);

  //sched->addTask(task, level->eachPatch(), m_sharedState->allMaterials());

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

//______________________________________________________________________
//
template <typename EXECUTION_SPACE, typename MEMORY_SPACE>
void Poisson1::timeAdvance(DetailedTask* task,
                            Task::CallBackEvent event,
                            const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            void* old_TaskGpuDW,
                            void* new_TaskGpuDW,
                            void* stream,
                            int deviceID)
{

  int matl = 0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
//Get the data.
#if !defined(UINTAH_ENABLE_KOKKOS)
    // The legacy Uintah CPU way
    constNCVariable<double> phi;
    NCVariable<double> newphi;
    old_dw->get(phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
    new_dw->allocateAndPut(newphi, phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

#elif !defined(HAVE_CUDA) && defined(UINTAH_ENABLE_KOKKOS)
    // Grab the variables then grab the Kokkos Views
    constNCVariable<double> NC_phi;
    NCVariable<double> NC_newphi;
    old_dw->get(NC_phi, phi_label, matl, patch, Ghost::AroundNodes, 1);
    new_dw->allocateAndPut(NC_newphi, phi_label, matl, patch);
    NC_newphi.copyPatch(NC_phi, NC_newphi.getLowIndex(), NC_newphi.getHighIndex());
    KokkosView3<const double> phi = NC_phi.getKokkosView();
    KokkosView3<double> newphi = NC_newphi.getKokkosView();

#elif defined(HAVE_CUDA) && defined(UINTAH_ENABLE_KOKKOS)
    //Note: this will only work if the task's using_device is set to true.
    //This section was only put here for future boilerplace
    // Get the variables directly from the GPU Data Warehouse, we can avoid
    // creating a temporary GPUGridVariable.
    //TODO: Need to copy patch faces (or just zero them out), we don't have anything like that yet, so
    //this will pull in garbage data from halos
    KokkosView3<const double> phi = old_dw->getGPUDW()->getKokkosView<const double>(phi_label->getName().c_str(), patch->getID(),  matl, 0);
    KokkosView3<double> newphi    = new_dw->getGPUDW()->getKokkosView<double>(phi_label->getName().c_str(), patch->getID(),  matl, 0);
#endif

    //Prepare the range
    double residual = 0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);
    h -= IntVector(patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1);

    Uintah::BlockRange range(l, h);

    TimeAdvanceFunctor<MEMORY_SPACE> func(phi, newphi);
    Uintah::parallel_reduce_sum<EXECUTION_SPACE>(range, func, residual);
    //new_dw->put(sum_vartype(residual), residual_label);
  }
}
