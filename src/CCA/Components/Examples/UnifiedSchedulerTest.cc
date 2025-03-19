/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <CCA/Components/Examples/UnifiedSchedulerTest.h>
#include <CCA/Components/Examples/ExamplesLabel.h>

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace Uintah;

UnifiedSchedulerTest::UnifiedSchedulerTest( const ProcessorGroup   * myworld
                                          , const MaterialManagerP   materialManager
                                          )
  : ApplicationCommon(myworld, materialManager)
{
  m_phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
  m_residual_label = VarLabel::create("residual", sum_vartype::getTypeDescription());
}

UnifiedSchedulerTest::~UnifiedSchedulerTest()
{
  VarLabel::destroy(m_phi_label);
  VarLabel::destroy(m_residual_label);
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::problemSetup( const ProblemSpecP & params
                                       , const ProblemSpecP & restart_prob_spec
                                       ,       GridP        & grid
                                       )
{
  ProblemSpecP ps = params->findBlock("UnifiedSchedulerTest");
  ps->require("delt", m_delt);
  m_simple_material = scinew SimpleMaterial();
  m_materialManager->registerSimpleMaterial(m_simple_material);
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleInitialize( const LevelP     & level
                                             ,       SchedulerP & sched
                                             )
{
  Task* task = scinew Task("UnifiedSchedulerTest::initialize", this, &UnifiedSchedulerTest::initialize);

  task->computesWithScratchGhost(m_phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  task->computes(m_residual_label);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleRestartInitialize( const LevelP     & level
                                                    ,       SchedulerP & sched
                                                    )
{
    // nothing to implement
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleComputeStableTimeStep( const LevelP     & level
                                                        ,       SchedulerP & sched
                                                        )
{
  Task* task = scinew Task("UnifiedSchedulerTest::computeStableTimeStep", this, &UnifiedSchedulerTest::computeStableTimeStep);

  task->needsLabel(Task::NewDW, m_residual_label);
  task->computes(getDelTLabel(), level.get_rep());
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::scheduleTimeAdvance( const LevelP     & level
                                              ,       SchedulerP & sched
                                              )
{
  Task* task = scinew Task("UnifiedSchedulerTest::timeAdvance", this, &UnifiedSchedulerTest::timeAdvance<UintahSpaces::GPU, UintahSpaces::DeviceSpace>);
//  Task* task = scinew Task("UnifiedSchedulerTest::timeAdvance1DP"    , this, &UnifiedSchedulerTest::timeAdvance1DP);
//  Task* task = scinew Task("UnifiedSchedulerTest::timeAdvance3DP"    , this, &UnifiedSchedulerTest::timeAdvance3DP);

  task->needsLabel(Task::OldDW, m_phi_label, Ghost::AroundNodes, 1);
  task->computesWithScratchGhost(m_phi_label, nullptr, Uintah::Task::NormalDomain, Ghost::AroundNodes, 1);
  task->computes(m_residual_label);
  sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::computeStableTimeStep( const ProcessorGroup * pg
                                                , const PatchSubset    * patches
                                                , const MaterialSubset * matls
                                                ,       DataWarehouse  * old_dw
                                                ,       DataWarehouse  * new_dw
                                                )
{
  if (pg->myRank() == 0) {
    sum_vartype residual;
    new_dw->get(residual, m_residual_label);
  }
  new_dw->put(delt_vartype(m_delt), getDelTLabel(), getLevel(patches));
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::initialize( const ProcessorGroup * pg
                                     , const PatchSubset    * patches
                                     , const MaterialSubset * matls
                                     ,       DataWarehouse  * /*old_dw*/
                                     ,       DataWarehouse  * new_dw
                                     )
{
  int matl = 0;
  for (int p = 0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    NCVariable<double> phi;
    new_dw->allocateAndPut(phi, m_phi_label, matl, patch, Ghost::AroundNodes, 1);
    phi.initialize(0.);

    for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face)) {
      if (patch->getBCType(face) == Patch::None) {
        int num_children = patch->getBCDataArray(face)->getNumberChildren(matl);
        for (int child = 0; child < num_children; ++child) {
          Iterator nbound_ptr, nu;
          const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);
          const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*>(bcb);
          double value = bc->getValue();
          for (nbound_ptr.reset(); !nbound_ptr.done(); ++nbound_ptr) {
            phi[*nbound_ptr] = value;
          }
          delete bcb;
        }
      }
    }
    new_dw->put(sum_vartype(-1), m_residual_label);
  }
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvance1DP( const ProcessorGroup * pg
                                         , const PatchSubset    * patches
                                         , const MaterialSubset * matls
                                         ,       DataWarehouse  * old_dw
                                         ,       DataWarehouse  * new_dw
                                         )
{

  int matl = 0;
  int ghostLayers = 1;

  // Do time steps
  int num_patches = patches->size();
  for (int p = 0; p < num_patches; ++p) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, m_phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, m_phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0.0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);

    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor ? 0 : 1);

    //__________________________________
    // 1D-Pointer Stencil
    double* phi_data = (double*)phi.getWindow()->getData()->getPointer();
    double* newphi_data = (double*)newphi.getWindow()->getData()->getPointer();

    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();
    int ghostLayers = 1;
    int ystride = yhigh + ghostLayers;
    int xstride = xhigh + ghostLayers;

    for (int k = l.z(); k < zhigh; k++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int i = l.x(); i < xhigh; i++) {

          // For an array of [ A ][ B ][ C ], we can index it thus:
          //          (a * B * C) + (b * C) + (c * 1)

          int idx = i + (j * xstride) + (k * xstride * ystride);

          int xminus = (i - 1) + (j * xstride) + (k * xstride * ystride);
          int xplus  = (i + 1) + (j * xstride) + (k * xstride * ystride);
          int yminus = i + ((j - 1) * xstride) + (k * xstride * ystride);
          int yplus  = i + ((j + 1) * xstride) + (k * xstride * ystride);
          int zminus = i + (j * xstride) + ((k - 1) * xstride * ystride);
          int zplus  = i + (j * xstride) + ((k + 1) * xstride * ystride);

          newphi_data[idx] = (1. / 6)
              * (phi_data[xminus] + phi_data[xplus] + phi_data[yminus] + phi_data[yplus] + phi_data[zminus] + phi_data[zplus]);

          double diff = newphi_data[idx] - phi_data[idx];
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), m_residual_label);
  } // end patch  loop
}

//______________________________________________________________________
//
void UnifiedSchedulerTest::timeAdvance3DP( const ProcessorGroup * pg
                                         , const PatchSubset    * patches
                                         , const MaterialSubset * matls
                                         , DataWarehouse        * old_dw
                                         , DataWarehouse        * new_dw
                                         )
{

  int matl = 0;
  int ghostLayers = 1;

  // Do time steps
  int num_patches = patches->size();
  for (int p = 0; p < num_patches; ++p) {
    const Patch* patch = patches->get(p);
    constNCVariable<double> phi;
    old_dw->get(phi, m_phi_label, matl, patch, Ghost::AroundNodes, ghostLayers);

    NCVariable<double> newphi;
    new_dw->allocateAndPut(newphi, m_phi_label, matl, patch);
    newphi.copyPatch(phi, newphi.getLowIndex(), newphi.getHighIndex());

    double residual = 0.0;
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    l += IntVector(patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1);

    h -= IntVector(patch->getBCType(Patch::xplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::yplus)  == Patch::Neighbor ? 0 : 1,
                   patch->getBCType(Patch::zplus)  == Patch::Neighbor ? 0 : 1);

    //__________________________________
    //  3D-Pointer Stencil
    int zhigh = h.z();
    int yhigh = h.y();
    int xhigh = h.x();

    for (int i = l.z(); i < zhigh; i++) {
      for (int j = l.y(); j < yhigh; j++) {
        for (int k = l.x(); k < xhigh; k++) {

          double xminus = phi(i - 1, j, k);
          double xplus  = phi(i + 1, j, k);
          double yminus = phi(i, j - 1, k);
          double yplus  = phi(i, j + 1, k);
          double zminus = phi(i, j, k - 1);
          double zplus  = phi(i, j, k + 1);

          newphi(i, j, k) = (1. / 6) * (xminus + xplus + yminus + yplus + zminus + zplus);

          double diff = newphi(i, j, k) - phi(i, j, k);
          residual += diff * diff;
        }
      }
    }
    new_dw->put(sum_vartype(residual), m_residual_label);
  }  // end patch  loop
}
