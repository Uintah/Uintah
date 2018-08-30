/*
 * DiscreteInterface.cc
 *
 *  Created on: Feb 18, 2017
 *      Author: jbhooper
 *
 *
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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/SimpleDiffusionContact.h>
using namespace Uintah;

SimpleSDInterface::SimpleSDInterface(
                                         ProblemSpecP     & probSpec  ,
                                         MaterialManagerP & simState  ,
                                         MPMFlags         * mFlags    ,
                                         MPMLabel         * mpmLabel
                                        )
                                        : SDInterfaceModel(probSpec, simState,
                                                           mFlags, mpmLabel)

{

}

SimpleSDInterface::~SimpleSDInterface()
{
}

void SimpleSDInterface::addComputesAndRequiresInterpolated(
                                                                   SchedulerP   & sched   ,
                                                             const PatchSet     * patches ,
                                                             const MaterialSet  * matls
                                                            )
{
  // Shouldn't need to directly modify the concentration.
}

void SimpleSDInterface::sdInterfaceInterpolated(
                                                  const ProcessorGroup  *         ,
                                                  const PatchSubset     * patches ,
                                                  const MaterialSubset  * matls   ,
                                                        DataWarehouse   * old_dw  ,
                                                        DataWarehouse   * new_dw
                                                 )
{

}

void SimpleSDInterface::addComputesAndRequiresDivergence(
                                                                 SchedulerP   & sched,
                                                           const PatchSet     * patches,
                                                           const MaterialSet  * matls
                                                          )
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  Task* task  = scinew Task("SimpleSDInterface::sdInterfaceDivergence", this,
                            &SimpleSDInterface::sdInterfaceDivergence);

  // Everthing needs to register to calculate the basic interface flux rate
  setBaseComputesAndRequiresDivergence(task, matls->getUnion());

  task->requires(Task::OldDW, d_mpm_lb->delTLabel);

  task->requires(Task::NewDW, d_mpm_lb->gVolumeLabel,                  gnone);
  task->requires(Task::NewDW, d_mpm_lb->gVolumeLabel,
                 d_materialManager->getAllInOneMatls(), Task::OutOfDomain, gnone);
  task->requires(Task::NewDW, d_mpm_lb->diffusion->gConcentration,     gnone);
  task->requires(Task::NewDW, d_mpm_lb->diffusion->gConcentration,
                 d_materialManager->getAllInOneMatls(), Task::OutOfDomain, gnone);

//  task->computes(sdInterfaceRate, mss);

  sched->addTask(task, patches, matls);
}

void SimpleSDInterface::sdInterfaceDivergence(
                                                const ProcessorGroup  *         ,
                                                const PatchSubset     * patches ,
                                                const MaterialSubset  * matls   ,
                                                      DataWarehouse   * oldDW   ,
                                                      DataWarehouse   * newDW
                                               )
{
  int numMatls = matls->size();

  delt_vartype delT;
  oldDW->get(delT, d_mpm_lb->delTLabel, getLevel(patches) );
  double delTInv = 1.0/delT;

  for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx)
  {
    const Patch* patch = patches->get(patchIdx);
    int totalDWI = d_materialManager->getAllInOneMatls()->get(0);

    constNCVariable<double>               gTotalVolume;
    newDW->get(gTotalVolume, d_mpm_lb->gVolumeLabel, totalDWI, patch,
               d_mpm_flags->d_particle_ghost_type,
               d_mpm_flags->d_particle_ghost_layer);

    std::vector<constNCVariable<double> > gVolume(numMatls);
    std::vector<constNCVariable<double> > gConc(numMatls);  // C_i,j = gConc[j][i]

    std::vector<NCVariable<double> > gdCdt_interface(numMatls);
    NCVariable<double> gdCdt_interface_total;
    newDW->allocateAndPut(gdCdt_interface_total,  sdInterfaceRate,  totalDWI, patch);
    gdCdt_interface_total.initialize(0.0);

    std::vector<NCVariable<int>    > gInterfaceFlag(numMatls);
    NCVariable<double> gInterfaceFlagGlobal;
    newDW->allocateAndPut(gInterfaceFlagGlobal,   sdInterfaceFlag,  totalDWI, patch);
    gInterfaceFlagGlobal.initialize(false);
    // Loop over materials and preload our arrays of material based nodal data
    std::vector<NCVariable<double> > gFluxAmount(numMatls);

    std::vector<double> massNormFactor(numMatls);

    for (int mIdx = 0; mIdx < numMatls; ++mIdx) {
      int dwi = matls->get(mIdx);
      newDW->get(gVolume[mIdx], d_mpm_lb->gVolumeLabel,              dwi, patch,
                 d_mpm_flags->d_particle_ghost_type,
                 d_mpm_flags->d_particle_ghost_layer);
      newDW->get(gConc[mIdx],   d_mpm_lb->diffusion->gConcentration, dwi, patch,
                 d_mpm_flags->d_particle_ghost_type,
                 d_mpm_flags->d_particle_ghost_layer);

      newDW->allocateAndPut(gdCdt_interface[mIdx], sdInterfaceRate, dwi, patch,
                            d_mpm_flags->d_particle_ghost_type,
                            d_mpm_flags->d_particle_ghost_layer);
      
      gdCdt_interface[mIdx].initialize(0);

      newDW->allocateAndPut(gInterfaceFlag[mIdx],  sdInterfaceFlag, dwi, patch,
                            d_mpm_flags->d_particle_ghost_type,
                            d_mpm_flags->d_particle_ghost_layer);
      gInterfaceFlag[mIdx].initialize(false);
      // Not in place yet, but should be for mass normalization.
      //massNormFactor[mIdx] = mpm_matl->getMassNormFactor();
      massNormFactor[mIdx] = 1.0;

      newDW->allocateTemporary(gFluxAmount[mIdx], patch);
      gFluxAmount[mIdx].initialize(0.0);

      for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
        IntVector node = *nIt;
        gFluxAmount[mIdx][node] = gVolume[mIdx][node] * massNormFactor[mIdx];
      }
    }

    // Determine nodes on an interfaace
    const double minPresence = 1e-100; // Min volume for material presence
    for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
      IntVector node            = *nIt;
      int       materialIndex   = 0;
      bool      interfaceFound  = false;
      while ((materialIndex < numMatls) && !interfaceFound) {
        double checkVolume = gVolume[materialIndex][node];
        if ((checkVolume > minPresence) && checkVolume != gTotalVolume[node] ) {
          interfaceFound = true;
        }
        ++materialIndex;
      }
      // Tag interface on material grid for all materials present
      if (interfaceFound) {
        gInterfaceFlagGlobal[node] = true;
        for (materialIndex = 0; materialIndex < numMatls; ++materialIndex) {
          if (gVolume[materialIndex][node] > minPresence) {
            gInterfaceFlag[materialIndex][node] = true;
          }
        }
      }
    }

    for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
      double numerator    = 0.0;
      double denominator  = 0.0;
      IntVector node = *nIt;
      if (gInterfaceFlagGlobal[node] == true) {
        for (int mIdx = 0; mIdx < numMatls; ++mIdx) {
          if (gInterfaceFlag[mIdx][node] == true) {
            numerator += (gConc[mIdx][node] * gFluxAmount[mIdx][node]);
            denominator += gFluxAmount[mIdx][node];
          }
        }
        double contactConcentration = numerator/denominator;
      // FIXME TODO -- Ensure gConc has already had the mass factor divided out
      // by now. -- JBH
        for (int mIdx = 0; mIdx < numMatls; ++mIdx) {
          double dCdt = delTInv *(contactConcentration - gConc[mIdx][node]);
          gdCdt_interface[mIdx][node] = dCdt;
          gdCdt_interface_total[node] += dCdt;
        } // Loop over materials
      }
    } // Loop over nodes
  } // Loop over patches
}

void SimpleSDInterface::outputProblemSpec(
                                            ProblemSpecP  & ps
                                           )
{
  ProblemSpecP sdim_ps = ps;
  sdim_ps = ps->appendChild("diffusion_interface");
  sdim_ps->appendElement("type","simple");
  d_materials_list.outputProblemSpec(sdim_ps);
}
