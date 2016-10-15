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

#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPMFVM/ESConductivityModel.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

using namespace Uintah;

static DebugStream cout_doing("ESMPM_DOING_COUT", false);

ESConductivityModel::ESConductivityModel(SimulationStateP& shared_state,
                                         MPMFlags* mpm_flags,
                                         MPMLabel* mpm_lb, FVMLabel* fvm_lb)
{
  d_shared_state = shared_state;
  d_mpm_flags = mpm_flags;
  d_mpm_lb = mpm_lb;
  d_fvm_lb = fvm_lb;

  d_TINY_RHO  = 1.e-12;
}

ESConductivityModel::~ESConductivityModel()
{

}

void ESConductivityModel::scheduleComputeConductivity(SchedulerP& sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* all_matls,
                                                      const MaterialSubset* one_matl)
{
  const Level* level = getLevel(patches);
  if(!d_mpm_flags->doMPMOnLevel(level->getIndex(), level->getGrid()->numLevels())){
    return;
  }

  printSchedule(patches, cout_doing, "ESConductivityModel::scheduleComputeConductivity");

  Task* task = scinew Task("ESConductivityModel::computeConductivity", this,
                           &ESConductivityModel::computeConductivity);

  task->requires(Task::NewDW, d_mpm_lb->gConcentrationLabel, Ghost::AroundCells, 1);
  task->requires(Task::NewDW, d_mpm_lb->gMassLabel,          Ghost::AroundCells, 1);

  task->computes(d_fvm_lb->fcxConductivity, one_matl, Task::OutOfDomain);
  task->computes(d_fvm_lb->fcyConductivity, one_matl, Task::OutOfDomain);
  task->computes(d_fvm_lb->fczConductivity, one_matl, Task::OutOfDomain);

  sched->addTask(task, level->eachPatch(), all_matls);
}

void ESConductivityModel::computeConductivity(const ProcessorGroup* pg,
                                              const PatchSubset* patches,
                                              const MaterialSubset*,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  IntVector i100(1,0,0);
  IntVector i110(1,1,0);
  IntVector i111(1,1,1);
  IntVector i011(0,1,1);
  IntVector i001(0,0,1);
  IntVector i101(1,0,1);
  IntVector i010(0,1,0);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, cout_doing,
                "Doing ESConductivityModel::computeConductivity");

    int num_matls = d_shared_state->getNumMPMMatls();
    Vector cell_dim = patch->getLevel()->dCell();
    double cell_vol = cell_dim.x() * cell_dim.y() * cell_dim.z();

    SFCXVariable<double> fcx_conductivity;
    SFCYVariable<double> fcy_conductivity;
    SFCZVariable<double> fcz_conductivity;

    SFCXVariable<double> fcx_mass;
    SFCYVariable<double> fcy_mass;
    SFCZVariable<double> fcz_mass;

    new_dw->allocateAndPut(fcx_conductivity,  d_fvm_lb->fcxConductivity,    0, patch);
    new_dw->allocateAndPut(fcy_conductivity,  d_fvm_lb->fcyConductivity,    0, patch);
    new_dw->allocateAndPut(fcz_conductivity,  d_fvm_lb->fczConductivity,    0, patch);

    new_dw->allocateTemporary(fcx_mass, patch, Ghost::None, 0);
    new_dw->allocateTemporary(fcy_mass, patch, Ghost::None, 0);
    new_dw->allocateTemporary(fcz_mass, patch, Ghost::None, 0);

    fcx_conductivity.initialize(0.0);
    fcy_conductivity.initialize(0.0);
    fcz_conductivity.initialize(0.0);

    fcx_mass.initialize(d_TINY_RHO * cell_vol);
    fcy_mass.initialize(d_TINY_RHO * cell_vol);
    fcz_mass.initialize(d_TINY_RHO * cell_vol);

    IntVector lowidx = patch->getCellLowIndex();
    IntVector highidx = patch->getExtraCellHighIndex();
    for(int m = 0; m < num_matls; m++){
      MPMMaterial* mpm_matl = d_shared_state->getMPMMaterial(m);
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> gconc;
      constNCVariable<double> gmass;

      new_dw->get(gconc, d_mpm_lb->gConcentrationLabel, dwi, patch, Ghost::AroundCells, 1);
      new_dw->get(gmass, d_mpm_lb->gMassLabel,          dwi, patch, Ghost::AroundCells, 1);


      for(CellIterator iter=CellIterator(lowidx, highidx); !iter.done(); iter++){
        IntVector c = *iter;

        fcx_conductivity[c] += .25 * gmass[c]        * gconc[c];
        fcx_conductivity[c] += .25 * gmass[c + i010] * gconc[c + i010];
        fcx_conductivity[c] += .25 * gmass[c + i011] * gconc[c + i011];
        fcx_conductivity[c] += .25 * gmass[c + i001] * gconc[c + i001];
        fcx_mass[c] += .25 * gmass[c];
        fcx_mass[c] += .25 * gmass[c + i010];
        fcx_mass[c] += .25 * gmass[c + i011];
        fcx_mass[c] += .25 * gmass[c + i001];

        fcy_conductivity[c] += .25 * gmass[c]        * gconc[c];
        fcy_conductivity[c] += .25 * gmass[c + i100] * gconc[c + i100];
        fcy_conductivity[c] += .25 * gmass[c + i101] * gconc[c + i101];
        fcy_conductivity[c] += .25 * gmass[c + i001] * gconc[c + i001];
        fcy_mass[c] += .25 * gmass[c];
        fcy_mass[c] += .25 * gmass[c + i100];
        fcy_mass[c] += .25 * gmass[c + i101];
        fcy_mass[c] += .25 * gmass[c + i001];

        fcz_conductivity[c] += .25 * gmass[c] * gconc[c];
        fcz_conductivity[c] += .25 * gmass[c + i100] * gconc[c + i100];
        fcz_conductivity[c] += .25 * gmass[c + i110] * gconc[c + i110];
        fcz_conductivity[c] += .25 * gmass[c + i010] * gconc[c + i010];
        fcz_mass[c] += .25 * gmass[c];
        fcz_mass[c] += .25 * gmass[c + i100];
        fcz_mass[c] += .25 * gmass[c + i110];
        fcz_mass[c] += .25 * gmass[c + i010];
      }
    } // End material loop

    for(CellIterator iter=CellIterator(lowidx, highidx); !iter.done(); iter++){
      IntVector c = *iter;
      fcx_conductivity[c] = fcx_conductivity[c] / fcx_mass[c];
      fcy_conductivity[c] = fcy_conductivity[c] / fcy_mass[c];
      fcz_conductivity[c] = fcz_conductivity[c] / fcz_mass[c];
    }
  } // End patch loop
}
