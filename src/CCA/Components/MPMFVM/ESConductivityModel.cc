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

#include <CCA/Components/FVM/FVMLabel.h>

#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>
#include <CCA/Components/MPMFVM/ESConductivityModel.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <Core/Math/MiscMath.h>

#include <vector>
#include <iostream>

using namespace Uintah;

static DebugStream cout_doing("ESMPM_DOING_COUT", false);

ESConductivityModel::ESConductivityModel(MaterialManagerP& materialManager,
                                         MPMFlags* mpm_flags,
                                         MPMLabel* mpm_lb, FVMLabel* fvm_lb,
                                         std::string& model_type)
{
  d_materialManager = materialManager;
  d_mpm_flags = mpm_flags;
  d_mpm_lb = mpm_lb;
  d_fvm_lb = fvm_lb;

  d_gac = Ghost::AroundCells;
  d_TINY_RHO  = 1.e-12;

  d_conductivity_equation = 0;

  d_model_type = model_type;
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

  //task->requires(Task::NewDW, d_mpm_lb->gConcentrationLabel, Ghost::AroundCells, 1);
  //task->requires(Task::NewDW, d_mpm_lb->gMassLabel,          Ghost::AroundCells, 1);
  task->requires(Task::OldDW, d_mpm_lb->diffusion->pConcentration, d_gac, 1);
  task->requires(Task::OldDW, d_mpm_lb->pXLabel,             d_gac, 1);

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
  std::vector<IntVector> ni(6);
  std::vector<double> S(6);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Vector cell_dim = patch->getLevel()->dCell();
    Point anchor = patch->getLevel()->getAnchor();
    //Point norm_pos = Point((pos - anchor)/cell_dim);
    double cell_vol = cell_dim.x() * cell_dim.y() * cell_dim.z();

    IntVector low_idx     = patch->getCellLowIndex();
    IntVector high_idx    = patch->getExtraCellHighIndex();

    SFCXVariable<double> fcx_conductivity;
    SFCYVariable<double> fcy_conductivity;
    SFCZVariable<double> fcz_conductivity;

    SFCXVariable<double> fcx_mass;
    SFCYVariable<double> fcy_mass;
    SFCZVariable<double> fcz_mass;

    new_dw->allocateAndPut(fcx_conductivity, d_fvm_lb->fcxConductivity, 0, patch, d_gac, 1);
    new_dw->allocateAndPut(fcy_conductivity, d_fvm_lb->fcyConductivity, 0, patch, d_gac, 1);
    new_dw->allocateAndPut(fcz_conductivity, d_fvm_lb->fczConductivity, 0, patch, d_gac, 1);

    new_dw->allocateTemporary(fcx_mass, patch, d_gac, 1);
    new_dw->allocateTemporary(fcy_mass, patch, d_gac, 1);
    new_dw->allocateTemporary(fcz_mass, patch, d_gac, 1);

    fcx_conductivity.initialize(0.0);
    fcy_conductivity.initialize(0.0);
    fcz_conductivity.initialize(0.0);

    fcx_mass.initialize(d_TINY_RHO * cell_vol);
    fcy_mass.initialize(d_TINY_RHO * cell_vol);
    fcz_mass.initialize(d_TINY_RHO * cell_vol);

    int numMatls = d_materialManager->getNumMatls( "MPM" );
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial* ) d_materialManager->getMaterial( "MPM", m );
      int dwi = mpm_matl->getDWIndex();

      d_conductivity_equation = mpm_matl->getScalarDiffusionModel()->getConductivityEquation();
      constParticleVariable<Point>  px;
      constParticleVariable<double> pconcentration;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, d_gac,
                                                       1, d_mpm_lb->pXLabel);

      old_dw->get(px,             d_mpm_lb->pXLabel,                  pset);
      old_dw->get(pconcentration, d_mpm_lb->diffusion->pConcentration,      pset);

      for (ParticleSubset::iterator iter  = pset->begin(); iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Point norm_pos = Point((px[idx] - anchor)/cell_dim);

        IntVector cell_idx(Floor(norm_pos.x()), Floor(norm_pos.y()),
                           Floor(norm_pos.z()));

        Point px(norm_pos.x() - (double)cell_idx.x(),
                 norm_pos.y() - (double)cell_idx.y(),
                 norm_pos.z() - (double)cell_idx.z());

        double conductivity = d_conductivity_equation->computeConductivity(pconcentration[idx]);

        ni[0] = cell_idx;                       // face center x-
        ni[1] = cell_idx + IntVector(1, 0, 0);  // face center x+
        ni[2] = cell_idx;                       // face center y-
        ni[3] = cell_idx + IntVector(0, 1, 0);  // face center y+
        ni[4] = cell_idx;                       // face center z-
        ni[5] = cell_idx + IntVector(0, 0, 1);  // face center z+

        S[0] = distanceFunc(px, Point(0.0, 0.5, 0.5));
        S[1] = distanceFunc(px, Point(1.0, 0.5, 0.5));
        S[2] = distanceFunc(px, Point(0.5, 0.0, 0.5));
        S[3] = distanceFunc(px, Point(0.5, 1.0, 0.5));
        S[4] = distanceFunc(px, Point(0.5, 0.5, 0.0));
        S[5] = distanceFunc(px, Point(0.5, 0.5, 1.0));

        if(cell_idx.x() < low_idx.x()){
          if(cell_idx.y() >= low_idx.y() && cell_idx.y() < high_idx.y()){
            if(cell_idx.z() >= low_idx.z() && cell_idx.z() < high_idx.z()){
              fcx_conductivity[ni[1]] += conductivity * S[1];
              fcx_mass[ni[1]] += S[1];
            }
          }
        }else if(cell_idx.x() >= high_idx.x()){
          if(cell_idx.y() >= low_idx.y() && cell_idx.y() < high_idx.y()){
            if(cell_idx.z() >= low_idx.z() && cell_idx.z() < high_idx.z()){
              fcx_conductivity[ni[0]] += conductivity * S[0];
              fcx_mass[ni[0]] += S[0];
            }
          }
        }else{
          if(cell_idx.y() < low_idx.y()){
            if(cell_idx.z() >= low_idx.z() && cell_idx.z() < high_idx.z()){
              fcy_conductivity[ni[3]] += conductivity * S[3];
              fcy_mass[ni[3]] += S[3];
            }
          }else if(cell_idx.y() >= high_idx.y()){
            if(cell_idx.z() >= low_idx.z() && cell_idx.z() < high_idx.z()){
              fcy_conductivity[ni[2]] += conductivity * S[2];
              fcy_mass[ni[2]] += S[2];
            }
          }else{
            if(cell_idx.z() < low_idx.z()){
              fcz_conductivity[ni[5]] += conductivity * S[5];
              fcz_mass[ni[5]] += S[5];
            }else if(cell_idx.z() >= high_idx.z()){
              fcz_conductivity[ni[4]] += conductivity * S[4];
              fcz_mass[ni[4]] += S[4];
            }else{
              fcx_conductivity[ni[0]] += conductivity * S[0];
              fcx_mass[ni[0]] += S[0];
              fcx_conductivity[ni[1]] += conductivity * S[1];
              fcx_mass[ni[1]] += S[1];
              fcy_conductivity[ni[2]] += conductivity * S[2];
              fcy_mass[ni[2]] += S[2];
              fcy_conductivity[ni[3]] += conductivity * S[3];
              fcy_mass[ni[3]] += S[3];
              fcz_conductivity[ni[4]] += conductivity * S[4];
              fcz_mass[ni[4]] += S[4];
              fcz_conductivity[ni[5]] += conductivity * S[5];
              fcz_mass[ni[5]] += S[5];
            }
          }
        }
      } // End Particle Loop
    } // End Material Loop

    for(CellIterator iter=CellIterator(low_idx, high_idx); !iter.done(); iter++){
      IntVector c = *iter;
      fcx_conductivity[c] = fcx_conductivity[c] / fcx_mass[c];
      fcy_conductivity[c] = fcy_conductivity[c] / fcy_mass[c];
      fcz_conductivity[c] = fcz_conductivity[c] / fcz_mass[c];
    } // End Cell Loop
  } // End Patch Loop
}

double ESConductivityModel::distanceFunc(Point p1, Point p2)
{
  double dist = (p1.x() - p2.x()) * (p1.x() - p2.x());
  dist += (p1.y() - p2.y()) * (p1.y() - p2.y());
  dist += (p1.z() - p2.z()) * (p1.z() - p2.z());

  return 1/dist;
}

/*
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

    int num_matls = d_materialManager->getNumMatls( "MPM" );
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
      MPMMaterial* mpm_matl = d_materialManager->getMaterial( "MPM", m);
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
*/
