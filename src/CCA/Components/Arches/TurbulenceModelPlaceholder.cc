/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//----- TurbulenceModelPlaceholder.cc --------------------------------------------------
#include <CCA/Components/Arches/TurbulenceModelPlaceholder.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Grid/Level.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/Array3.h>
#include <iostream>

using namespace std;

using namespace Uintah;

//****************************************************************************
TurbulenceModelPlaceholder::TurbulenceModelPlaceholder(const ArchesLabel* label,
                                   const MPMArchesLabel* MAlb,
                                   PhysicalConstants* phyConsts,
                                   BoundaryCondition* bndry_cond):
TurbulenceModel(label, MAlb),
d_physicalConsts(phyConsts),
d_boundaryCondition(bndry_cond)
{}

//****************************************************************************
TurbulenceModelPlaceholder::~TurbulenceModelPlaceholder()
{}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object
//****************************************************************************
double
TurbulenceModelPlaceholder::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void
TurbulenceModelPlaceholder::problemSetup(const ProblemSpecP& params)
{
  d_turbPrNo = 0.4;
  problemSetupCommon( params );
}

//****************************************************************************
// Schedule recomputation of the turbulence sub model
//****************************************************************************
void
TurbulenceModelPlaceholder::sched_reComputeTurbSubmodel( SchedulerP& sched,
                                                         const LevelP& level,
                                                         const MaterialSet* matls,
                                                         const TimeIntegratorLabel* timelabels)
{
  string taskname =  "TurbulenceModelPlaceholder::ReTurbSubmodel" +
  timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &TurbulenceModelPlaceholder::reComputeTurbSubmodel,
                          timelabels);

  // Requires
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel,  gn);
  //tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, gn, 0);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gac, 1);
  // for multimaterial
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
  }

  tsk->modifies(d_lab->d_viscosityCTSLabel);
  tsk->modifies(d_lab->d_turbViscosLabel);

  sched->addTask(tsk, level->eachPatch(), matls);
}


void
TurbulenceModelPlaceholder::reComputeTurbSubmodel(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw,
                                        const TimeIntegratorLabel* timelabels)
{
  //  double time = d_lab->d_sharedState->getElapsedSimTime();

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    //constCCVariable<double> density;

    CCVariable<double> viscosity; // this is the total viscosity that arches uses
    CCVariable<double> turbViscosity;


    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;


    new_dw->getModifiable(viscosity,     d_lab->d_viscosityCTSLabel,indx, patch);
    new_dw->getModifiable(turbViscosity, d_lab->d_turbViscosLabel, indx, patch );

//    new_dw->get(density,     d_lab->d_densityCPLabel,      indx, patch, gn,  0);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch,gn, 0);
    }
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

    // get physical constants
    double mol_viscos; // molecular viscosity
    mol_viscos = d_physicalConsts->getMolecularViscosity();

    // Get the patch and variable details
    // compatible with fortran index
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    viscosity.initialize(0.0);

    turbViscosity.copyData(viscosity);
    for ( CellIterator iter=patch->getCellIterator(); !iter.done(); ++iter ){
      IntVector c = *iter;
      viscosity[c] = viscosity[c] + mol_viscos;
    }

    //__________________________________
    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;

    int wall_celltypeval = d_boundaryCondition->wallCellType();
    if (xminus) {         // xminus
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);

          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }
    if (xplus) {          // xplus
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);

          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }
    if (yminus) {         // yminus
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);

          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }
    if (yplus) {          // yplus
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

          IntVector currCell(colX, colY+1, colZ);
          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }
    if (zminus) {         // zminus
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);

          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }
    if (zplus) {          // zplus
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);

          if (cellType[currCell] != wall_celltypeval){
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *density[currCell]/density[IntVector(colX,colY,colZ)];
          }
        }
      }
    }

    if (d_MAlab) {
      IntVector indexLow = patch->getExtraCellLowIndex();
      IntVector indexHigh = patch->getExtraCellHighIndex();
      for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
            // Store current cell
            IntVector currCell(colX, colY, colZ);
            viscosity[currCell] *=  voidFraction[currCell];
            turbViscosity[currCell] *=  voidFraction[currCell];
          }
        }
      }
    }
  }
}
