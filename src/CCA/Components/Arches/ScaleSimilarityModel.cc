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

//----- ScaleSimilarityModel.cc --------------------------------------------------

#include <CCA/Components/Arches/ScaleSimilarityModel.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Grid/Level.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/Array3.h>
#include <iostream>
using namespace std;

using namespace Uintah;


//****************************************************************************
// Default constructor for SmagorinskyModel
//****************************************************************************
ScaleSimilarityModel::ScaleSimilarityModel(const ArchesLabel* label, 
                                           const MPMArchesLabel* MAlb,
                                           PhysicalConstants* phyConsts,
                                           BoundaryCondition* bndry_cond):
                                           TurbulenceModel(label, MAlb),
                                           d_physicalConsts(phyConsts),
                                           d_boundaryCondition(bndry_cond)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
ScaleSimilarityModel::~ScaleSimilarityModel()
{
}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object 
//****************************************************************************
double 
ScaleSimilarityModel::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}


//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
ScaleSimilarityModel::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params->findBlock("ScaleSimilarity");
  db->require("cf", d_CF);

  problemSetupCommon( params ); 

}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
ScaleSimilarityModel::sched_reComputeTurbSubmodel(SchedulerP& sched, 
                                                  const LevelP& level,
                                                  const MaterialSet* matls,
                                                  const TimeIntegratorLabel* timelabels)
{
//  SmagorinskyModel::sched_reComputeTurbSubmodel(sched, patches, matls,
//                                                timelabels);

  string taskname =  "ScaleSimilarityModel::ReTurbSubmodel" +
                     timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
                            &ScaleSimilarityModel::reComputeTurbSubmodel,
                            timelabels);

  // Requires
  // Assuming one layer of ghost cells
  // initialize with the value of zero at the physical bc's
  // construct a stress tensor and stored as a array with the following order
  // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
  
  Ghost::GhostType  gn = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel,       gac, 1);

  // for multimaterial
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
  }
  
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_stressTensorCompLabel, d_lab->d_tensorMatl, oams);
  }
  else {
    tsk->modifies(d_lab->d_stressTensorCompLabel, d_lab->d_tensorMatl, oams);
  }

  sched->addTask(tsk, level->eachPatch(), matls);
}

//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
ScaleSimilarityModel::reComputeTurbSubmodel(const ProcessorGroup* pc,
                                            const PatchSubset* patches,
                                            const MaterialSubset*,
                                            DataWarehouse*,
                                            DataWarehouse* new_dw,
                                            const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<Vector> Vel; 
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume; 
    
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(Vel,    d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    new_dw->get(den,    d_lab->d_densityCPLabel,      indx, patch, gac, 1);
    
    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch,gn, 0);
    }
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch,gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch,gac, 1);

    // Get the patch and variable details
    // compatible with fortran index
    double CF = d_CF;
    StencilMatrix<CCVariable<double> > stressTensorCoeff; //9 point tensor

  // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_tensorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        new_dw->allocateAndPut(stressTensorCoeff[ii],  d_lab->d_stressTensorCompLabel, ii, patch);
      }else{
        new_dw->getModifiable(stressTensorCoeff[ii],   d_lab->d_stressTensorCompLabel, ii, patch);
      }
      stressTensorCoeff[ii].initialize(0.0);
    }


    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian


    int numGC = 1;
    IntVector idxLo = patch->getExtraCellLowIndex(numGC);
    IntVector idxHi = patch->getExtraCellHighIndex(numGC);
    Array3<double> denUU(idxLo, idxHi);
    denUU.initialize(0.0);
    Array3<double> denUV(idxLo, idxHi);
    denUV.initialize(0.0);
    Array3<double> denUW(idxLo, idxHi);
    denUW.initialize(0.0);
    Array3<double> denVV(idxLo, idxHi);
    denVV.initialize(0.0);
    Array3<double> denVW(idxLo, idxHi);
    denVW.initialize(0.0);
    Array3<double> denWW(idxLo, idxHi);
    denWW.initialize(0.0);
    Array3<double> denPhiU(idxLo, idxHi);
    denPhiU.initialize(0.0);
    Array3<double> denPhiV(idxLo, idxHi);
    denPhiV.initialize(0.0);
    Array3<double> denPhiW(idxLo, idxHi);
    denPhiW.initialize(0.0);
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;
    
    int startZ = idxLo.z();
    if (zminus) startZ++;
    int endZ = idxHi.z();
    if (zplus) endZ--;
    int startY = idxLo.y();
    if (yminus) startY++;
    int endY = idxHi.y();
    if (yplus) endY--;
    int startX = idxLo.x();
    if (xminus) startX++;
    int endX = idxHi.x();
    if (xplus) endX--;
    
    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          denUU[currCell] = Vel[currCell].x() * Vel[currCell].x(); 
          denUV[currCell] = Vel[currCell].x() * Vel[currCell].y();
          denUW[currCell] = Vel[currCell].x() * Vel[currCell].z();
          denVV[currCell] = Vel[currCell].y() * Vel[currCell].y();
          denVW[currCell] = Vel[currCell].y() * Vel[currCell].z();
          denWW[currCell] = Vel[currCell].z() * Vel[currCell].z();
        }
      }
    }

#if 1
    if (xminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, colZ);
          IntVector prevCell(startX, colY, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }
    if (xplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, colZ);
          IntVector prevCell(endX-1, colY, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }
    if (yminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, startY-1, colZ);
          IntVector prevCell(colX, startY, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }
    if (yplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, endY, colZ);
          IntVector prevCell(colX, endY-1, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }
    if (zminus) { 
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, startZ-1);
          IntVector prevCell(colX, colY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }
    if (zplus) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, endZ);
          IntVector prevCell(colX, colY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
    }

    // fill the corner cells
    if (xminus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, startY-1, colZ);
          IntVector prevCell(startX, startY, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, endY, colZ);
          IntVector prevCell(startX, endY-1, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, startZ-1);
          IntVector prevCell(startX, colY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, endZ);
          IntVector prevCell(startX, colY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(startX-1, startY-1, startZ-1);
        IntVector prevCell(startX, startY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(startX-1, startY-1, endZ);
        IntVector prevCell(startX, startY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(startX-1, endY, startZ-1);
        IntVector prevCell(startX, endY-1, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(startX-1, endY, endZ);
        IntVector prevCell(startX, endY-1, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
    }
    if (xplus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, startY-1, colZ);
          IntVector prevCell(endX-1, startY, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, endY, colZ);
          IntVector prevCell(endX-1, endY-1, colZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, startZ-1);
          IntVector prevCell(endX-1, colY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, endZ);
          IntVector prevCell(endX-1, colY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(endX, startY-1, startZ-1);
        IntVector prevCell(endX-1, startY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(endX, startY-1, endZ);
        IntVector prevCell(endX-1, startY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(endX, endY, startZ-1);
        IntVector prevCell(endX-1, endY-1, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(endX, endY, endZ);
        IntVector prevCell(endX-1, endY-1, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
        
    }
    // for yminus&&zminus fill the corner cells for all internal x
    if (yminus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, startZ-1);
        IntVector prevCell(colX, startY, startZ);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
    }
    if (yminus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, endZ);
        IntVector prevCell(colX, startY, endZ-1);
          denUU[currCell] = denUU[prevCell];
          denUV[currCell] = denUV[prevCell];
          denUW[currCell] = denUW[prevCell];
          denVV[currCell] = denVV[prevCell];
          denVW[currCell] = denVW[prevCell];
          denWW[currCell] = denWW[prevCell];
          denPhiU[currCell] = denPhiU[prevCell];
          denPhiV[currCell] = denPhiV[prevCell];
          denPhiW[currCell] = denPhiW[prevCell];
      }
    }
    if (yplus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, startZ-1);
        IntVector prevCell(colX, endY-1, startZ);
        denUU[currCell] = denUU[prevCell];
        denUV[currCell] = denUV[prevCell];
        denUW[currCell] = denUW[prevCell];
        denVV[currCell] = denVV[prevCell];
        denVW[currCell] = denVW[prevCell];
        denWW[currCell] = denWW[prevCell];
        denPhiU[currCell] = denPhiU[prevCell];
        denPhiV[currCell] = denPhiV[prevCell];
        denPhiW[currCell] = denPhiW[prevCell];
      }
    }
    if (yplus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, endZ);
        IntVector prevCell(colX, endY-1, endZ-1);
        denUU[currCell] = denUU[prevCell];
        denUV[currCell] = denUV[prevCell];
        denUW[currCell] = denUW[prevCell];
        denVV[currCell] = denVV[prevCell];
        denVW[currCell] = denVW[prevCell];
        denWW[currCell] = denWW[prevCell];
        denPhiU[currCell] = denPhiU[prevCell];
        denPhiV[currCell] = denPhiV[prevCell];
        denPhiW[currCell] = denPhiW[prevCell];
      }
    }        

#endif
    Array3<double> filterdenUU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenUU.initialize(0.0);
    Array3<double> filterdenUV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenUV.initialize(0.0);
    Array3<double> filterdenUW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenUW.initialize(0.0);
    Array3<double> filterdenVV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenVV.initialize(0.0);
    Array3<double> filterdenVW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenVW.initialize(0.0);
    Array3<double> filterdenWW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenWW.initialize(0.0);
    Array3<double> filterDen(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterDen.initialize(0.0);
    Array3<double> filterUVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterUVel.initialize(0.0);
    Array3<double> filterVVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterVVel.initialize(0.0);
    Array3<double> filterWVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterWVel.initialize(0.0);
    Array3<double> filterPhi(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterPhi.initialize(0.0);
    Array3<double> filterdenPhiU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenPhiU.initialize(0.0);
    Array3<double> filterdenPhiV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenPhiV.initialize(0.0);
    Array3<double> filterdenPhiW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterdenPhiW.initialize(0.0);
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    d_filter->applyFilter_noPetsc(pc, patch, Vel, filterVolume, cellType, filterUVel, 0);
    d_filter->applyFilter_noPetsc(pc, patch, Vel, filterVolume, cellType, filterVVel, 1);
    d_filter->applyFilter_noPetsc(pc, patch, Vel, filterVolume, cellType, filterWVel, 2);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denUU, filterVolume, cellType, filterdenUU);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denUV, filterVolume, cellType, filterdenUV);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denUW, filterVolume, cellType, filterdenUW);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denVV, filterVolume, cellType, filterdenVV);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denVW, filterVolume, cellType, filterdenVW);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denWW, filterVolume, cellType, filterdenWW);
    
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denPhiU, filterVolume, cellType, filterdenPhiU);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denPhiV, filterVolume, cellType, filterdenPhiV);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, denPhiW, filterVolume, cellType, filterdenPhiW);

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          // compute stress tensor
          // index 0: T11, 1:T12, 2:T13, 3:T21, 4:T22, 5:T23, 6:T31, 7:T32, 8:T33
          (stressTensorCoeff[0])[currCell] = CF*den[currCell]*(filterdenUU[currCell] -
                                                 filterUVel[currCell]*
                                                 filterUVel[currCell]);
          (stressTensorCoeff[1])[currCell] = CF*den[currCell]*(filterdenUV[currCell] -
                                                 filterUVel[currCell]*
                                                 filterVVel[currCell]);
          (stressTensorCoeff[2])[currCell] = CF*den[currCell]*(filterdenUW[currCell] -
                                                 filterUVel[currCell]*
                                                 filterWVel[currCell]);
          (stressTensorCoeff[3])[currCell] = (stressTensorCoeff[1])[currCell];
          (stressTensorCoeff[4])[currCell] = CF*den[currCell]*(filterdenVV[currCell] -
                                                 filterVVel[currCell]*
                                                 filterVVel[currCell]);
          (stressTensorCoeff[5])[currCell] = CF*den[currCell]*(filterdenVW[currCell] -
                                                 filterVVel[currCell]*
                                                 filterWVel[currCell]);
          (stressTensorCoeff[6])[currCell] = (stressTensorCoeff[2])[currCell];
          (stressTensorCoeff[7])[currCell] = (stressTensorCoeff[5])[currCell];
          (stressTensorCoeff[8])[currCell] = CF*den[currCell]*(filterdenWW[currCell] -
                                                 filterWVel[currCell]*
                                                 filterWVel[currCell]);

        }
      }
    }
  }
}
