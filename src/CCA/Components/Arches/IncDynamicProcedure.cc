/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

//----- IncDynamicProcedure.cc --------------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Arches/IncDynamicProcedure.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Thread/Time.h>

using namespace std;
using namespace Uintah;

// flag to enable filter check
// need even grid size, unfiltered values are +-1; filtered value should be 0
// #define FILTER_CHECK
#ifdef FILTER_CHECK
#include <Core/Math/MiscMath.h>
#endif

//****************************************************************************
// Default constructor for IncDynamicProcedure
//****************************************************************************
IncDynamicProcedure::IncDynamicProcedure(const ArchesLabel* label, 
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
IncDynamicProcedure::~IncDynamicProcedure()
{
}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object 
//****************************************************************************
double 
IncDynamicProcedure::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
IncDynamicProcedure::problemSetup(const ProblemSpecP& params)
{
  problemSetupCommon( params ); 
  ProblemSpecP db = params->findBlock("Turbulence");

  // actually, Shmidt number, not Prandtl number
  d_turbPrNo = 0.0;
  if (db->findBlock("turbulentPrandtlNumber")){
    db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
  }
  db->getWithDefault("filter_cs_squared",d_filter_cs_squared,false);

}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
IncDynamicProcedure::sched_reComputeTurbSubmodel( SchedulerP& sched, 
                                                  const LevelP& level,
                                                  const MaterialSet* matls,
                                                  const TimeIntegratorLabel* timelabels)
{
  string taskname =  "IncDynamicProcedure::reComputeTurbSubmodel" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &IncDynamicProcedure::reComputeTurbSubmodel,
                          timelabels);

  // Requires
  // Assuming one layer of ghost cells
  // initialize with the value of zero at the physical bc's
  // construct a stress tensor and stored as a array with the following order
  // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
  Ghost::GhostType  gn  = Ghost::None; 
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);
      
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);
  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_strainTensorCompLabel, d_lab->d_symTensorMatl,
                  oams);
  }else{ 
    tsk->modifies(d_lab->d_strainTensorCompLabel, d_lab->d_symTensorMatl,
                  oams);
  }
    
  sched->addTask(tsk, level->eachPatch(), matls);

  //__________________________________
  taskname =  "IncDynamicProcedure::reComputeFilterValues" +
                     timelabels->integrator_step_name;
  tsk = scinew Task(taskname, this,
                    &IncDynamicProcedure::reComputeFilterValues,
                    timelabels);

  // Requires
  // Assuming one layer of ghost cells
  // initialize with the value of zero at the physical bc's
  // construct a stress tensor and stored as a array with the following order
  // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
  
  tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_strainTensorCompLabel,
                d_lab->d_symTensorMatl, oams, gac, 1);
  
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gac, 1);
  
  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_strainMagnitudeLabel);
    tsk->computes(d_lab->d_strainMagnitudeMLLabel);
    tsk->computes(d_lab->d_strainMagnitudeMMLabel);
  }
  else {
    tsk->modifies(d_lab->d_strainMagnitudeLabel);
    tsk->modifies(d_lab->d_strainMagnitudeMLLabel);
    tsk->modifies(d_lab->d_strainMagnitudeMMLabel);
  }
    
  sched->addTask(tsk, level->eachPatch(), matls);
  //__________________________________
  taskname =  "IncDynamicProcedure::reComputeSmagCoeff" +
               timelabels->integrator_step_name;
  tsk = scinew Task(taskname, this,
                    &IncDynamicProcedure::reComputeSmagCoeff,
                    timelabels);

  // Requires
  // Assuming one layer of ghost cells
  // initialize with the value of zero at the physical bc's
  // construct a stress tensor and stored as an array with the following order
  // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,         gac,1);
  tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeLabel,   gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMLLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMMLabel, gac, 1);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,          gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel,      gac, 1); 

  // for multimaterial
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
  }
  
  // Computes
  tsk->modifies(d_lab->d_viscosityCTSLabel);
  tsk->modifies(d_lab->d_turbViscosLabel);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_CsLabel);
  }else{ 
    tsk->modifies(d_lab->d_CsLabel);
  }  
  sched->addTask(tsk, level->eachPatch(), matls);
}


//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
IncDynamicProcedure::reComputeTurbSubmodel(const ProcessorGroup*,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse*,
                                           DataWarehouse* new_dw,
                                           const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<Vector> VelCC;
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel,     d_lab->d_uVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(vVel,     d_lab->d_vVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(wVel,     d_lab->d_wVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(VelCC,    d_lab->d_CCVelocityLabel,     indx, patch, gac, 1);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    
    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<CCVariable<double> > SIJ;    //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
        new_dw->allocateAndPut(SIJ[ii], 
                             d_lab->d_strainTensorCompLabel, ii, patch);
      }else{ 
        new_dw->getModifiable(SIJ[ii], 
                             d_lab->d_strainTensorCompLabel, ii, patch);
      }
      SIJ[ii].initialize(0.0);
    }
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();
    int startZ = indexLow.z();
    int endZ = indexHigh.z()+1;
    int startY = indexLow.y();
    int endY = indexHigh.y()+1;
    int startX = indexLow.x();
    int endX = indexHigh.x()+1;

    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          double uep, uwp, unp, usp, utp, ubp;
          double vnp, vsp, vep, vwp, vtp, vbp;
          double wnp, wsp, wep, wwp, wtp, wbp;

          uep = uVel[IntVector(colX+1,colY,colZ)];
          uwp = uVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          unp = 0.5*VelCC[IntVector(colX,colY+1,colZ)].x();
          usp = 0.5*VelCC[IntVector(colX,colY-1,colZ)].x();
          utp = 0.5*VelCC[IntVector(colX,colY,colZ+1)].x();
          ubp = 0.5*VelCC[IntVector(colX,colY,colZ-1)].x();

          vnp = vVel[IntVector(colX,colY+1,colZ)];
          vsp = vVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          vep = 0.5*VelCC[IntVector(colX+1,colY,colZ)].y();
          vwp = 0.5*VelCC[IntVector(colX-1,colY,colZ)].y();
          vtp = 0.5*VelCC[IntVector(colX,colY,colZ+1)].y();
          vbp = 0.5*VelCC[IntVector(colX,colY,colZ-1)].y();

          wtp = wVel[IntVector(colX,colY,colZ+1)];
          wbp = wVel[currCell];
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          wep = 0.5*VelCC[IntVector(colX+1,colY,colZ)].z();
          wwp = 0.5*VelCC[IntVector(colX-1,colY,colZ)].z();
          wnp = 0.5*VelCC[IntVector(colX,colY+1,colZ)].z();
          wsp = 0.5*VelCC[IntVector(colX,colY-1,colZ)].z();


          //     calculate the grid strain rate tensor

          double sewcur = cellinfo->sew[colX];
          double snscur = cellinfo->sns[colY];
          double stbcur = cellinfo->stb[colZ];

          (SIJ[0])[currCell] = (uep-uwp)/sewcur;
          (SIJ[1])[currCell] = (vnp-vsp)/snscur;
          (SIJ[2])[currCell] = (wtp-wbp)/stbcur;
          (SIJ[3])[currCell] = 0.5*((unp-usp)/snscur + 
                               (vep-vwp)/sewcur);
          (SIJ[4])[currCell] = 0.5*((utp-ubp)/stbcur + 
                               (wep-wwp)/sewcur);
          (SIJ[5])[currCell] = 0.5*((vtp-vbp)/stbcur + 
                               (wnp-wsp)/snscur);

        }
      }
    }
    if (xminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, colZ);
          IntVector prevCell(startX, colY, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
    }
    if (xplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, colZ);
          IntVector prevCell(endX-1, colY, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
    }
    if (yminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, startY-1, colZ);
          IntVector prevCell(colX, startY, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
    }
    if (yplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, endY, colZ);
          IntVector prevCell(colX, endY-1, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
    }
    if (zminus) { 
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, startZ-1);
          IntVector prevCell(colX, colY, startZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
                               
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
                               
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
                               

        }
      }
    }
    if (zplus) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, endZ);
          IntVector prevCell(colX, colY, endZ-1);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
                               
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
                               
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
                               

        }
      }
    }
    // fill the corner cells
    if (xminus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, startY-1, colZ);
          IntVector prevCell(startX, startY, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, endY, colZ);
          IntVector prevCell(startX, endY-1, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, startZ-1);
          IntVector prevCell(startX, colY, startZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, endZ);
          IntVector prevCell(startX, colY, endZ-1);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(startX-1, startY-1, startZ-1);
        IntVector prevCell(startX, startY, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(startX-1, startY-1, endZ);
        IntVector prevCell(startX, startY, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(startX-1, endY, startZ-1);
        IntVector prevCell(startX, endY-1, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(startX-1, endY, endZ);
        IntVector prevCell(startX, endY-1, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
        
    }
    if (xplus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, startY-1, colZ);
          IntVector prevCell(endX-1, startY, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, endY, colZ);
          IntVector prevCell(endX-1, endY-1, colZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, startZ-1);
          IntVector prevCell(endX-1, colY, startZ);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, endZ);
          IntVector prevCell(endX-1, colY, endZ-1);
          (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
          (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
          (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
          (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
          (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
          (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(endX, startY-1, startZ-1);
        IntVector prevCell(endX-1, startY, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(endX, startY-1, endZ);
        IntVector prevCell(endX-1, startY, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(endX, endY, startZ-1);
        IntVector prevCell(endX-1, endY-1, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(endX, endY, endZ);
        IntVector prevCell(endX-1, endY-1, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
        
    }
    // for yminus&&zminus fill the corner cells for all internal x
    if (yminus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, startZ-1);
        IntVector prevCell(colX, startY, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
    }
    if (yminus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, endZ);
        IntVector prevCell(colX, startY, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
    }
    if (yplus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, startZ-1);
        IntVector prevCell(colX, endY-1, startZ);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
    }
    if (yplus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, endZ);
        IntVector prevCell(colX, endY-1, endZ-1);
        (SIJ[0])[currCell] =  (SIJ[0])[prevCell];
        (SIJ[1])[currCell] =  (SIJ[1])[prevCell];
        (SIJ[2])[currCell] =  (SIJ[2])[prevCell];
        (SIJ[3])[currCell] =  (SIJ[3])[prevCell];
        (SIJ[4])[currCell] =  (SIJ[4])[prevCell];
        (SIJ[5])[currCell] =  (SIJ[5])[prevCell];
      }
    }        
  }
}



//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
IncDynamicProcedure::reComputeFilterValues(const ProcessorGroup* pc,
                                           const PatchSubset* patches,
                                           const MaterialSubset*,
                                           DataWarehouse*,
                                           DataWarehouse* new_dw,
                                           const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    TAU_PROFILE_TIMER(compute1, "Compute1", "[reComputeFilterValues::compute1]" , TAU_USER);
    TAU_PROFILE_TIMER(compute2, "Compute2", "[reComputeFilterValues::compute2]" , TAU_USER);
    TAU_PROFILE_TIMER(compute3, "Compute3", "[reComputeFilterValues::compute3]" , TAU_USER);
    TAU_PROFILE_TIMER(compute4, "Compute4", "[reComputeFilterValues::compute4]" , TAU_USER);
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<Vector> ccVel;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume; 
    // Get the velocity, density and viscosity from the old data warehouse

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(ccVel,   d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    new_dw->get(cellType, d_lab->d_cellTypeLabel,       indx, patch, gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel,       indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    
    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<constCCVariable<double> > SIJ; //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++){
      new_dw->get(SIJ[ii], d_lab->d_strainTensorCompLabel, ii, patch,
                  gac, 1);
    }
//    StencilMatrix<Array3<double> > LIJ;    //6 point tensor
//    StencilMatrix<Array3<double> > MIJ;    //6 point tensor
    StencilMatrix<Array3<double> > SHATIJ; //6 point tensor
    StencilMatrix<Array3<double> > betaIJ;  //6 point tensor
    StencilMatrix<Array3<double> > betaHATIJ; //6 point tensor

    IntVector idxLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector idxHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

    int tensorSize = 6; //  1-> 11, 2->22, 3->33, 4 ->12, 5->13, 6->23
    for (int ii = 0; ii < tensorSize; ii++) {
//      LIJ[ii].resize(idxLo, idxHi);
//      LIJ[ii].initialize(0.0);
//      MIJ[ii].resize(idxLo, idxHi);
//      MIJ[ii].initialize(0.0);
      SHATIJ[ii].resize(idxLo, idxHi);
      SHATIJ[ii].initialize(0.0);
      betaIJ[ii].resize(idxLo, idxHi);
      betaIJ[ii].initialize(0.0);
      betaHATIJ[ii].resize(idxLo, idxHi);
      betaHATIJ[ii].initialize(0.0);
    }  // allocate stress tensor coeffs
    CCVariable<double> IsImag;
    CCVariable<double> MLI;
    CCVariable<double> MMI;
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    new_dw->allocateAndPut(IsImag, 
                           d_lab->d_strainMagnitudeLabel, indx, patch);
    new_dw->allocateAndPut(MLI, 
                           d_lab->d_strainMagnitudeMLLabel, indx, patch);
    new_dw->allocateAndPut(MMI, 
                           d_lab->d_strainMagnitudeMMLabel, indx, patch);
    }else {
    new_dw->getModifiable(IsImag, 
                           d_lab->d_strainMagnitudeLabel, indx, patch);
    new_dw->getModifiable(MLI, 
                           d_lab->d_strainMagnitudeMLLabel, indx, patch);
    new_dw->getModifiable(MMI, 
                           d_lab->d_strainMagnitudeMMLabel, indx, patch);
    }
    IsImag.initialize(0.0);
    MLI.initialize(0.0);
    MMI.initialize(0.0);


    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian


    Array3<double> IsI(idxLo, idxHi); // magnitude of strain rate
    IsI.initialize(0.0);
//    Array3<double> IshatI(idxLo, idxHi); // magnitude of test filter strain rate
//    IshatI.initialize(0.0);
    Array3<double> UU(idxLo, idxHi);
    Array3<double> UV(idxLo, idxHi);
    Array3<double> UW(idxLo, idxHi);
    Array3<double> VV(idxLo, idxHi);
    Array3<double> VW(idxLo, idxHi);
    Array3<double> WW(idxLo, idxHi);
    UU.initialize(0.0);
    UV.initialize(0.0);
    UW.initialize(0.0);
    VV.initialize(0.0);
    VW.initialize(0.0);
    WW.initialize(0.0);
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
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

  TAU_PROFILE_START(compute1);
    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, colZ);
          // calculate absolute value of the grid strain rate
          // computes for the ghost cells too
          double sij0 = (SIJ[0])[currCell];
          double sij1 = (SIJ[1])[currCell];
          double sij2 = (SIJ[2])[currCell];
          double sij3 = (SIJ[3])[currCell];
          double sij4 = (SIJ[4])[currCell];
          double sij5 = (SIJ[5])[currCell];
          double isi_cur = sqrt(2.0*(sij0*sij0 + sij1*sij1 + sij2*sij2 +
                                     2.0*(sij3*sij3 + sij4*sij4 + sij5*sij5)));
          double uvel_cur = ccVel[currCell].x();
          double vvel_cur = ccVel[currCell].y();
          double wvel_cur = ccVel[currCell].z();

          IsI[currCell] = isi_cur; 

          //    calculate the grid filtered stress tensor, beta

          (betaIJ[0])[currCell] = isi_cur*sij0;
          (betaIJ[1])[currCell] = isi_cur*sij1;
          (betaIJ[2])[currCell] = isi_cur*sij2;
          (betaIJ[3])[currCell] = isi_cur*sij3;
          (betaIJ[4])[currCell] = isi_cur*sij4;
          (betaIJ[5])[currCell] = isi_cur*sij5;
          // required to compute Leonard term
          UU[currCell] = uvel_cur*uvel_cur;
          UV[currCell] = uvel_cur*vvel_cur;
          UW[currCell] = uvel_cur*wvel_cur;
          VV[currCell] = vvel_cur*vvel_cur;
          VW[currCell] = vvel_cur*wvel_cur;
          WW[currCell] = wvel_cur*wvel_cur;
        }
      }
    }
  TAU_PROFILE_STOP(compute1);
    Array3<double> filterUU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterUU.initialize(0.0);
    Array3<double> filterUV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterUV.initialize(0.0);
    Array3<double> filterUW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterUW.initialize(0.0);
    Array3<double> filterVV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterVV.initialize(0.0);
    Array3<double> filterVW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterVW.initialize(0.0);
    Array3<double> filterWW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterWW.initialize(0.0);
    Array3<double> filterUVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterUVel.initialize(0.0);
    Array3<double> filterVVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterVVel.initialize(0.0);
    Array3<double> filterWVel(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterWVel.initialize(0.0);
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();
    double start_turbTime = Time::currentSeconds();

    d_filter->applyFilter_noPetsc(pc, patch, ccVel, filterVolume, cellType, filterUVel, 0);
    d_filter->applyFilter_noPetsc(pc, patch, ccVel, filterVolume, cellType, filterVVel, 1);
    d_filter->applyFilter_noPetsc(pc, patch, ccVel, filterVolume, cellType, filterWVel, 2);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, UU, filterVolume, cellType, filterUU);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, UV, filterVolume, cellType, filterUV);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, UW, filterVolume, cellType, filterUW);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, VV, filterVolume, cellType, filterVV);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, VW, filterVolume, cellType, filterVW);
    d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch, WW, filterVolume, cellType, filterWW);

    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      d_filter->applyFilter_noPetsc< constCCVariable<double> >(pc, patch, SIJ[ii], filterVolume, cellType, SHATIJ[ii]);
      d_filter->applyFilter_noPetsc< Array3<double> >(pc, patch,betaIJ[ii], filterVolume, cellType, betaHATIJ[ii]);
    }

    string msg = "Time for the Filter operation in Turbulence Model: ";
    proc0cerr << msg << Time::currentSeconds() - start_turbTime << " seconds\n";

    TAU_PROFILE_START( compute2 );

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double delta = cellinfo->sew[colX]*
                         cellinfo->sns[colY]*cellinfo->stb[colZ];
          double filter = pow(delta, 1.0/3.0);

          // test filter width is assumed to be twice that of the basic filter
          // needs following modifications:
          // a) make the test filter work for anisotropic grid
          // b) generalize the filter operation
          double shatij0 = (SHATIJ[0])[currCell];
          double shatij1 = (SHATIJ[1])[currCell];
          double shatij2 = (SHATIJ[2])[currCell];
          double shatij3 = (SHATIJ[3])[currCell];
          double shatij4 = (SHATIJ[4])[currCell];
          double shatij5 = (SHATIJ[5])[currCell];
          double IshatIcur = sqrt(2.0*(shatij0*shatij0 + shatij1*shatij1 +
                                       shatij2*shatij2 + 2.0*(shatij3*shatij3 + 
                                       shatij4*shatij4 + shatij5*shatij5)));

//          IshatI[currCell] = IshatIcur;

          IsImag[currCell] = IsI[currCell]; 


          double MIJ0cur = 2.0*(filter*filter)*
                               ((betaHATIJ[0])[currCell]-
                                2.0*2.0*IshatIcur*shatij0);
//          (MIJ[0])[currCell] = MIJ0cur;
          double MIJ1cur = 2.0*(filter*filter)*
                               ((betaHATIJ[1])[currCell]-
                                2.0*2.0*IshatIcur*shatij1);
//          (MIJ[1])[currCell] = MIJ1cur;
          double MIJ2cur = 2.0*(filter*filter)*
                               ((betaHATIJ[2])[currCell]-
                                2.0*2.0*IshatIcur*shatij2);
//          (MIJ[2])[currCell] = MIJ2cur;
          double MIJ3cur = 2.0*(filter*filter)*
                               ((betaHATIJ[3])[currCell]-
                                2.0*2.0*IshatIcur*shatij3);
//          (MIJ[3])[currCell] = MIJ3cur;
          double MIJ4cur = 2.0*(filter*filter)*
                               ((betaHATIJ[4])[currCell]-
                                2.0*2.0*IshatIcur*shatij4);
//          (MIJ[4])[currCell] = MIJ4cur;
          double MIJ5cur =  2.0*(filter*filter)*
                               ((betaHATIJ[5])[currCell]-
                                2.0*2.0*IshatIcur*shatij5);
//          (MIJ[5])[currCell] = MIJ5cur; 


          // compute Leonard stress tensor
          // index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
          double filterUVelcur = filterUVel[currCell];
          double filterVVelcur = filterVVel[currCell];
          double filterWVelcur = filterWVel[currCell];
          double LIJ0cur = (filterUU[currCell] -
                                filterUVelcur*
                                filterUVelcur);
//          (LIJ[0])[currCell] = LIJ0cur;
          double LIJ1cur = (filterVV[currCell] -
                                filterVVelcur*
                                filterVVelcur);
//          (LIJ[1])[currCell] = LIJ1cur; 
          double LIJ2cur = (filterWW[currCell] -
                                filterWVelcur*
                                filterWVelcur);
//          (LIJ[2])[currCell] = LIJ2cur;
          double LIJ3cur = (filterUV[currCell] -
                                filterUVelcur*
                                filterVVelcur);
//          (LIJ[3])[currCell] = LIJ3cur;
          double LIJ4cur = (filterUW[currCell] -
                                filterUVelcur*
                                filterWVelcur);
//          (LIJ[4])[currCell] = LIJ4cur;
          double LIJ5cur = (filterVW[currCell] -
                                filterVVelcur*
                                filterWVelcur);
//          (LIJ[5])[currCell] = LIJ5cur;

          // compute the magnitude of ML and MM
          MLI[currCell] = MIJ0cur*LIJ0cur +
                         MIJ1cur*LIJ1cur +
                         MIJ2cur*LIJ2cur +
                         2.0*(MIJ3cur*LIJ3cur +
                              MIJ4cur*LIJ4cur +
                              MIJ5cur*LIJ5cur );
          MMI[currCell] = MIJ0cur*MIJ0cur +
                         MIJ1cur*MIJ1cur +
                         MIJ2cur*MIJ2cur +
                         2.0*(MIJ3cur*MIJ3cur +
                              MIJ4cur*MIJ4cur +
                              MIJ5cur*MIJ5cur );
                // calculate absolute value of the grid strain rate
        }
      }
    }
  TAU_PROFILE_STOP(compute2);
  TAU_PROFILE_START(compute3);
    startZ = indexLow.z();
    endZ = indexHigh.z()+1;
    startY = indexLow.y();
    endY = indexHigh.y()+1;
    startX = indexLow.x();
    endX = indexHigh.x()+1;
    
    if (xminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, colZ);
          IntVector prevCell(startX, colY, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    if (xplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, colZ);
          IntVector prevCell(endX-1, colY, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    if (yminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, startY-1, colZ);
          IntVector prevCell(colX, startY, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    if (yplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, endY, colZ);
          IntVector prevCell(colX, endY-1, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    if (zminus) { 
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, startZ-1);
          IntVector prevCell(colX, colY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    if (zplus) {
      for (int colY = startY; colY < endY; colY ++) {
        for (int colX = startX; colX < endX; colX ++) {
          IntVector currCell(colX, colY, endZ);
          IntVector prevCell(colX, colY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
    }
    // fill the corner cells
    if (xminus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, startY-1, colZ);
          IntVector prevCell(startX, startY, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(startX-1, endY, colZ);
          IntVector prevCell(startX, endY-1, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, startZ-1);
          IntVector prevCell(startX, colY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(startX-1, colY, endZ);
          IntVector prevCell(startX, colY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(startX-1, startY-1, startZ-1);
        IntVector prevCell(startX, startY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(startX-1, startY-1, endZ);
        IntVector prevCell(startX, startY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(startX-1, endY, startZ-1);
        IntVector prevCell(startX, endY-1, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(startX-1, endY, endZ);
        IntVector prevCell(startX, endY-1, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
        
    }
    if (xplus) {
      if (yminus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, startY-1, colZ);
          IntVector prevCell(endX-1, startY, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (yplus) {
        for (int colZ = startZ; colZ < endZ; colZ ++) {
          IntVector currCell(endX, endY, colZ);
          IntVector prevCell(endX-1, endY-1, colZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (zminus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, startZ-1);
          IntVector prevCell(endX-1, colY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (zplus) {
        for (int colY = startY; colY < endY; colY ++) {
          IntVector currCell(endX, colY, endZ);
          IntVector prevCell(endX-1, colY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
        }
      }
      if (yminus&&zminus) {
        IntVector currCell(endX, startY-1, startZ-1);
        IntVector prevCell(endX-1, startY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yminus&&zplus) {
        IntVector currCell(endX, startY-1, endZ);
        IntVector prevCell(endX-1, startY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yplus&&zminus) {
        IntVector currCell(endX, endY, startZ-1);
        IntVector prevCell(endX-1, endY-1, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
      if (yplus&&zplus) {
        IntVector currCell(endX, endY, endZ);
        IntVector prevCell(endX-1, endY-1, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
        
    }
    // for yminus&&zminus fill the corner cells for all internal x
    if (yminus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, startZ-1);
        IntVector prevCell(colX, startY, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
    }
    if (yminus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, startY-1, endZ);
        IntVector prevCell(colX, startY, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
    }
    if (yplus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, startZ-1);
        IntVector prevCell(colX, endY-1, startZ);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
    }
    if (yplus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
        IntVector currCell(colX, endY, endZ);
        IntVector prevCell(colX, endY-1, endZ-1);
          MLI[currCell] = MLI[prevCell];
          MMI[currCell] = MMI[prevCell];
      }
    }        

  TAU_PROFILE_STOP(compute3);

  }
}



//______________________________________________________________________
//
void 
IncDynamicProcedure::reComputeSmagCoeff(const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset*,
                                        DataWarehouse*,
                                        DataWarehouse* new_dw,
                                        const TimeIntegratorLabel* timelabels)
{
//  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> IsI;
    constCCVariable<double> MLI;
    constCCVariable<double> MMI;
    CCVariable<double> Cs; //smag coeff 
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume; 
    CCVariable<double> viscosity;
    CCVariable<double> turbViscosity; 

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
       new_dw->allocateAndPut(Cs, d_lab->d_CsLabel, indx, patch);
    }else{
       new_dw->getModifiable(Cs, d_lab->d_CsLabel, indx, patch);
    }
    Cs.initialize(0.0);

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->getModifiable(viscosity, d_lab->d_viscosityCTSLabel, indx, patch);
    new_dw->getModifiable(turbViscosity, d_lab->d_turbViscosLabel, indx, patch);
    
    new_dw->get(IsI,d_lab->d_strainMagnitudeLabel,    indx, patch, gn, 0);
    // using a box filter of 2*delta...will require more ghost cells if the size of filter is increased
    new_dw->get(MLI, d_lab->d_strainMagnitudeMLLabel, indx, patch, gac, 1);  
    new_dw->get(MMI, d_lab->d_strainMagnitudeMMLabel, indx, patch, gac, 1);
    new_dw->get(den, d_lab->d_densityCPLabel,         indx, patch, gac, 1);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
    }
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // get physical constants
    double viscos; // molecular viscosity
    viscos = d_physicalConsts->getMolecularViscosity();
 

    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian
    Array3<double> MLHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of strain rate
    MLHatI.initialize(0.0);
    Array3<double> MMHatI(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex()); // magnitude of test filter strain rate
    MMHatI.initialize(0.0);
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    d_filter->applyFilter_noPetsc< constCCVariable<double> >(pc, patch, MLI, filterVolume, cellType, MLHatI);
    d_filter->applyFilter_noPetsc< constCCVariable<double> >(pc, patch, MMI, filterVolume, cellType, MMHatI);

    CCVariable<double> tempCs;
    tempCs.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    tempCs.initialize(0.0);
          //     calculate the local Smagorinsky coefficient
          //     perform "clipping" in case MLij is negative...
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double value;
          if ((MMHatI[currCell] < 1.0e-10)||(MLHatI[currCell] < 1.0e-7))
            value = 0.0;
          else
            value = MLHatI[currCell]/MMHatI[currCell];
          tempCs[currCell] = value;
        }
      }
    }

    if ((d_filter_cs_squared)&&(!(d_3d_periodic))) {
    // filtering for periodic case is not implemented 
    // if it needs to be then tempCs will require 1 layer of boundary cells to be computed
    d_filter->applyFilter_noPetsc<CCVariable<double> >(pc, patch, tempCs, filterVolume, cellType, Cs);
    }
    else
    Cs.copy(tempCs, tempCs.getLowIndex(),
                      tempCs.getHighIndex());

    double factor = 1.0;
#if 0
    if (time < 2.0)
      factor = (time+0.000001)*0.5;
#endif

    if (d_MAlab) {

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            double delta = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);
            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);
            viscosity[currCell] =  Cs[currCell] * Cs[currCell] * filter * filter *
              IsI[currCell] * den[currCell] + viscos*voidFraction[currCell];

            turbViscosity[currCell] = viscosity[currCell] - viscos*voidFraction[currCell]; 
          }
        }
      }
    }
    else {
      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
            IntVector currCell(colX, colY, colZ);
            double delta = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);
            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);
            viscosity[currCell] =  Cs[currCell] * Cs[currCell] * filter * filter *
              IsI[currCell] * den[currCell] + viscos;

            turbViscosity[currCell] = viscosity[currCell] - viscos; 
          }
        }
      }
    }

    // boundary conditions...make a separate function apply Boundary
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int wall_celltypeval = d_boundaryCondition->wallCellType();
    if (xminus) {
      int colX = indexLow.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (xplus) {
      int colX =  indexHigh.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (yminus) {
      int colY = indexLow.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (yplus) {
      int colY =  indexHigh.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (zminus) {
      int colZ = indexLow.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }
    if (zplus) {
      int colZ =  indexHigh.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          if (cellType[currCell] != wall_celltypeval)
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
//            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
//                    *den[currCell]/den[IntVector(colX,colY,colZ)];
        }
      }
    }

  }
}

