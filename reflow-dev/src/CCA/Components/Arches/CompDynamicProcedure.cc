/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

//----- CompDynamicProcedure.cc --------------------------------------------------

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Arches/CompDynamicProcedure.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/StencilMatrix.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Exceptions/ProblemSetupException.h>
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
// Default constructor for CompDynamicProcedure
//****************************************************************************
CompDynamicProcedure::CompDynamicProcedure(const ArchesLabel* label, 
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
CompDynamicProcedure::~CompDynamicProcedure()
{
}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object 
//****************************************************************************
double 
CompDynamicProcedure::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
  void 
CompDynamicProcedure::problemSetup(const ProblemSpecP& params)
{
  problemSetupCommon( params ); 
  ProblemSpecP db = params->findBlock("Turbulence");

  db->getWithDefault("filter_cs_squared",d_filter_cs_squared,false);

}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
  void 
CompDynamicProcedure::sched_reComputeTurbSubmodel(SchedulerP& sched, 
    const PatchSet* patches,
    const MaterialSet* matls,
    const TimeIntegratorLabel* timelabels)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None;
  Task::MaterialDomainSpec oams = Task::OutOfDomain;  //outside of arches matlSet.
  {
    string taskname =  "CompDynamicProcedure::reComputeTurbSubmodel" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
        &CompDynamicProcedure::reComputeTurbSubmodel,
        timelabels);


    // Requires
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,     gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,      gac, 2);
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 


    int mmWallID = d_boundaryCondition->getMMWallId();
    if (mmWallID > 0)
      tsk->requires(Task::NewDW, d_lab->d_denRefArrayLabel, Ghost::None, 0);

    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_filterRhoULabel);
      tsk->computes(d_lab->d_filterRhoVLabel);
      tsk->computes(d_lab->d_filterRhoWLabel);
      tsk->computes(d_lab->d_filterRhoLabel);
    }
    else {
      tsk->modifies(d_lab->d_filterRhoULabel);
      tsk->modifies(d_lab->d_filterRhoVLabel);
      tsk->modifies(d_lab->d_filterRhoWLabel);
      tsk->modifies(d_lab->d_filterRhoLabel);
    }  

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeStrainRateTensors" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
        &CompDynamicProcedure::reComputeStrainRateTensors,
        timelabels);
    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel,gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoULabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoVLabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoWLabel,    gaf, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoLabel,     gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);


    // Computes
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->computes(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
    } else {
      tsk->modifies(d_lab->d_strainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
      tsk->modifies(d_lab->d_filterStrainTensorCompLabel,
          d_lab->d_symTensorMatl, oams);
    }  
    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeFilterValues" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
        &CompDynamicProcedure::reComputeFilterValues,
        timelabels);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}

    tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_filterRhoLabel,      gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 2);

    tsk->requires(Task::NewDW, d_lab->d_strainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

    tsk->requires(Task::NewDW, d_lab->d_filterStrainTensorCompLabel,
        d_lab->d_symTensorMatl, oams,gac, 1);

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

    sched->addTask(tsk, patches, matls);
  }
  //__________________________________
  {
    string taskname =  "CompDynamicProcedure::reComputeSmagCoeff" +
      timelabels->integrator_step_name;
    Task* tsk = scinew Task(taskname, this,
        &CompDynamicProcedure::reComputeSmagCoeff,
        timelabels);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as an array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,         gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeLabel,   gn, 0);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMLLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMMLabel, gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,          gac, 1);
    tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
    tsk->requires(Task::NewDW, d_lab->d_filterVolumeLabel, gn); 
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);

    // for multimaterial
    if (d_MAlab){
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
    }

    // Computes
    tsk->modifies(d_lab->d_viscosityCTSLabel);
    tsk->modifies(d_lab->d_turbViscosLabel); 

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      tsk->computes(d_lab->d_CsLabel);
    }
    else {
      tsk->modifies(d_lab->d_CsLabel);
    }

    sched->addTask(tsk, patches, matls);
  }
}


//****************************************************************************
// Actual recompute 
//****************************************************************************
  void 
CompDynamicProcedure::reComputeTurbSubmodel(const ProcessorGroup* pc,
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
    constCCVariable<double> density;
    constCCVariable<int> cellType;
    constCCVariable<double> filterVolume; 

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    // Get the velocity
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vVel, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wVel, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(density, d_lab->d_densityCPLabel,  indx, patch, gac, 2);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0); 

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 2);


    SFCXVariable<double> filterRhoU;
    SFCYVariable<double> filterRhoV;
    SFCZVariable<double> filterRhoW;
    CCVariable<double> filterRho;
    CCVariable<double> filterRhoF;
    CCVariable<double> filterRhoE;
    CCVariable<double> filterRhoRF;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->allocateAndPut(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->allocateAndPut(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->allocateAndPut(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);

    }
    else {
      new_dw->getModifiable(filterRhoU, d_lab->d_filterRhoULabel, indx, patch);
      new_dw->getModifiable(filterRhoV, d_lab->d_filterRhoVLabel, indx, patch);
      new_dw->getModifiable(filterRhoW, d_lab->d_filterRhoWLabel, indx, patch);
      new_dw->getModifiable(filterRho,  d_lab->d_filterRhoLabel,  indx, patch);
    }
    filterRhoU.initialize(0.0);
    filterRhoV.initialize(0.0);
    filterRhoW.initialize(0.0);
    filterRho.initialize(0.0);

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    IntVector indexLowU = patch->getSFCXFORTLowIndex__Old();
    IntVector indexHighU = patch->getSFCXFORTHighIndex__Old();
    IntVector indexLowV = patch->getSFCYFORTLowIndex__Old();
    IntVector indexHighV = patch->getSFCYFORTHighIndex__Old();
    IntVector indexLowW = patch->getSFCZFORTLowIndex__Old();
    IntVector indexHighW = patch->getSFCZFORTHighIndex__Old();

    if (xminus) indexLowU -= IntVector(1,0,0); 
    if (yminus) indexLowV -= IntVector(0,1,0); 
    if (zminus) indexLowW -= IntVector(0,0,1); 
    if (xplus) indexHighU += IntVector(1,0,0); 
    if (yplus) indexHighV += IntVector(0,1,0); 
    if (zplus) indexHighW += IntVector(0,0,1); 

    int flowID = -1;
    int mmWallID = d_boundaryCondition->getMMWallId();
    for (int colZ = indexLowU.z(); colZ <= indexHighU.z(); colZ ++) {
      for (int colY = indexLowU.y(); colY <= indexHighU.y(); colY ++) {
        for (int colX = indexLowU.x(); colX <= indexHighU.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((xplus)&&((colX == indexHighU.x())||(colX == indexHighU.x()-1)))
            shift = IntVector(-1,0,0);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(1,0,0)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(1,0,0)] != mmWallID) {
                      double vol = cellinfo->sewu[colX+ii]*
                        cellinfo->sns[colY+jj]*
                        cellinfo->stb[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoU[currCell] += vol*uVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(1,0,0)]);
                      totalVol += vol;
                    }
                }
              }
            }
            filterRhoU[currCell] /= totalVol;
          }
        }
      }
    }
    // assuming SFCY still stored with z being outer index and x inner index
    for (int colZ = indexLowV.z(); colZ <= indexHighV.z(); colZ ++) {
      for (int colY = indexLowV.y(); colY <= indexHighV.y(); colY ++) {
        for (int colX = indexLowV.x(); colX <= indexHighV.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((yplus)&&((colY == indexHighV.y())||(colY == indexHighV.y()-1)))
            shift = IntVector(0,-1,0);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(0,1,0)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(0,1,0)] != mmWallID) {
                      double vol = cellinfo->sew[colX+ii]*
                        cellinfo->snsv[colY+jj]*
                        cellinfo->stb[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoV[currCell] += vol*vVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(0,1,0)]);
                      totalVol += vol;
                    }
                }
              }
            }

            filterRhoV[currCell] /= totalVol;
          }
        }
      }
    }
    // assuming SFCZ still stored with z being outer index and x inner index
    for (int colZ = indexLowW.z(); colZ <= indexHighW.z(); colZ ++) {
      for (int colY = indexLowW.y(); colY <= indexHighW.y(); colY ++) {
        for (int colX = indexLowW.x(); colX <= indexHighW.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          IntVector shift(0,0,0);
          if ((zplus)&&((colZ == indexHighW.z())||(colZ == indexHighW.z()-1))) 
            shift = IntVector(0,0,-1);
          int bndry_count=0;
          if  (!(cellType[currCell + shift - IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(1,0,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,1,0)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift - IntVector(0,0,1)] == flowID))
            bndry_count++;
          if  (!(cellType[currCell + shift + IntVector(0,0,1)] == flowID))
            bndry_count++;
          bool corner = (bndry_count==3);
          double totalVol = 0.0;
          if ((cellType[currCell+shift] == flowID)&&
              (cellType[currCell+shift-IntVector(0,0,1)] != mmWallID)) {
            for (int kk = -1; kk <= 1; kk ++) {
              for (int jj = -1; jj <= 1; jj ++) {
                for (int ii = -1; ii <= 1; ii ++) {
                  IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
                  // on the boundary
                  if (cellType[filterCell+shift] != flowID) {
                    // intrusion
                    if (filterCell+shift == currCell+shift) {
                      // do nothing here, assuming intrusion velocity is 0
                    }
                  }
                  // inside the domain
                  else
                    if (cellType[filterCell+shift-IntVector(0,0,1)] != mmWallID) {
                      double vol = cellinfo->sew[colX+ii]*
                        cellinfo->sns[colY+jj]*
                        cellinfo->stbw[colZ+kk];
                      if (!(corner)) vol *= (1.0-0.5*abs(ii))*
                        (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
                      filterRhoW[currCell] += vol*wVel[filterCell]*
                        0.5*(density[filterCell]+
                            density[filterCell-IntVector(0,0,1)]);
                      totalVol += vol;
                    }
                }
              }
            }

            filterRhoW[currCell] /= totalVol;
          }
        }
      }
    }

    int ngc = 1;
    IntVector idxLo = patch->getExtraCellLowIndex(ngc);
    IntVector idxHi = patch->getExtraCellHighIndex(ngc);
    Array3<double> rhoF(idxLo, idxHi);
    Array3<double> rhoE(idxLo, idxHi);
    Array3<double> rhoRF(idxLo, idxHi);
    rhoF.initialize(0.0);
    rhoE.initialize(0.0);
    rhoRF.initialize(0.0);

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

    filterRho.copy(density, patch->getExtraCellLowIndex(),
        patch->getExtraCellHighIndex());
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, density, filterVolume, cellType, filterRho); 

    // making filterRho nonzero 
    if (mmWallID > 0) {

      constCCVariable<double> ref_density; 
      new_dw->get(ref_density, d_lab->d_denRefArrayLabel, indx, patch, Ghost::None, 0); 

      idxLo = patch->getExtraCellLowIndex();
      idxHi = patch->getExtraCellHighIndex();

      for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
          for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {

            IntVector currCell(colX, colY, colZ);

            if (filterRho[currCell] < 1.0e-15) 
              filterRho[currCell]=ref_density[currCell];

          }
        }
      }
    }
  }
}
//****************************************************************************
// Actual recompute 
//****************************************************************************
  void 
CompDynamicProcedure::reComputeStrainRateTensors(const ProcessorGroup*,
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
    constSFCXVariable<double> filterRhoU;
    constSFCYVariable<double> filterRhoV;
    constSFCZVariable<double> filterRhoW;
    constCCVariable<double> filterRho;
    constCCVariable<double> filterRhoF;
    constCCVariable<double> filterRhoE;
    constCCVariable<double> filterRhoRF;

    // Get the velocity
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    new_dw->get(uVel,       d_lab->d_uVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(vVel,       d_lab->d_vVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(wVel,       d_lab->d_wVelocitySPBCLabel,  indx, patch, gaf, 1);
    new_dw->get(VelCC,      d_lab->d_CCVelocityLabel,     indx, patch, gac, 1);
    new_dw->get(filterRhoU, d_lab->d_filterRhoULabel,     indx, patch, gaf, 1);
    new_dw->get(filterRhoV, d_lab->d_filterRhoVLabel,     indx, patch, gaf, 1);
    new_dw->get(filterRhoW, d_lab->d_filterRhoWLabel,     indx, patch, gaf, 1);
    new_dw->get(filterRho,  d_lab->d_filterRhoLabel,      indx, patch, gac, 1);

    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<CCVariable<double> > SIJ;    //6 point tensor
    StencilMatrix<CCVariable<double> > filterSIJ;    //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
        new_dw->allocateAndPut(SIJ[ii],       d_lab->d_strainTensorCompLabel,       ii, patch);
        new_dw->allocateAndPut(filterSIJ[ii], d_lab->d_filterStrainTensorCompLabel, ii, patch);
      }else {
        new_dw->getModifiable(SIJ[ii],        d_lab->d_strainTensorCompLabel,       ii, patch);
        new_dw->getModifiable(filterSIJ[ii],  d_lab->d_filterStrainTensorCompLabel, ii, patch);
      }
      SIJ[ii].initialize(0.0);
      filterSIJ[ii].initialize(0.0);
    }

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    for (int colZ =indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ);
          double sewcur = cellinfo->sew[colX];
          double snscur = cellinfo->sns[colY];
          double stbcur = cellinfo->stb[colZ];
          double efaccur = cellinfo->efac[colX];
          double wfaccur = cellinfo->wfac[colX];
          double nfaccur = cellinfo->nfac[colY];
          double sfaccur = cellinfo->sfac[colY];
          double tfaccur = cellinfo->tfac[colZ];
          double bfaccur = cellinfo->bfac[colZ];

          double uep, uwp, unp, usp, utp, ubp;
          double vnp, vsp, vep, vwp, vtp, vbp;
          double wtp, wbp, wep, wwp, wnp, wsp;

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
          (SIJ[0])[currCell] = (uep-uwp)/sewcur;
          (SIJ[1])[currCell] = (vnp-vsp)/snscur;
          (SIJ[2])[currCell] = (wtp-wbp)/stbcur;
          (SIJ[3])[currCell] = 0.5*((unp-usp)/snscur + 
              (vep-vwp)/sewcur);
          (SIJ[4])[currCell] = 0.5*((utp-ubp)/stbcur + 
              (wep-wwp)/sewcur);
          (SIJ[5])[currCell] = 0.5*((vtp-vbp)/stbcur + 
              (wnp-wsp)/snscur);

          double fuep, fuwp, funp, fusp, futp, fubp;
          double fvnp, fvsp, fvep, fvwp, fvtp, fvbp;
          double fwtp, fwbp, fwep, fwwp, fwnp, fwsp;

          fuep = filterRhoU[IntVector(colX+1,colY,colZ)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX+1,colY,colZ)]));
          fuwp = filterRhoU[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX-1,colY,colZ)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          funp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX+1,colY+1,colZ)])) +
              wfaccur * filterRhoU[IntVector(colX,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX-1,colY+1,colZ)])));
          fusp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX+1,colY-1,colZ)])) +
              wfaccur * filterRhoU[IntVector(colX,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX-1,colY-1,colZ)])));
          futp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX+1,colY,colZ+1)])) +
              wfaccur * filterRhoU[IntVector(colX,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX-1,colY,colZ+1)])));
          fubp = 0.5*(efaccur * filterRhoU[IntVector(colX+1,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX+1,colY,colZ-1)])) +
              wfaccur * filterRhoU[IntVector(colX,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX-1,colY,colZ-1)])));

          fvnp = filterRhoV[IntVector(colX,colY+1,colZ)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY+1,colZ)]));
          fvsp = filterRhoV[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY-1,colZ)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          fvep = 0.5*(nfaccur * filterRhoV[IntVector(colX+1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY+1,colZ)])) +
              sfaccur * filterRhoV[IntVector(colX+1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY-1,colZ)])));
          fvwp = 0.5*(nfaccur * filterRhoV[IntVector(colX-1,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY+1,colZ)])) +
              sfaccur * filterRhoV[IntVector(colX-1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY-1,colZ)])));
          fvtp = 0.5*(nfaccur * filterRhoV[IntVector(colX,colY+1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX,colY+1,colZ+1)])) +
              sfaccur * filterRhoV[IntVector(colX,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ+1)] +
                    filterRho[IntVector(colX,colY-1,colZ+1)])));
          fvbp = 0.5*(nfaccur * filterRhoV[IntVector(colX,colY+1,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX,colY+1,colZ-1)])) +
              sfaccur * filterRhoV[IntVector(colX,colY,colZ-1)]/
              (0.5*(filterRho[IntVector(colX,colY,colZ-1)] +
                    filterRho[IntVector(colX,colY-1,colZ-1)])));

          fwtp = filterRhoW[IntVector(colX,colY,colZ+1)]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY,colZ+1)]));
          fwbp = filterRhoW[currCell]/
            (0.5*(filterRho[currCell] +
                  filterRho[IntVector(colX,colY,colZ-1)]));
          // colX,coly,colZ component cancels out when computing derivative,
          // so it has been ommited
          fwep = 0.5*(tfaccur * filterRhoW[IntVector(colX+1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX+1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX+1,colY,colZ)] +
                    filterRho[IntVector(colX+1,colY,colZ-1)])));
          fwwp = 0.5*(tfaccur * filterRhoW[IntVector(colX-1,colY,colZ+1)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX-1,colY,colZ)]/
              (0.5*(filterRho[IntVector(colX-1,colY,colZ)] +
                    filterRho[IntVector(colX-1,colY,colZ-1)])));
          fwnp = 0.5*(tfaccur * filterRhoW[IntVector(colX,colY+1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX,colY+1,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX,colY+1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY+1,colZ)] +
                    filterRho[IntVector(colX,colY+1,colZ-1)])));
          fwsp = 0.5*(tfaccur * filterRhoW[IntVector(colX,colY-1,colZ+1)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX,colY-1,colZ+1)])) +
              bfaccur * filterRhoW[IntVector(colX,colY-1,colZ)]/
              (0.5*(filterRho[IntVector(colX,colY-1,colZ)] +
                    filterRho[IntVector(colX,colY-1,colZ-1)])));

          //     calculate the filtered strain rate tensor
          (filterSIJ[0])[currCell] = (fuep-fuwp)/sewcur;
          (filterSIJ[1])[currCell] = (fvnp-fvsp)/snscur;
          (filterSIJ[2])[currCell] = (fwtp-fwbp)/stbcur;
          (filterSIJ[3])[currCell] = 0.5*((funp-fusp)/snscur + 
              (fvep-fvwp)/sewcur);
          (filterSIJ[4])[currCell] = 0.5*((futp-fubp)/stbcur + 
              (fwep-fwwp)/sewcur);
          (filterSIJ[5])[currCell] = 0.5*((fvtp-fvbp)/stbcur + 
              (fwnp-fwsp)/snscur);

        }
      }
    }
  }
}



//****************************************************************************
// Actual recompute 
//****************************************************************************
  void 
CompDynamicProcedure::reComputeFilterValues(const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw,
    const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    TAU_PROFILE_TIMER(compute1, "Compute1", "[reComputeFilterValues::compute1]" , TAU_USER);
    TAU_PROFILE_TIMER(compute2, "Compute2", "[reComputeFilterValues::compute2]" , TAU_USER);
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int indx = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<Vector> ccVel;
    constCCVariable<double> den;
    constCCVariable<double> filterRho;
    constCCVariable<double> filterRhoF;
    constCCVariable<double> filterRhoE;
    constCCVariable<double> filterRhoRF;
    constCCVariable<double> filterVolume; 
    constCCVariable<int> cellType; 


    // Get the velocity and density
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, Ghost::None, 0); 
    new_dw->get(cellType,     d_lab->d_cellTypeLabel,     indx, patch, gac, 2);
    new_dw->get(ccVel,     d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    new_dw->get(den,       d_lab->d_densityCPLabel,      indx, patch, gac, 1);
    new_dw->get(filterRho, d_lab->d_filterRhoLabel,      indx, patch, gac, 1);


    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();


    IntVector idxLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
    IntVector idxHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

    StencilMatrix<constCCVariable<double> > SIJ; //6 point tensor
    StencilMatrix<constCCVariable<double> > SHATIJ; //6 point tensor
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      new_dw->get(SIJ[ii],    d_lab->d_strainTensorCompLabel,      ii, patch,gac, 1);
      new_dw->get(SHATIJ[ii], d_lab->d_filterStrainTensorCompLabel,ii, patch, gac, 1);
    }

    StencilMatrix<Array3<double> > betaIJ;    //6 point tensor
    StencilMatrix<Array3<double> > betaHATIJ; //6 point tensor
    //  0-> 11, 1->22, 2->33, 3 ->12, 4->13, 5->23
    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      betaIJ[ii].resize(idxLo, idxHi);
      betaIJ[ii].initialize(0.0);
      betaHATIJ[ii].resize(idxLo, idxHi);
      betaHATIJ[ii].initialize(0.0);
    }  // allocate stress tensor coeffs

    CCVariable<double> IsImag;
    CCVariable<double> MLI;
    CCVariable<double> MMI;
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(IsImag, d_lab->d_strainMagnitudeLabel,   indx, patch);
      new_dw->allocateAndPut(MLI,    d_lab->d_strainMagnitudeMLLabel, indx, patch);
      new_dw->allocateAndPut(MMI,    d_lab->d_strainMagnitudeMMLabel, indx, patch);
    }
    else {
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
    Array3<double> rhoU(idxLo, idxHi);
    Array3<double> rhoV(idxLo, idxHi);
    Array3<double> rhoW(idxLo, idxHi);
    Array3<double> rhoUU(idxLo, idxHi);
    Array3<double> rhoUV(idxLo, idxHi);
    Array3<double> rhoUW(idxLo, idxHi);
    Array3<double> rhoVV(idxLo, idxHi);
    Array3<double> rhoVW(idxLo, idxHi);
    Array3<double> rhoWW(idxLo, idxHi);
    IsI.initialize(0.0);
    rhoU.initialize(0.0);
    rhoV.initialize(0.0);
    rhoW.initialize(0.0);
    rhoUU.initialize(0.0);
    rhoUV.initialize(0.0);
    rhoUW.initialize(0.0);
    rhoVV.initialize(0.0);
    rhoVW.initialize(0.0);
    rhoWW.initialize(0.0);
    Array3<double> rhoFU;
    Array3<double> rhoFV;
    Array3<double> rhoFW;
    Array3<double> rhoEU;
    Array3<double> rhoEV;
    Array3<double> rhoEW;
    Array3<double> rhoRFU;
    Array3<double> rhoRFV;
    Array3<double> rhoRFW;
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
          // trace has been neglected
          //        double trace = (sij0 + sij1 + sij2)/3.0;
          double trace = 0.0;
          double uvel_cur = ccVel[currCell].x();
          double vvel_cur = ccVel[currCell].y();
          double wvel_cur = ccVel[currCell].z();
          double den_cur = den[currCell];

          IsI[currCell] = isi_cur; 

          //    calculate the grid filtered stress tensor, beta

          (betaIJ[0])[currCell] = den_cur*isi_cur*(sij0-trace);
          (betaIJ[1])[currCell] = den_cur*isi_cur*(sij1-trace);
          (betaIJ[2])[currCell] = den_cur*isi_cur*(sij2-trace);
          (betaIJ[3])[currCell] = den_cur*isi_cur*sij3;
          (betaIJ[4])[currCell] = den_cur*isi_cur*sij4;
          (betaIJ[5])[currCell] = den_cur*isi_cur*sij5;
          // required to compute Leonard term
          rhoUU[currCell] = den_cur*uvel_cur*uvel_cur;
          rhoUV[currCell] = den_cur*uvel_cur*vvel_cur;
          rhoUW[currCell] = den_cur*uvel_cur*wvel_cur;
          rhoVV[currCell] = den_cur*vvel_cur*vvel_cur;
          rhoVW[currCell] = den_cur*vvel_cur*wvel_cur;
          rhoWW[currCell] = den_cur*wvel_cur*wvel_cur;
          rhoU[currCell] = den_cur*uvel_cur;
          rhoV[currCell] = den_cur*vvel_cur;
          rhoW[currCell] = den_cur*wvel_cur;

        }
      }
    }
    TAU_PROFILE_STOP(compute1);
    Array3<double> filterRhoUU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUU.initialize(0.0);
    Array3<double> filterRhoUV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUV.initialize(0.0);
    Array3<double> filterRhoUW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoUW.initialize(0.0);
    Array3<double> filterRhoVV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoVV.initialize(0.0);
    Array3<double> filterRhoVW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoVW.initialize(0.0);
    Array3<double> filterRhoWW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoWW.initialize(0.0);
    Array3<double> filterRhoU(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoU.initialize(0.0);
    Array3<double> filterRhoV(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoV.initialize(0.0);
    Array3<double> filterRhoW(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRhoW.initialize(0.0);

    Array3<double> filterRhoFU;
    Array3<double> filterRhoFV;
    Array3<double> filterRhoFW;
    Array3<double> filterRhoEU;
    Array3<double> filterRhoEV;
    Array3<double> filterRhoEW;
    Array3<double> filterRhoRFU;
    Array3<double> filterRhoRFV;
    Array3<double> filterRhoRFW;

    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();
    double start_turbTime = Time::currentSeconds();

    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoU,   filterVolume, cellType, filterRhoU);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoV,   filterVolume, cellType, filterRhoV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoW,   filterVolume, cellType, filterRhoW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUU,  filterVolume, cellType,  filterRhoUU);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUV,  filterVolume, cellType,  filterRhoUV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoUW,  filterVolume, cellType,  filterRhoUW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoVV,  filterVolume, cellType,  filterRhoVV);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoVW,  filterVolume, cellType,  filterRhoVW);
    d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoWW,  filterVolume, cellType,  filterRhoWW);

    for (int ii = 0; ii < d_lab->d_symTensorMatl->size(); ii++) {
      d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, betaIJ[ii], filterVolume, cellType, betaHATIJ[ii]);
    }

    string msg = "Time for the Filter operation in Turbulence Model: ";
    if (Uintah::Parallel::getNumThreads() > 1) {
      proc0thread0cerr << msg << Time::currentSeconds() - start_turbTime << " seconds\n";
    } else {
      proc0cerr << msg << Time::currentSeconds() - start_turbTime << " seconds\n";
    }

    TAU_PROFILE_START(compute2);

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
          double filterDencur = filterRho[currCell];
          //        ignoring the trace
          //        double trace = (shatij0 + shatij1 + shatij2)/3.0;
          double trace = 0.0;

          IsImag[currCell] = IsI[currCell]; 

          double MIJ0cur = 2.0*filter*filter*
            ((betaHATIJ[0])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij0-trace));
          double MIJ1cur = 2.0*filter*filter*
            ((betaHATIJ[1])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij1-trace));
          double MIJ2cur = 2.0*filter*filter*
            ((betaHATIJ[2])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*(shatij2-trace));
          double MIJ3cur = 2.0*filter*filter*
            ((betaHATIJ[3])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij3);
          double MIJ4cur = 2.0*filter*filter*
            ((betaHATIJ[4])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij4);
          double MIJ5cur =  2.0*filter*filter*
            ((betaHATIJ[5])[currCell]-
             2.0*2.0*filterDencur*IshatIcur*shatij5);


          // compute Leonard stress tensor
          // index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
          double filterRhoUcur = filterRhoU[currCell];
          double filterRhoVcur = filterRhoV[currCell];
          double filterRhoWcur = filterRhoW[currCell];
          double LIJ0cur = filterRhoUU[currCell] -
            filterRhoUcur*filterRhoUcur/filterDencur;
          double LIJ1cur = filterRhoVV[currCell] -
            filterRhoVcur*filterRhoVcur/filterDencur;
          double LIJ2cur = filterRhoWW[currCell] -
            filterRhoWcur*filterRhoWcur/filterDencur;
          double LIJ3cur = filterRhoUV[currCell] -
            filterRhoUcur*filterRhoVcur/filterDencur;
          double LIJ4cur = filterRhoUW[currCell] -
            filterRhoUcur*filterRhoWcur/filterDencur;
          double LIJ5cur = filterRhoVW[currCell] -
            filterRhoVcur*filterRhoWcur/filterDencur;

          // Explicitly making LIJ traceless here
          // Actually, trace has been ignored          
          //        double LIJtrace = (LIJ0cur + LIJ1cur + LIJ2cur)/3.0;
          double LIJtrace = 0.0;
          LIJ0cur = LIJ0cur - LIJtrace;
          LIJ1cur = LIJ1cur - LIJtrace;
          LIJ2cur = LIJ2cur - LIJtrace;

          // compute the magnitude of ML and MM
          MLI[currCell] = MIJ0cur*LIJ0cur +
            MIJ1cur*LIJ1cur +
            MIJ2cur*LIJ2cur +
            2.0*(MIJ3cur*LIJ3cur +
                MIJ4cur*LIJ4cur +
                MIJ5cur*LIJ5cur );
          // calculate absolute value of the grid strain rate
          MMI[currCell] = MIJ0cur*MIJ0cur +
            MIJ1cur*MIJ1cur +
            MIJ2cur*MIJ2cur +
            2.0*(MIJ3cur*MIJ3cur +
                MIJ4cur*MIJ4cur +
                MIJ5cur*MIJ5cur );

        }
      }
    }
    TAU_PROFILE_STOP(compute2);

  }
}



//______________________________________________________________________
//
  void 
CompDynamicProcedure::reComputeSmagCoeff(const ProcessorGroup* pc,
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
    CCVariable<double> ShF; //Shmidt number 
    CCVariable<double> ShE; //Shmidt number 
    CCVariable<double> ShRF; //Shmidt number 
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    CCVariable<double> viscosity;
    CCVariable<double> turbViscosity; 
    constCCVariable<double> filterVolume; 
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(Cs, d_lab->d_CsLabel, indx, patch);
    }
    else {
      new_dw->getModifiable(Cs, d_lab->d_CsLabel, indx, patch);
    }
    Cs.initialize(0.0);

    new_dw->getModifiable(viscosity,         d_lab->d_viscosityCTSLabel,        indx, patch);
    new_dw->getModifiable(turbViscosity,            d_lab->d_turbViscosLabel,              indx, patch);

    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    new_dw->get(IsI, d_lab->d_strainMagnitudeLabel,   indx, patch,   gn, 0);
    // using a box filter of 2*delta...will require more ghost cells if the size of filter is increased
    new_dw->get(MLI, d_lab->d_strainMagnitudeMLLabel, indx, patch, gac, 1);
    new_dw->get(MMI, d_lab->d_strainMagnitudeMMLabel, indx, patch, gac, 1);
    new_dw->get(filterVolume, d_lab->d_filterVolumeLabel, indx, patch, gn, 0); 
    new_dw->get(den, d_lab->d_densityCPLabel, indx, patch,gac, 1);

    if (d_MAlab){
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, indx, patch, gn, 0);
    }

    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);

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
    MLHatI.initialize(0.0);
    IntVector indexLow = patch->getFortranCellLowIndex();
    IntVector indexHigh = patch->getFortranCellHighIndex();

    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, MLI, filterVolume, cellType, MLHatI);
    d_filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, MMI, filterVolume, cellType, MMHatI);

    CCVariable<double> tempCs;
    tempCs.allocate(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    tempCs.initialize(0.0);
    CCVariable<double> tempShF;
    CCVariable<double> tempShE;
    CCVariable<double> tempShRF;
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
      d_filter->applyFilter_noPetsc<Array3<double> >(pc, patch, tempCs, filterVolume, cellType, Cs);
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
            double delta = cellinfo->sew[colX]*
              cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);

            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);

            viscosity[currCell] =  Cs[currCell] * Cs[currCell] *
              filter * filter *
              IsI[currCell] * den[currCell] +
              viscos*voidFraction[currCell];

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
            double delta = cellinfo->sew[colX]*
              cellinfo->sns[colY]*cellinfo->stb[colZ];
            double filter = pow(delta, 1.0/3.0);

            Cs[currCell] = Min(Cs[currCell],10.0);
            Cs[currCell] = factor * sqrt(Cs[currCell]);
            viscosity[currCell] =  Cs[currCell] * Cs[currCell] *
              filter * filter *
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
    int wall_celltypeval = BoundaryCondition::WALL; //d_boundaryCondition->wallCellType();
    if (xminus) {
      int colX = indexLow.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                             *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }
    if (xplus) {
      int colX =  indexHigh.x();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }
    if (yminus) {
      int colY = indexLow.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }
    if (yplus) {
      int colY =  indexHigh.y();
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }
    if (zminus) {
      int colZ = indexLow.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }
    if (zplus) {
      int colZ =  indexHigh.z();
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
        for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          if (cellType[currCell] != wall_celltypeval) {
            viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
            turbViscosity[currCell] = turbViscosity[IntVector(colX,colY,colZ)];
            //          viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)]
            //                    *den[currCell]/den[IntVector(colX,colY,colZ)];
          }          
        }
      }
    }

  }
}
