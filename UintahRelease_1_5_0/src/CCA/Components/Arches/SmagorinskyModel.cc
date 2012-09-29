/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- SmagorinskyModel.cc --------------------------------------------------

#include <CCA/Components/Arches/SmagorinskyModel.h>
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

#include <CCA/Components/Arches/fortran/scalarvarmodel_fort.h>

//****************************************************************************
// Default constructor for SmagorinkyModel
//****************************************************************************
SmagorinskyModel::SmagorinskyModel(const ArchesLabel* label, 
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
SmagorinskyModel::~SmagorinskyModel()
{
}

//****************************************************************************
//  Get the molecular viscosity from the Physical Constants object 
//****************************************************************************
double 
SmagorinskyModel::getMolecularViscosity() const {
  return d_physicalConsts->getMolecularViscosity();
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
SmagorinskyModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Turbulence");
  db->require("cf", d_CF);
  db->require("fac_mesh", d_factorMesh);
  db->require("filterl", d_filterl);
  if (d_calcVariance) {
    proc0cout << "Smagorinsky type model will be used to model variance" << endl;
    db->require("variance_coefficient",d_CFVar); // const reqd by variance eqn

    db->getWithDefault("mixture_fraction_label",d_mix_frac_label_name, "scalarSP"); 
    d_mf_label = VarLabel::find( d_mix_frac_label_name );
    proc0cout << "Using " << *d_mf_label << " to compute scalar variance." << endl;
    
    proc0cout << "WARNING! Scalar filtering for variance limit is not supported" << endl;
    proc0cout << "by this model. Possibly high variance values would be generated" << endl;
  }

  // actually, Shmidt number, not Prandtl number
  d_turbPrNo = 0.0;
  if (db->findBlock("turbulentPrandtlNumber")){
    db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
  }
}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
SmagorinskyModel::sched_reComputeTurbSubmodel(SchedulerP& sched, 
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels)
{
  string taskname =  "SmagorinskyModel::ReTurbSubmodel" +
                     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
                          &SmagorinskyModel::reComputeTurbSubmodel,
                          timelabels);

  // Requires
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf = Ghost::AroundFaces;
  Ghost::GhostType  gn = Ghost::None; 
   
  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, gn);
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel,      gn, 0);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel,  gaf, 1);
  tsk->requires(Task::NewDW, d_lab->d_CCVelocityLabel, gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,       gac, 1);
  // for multimaterial
  if (d_MAlab){
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, gn, 0);
  }

  tsk->modifies(d_lab->d_viscosityCTSLabel);
  tsk->modifies(d_lab->d_tauSGSLabel); 

  sched->addTask(tsk, patches, matls);
}


//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
SmagorinskyModel::reComputeTurbSubmodel(const ProcessorGroup*,
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

    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    
    constCCVariable<Vector> VelocityCC;
    constCCVariable<double> density;
    
    CCVariable<double> viscosity;
    CCVariable<double> tauSGS; 
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gn = Ghost::None; 
     
    
    new_dw->getModifiable(viscosity, d_lab->d_viscosityCTSLabel,indx, patch);
    new_dw->getModifiable(tauSGS,    d_lab->d_tauSGSLabel, indx, patch ); 
                           
    new_dw->get(uVelocity, d_lab->d_uVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPBCLabel, indx, patch, gaf, 1);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, indx, patch, gaf, 1);
                
    new_dw->get(density,     d_lab->d_densityCPLabel,      indx, patch, gn,  0);
    new_dw->get(VelocityCC, d_lab->d_CCVelocityLabel, indx, patch, gac, 1);
    
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
    double CF = d_CF;
#if 0
    if (time < 2.0 ) 
      CF *= (time+ 0.0001)*0.5;
#endif      

    Vector Dx = patch->dCell(); 
    double dmesh = Dx.x()*Dx.y()*Dx.z(); 
    dmesh = pow(dmesh,1.0/3.0);

    double pmixl = CF * max(d_filterl, d_factorMesh*dmesh); 

    for ( CellIterator iter=patch->getCellIterator(); !iter.done(); ++iter ){ 

      IntVector c = *iter; 

      viscosity[c] = compute_smag_viscos( uVelocity, vVelocity, wVelocity, 
          VelocityCC, density, pmixl, Dx, c ); 

      tauSGS[c] = viscosity[c]; 

      viscosity[c] += mol_viscos; 

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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] = tauSGS[IntVector(colX,colY,colZ)];
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
            tauSGS[currCell] *=  voidFraction[currCell];
          }
        }
      }
    }
  }
}


//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
SmagorinskyModel::sched_computeScalarVariance(SchedulerP& sched, 
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels)
{
  string taskname =  "SmagorinskyModel::computeScalarVaraince" +
                     timelabels->integrator_step_name;
                     
  Task* tsk = scinew Task(taskname, this,
                          &SmagorinskyModel::computeScalarVariance,
                          timelabels);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  Ghost::GhostType  gac = Ghost::AroundCells;
  tsk->requires(Task::NewDW, d_mf_label,             gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, gac, 1);

  // Computes
  if ((timelabels->integrator_step_number == TimeIntegratorStepNumber::First)) {
    tsk->computes(d_lab->d_scalarVarSPLabel);
    tsk->computes(d_lab->d_normalizedScalarVarLabel);
  }
  else {
    tsk->modifies(d_lab->d_scalarVarSPLabel);
    tsk->modifies(d_lab->d_normalizedScalarVarLabel);
  }

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
//
void 
SmagorinskyModel::computeScalarVariance(const ProcessorGroup*,
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
    constCCVariable<double> scalar;
    CCVariable<double> scalarVar;
    CCVariable<double> normalizedScalarVar;
    
    // Get the velocity, density and viscosity from the old data warehouse
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get(scalar, d_mf_label,  indx, patch,gac, 1);
    
    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      new_dw->allocateAndPut(scalarVar,           d_lab->d_scalarVarSPLabel,         indx,patch);
      new_dw->allocateAndPut(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, indx,patch);
    }
    else {
      new_dw->getModifiable(scalarVar,           d_lab->d_scalarVarSPLabel,         indx,patch);
      new_dw->getModifiable(normalizedScalarVar, d_lab->d_normalizedScalarVarLabel, indx,patch);
    }
    scalarVar.initialize(0.0);
    normalizedScalarVar.initialize(0.0);

    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // compatible with fortran index
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    double CFVar = d_CFVar;
#if 0
    if (time < 2.0 ) 
      CFVar *= (time+ 0.0001)*0.5;
#endif
    fort_scalarvarmodel(scalar, idxLo, idxHi, scalarVar, cellinfo->dxpw,
                        cellinfo->dyps, cellinfo->dzpb, cellinfo->sew,
                        cellinfo->sns, cellinfo->stb, CFVar, d_factorMesh,
                        d_filterl);

    double small = 1.0e-10;
    double var_limit = 0.0;
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;

      // check variance bounds and normalize
      var_limit = scalar[c] * (1.0 - scalar[c]);

      if(scalarVar[c] < small){
        scalarVar[c] = 0.0;
      }
      if(scalarVar[c] > var_limit){
        scalarVar[c] = var_limit;
      }
      normalizedScalarVar[c] = scalarVar[c]/(var_limit+small);
    }

    
    //__________________________________
    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;
    int outlet_celltypeval = d_boundaryCondition->outletCellType();
    int pressure_celltypeval = d_boundaryCondition->pressureCellType();
    
    if (xminus) {       // xminus
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    
    if (xplus) {      // xplus
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (yminus) {   // yminus
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (yplus) {    // yplus
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (zminus) {   // zminus
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          if ((cellType[currCell] == outlet_celltypeval)||
            (cellType[currCell] == pressure_celltypeval))
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
        }
      }
    }
    if (zplus) {  // zplus
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
            
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]) {
              scalarVar[currCell] = scalarVar[IntVector(colX,colY,colZ)];
              normalizedScalarVar[currCell] = 
                          normalizedScalarVar[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
  }
}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
SmagorinskyModel::sched_computeScalarDissipation(SchedulerP& sched, 
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls,
                                                 const TimeIntegratorLabel* timelabels)
{
  string taskname =  "SmagorinskyModel::computeScalarDissipation" +
                     timelabels->integrator_step_name;
 
  Task* tsk = scinew Task(taskname, this,
                          &SmagorinskyModel::computeScalarDissipation,
                          timelabels);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  // assuming scalar dissipation is computed before turbulent viscosity calculation 
  Ghost::GhostType  gac = Ghost::AroundCells;
  tsk->requires(Task::NewDW, d_mf_label,                gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,gac, 1);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel,    gac, 1);

  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First ){
    tsk->computes(d_lab->d_scalarDissSPLabel);
  }else{
    tsk->modifies(d_lab->d_scalarDissSPLabel);
  }

  tsk->requires(Task::NewDW, d_lab->d_cellInfoLabel, Ghost::None);

  sched->addTask(tsk, patches, matls);
}

//______________________________________________________________________
//
void 
SmagorinskyModel::computeScalarDissipation(const ProcessorGroup*,
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
    constCCVariable<double> viscosity;
    constCCVariable<double> scalar;
    CCVariable<double> scalarDiss;  // dissipation..chi
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    new_dw->get(scalar,    d_mf_label,                 indx, patch, gac, 1);
    new_dw->get(viscosity, d_lab->d_viscosityCTSLabel, indx, patch,gac, 1);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut(scalarDiss, d_lab->d_scalarDissSPLabel,indx, patch);
    }else{
      new_dw->getModifiable(scalarDiss, d_lab->d_scalarDissSPLabel, indx, patch);
    }
    scalarDiss.initialize(0.0);

    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel, indx, patch, gac, 1);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, indx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      int colX = c.x();
      int colY = c.y();
      int colZ = c.z();

      double scale = 0.5*(scalar[c] + scalar[IntVector(colX+1,colY,colZ)]);
      double scalw = 0.5*(scalar[c] + scalar[IntVector(colX-1,colY,colZ)]);
      double scaln = 0.5*(scalar[c] + scalar[IntVector(colX,colY+1,colZ)]);
      double scals = 0.5*(scalar[c] + scalar[IntVector(colX,colY-1,colZ)]);
      double scalt = 0.5*(scalar[c] + scalar[IntVector(colX,colY,colZ+1)]);
      double scalb = 0.5*(scalar[c] + scalar[IntVector(colX,colY,colZ-1)]);
      
      double dfdx = (scale-scalw)/cellinfo->sew[colX];
      double dfdy = (scaln-scals)/cellinfo->sns[colY];
      double dfdz = (scalt-scalb)/cellinfo->stb[colZ];
      scalarDiss[c] = viscosity[c]/d_turbPrNo*
                            (dfdx*dfdx + dfdy*dfdy + dfdz*dfdz); 
    }

    
    //__________________________________
    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;
    int outlet_celltypeval = d_boundaryCondition->outletCellType();
    int pressure_celltypeval = d_boundaryCondition->pressureCellType();
    
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();
    
    if (xminus) {
      int colX = idxLo.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX-1, colY, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
            
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (xplus) {
      int colX = idxHi.x();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
          IntVector currCell(colX+1, colY, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
          
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (yminus) {
      int colY = idxLo.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY-1, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (yplus) {
      int colY = idxHi.y();
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY+1, colZ);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (zminus) {
      int colZ = idxLo.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ-1);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
    if (zplus) {
      int colZ = idxHi.z();
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
          IntVector currCell(colX, colY, colZ+1);
          
          if ((cellType[currCell] == outlet_celltypeval)||
              (cellType[currCell] == pressure_celltypeval)){
              
            if (scalar[currCell] == scalar[IntVector(colX,colY,colZ)]){
              scalarDiss[currCell] = scalarDiss[IntVector(colX,colY,colZ)];
            }
          }
        }
      }
    }
  }
}
