//----- DynamicProcedure.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/DynamicProcedure.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/Core/Grid/Stencil.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <iostream>

using namespace std;

using namespace Uintah;
using namespace SCIRun;


//****************************************************************************
// Default constructor for SmagorinskyModel
//****************************************************************************
DynamicProcedure::DynamicProcedure(const ArchesLabel* label, 
				   const MPMArchesLabel* MAlb,
				   PhysicalConstants* phyConsts,
				   BoundaryCondition* bndry_cond):
                                    SmagorinskyModel(label, MAlb, phyConsts,
						    bndry_cond)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
DynamicProcedure::~DynamicProcedure()
{
}


//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
DynamicProcedure::problemSetup(const ProblemSpecP& params)
{
  SmagorinskyModel::problemSetup(params);

}

//****************************************************************************
// Schedule compute 
//****************************************************************************
void 
DynamicProcedure::sched_computeTurbSubmodel(const LevelP& level,
						SchedulerP& sched, 
						const PatchSet* patches,
						const MaterialSet* matls)
{
#ifdef PetscFilter
  d_filter->sched_buildFilterMatrix(level, sched);
#endif
  SmagorinskyModel::sched_computeTurbSubmodel(level, sched, patches, matls);
  Task* tsk = scinew Task("DynamicProcedure::TurbSubmodel",
			  this,
			  &DynamicProcedure::computeTurbSubmodel);
  
  sched->addTask(tsk, patches, matls);


}

void
DynamicProcedure::computeTurbSubmodel(const ProcessorGroup* pg,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
#ifdef PetscFilter
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    d_filter->setFilterMatrix(pg, patch, cellinfo);
#endif
  }
}




//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
DynamicProcedure::sched_reComputeTurbSubmodel(SchedulerP& sched, 
					      const PatchSet* patches,
					      const MaterialSet* matls,
					    const int Runge_Kutta_current_step,
					    const bool Runge_Kutta_last_step)
{
  {
    Task* tsk = scinew Task("DynamicProcedure::ReTurbSubmodel",
			    this,
			    &DynamicProcedure::reComputeTurbSubmodel,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}

  if (Runge_Kutta_last_step) {
    tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		  Ghost::AroundFaces,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, 
		  Ghost::AroundFaces,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, 
		  Ghost::AroundFaces,
		  Arches::ONEGHOSTCELL);
  }
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
		tsk->requires(Task::NewDW, d_lab->d_uVelocityPredLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
		tsk->requires(Task::NewDW, d_lab->d_vVelocityPredLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
		tsk->requires(Task::NewDW, d_lab->d_wVelocityPredLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
		tsk->requires(Task::NewDW, d_lab->d_uVelocityIntermLabel,
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
		tsk->requires(Task::NewDW, d_lab->d_vVelocityIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
		tsk->requires(Task::NewDW, d_lab->d_wVelocityIntermLabel, 
		  Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }
    
        
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    // Computes
  if (Runge_Kutta_current_step == Arches::FIRST) 
    tsk->computes(d_lab->d_strainTensorCompLabel, d_lab->d_stressSymTensorMatl,
		  Task::OutOfDomain);
  else 
    tsk->modifies(d_lab->d_strainTensorCompLabel, d_lab->d_stressSymTensorMatl,
		  Task::OutOfDomain);
    
    sched->addTask(tsk, patches, matls);
  }
  {
    Task* tsk = scinew Task("DynamicProcedure::ReComputeFilterValues",
			    this,
			    &DynamicProcedure::reComputeFilterValues,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as a array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
    
  if (Runge_Kutta_last_step) {
    tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel,
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
  }
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityPredLabel,
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityPredLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityPredLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
    tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityIntermLabel,
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityIntermLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityIntermLabel, 
		  Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }
    tsk->requires(Task::NewDW, d_lab->d_strainTensorCompLabel, d_lab->d_stressSymTensorMatl,
		  Task::OutOfDomain, Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    // Computes
  if (Runge_Kutta_current_step == Arches::FIRST) {
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
  {
    Task* tsk = scinew Task("DynamicProcedure::reComputeSmagCoeff",
			    this,
			    &DynamicProcedure::reComputeSmagCoeff,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

    // Requires
    // Assuming one layer of ghost cells
    // initialize with the value of zero at the physical bc's
    // construct a stress tensor and stored as an array with the following order
    // {t11, t12, t13, t21, t22, t23, t31, t23, t33}
  if (Runge_Kutta_last_step) 
    tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    tsk->requires(Task::NewDW, d_lab->d_densityPredLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
    tsk->requires(Task::NewDW, d_lab->d_densityIntermLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMLLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_strainMagnitudeMMLabel, Ghost::AroundCells,
		  Arches::ONEGHOSTCELL);
    tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, Ghost::None,
		  Arches::ZEROGHOSTCELLS);

    // for multimaterial
    if (d_MAlab)
      tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    
    tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    // Computes
  if (Runge_Kutta_last_step)
    tsk->computes(d_lab->d_viscosityCTSLabel);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    tsk->computes(d_lab->d_viscosityPredLabel);
	 break;

	 case Arches::SECOND:
    tsk->computes(d_lab->d_viscosityIntermLabel);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }
  if (Runge_Kutta_current_step == Arches::FIRST) 
    tsk->computes(d_lab->d_CsLabel);
  else 
    tsk->modifies(d_lab->d_CsLabel);
    
    sched->addTask(tsk, patches, matls);
  }
}


//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
DynamicProcedure::reComputeTurbSubmodel(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw,
					const int Runge_Kutta_current_step,
					const bool Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse

  if (Runge_Kutta_last_step) {
    new_dw->get(uVel,d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVel,d_lab->d_vVelocitySPBCLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVel, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  }
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    new_dw->get(uVel,d_lab->d_uVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVel,d_lab->d_vVelocityPredLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVel, d_lab->d_wVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
    new_dw->get(uVel,d_lab->d_uVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVel,d_lab->d_vVelocityIntermLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVel, d_lab->d_wVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }

    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    
    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<CCVariable<double> > SIJ;    //6 point tensor
    for (int ii = 0; ii < d_lab->d_stressSymTensorMatl->size(); ii++) {
    if (Runge_Kutta_current_step == Arches::FIRST) 
      new_dw->allocateAndPut(SIJ[ii], 
			     d_lab->d_strainTensorCompLabel, ii, patch);
    else 
      new_dw->getModifiable(SIJ[ii], 
			     d_lab->d_strainTensorCompLabel, ii, patch);
      SIJ[ii].initialize(0.0);
    }
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
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
	  unp = 0.25*(uVel[IntVector(colX+1,colY,colZ)] + uVel[currCell]
		      + uVel[IntVector(colX+1,colY+1,colZ)] 
		      + uVel[IntVector(colX,colY+1,colZ)]);
	  usp = 0.25*(uVel[IntVector(colX+1,colY,colZ)] + uVel[currCell] +
		      uVel[IntVector(colX+1,colY-1,colZ)] +
		      uVel[IntVector(colX,colY-1,colZ)]);
	  utp = 0.25*(uVel[IntVector(colX+1,colY,colZ)] + uVel[currCell] +
		      uVel[IntVector(colX+1,colY,colZ+1)] + 
		      uVel[IntVector(colX,colY,colZ+1)]);
	  ubp = 0.25*(uVel[IntVector(colX+1,colY,colZ)] + uVel[currCell] + 
		      uVel[IntVector(colX+1,colY,colZ-1)] + 
		      uVel[IntVector(colX,colY,colZ-1)]);
	  vnp = vVel[IntVector(colX,colY+1,colZ)];
	  vsp = vVel[currCell];
	  vep = 0.25*(vVel[IntVector(colX,colY+1,colZ)] + vVel[currCell] +
		      vVel[IntVector(colX+1,colY+1,colZ)] + 
		      vVel[IntVector(colX+1,colY,colZ)]);
	  vwp = 0.25*(vVel[IntVector(colX,colY+1,colZ)] + vVel[currCell] +
		      vVel[IntVector(colX-1,colY+1,colZ)] + 
		      vVel[IntVector(colX-1,colY,colZ)]);
	  vtp = 0.25*(vVel[IntVector(colX,colY+1,colZ)] + vVel[currCell] + 
		      vVel[IntVector(colX,colY+1,colZ+1)] + 
		      vVel[IntVector(colX,colY,colZ+1)]);
	  vbp = 0.25*(vVel[IntVector(colX,colY+1,colZ)] + vVel[currCell] +
		      vVel[IntVector(colX,colY+1,colZ-1)] + 
		      vVel[IntVector(colX,colY,colZ-1)]);

	  wtp = wVel[IntVector(colX,colY,colZ+1)];
	  wbp = wVel[currCell];
	  wep = 0.25*(wVel[IntVector(colX,colY,colZ+1)] + wVel[currCell] + 
		      wVel[IntVector(colX+1,colY,colZ+1)] + 
		      wVel[IntVector(colX+1,colY,colZ)]);
	  wwp = 0.25*(wVel[IntVector(colX,colY,colZ+1)] + wVel[currCell] +
		      wVel[IntVector(colX-1,colY,colZ+1)] + 
		      wVel[IntVector(colX-1,colY,colZ)]);
	  wnp = 0.25*(wVel[IntVector(colX,colY,colZ+1)] + wVel[currCell] + 
		      wVel[IntVector(colX,colY+1,colZ+1)] + 
		      wVel[IntVector(colX,colY+1,colZ)]);
	  wsp = 0.25*(wVel[IntVector(colX,colY,colZ+1)] + wVel[currCell] +
		      wVel[IntVector(colX,colY-1,colZ+1)] + 
		      wVel[IntVector(colX,colY-1,colZ)]);

	  //     calculate the grcolXd stracolXn rate tensor

	  (SIJ[0])[currCell] = (uep-uwp)/cellinfo->sew[colX];
	  (SIJ[1])[currCell] = (vnp-vsp)/cellinfo->sns[colY];
	  (SIJ[2])[currCell] = (wtp-wbp)/cellinfo->stb[colZ];
	  (SIJ[3])[currCell] = 0.5*((unp-usp)/cellinfo->sns[colY] + 
			       (vep-vwp)/cellinfo->sew[colX]);
	  (SIJ[4])[currCell] = 0.5*((utp-ubp)/cellinfo->stb[colZ] + 
			       (wep-wwp)/cellinfo->sew[colX]);
	  (SIJ[5])[currCell] = 0.5*((vtp-vbp)/cellinfo->stb[colZ] + 
			       (wnp-wsp)/cellinfo->sns[colY]);

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
DynamicProcedure::reComputeFilterValues(const ProcessorGroup* pc,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw,
					const int Runge_Kutta_current_step,
					const bool Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> ccuVel;
    constCCVariable<double> ccvVel;
    constCCVariable<double> ccwVel;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse

  if (Runge_Kutta_last_step) {
    new_dw->get(ccuVel, d_lab->d_newCCUVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccvVel, d_lab->d_newCCVVelocityLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccwVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  }
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    new_dw->get(ccuVel, d_lab->d_newCCUVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccvVel, d_lab->d_newCCVVelocityPredLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccwVel, d_lab->d_newCCWVelocityPredLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
    new_dw->get(ccuVel, d_lab->d_newCCUVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccvVel, d_lab->d_newCCVVelocityIntermLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(ccwVel, d_lab->d_newCCWVelocityIntermLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }

    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    
    // Get the patch and variable details
    // compatible with fortran index
    StencilMatrix<constCCVariable<double> > SIJ; //6 point tensor
    for (int ii = 0; ii < d_lab->d_stressSymTensorMatl->size(); ii++)
      new_dw->get(SIJ[ii], d_lab->d_strainTensorCompLabel, ii, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    StencilMatrix<Array3<double> > LIJ;    //6 point tensor
    StencilMatrix<Array3<double> > MIJ;    //6 point tensor
    StencilMatrix<Array3<double> > SHATIJ; //6 point tensor
    StencilMatrix<Array3<double> > betaIJ;  //6 point tensor
    StencilMatrix<Array3<double> > betaHATIJ; //6 point tensor
    int numGC = 1;
    IntVector idxLo = patch->getGhostCellLowIndex(numGC);
    IntVector idxHi = patch->getGhostCellHighIndex(numGC);

    int tensorSize = 6; //  1-> 11, 2->22, 3->33, 4 ->12, 5->13, 6->23
    for (int ii = 0; ii < tensorSize; ii++) {
      LIJ[ii].resize(idxLo, idxHi);
      LIJ[ii].initialize(0.0);
      MIJ[ii].resize(idxLo, idxHi);
      MIJ[ii].initialize(0.0);
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
    if (Runge_Kutta_current_step == Arches::FIRST) {
    new_dw->allocateAndPut(IsImag, 
			   d_lab->d_strainMagnitudeLabel, matlIndex, patch);
    new_dw->allocateAndPut(MLI, 
			   d_lab->d_strainMagnitudeMLLabel, matlIndex, patch);
    new_dw->allocateAndPut(MMI, 
			   d_lab->d_strainMagnitudeMMLabel, matlIndex, patch);
    }
    else {
    new_dw->getModifiable(IsImag, 
			   d_lab->d_strainMagnitudeLabel, matlIndex, patch);
    new_dw->getModifiable(MLI, 
			   d_lab->d_strainMagnitudeMLLabel, matlIndex, patch);
    new_dw->getModifiable(MMI, 
			   d_lab->d_strainMagnitudeMMLabel, matlIndex, patch);
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
    Array3<double> IshatI(idxLo, idxHi); // magnitude of test filter strain rate
    IshatI.initialize(0.0);
    Array3<double> UU(idxLo, idxHi);
    Array3<double> UV(idxLo, idxHi);
    Array3<double> UW(idxLo, idxHi);
    Array3<double> VV(idxLo, idxHi);
    Array3<double> VW(idxLo, idxHi);
    Array3<double> WW(idxLo, idxHi);
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

    for (int colZ = startZ; colZ < endZ; colZ ++) {
      for (int colY = startY; colY < endY; colY ++) {
	for (int colX = startX; colX < endX; colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  // calculate absolute value of the grid strain rate
          // computes for the ghost cells too
	  IsI[currCell] = sqrt(2*((SIJ[0])[currCell]*(SIJ[0])[currCell] + 
				  (SIJ[1])[currCell]*(SIJ[1])[currCell] +
				  (SIJ[2])[currCell]*(SIJ[2])[currCell] +
				  2*((SIJ[3])[currCell]*(SIJ[3])[currCell] + 
				     (SIJ[4])[currCell]*(SIJ[4])[currCell] +
				     (SIJ[5])[currCell]*(SIJ[5])[currCell])));

	  //    calculate the grid filtered stress tensor, beta

	  (betaIJ[0])[currCell] = IsI[currCell]*(SIJ[0])[currCell];
	  (betaIJ[1])[currCell] = IsI[currCell]*(SIJ[1])[currCell];
	  (betaIJ[2])[currCell] = IsI[currCell]*(SIJ[2])[currCell];
	  (betaIJ[3])[currCell] = IsI[currCell]*(SIJ[3])[currCell];
	  (betaIJ[4])[currCell] = IsI[currCell]*(SIJ[4])[currCell];
	  (betaIJ[5])[currCell] = IsI[currCell]*(SIJ[5])[currCell];
	  // required to compute Leonard term
	  UU[currCell] = ccuVel[currCell]*ccuVel[currCell];
	  UV[currCell] = ccuVel[currCell]*ccvVel[currCell];
	  UW[currCell] = ccuVel[currCell]*ccwVel[currCell];
	  VV[currCell] = ccvVel[currCell]*ccvVel[currCell];
	  VW[currCell] = ccvVel[currCell]*ccwVel[currCell];
	  WW[currCell] = ccwVel[currCell]*ccwVel[currCell];
	}
      }
    }
#ifndef PetscFilter
    if (xminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(startX-1, colY, colZ);
	  IntVector prevCell(startX, colY, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
    }
    if (xplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(endX, colY, colZ);
	  IntVector prevCell(endX-1, colY, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
    }
    if (yminus) { 
      for (int colZ = startZ; colZ < endZ; colZ ++) {
	for (int colX = startX; colX < endX; colX ++) {
	  IntVector currCell(colX, startY-1, colZ);
	  IntVector prevCell(colX, startY, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
    }
    if (yplus) {
      for (int colZ = startZ; colZ < endZ; colZ ++) {
	for (int colX = startX; colX < endX; colX ++) {
	  IntVector currCell(colX, endY, colZ);
	  IntVector prevCell(colX, endY-1, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
    }
    if (zminus) { 
      for (int colY = startY; colY < endY; colY ++) {
	for (int colX = startX; colX < endX; colX ++) {
	  IntVector currCell(colX, colY, startZ-1);
	  IntVector prevCell(colX, colY, startZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
			       
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
			       
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
			       

	}
      }
    }
    if (zplus) {
      for (int colY = startY; colY < endY; colY ++) {
	for (int colX = startX; colX < endX; colX ++) {
	  IntVector currCell(colX, colY, endZ);
	  IntVector prevCell(colX, colY, endZ-1);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
			       
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
			       
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
			       

	}
      }
    }
    // fill the corner cells
    if (xminus) {
      if (yminus) {
	for (int colZ = startZ; colZ < endZ; colZ ++) {
	  IntVector currCell(startX-1, startY-1, colZ);
	  IntVector prevCell(startX, startY, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (yplus) {
	for (int colZ = startZ; colZ < endZ; colZ ++) {
	  IntVector currCell(startX-1, endY, colZ);
	  IntVector prevCell(startX, endY-1, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (zminus) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(startX-1, colY, startZ-1);
	  IntVector prevCell(startX, colY, startZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (zplus) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(startX-1, colY, endZ);
	  IntVector prevCell(startX, colY, endZ-1);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (yminus&&zminus) {
	IntVector currCell(startX-1, startY-1, startZ-1);
	IntVector prevCell(startX, startY, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yminus&&zplus) {
	IntVector currCell(startX-1, startY-1, endZ);
	IntVector prevCell(startX, startY, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yplus&&zminus) {
	IntVector currCell(startX-1, endY, startZ-1);
	IntVector prevCell(startX, endY-1, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yplus&&zplus) {
	IntVector currCell(startX-1, endY, endZ);
	IntVector prevCell(startX, endY-1, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
	
    }
    if (xplus) {
      if (yminus) {
	for (int colZ = startZ; colZ < endZ; colZ ++) {
	  IntVector currCell(endX, startY-1, colZ);
	  IntVector prevCell(endX-1, startY, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (yplus) {
	for (int colZ = startZ; colZ < endZ; colZ ++) {
	  IntVector currCell(endX, endY, colZ);
	  IntVector prevCell(endX-1, endY-1, colZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (zminus) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(endX, colY, startZ-1);
	  IntVector prevCell(endX-1, colY, startZ);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (zplus) {
	for (int colY = startY; colY < endY; colY ++) {
	  IntVector currCell(endX, colY, endZ);
	  IntVector prevCell(endX-1, colY, endZ-1);
	  (betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	  (betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	  (betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	  (betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	  (betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	  (betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
	}
      }
      if (yminus&&zminus) {
	IntVector currCell(endX, startY-1, startZ-1);
	IntVector prevCell(endX-1, startY, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yminus&&zplus) {
	IntVector currCell(endX, startY-1, endZ);
	IntVector prevCell(endX-1, startY, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yplus&&zminus) {
	IntVector currCell(endX, endY, startZ-1);
	IntVector prevCell(endX-1, endY-1, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
      if (yplus&&zplus) {
	IntVector currCell(endX, endY, endZ);
	IntVector prevCell(endX-1, endY-1, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
	
    }
    // for yminus&&zminus fill the corner cells for all internal x
    if (yminus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
	IntVector currCell(colX, startY-1, startZ-1);
	IntVector prevCell(colX, startY, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
    }
    if (yminus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
	IntVector currCell(colX, startY-1, endZ);
	IntVector prevCell(colX, startY, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
    }
    if (yplus&&zminus) {
      for (int colX = startX; colX < endX; colX++) {
	IntVector currCell(colX, endY, startZ-1);
	IntVector prevCell(colX, endY-1, startZ);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	  UU[currCell] = UU[prevCell];
	  UV[currCell] = UV[prevCell];
	  UW[currCell] = UW[prevCell];
	  VV[currCell] = VV[prevCell];
	  VW[currCell] = VW[prevCell];
	  WW[currCell] = WW[prevCell];
      }
    }
    if (yplus&&zplus) {
      for (int colX = startX; colX < endX; colX++) {
	IntVector currCell(colX, endY, endZ);
	IntVector prevCell(colX, endY-1, endZ-1);
	(betaIJ[0])[currCell] =  (betaIJ[0])[prevCell];
	(betaIJ[1])[currCell] =  (betaIJ[1])[prevCell];
	(betaIJ[2])[currCell] =  (betaIJ[2])[prevCell];
	(betaIJ[3])[currCell] =  (betaIJ[3])[prevCell];
	(betaIJ[4])[currCell] =  (betaIJ[4])[prevCell];
	(betaIJ[5])[currCell] =  (betaIJ[5])[prevCell];
	UU[currCell] = UU[prevCell];
	UV[currCell] = UV[prevCell];
	UW[currCell] = UW[prevCell];
	VV[currCell] = VV[prevCell];
	VW[currCell] = VW[prevCell];
	WW[currCell] = WW[prevCell];
      }
    }	
#endif
    Array3<double> filterUU(patch->getLowIndex(), patch->getHighIndex());
    filterUU.initialize(0.0);
    Array3<double> filterUV(patch->getLowIndex(), patch->getHighIndex());
    filterUV.initialize(0.0);
    Array3<double> filterUW(patch->getLowIndex(), patch->getHighIndex());
    filterUW.initialize(0.0);
    Array3<double> filterVV(patch->getLowIndex(), patch->getHighIndex());
    filterVV.initialize(0.0);
    Array3<double> filterVW(patch->getLowIndex(), patch->getHighIndex());
    filterVW.initialize(0.0);
    Array3<double> filterWW(patch->getLowIndex(), patch->getHighIndex());
    filterWW.initialize(0.0);
    Array3<double> filterUVel(patch->getLowIndex(), patch->getHighIndex());
    filterUVel.initialize(0.0);
    Array3<double> filterVVel(patch->getLowIndex(), patch->getHighIndex());
    filterVVel.initialize(0.0);
    Array3<double> filterWVel(patch->getLowIndex(), patch->getHighIndex());
    filterWVel.initialize(0.0);
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
#ifdef PetscFilter
#if 0
    cerr << "In the Dynamic Procedure print ccuvel" << endl;
    ccuVel.print(cerr);
#endif

    d_filter->applyFilter(pc, patch,ccuVel, filterUVel);
#if 0
    cerr << "In the Dynamic Procedure after filter" << endl;
    filterUVel.print(cerr);
#endif
    d_filter->applyFilter(pc, patch,ccvVel, filterVVel);
    d_filter->applyFilter(pc, patch,ccwVel, filterWVel);
    d_filter->applyFilter(pc, patch,UU, filterUU);
    d_filter->applyFilter(pc, patch,UV, filterUV);
    d_filter->applyFilter(pc, patch,UW, filterUW);
    d_filter->applyFilter(pc, patch,VV, filterVV);
    d_filter->applyFilter(pc, patch,VW, filterVW);
    d_filter->applyFilter(pc, patch,WW, filterWW);
    for (int ii = 0; ii < d_lab->d_stressSymTensorMatl->size(); ii++) {
      d_filter->applyFilter(pc, patch,SIJ[ii], SHATIJ[ii]);
      d_filter->applyFilter(pc, patch,betaIJ[ii], betaHATIJ[ii]);
    }

#else
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double delta = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	  double filter = pow(delta, 1.0/3.0);
	  double cube_delta = (2.0*cellinfo->sew[colX])*(2.0*cellinfo->sns[colY])*
                 	    (2.0*cellinfo->stb[colZ]);
	  double invDelta = 1.0/cube_delta;
	  filterUVel[currCell] = 0.0;
	  filterVVel[currCell] = 0.0;
	  filterWVel[currCell] = 0.0;
	  filterUU[currCell] = 0.0;
	  filterUV[currCell] = 0.0;
	  filterUW[currCell] = 0.0;
	  filterVV[currCell] = 0.0;
	  filterVW[currCell] = 0.0;
	  filterWW[currCell] = 0.0;
	  
	  for (int kk = -1; kk <= 1; kk ++) {
	    for (int jj = -1; jj <= 1; jj ++) {
	      for (int ii = -1; ii <= 1; ii ++) {
		IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		             cellinfo->stb[colZ+kk]*
		             (1.0-0.5*abs(ii))*
		             (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
		filterUVel[currCell] += ccuVel[filterCell]*vol*invDelta; 
		filterVVel[currCell] += ccvVel[filterCell]*vol*invDelta; 
		filterWVel[currCell] += ccwVel[filterCell]*vol*invDelta;
		filterUU[currCell] += UU[filterCell]*vol*invDelta;  
		filterUV[currCell] += UV[filterCell]*vol*invDelta;  
		filterUW[currCell] += UW[filterCell]*vol*invDelta;  
		filterVV[currCell] += VV[filterCell]*vol*invDelta;  
		filterVW[currCell] += VW[filterCell]*vol*invDelta;  
		filterWW[currCell] += WW[filterCell]*vol*invDelta;  

		(SHATIJ[0])[currCell] += (SIJ[0])[filterCell]*vol*invDelta;
		(SHATIJ[1])[currCell] += (SIJ[1])[filterCell]*vol*invDelta;
		(SHATIJ[2])[currCell] += (SIJ[2])[filterCell]*vol*invDelta;
		(SHATIJ[3])[currCell] += (SIJ[3])[filterCell]*vol*invDelta;
		(SHATIJ[4])[currCell] += (SIJ[4])[filterCell]*vol*invDelta;
		(SHATIJ[5])[currCell] += (SIJ[5])[filterCell]*vol*invDelta;
				     

		(betaHATIJ[0])[currCell] += (betaIJ[0])[filterCell]*vol*invDelta;
		(betaHATIJ[1])[currCell] += (betaIJ[1])[filterCell]*vol*invDelta;
		(betaHATIJ[2])[currCell] += (betaIJ[2])[filterCell]*vol*invDelta;
		(betaHATIJ[3])[currCell] += (betaIJ[3])[filterCell]*vol*invDelta;
		(betaHATIJ[4])[currCell] += (betaIJ[4])[filterCell]*vol*invDelta;
		(betaHATIJ[5])[currCell] += (betaIJ[5])[filterCell]*vol*invDelta;
	      }
	    }
	  }
	}
      }
    }
#endif  
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double delta = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	  double filter = pow(delta, 1.0/3.0);

	  // test filter width is assumed to be twice that of the basic filter
	  // needs following modifications:
	  // a) make the test filter work for anisotropic grid
          // b) generalize the filter operation
	  IsImag[currCell] = IsI[currCell];
	  IshatI[currCell] = sqrt(2.0*((SHATIJ[0])[currCell]*(SHATIJ[0])[currCell] + 
				     (SHATIJ[1])[currCell]*(SHATIJ[1])[currCell] +
				     (SHATIJ[2])[currCell]*(SHATIJ[2])[currCell] +
				     2.0*((SHATIJ[3])[currCell]*(SHATIJ[3])[currCell] + 
					(SHATIJ[4])[currCell]*(SHATIJ[4])[currCell] +
					(SHATIJ[5])[currCell]*(SHATIJ[5])[currCell])));
	  (MIJ[0])[currCell] = 2.0*(filter*filter)*
	                       ((betaHATIJ[0])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[0])[currCell]);
	  (MIJ[1])[currCell] = 2.0*(filter*filter)*
	                       ((betaHATIJ[1])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[1])[currCell]);
	  (MIJ[2])[currCell] = 2.0*(filter*filter)*
	                       ((betaHATIJ[2])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[2])[currCell]);
	  (MIJ[3])[currCell] = 2.0*(filter*filter)*
	                       ((betaHATIJ[3])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[3])[currCell]);
	  (MIJ[4])[currCell] = 2.0*(filter*filter)*
	                       ((betaHATIJ[4])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[4])[currCell]);
	  (MIJ[5])[currCell] =  2.0*(filter*filter)*
	                       ((betaHATIJ[5])[currCell]-
				2.0*2.0*IshatI[currCell]*(SHATIJ[5])[currCell]);


	  // compute Leonard stress tensor
	  // index 0: L11, 1:L22, 2:L33, 3:L12, 4:L13, 5:L23
	  (LIJ[0])[currCell] = (filterUU[currCell] -
				filterUVel[currCell]*
				filterUVel[currCell]);
	  (LIJ[1])[currCell] = (filterVV[currCell] -
				filterVVel[currCell]*
				filterVVel[currCell]);
	  (LIJ[2])[currCell] = (filterWW[currCell] -
				filterWVel[currCell]*
				filterWVel[currCell]);
	  (LIJ[3])[currCell] = (filterUV[currCell] -
				filterUVel[currCell]*
				filterVVel[currCell]);
	  (LIJ[4])[currCell] = (filterUW[currCell] -
				filterUVel[currCell]*
				filterWVel[currCell]);
	  (LIJ[5])[currCell] = (filterVW[currCell] -
				filterVVel[currCell]*
				filterWVel[currCell]);

	  // compute the magnitude of ML and MM
	  MLI[currCell] = (MIJ[0])[currCell]*(LIJ[0])[currCell] +
	                 (MIJ[1])[currCell]*(LIJ[1])[currCell] +
	                 (MIJ[2])[currCell]*(LIJ[2])[currCell] +
                         2.0*((MIJ[3])[currCell]*(LIJ[3])[currCell] +
			      (MIJ[4])[currCell]*(LIJ[4])[currCell] +
			      (MIJ[5])[currCell]*(LIJ[5])[currCell] );
	  MMI[currCell] = (MIJ[0])[currCell]*(MIJ[0])[currCell] +
	                 (MIJ[1])[currCell]*(MIJ[1])[currCell] +
	                 (MIJ[2])[currCell]*(MIJ[2])[currCell] +
                         2.0*((MIJ[3])[currCell]*(MIJ[3])[currCell] +
			      (MIJ[4])[currCell]*(MIJ[4])[currCell] +
			      (MIJ[5])[currCell]*(MIJ[5])[currCell] );
		// calculate absolute value of the grid strain rate
	}
      }
    }
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


  }
}




void 
DynamicProcedure::reComputeSmagCoeff(const ProcessorGroup* pc,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse*,
				     DataWarehouse* new_dw,
				     const int Runge_Kutta_current_step,
				     const bool Runge_Kutta_last_step)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> IsI;
    constCCVariable<double> MLI;
    constCCVariable<double> MMI;
    CCVariable<double> Cs; //smag coeff 
    constCCVariable<double> den;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    CCVariable<double> viscosity;
    if (Runge_Kutta_current_step == Arches::FIRST) 
       new_dw->allocateAndPut(Cs, d_lab->d_CsLabel, matlIndex, patch);
    else
       new_dw->getModifiable(Cs, d_lab->d_CsLabel, matlIndex, patch);
    Cs.initialize(0.0);
    if (Runge_Kutta_last_step)
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityPredLabel, matlIndex, patch);
	 break;

	 case Arches::SECOND:
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityIntermLabel, matlIndex, patch);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }
    new_dw->copyOut(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(IsI,d_lab->d_strainMagnitudeLabel, matlIndex, patch, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
    // using a box filter of 2*delta...will require more ghost cells if the size of filter is increased
    new_dw->get(MLI,d_lab->d_strainMagnitudeMLLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(MMI, d_lab->d_strainMagnitudeMMLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    if (Runge_Kutta_last_step) 
       new_dw->get(den, d_lab->d_densityCPLabel, matlIndex, patch,
		   Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
       new_dw->get(den, d_lab->d_densityPredLabel, matlIndex, patch,
		   Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
       new_dw->get(den, d_lab->d_densityIntermLabel, matlIndex, patch,
		   Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
    }
    if (d_MAlab)
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // get physical constants
    double viscos; // molecular viscosity
    viscos = d_physicalConsts->getMolecularViscosity();
 

    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian
    Array3<double> MLHatI(patch->getLowIndex(), patch->getHighIndex()); // magnitude of strain rate
    MLHatI.initialize(0.0);
    Array3<double> MMHatI(patch->getLowIndex(), patch->getHighIndex()); // magnitude of test filter strain rate
    MMHatI.initialize(0.0);
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch, MLI, MLHatI);
    d_filter->applyFilter(pc, patch, MMI, MMHatI);
#else
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double cube_delta = (2.0*cellinfo->sew[colX])*(2.0*cellinfo->sns[colY])*
                 	    (2.0*cellinfo->stb[colZ]);
	  double delta = pow(cellinfo->sew[colX]*cellinfo->sns[colY]*
                 	      cellinfo->stb[colZ],1.0/3.0);
	  double invDelta = 1.0/cube_delta;
	  for (int kk = -1; kk <= 1; kk ++) {
	    for (int jj = -1; jj <= 1; jj ++) {
	      for (int ii = -1; ii <= 1; ii ++) {
		IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		             cellinfo->stb[colZ+kk]*
		             (1.0-0.5*abs(ii))*
		             (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
				     
		// calculate absolute value of the grid strain rate
		MLHatI[currCell] += MLI[filterCell]*vol*invDelta;
		MMHatI[currCell] += MMI[filterCell]*vol*invDelta;
	      }
	    }
	  }
	}
      }
    }
#endif
    
	  //     calculate the local Smagorinsky coefficient
	  //     perform "clipping" in case MLij is negative...
    double factor = 1.0;
#if 1
    if (time < 2.0)
      factor = (time+0.000001)*0.5;
#endif
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double delta = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	  double filter = pow(delta, 1.0/3.0);
	  if (MLHatI[currCell] < 0.0)
	    MLHatI[currCell] = 0.0;

	  //     calculate the effective viscosity

	  //     handle the case where we divide by zero

	  if ((MMHatI[currCell] < 1.0e-20)||(MLHatI[currCell]/MMHatI[currCell] > 100.0)) 
	    {
	      viscosity[currCell] = viscos;
	      Cs[currCell] = sqrt(MLHatI[currCell]/MMHatI[currCell]);
	    }
	  else {
	    viscosity[currCell] = factor*factor*
	                          MLHatI[currCell]/MMHatI[currCell]
	                          *filter*filter*IsI[currCell]*den[currCell] + viscos;
	    Cs[currCell] = sqrt(MLHatI[currCell]/MMHatI[currCell]);
	    Cs[currCell] *= factor;
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
    int wallID = d_boundaryCondition->wallCellType();
    if (xminus) {
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
	for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
	  int colX = indexLow.x();
	  IntVector currCell(colX-1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (xplus) {
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
	for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
	  int colX =  indexHigh.x();
	  IntVector currCell(colX+1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yminus) {
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
	for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
	  int colY = indexLow.y();
	  IntVector currCell(colX, colY-1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yplus) {
      for (int colZ = indexLow.z(); colZ <=  indexHigh.z(); colZ ++) {
	for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
	  int colY =  indexHigh.y();
	  IntVector currCell(colX, colY+1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zminus) {
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
	  int colZ = indexLow.z();
	  IntVector currCell(colX, colY, colZ-1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zplus) {
      for (int colY = indexLow.y(); colY <=  indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <=  indexHigh.x(); colX ++) {
	  int colZ =  indexHigh.z();
	  IntVector currCell(colX, colY, colZ+1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }

  }
}


void 
DynamicProcedure::sched_computeScalarVariance(SchedulerP& sched, 
					      const PatchSet* patches,
					      const MaterialSet* matls,
					    const int Runge_Kutta_current_step,
					    const bool Runge_Kutta_last_step)
{
  Task* tsk = scinew Task("DynamicProcedure::computeScalarVar",
			  this,
			  &DynamicProcedure::computeScalarVariance,
			  Runge_Kutta_current_step, Runge_Kutta_last_step);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  if (Runge_Kutta_last_step)
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
  tsk->requires(Task::NewDW, d_lab->d_scalarIntermLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
  }

  // Computes
  if (Runge_Kutta_current_step == Arches::FIRST)
     tsk->computes(d_lab->d_scalarVarSPLabel);
  else
     tsk->modifies(d_lab->d_scalarVarSPLabel);

  sched->addTask(tsk, patches, matls);
}


void 
DynamicProcedure::computeScalarVariance(const ProcessorGroup* pc,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw,
					const int Runge_Kutta_current_step,
					const bool Runge_Kutta_last_step)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> scalar;
    CCVariable<double> scalarVar;
    // Get the velocity, density and viscosity from the old data warehouse
    if (Runge_Kutta_last_step)
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    else { 
	 switch (Runge_Kutta_current_step) {
	 case Arches::FIRST:
    new_dw->get(scalar, d_lab->d_scalarPredLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 case Arches::SECOND:
    new_dw->get(scalar, d_lab->d_scalarIntermLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
	 break;

	 default:
		throw InvalidValue("Invalid Runge-Kutta step in DynamicProcedure");
	 }
    }
    if (Runge_Kutta_current_step == Arches::FIRST)
    	new_dw->allocateAndPut(scalarVar, d_lab->d_scalarVarSPLabel, matlIndex,
			       patch);
    else
    	new_dw->getModifiable(scalarVar, d_lab->d_scalarVarSPLabel, matlIndex,
			       patch);
    scalarVar.initialize(0.0);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    int numGC = 1;
    IntVector idxLo = patch->getGhostCellLowIndex(numGC);
    IntVector idxHi = patch->getGhostCellHighIndex(numGC);
    Array3<double> phiSqr(idxLo, idxHi);

    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  phiSqr[currCell] = scalar[currCell]*scalar[currCell];
	}
      }
    }

    Array3<double> filterPhi(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterPhiSqr(patch->getLowIndex(), patch->getHighIndex());
    filterPhi.initialize(0.0);
    filterPhiSqr.initialize(0.0);

    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch,scalar, filterPhi);
    d_filter->applyFilter(pc, patch,phiSqr, filterPhiSqr);
#else

    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double cube_delta = (2.0*cellinfo->sew[colX])*(2.0*cellinfo->sns[colY])*
                 	    (2.0*cellinfo->stb[colZ]);
	  double invDelta = 1.0/cube_delta;
	  for (int kk = -1; kk <= 1; kk ++) {
	    for (int jj = -1; jj <= 1; jj ++) {
	      for (int ii = -1; ii <= 1; ii ++) {
		IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		             cellinfo->stb[colZ+kk]*
		             (1.0-0.5*abs(ii))*
		             (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
		filterPhi[currCell] += scalar[filterCell]*vol; 
		filterPhiSqr[currCell] += phiSqr[filterCell]*vol; 
	      }
	    }
	  }
	  
	  filterPhi[currCell] *= invDelta;
	  filterPhiSqr[currCell] *= invDelta;
	}
      }
    }
#endif
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);


	  // compute scalar variance
	  scalarVar[currCell] = d_CFVar*(filterPhiSqr[currCell]-
					 (filterPhi[currCell]*filterPhi[currCell]));
	}
      }
    }
    // Put the calculated viscosityvalue into the new data warehouse
  }
}
