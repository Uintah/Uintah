//----- ScaleSimilarityModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
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
ScaleSimilarityModel::ScaleSimilarityModel(const ArchesLabel* label, 
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
ScaleSimilarityModel::~ScaleSimilarityModel()
{
}


//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
ScaleSimilarityModel::problemSetup(const ProblemSpecP& params)
{
  SmagorinskyModel::problemSetup(params);
  ProblemSpecP db = params->findBlock("ScaleSimilarity");
  db->require("cf", d_CF);
}

//****************************************************************************
// Schedule compute 
//****************************************************************************
void 
ScaleSimilarityModel::sched_computeTurbSubmodel(const LevelP& level,
						SchedulerP& sched, 
						const PatchSet* patches,
						const MaterialSet* matls)
{
#ifdef PetscFilter
  d_filter->sched_buildFilterMatrix(level, sched);
#endif
  SmagorinskyModel::sched_computeTurbSubmodel(level, sched, patches, matls);
  Task* tsk = scinew Task("ScaleSimilarityModel::TurbSubmodel",
			  this,
			  &ScaleSimilarityModel::computeTurbSubmodel);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);


  // Computes
  tsk->computes(d_lab->d_stressTensorCompLabel, d_lab->d_stressTensorMatl,
		Task::OutOfDomain);
  tsk->computes(d_lab->d_scalarFluxCompLabel, d_lab->d_scalarFluxMatl,
		Task::OutOfDomain);

  
  sched->addTask(tsk, patches, matls);

}

//****************************************************************************
// Actual compute 
//****************************************************************************
void 
ScaleSimilarityModel::computeTurbSubmodel(const ProcessorGroup* pg,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
#ifdef PetscFilter
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    constCCVariable<int> cellType;
    new_dw->get(cellType, d_lab->d_cellTypeLabel,
		matlIndex, patch, Ghost::AroundCells, Arches::ONEGHOSTCELL);
 
    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    d_filter->setFilterMatrix(pg, patch, cellinfo, cellType);
#endif
    StencilMatrix<CCVariable<double> > stressTensorCoeff; //9 point tensor

  // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
      new_dw->allocateAndPut(stressTensorCoeff[ii], 
		       d_lab->d_stressTensorCompLabel, ii, patch);
      stressTensorCoeff[ii].initialize(0.0);
    }

    StencilMatrix<CCVariable<double> > scalarFluxCoeff; //9 point tensor

  // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
      new_dw->allocateAndPut(scalarFluxCoeff[ii], 
		       d_lab->d_scalarFluxCompLabel, ii, patch);
      scalarFluxCoeff[ii].initialize(0.0);
    }
#if 0
    for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) 
      new_dw->put(stressTensorCoeff[ii], 
		  d_lab->d_stressTensorCompLabel, ii, patch);

    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) 
      new_dw->put(scalarFluxCoeff[ii], 
		  d_lab->d_scalarFluxCompLabel, ii, patch);
#endif


  }
}



//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
ScaleSimilarityModel::sched_reComputeTurbSubmodel(SchedulerP& sched, 
					      const PatchSet* patches,
					      const MaterialSet* matls,
				        const TimeIntegratorLabel* timelabels)
{
  SmagorinskyModel::sched_reComputeTurbSubmodel(sched, patches, matls,
						timelabels);

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
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_newCCUVelocityLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // for multimaterial
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);


      // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
    tsk->computes(d_lab->d_stressTensorCompLabel, d_lab->d_stressTensorMatl,
		  Task::OutOfDomain);

    tsk->computes(d_lab->d_scalarFluxCompLabel, d_lab->d_scalarFluxMatl,
		  Task::OutOfDomain);
  }
  else {
    tsk->modifies(d_lab->d_stressTensorCompLabel, d_lab->d_stressTensorMatl,
		  Task::OutOfDomain);

    tsk->modifies(d_lab->d_scalarFluxCompLabel, d_lab->d_scalarFluxMatl,
		  Task::OutOfDomain);
  }

  sched->addTask(tsk, patches, matls);
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
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> uVel;
    constCCVariable<double> vVel;
    constCCVariable<double> wVel;
    constCCVariable<double> den;
    constCCVariable<double> scalar;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse

    new_dw->get(uVel,d_lab->d_newCCUVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(vVel,d_lab->d_newCCVVelocityLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(wVel, d_lab->d_newCCWVelocityLabel, matlIndex, patch, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(den, d_lab->d_densityCPLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    
    if (d_MAlab)
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::None, Arches::ZEROGHOSTCELLS);

    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);

#ifndef PetscFilter
    // Get the PerPatch CellInformation data

    PerPatch<CellInformationP> cellInfoP;
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    CellInformation* cellinfo = cellInfoP.get().get_rep();
#endif
    
    // Get the patch and variable details
    // compatible with fortran index
    double CF = d_CF;
    StencilMatrix<CCVariable<double> > stressTensorCoeff; //9 point tensor

  // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        new_dw->allocateAndPut(stressTensorCoeff[ii], 
			       d_lab->d_stressTensorCompLabel, ii, patch);
      else
        new_dw->getModifiable(stressTensorCoeff[ii], 
			      d_lab->d_stressTensorCompLabel, ii, patch);
      stressTensorCoeff[ii].initialize(0.0);
    }


    // compute test filtered velocities, density and product 
    // (den*u*u, den*u*v, den*u*w, den*v*v,
    // den*v*w, den*w*w)
    // using a box filter, generalize it to use other filters such as Gaussian


    // computing turbulent scalar flux
    StencilMatrix<CCVariable<double> > scalarFluxCoeff; //9 point tensor

    // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
      if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
        new_dw->allocateAndPut(scalarFluxCoeff[ii], 
			       d_lab->d_scalarFluxCompLabel, ii, patch);
      else
        new_dw->getModifiable(scalarFluxCoeff[ii], 
			       d_lab->d_scalarFluxCompLabel, ii, patch);
      scalarFluxCoeff[ii].initialize(0.0);
    }

    int numGC = 1;
    IntVector idxLo = patch->getGhostCellLowIndex(numGC);
    IntVector idxHi = patch->getGhostCellHighIndex(numGC);
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
	  denUU[currCell] = uVel[currCell]*uVel[currCell];
	  denUV[currCell] = uVel[currCell]*vVel[currCell];
	  denUW[currCell] = uVel[currCell]*wVel[currCell];
	  denVV[currCell] = vVel[currCell]*vVel[currCell];
	  denVW[currCell] = vVel[currCell]*wVel[currCell];
	  denWW[currCell] = wVel[currCell]*wVel[currCell];
	  denPhiU[currCell] = scalar[currCell]*uVel[currCell];
	  denPhiV[currCell] = scalar[currCell]*vVel[currCell];
	  denPhiW[currCell] = scalar[currCell]*wVel[currCell];
	}
      }
    }
    //#ifndef PetscFilter
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
    Array3<double> filterdenUU(patch->getLowIndex(), patch->getHighIndex());
    filterdenUU.initialize(0.0);
    Array3<double> filterdenUV(patch->getLowIndex(), patch->getHighIndex());
    filterdenUV.initialize(0.0);
    Array3<double> filterdenUW(patch->getLowIndex(), patch->getHighIndex());
    filterdenUW.initialize(0.0);
    Array3<double> filterdenVV(patch->getLowIndex(), patch->getHighIndex());
    filterdenVV.initialize(0.0);
    Array3<double> filterdenVW(patch->getLowIndex(), patch->getHighIndex());
    filterdenVW.initialize(0.0);
    Array3<double> filterdenWW(patch->getLowIndex(), patch->getHighIndex());
    filterdenWW.initialize(0.0);
    Array3<double> filterDen(patch->getLowIndex(), patch->getHighIndex());
    filterDen.initialize(0.0);
    Array3<double> filterUVel(patch->getLowIndex(), patch->getHighIndex());
    filterUVel.initialize(0.0);
    Array3<double> filterVVel(patch->getLowIndex(), patch->getHighIndex());
    filterVVel.initialize(0.0);
    Array3<double> filterWVel(patch->getLowIndex(), patch->getHighIndex());
    filterWVel.initialize(0.0);
    Array3<double> filterPhi(patch->getLowIndex(), patch->getHighIndex());
    filterPhi.initialize(0.0);
    Array3<double> filterdenPhiU(patch->getLowIndex(), patch->getHighIndex());
    filterdenPhiU.initialize(0.0);
    Array3<double> filterdenPhiV(patch->getLowIndex(), patch->getHighIndex());
    filterdenPhiV.initialize(0.0);
    Array3<double> filterdenPhiW(patch->getLowIndex(), patch->getHighIndex());
    filterdenPhiW.initialize(0.0);
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch,uVel, filterUVel);
#if 0
    cerr << "In the Scale Similarity print vVel" << endl;
    vVel.print(cerr);
#endif

    d_filter->applyFilter(pc, patch,vVel, filterVVel);
#if 0
    cerr << "In the Scale Similarity model after filter print filterVVel" << endl;
    filterVVel.print(cerr);
#endif

    d_filter->applyFilter(pc, patch,wVel, filterWVel);
    d_filter->applyFilter(pc, patch,denUU, filterdenUU);
    d_filter->applyFilter(pc, patch,denUV, filterdenUV);
    d_filter->applyFilter(pc, patch,denUW, filterdenUW);
    d_filter->applyFilter(pc, patch,denVV, filterdenVV);
    d_filter->applyFilter(pc, patch,denVW, filterdenVW);
    d_filter->applyFilter(pc, patch,denWW, filterdenWW);
    d_filter->applyFilter(pc, patch,scalar, filterPhi);
    d_filter->applyFilter(pc, patch,denPhiU, filterdenPhiU);
    d_filter->applyFilter(pc, patch,denPhiV, filterdenPhiV);
    d_filter->applyFilter(pc, patch,denPhiW, filterdenPhiW);
#else
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double cube_delta = (2.0*cellinfo->sew[colX])*(2.0*cellinfo->sns[colY])*
                 	    (2.0*cellinfo->stb[colZ]);
	  //double invDelta = 1.0/cube_delta;
	  filterDen[currCell] = 0.0;
	  filterUVel[currCell] = 0.0;
	  filterVVel[currCell] = 0.0;
	  filterWVel[currCell] = 0.0;
	  filterdenUU[currCell] = 0.0;
	  filterdenUV[currCell] = 0.0;
	  filterdenUW[currCell] = 0.0;
	  filterdenVV[currCell] = 0.0;
	  filterdenVW[currCell] = 0.0;
	  filterdenWW[currCell] = 0.0;
	  filterPhi[currCell] = 0.0;
	  filterdenPhiU[currCell] = 0.0;
	  filterdenPhiV[currCell] = 0.0;
	  filterdenPhiW[currCell] = 0.0;
	  double totalVol = 0;
	  for (int kk = -1; kk <= 1; kk ++) {
	    for (int jj = -1; jj <= 1; jj ++) {
	      for (int ii = -1; ii <= 1; ii ++) {
		IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		             cellinfo->stb[colZ+kk]*
		             (1.0-0.5*abs(ii))*
		             (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
		totalVol += vol;
		//		filterDen[currCell] += den[filterCell]*vol; 
		filterDen[currCell] += den[currCell]*vol; 
		filterUVel[currCell] += uVel[filterCell]*vol; 
		filterVVel[currCell] += vVel[filterCell]*vol; 
		filterWVel[currCell] += wVel[filterCell]*vol;
		filterdenUU[currCell] += denUU[filterCell]*vol;  
		filterdenUV[currCell] += denUV[filterCell]*vol;  
		filterdenUW[currCell] += denUW[filterCell]*vol;  
		filterdenVV[currCell] += denVV[filterCell]*vol;  
		filterdenVW[currCell] += denVW[filterCell]*vol;  
		filterdenWW[currCell] += denWW[filterCell]*vol;  
		filterPhi[currCell] += scalar[filterCell]*vol; 
		filterdenPhiU[currCell] += denPhiU[filterCell]*vol;  
		filterdenPhiV[currCell] += denPhiV[filterCell]*vol;  
		filterdenPhiW[currCell] += denPhiW[filterCell]*vol;  
	      }
	    }
	  }
	  
	  filterDen[currCell] /= totalVol;
	  filterUVel[currCell] /= totalVol;
	  filterVVel[currCell] /= totalVol;
	  filterWVel[currCell] /= totalVol;
	  filterdenUU[currCell] /= totalVol;
	  filterdenUV[currCell] /= totalVol;
	  filterdenUW[currCell] /= totalVol;
	  filterdenVV[currCell] /= totalVol;
	  filterdenVW[currCell] /= totalVol;
	  filterdenWW[currCell] /= totalVol;
	  filterPhi[currCell] /= totalVol;
	  filterdenPhiU[currCell] /= totalVol;
	  filterdenPhiV[currCell] /= totalVol;
	  filterdenPhiW[currCell] /= totalVol;
	}
      }
    }
#endif
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

	  // scalar fluxes uf, vf, wf
	  (scalarFluxCoeff[0])[currCell] = CF*den[currCell]*(filterdenPhiU[currCell] -
						 filterPhi[currCell]*
						 filterUVel[currCell]);
	  (scalarFluxCoeff[1])[currCell] = CF*den[currCell]*
                                                (filterdenPhiV[currCell] -
						 filterPhi[currCell]*
						 filterVVel[currCell]);
	  (scalarFluxCoeff[2])[currCell] = CF*den[currCell]*
	                                       (filterdenPhiW[currCell] -
						 filterPhi[currCell]*
						 filterWVel[currCell]);

	}
      }
    }
#if 0
    // compute stress tensor
    for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) 
      new_dw->put(stressTensorCoeff[ii], 
		  d_lab->d_stressTensorCompLabel, ii, patch);

    for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) 
      new_dw->put(scalarFluxCoeff[ii], 
		  d_lab->d_scalarFluxCompLabel, ii, patch);
#endif

  }
}

void 
ScaleSimilarityModel::sched_computeScalarVariance(SchedulerP& sched, 
					      const PatchSet* patches,
					      const MaterialSet* matls,
			    		 const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ScaleSimilarityModel::computeScalarVaraince" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ScaleSimilarityModel::computeScalarVariance,
			  timelabels);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
     tsk->computes(d_lab->d_scalarVarSPLabel);
  else
     tsk->modifies(d_lab->d_scalarVarSPLabel);

  sched->addTask(tsk, patches, matls);
}


void 
ScaleSimilarityModel::computeScalarVariance(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw,
			    		const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> scalar;
    CCVariable<double> scalarVar;
    // Get the velocity, density and viscosity from the old data warehouse
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
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


	  // compute scalar variance
	  scalarVar[currCell] = d_CF*(filterPhiSqr[currCell]-
				      (filterPhi[currCell]*filterPhi[currCell]));
	}
      }
    }
    // Put the calculated viscosityvalue into the new data warehouse
#if 0
    new_dw->put(scalarVar, d_lab->d_scalarVarSPLabel, matlIndex, patch);
#endif
  }
}


void 
ScaleSimilarityModel::sched_computeScalarDissipation(SchedulerP& sched, 
						 const PatchSet* patches,
						 const MaterialSet* matls,
			    		 const TimeIntegratorLabel* timelabels)
{
  string taskname =  "ScaleSimilarityModel::computeScalarDissipation" +
		     timelabels->integrator_step_name;
  Task* tsk = scinew Task(taskname, this,
			  &ScaleSimilarityModel::computeScalarDissipation,
			  timelabels);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  // assuming scalar dissipation is computed before turbulent viscosity calculation 
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_viscosityCTSLabel,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
     tsk->requires(Task::OldDW, d_lab->d_scalarFluxCompLabel,
		   d_lab->d_scalarFluxMatl,
		   Task::OutOfDomain, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  // Computes
     tsk->computes(d_lab->d_scalarDissSPLabel);
  }
  else {
     tsk->requires(Task::NewDW, d_lab->d_scalarFluxCompLabel,
		   d_lab->d_scalarFluxMatl,
		   Task::OutOfDomain, Ghost::AroundCells, Arches::ONEGHOSTCELL);
  // Computes
     tsk->modifies(d_lab->d_scalarDissSPLabel);
  }

  sched->addTask(tsk, patches, matls);
}




void 
ScaleSimilarityModel::computeScalarDissipation(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse* old_dw,
					DataWarehouse* new_dw,
			    		const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> viscosity;
    constCCVariable<double> scalar;
    CCVariable<double> scalarDiss;  // dissipation..chi
    StencilMatrix<constCCVariable<double> > scalarFlux; //3 point stencil

    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First) {
      for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
        old_dw->get(scalarFlux[ii], 
		    d_lab->d_scalarFluxCompLabel, ii, patch,
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
      }
      new_dw->allocateAndPut(scalarDiss, d_lab->d_scalarDissSPLabel,
			     matlIndex, patch);
    }
    else {
      for (int ii = 0; ii < d_lab->d_scalarFluxMatl->size(); ii++) {
        new_dw->get(scalarFlux[ii], 
		    d_lab->d_scalarFluxCompLabel, ii, patch,
		    Ghost::AroundCells, Arches::ONEGHOSTCELL);
      }
      new_dw->getModifiable(scalarDiss, d_lab->d_scalarDissSPLabel,
			    matlIndex, patch);
    }
    scalarDiss.initialize(0.0);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // compatible with fortran index
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double scale = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX+1,colY,colZ)]);
	  double scalw = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX-1,colY,colZ)]);
	  double scaln = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX,colY+1,colZ)]);
	  double scals = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX,colY-1,colZ)]);
	  double scalt = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX,colY,colZ+1)]);
	  double scalb = 0.5*(scalar[currCell]+
			      scalar[IntVector(colX,colY,colZ-1)]);
	  double dfdx = (scale-scalw)/cellinfo->sew[colX];
	  double dfdy = (scaln-scals)/cellinfo->sns[colY];
	  double dfdz = (scalt-scalb)/cellinfo->stb[colZ];
	  // molecular diffusivity
	  scalarDiss[currCell] = 2.0*viscosity[currCell]/d_turbPrNo*
	                        (dfdx*dfdx + dfdy*dfdy + dfdz*dfdz); 
	  double turbProduction = -2.0*((scalarFlux[0])[currCell]*dfdx+
				       (scalarFlux[1])[currCell]*dfdy+
				       (scalarFlux[2])[currCell]*dfdz);
	  if (turbProduction > 0)
	    scalarDiss[currCell] += turbProduction;
	  if (scalarDiss[currCell] < 0.0)
	    scalarDiss[currCell] = 0.0;
	}
      }
    }
  }
}
