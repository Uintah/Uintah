//----- ScaleSimilarityModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/ScaleSimilarityModel.h>
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
ScaleSimilarityModel::ScaleSimilarityModel(const ArchesLabel* label, 
				   const MPMArchesLabel* MAlb,
				   PhysicalConstants* phyConsts,
				   BoundaryCondition* bndry_cond):
                                    SmagorinskyModel(label, MAlb, phyConsts,
						    bndry_cond), 
                                    d_lab(label), d_MAlab(MAlb)
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
ScaleSimilarityModel::sched_computeTurbSubmodel(SchedulerP& sched, 
						const PatchSet* patches,
						const MaterialSet* matls)
{
  SmagorinskyModel::sched_computeTurbSubmodel(sched, patches, matls);
  Task* tsk = scinew Task("ScaleSimilarityModel::TurbSubmodel",
			  this,
			  &ScaleSimilarityModel::computeTurbSubmodel);
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
ScaleSimilarityModel::computeTurbSubmodel(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

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
					      const MaterialSet* matls)
{
  SmagorinskyModel::sched_reComputeTurbSubmodel(sched, patches, matls);
  Task* tsk = scinew Task("ScaleSimilarityModel::ReTurbSubmodel",
			  this,
			  &ScaleSimilarityModel::reComputeTurbSubmodel);

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
		Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_newCCVVelocityLabel, 
		Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_newCCWVelocityLabel, 
		Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
  // for multimaterial
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

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
// Actual recompute 
//****************************************************************************
void 
ScaleSimilarityModel::reComputeTurbSubmodel(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw)
{
  double time = d_lab->d_sharedState->getElapsedTime();
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
    new_dw->get(den, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
    
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
    
    
    // Get the patch and variable details
    // compatible with fortran index
    double CF = d_CF;
    StencilMatrix<CCVariable<double> > stressTensorCoeff; //9 point tensor

  // allocate stress tensor coeffs
    for (int ii = 0; ii < d_lab->d_stressTensorMatl->size(); ii++) {
      new_dw->allocateAndPut(stressTensorCoeff[ii], 
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
      new_dw->allocateAndPut(scalarFluxCoeff[ii], 
		       d_lab->d_scalarFluxCompLabel, ii, patch);
      scalarFluxCoeff[ii].initialize(0.0);
    }

    int numGC = 1;
    IntVector idxLo = patch->getGhostCellLowIndex(numGC);
    IntVector idxHi = patch->getGhostCellHighIndex(numGC);
    Array3<double> denUU(idxLo, idxHi);
    Array3<double> denUV(idxLo, idxHi);
    Array3<double> denUW(idxLo, idxHi);
    Array3<double> denVV(idxLo, idxHi);
    Array3<double> denVW(idxLo, idxHi);
    Array3<double> denWW(idxLo, idxHi);
    Array3<double> denPhiU(idxLo, idxHi);
    Array3<double> denPhiV(idxLo, idxHi);
    Array3<double> denPhiW(idxLo, idxHi);
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  denUU[currCell] = den[currCell]*uVel[currCell]*uVel[currCell];
	  denUV[currCell] = den[currCell]*uVel[currCell]*vVel[currCell];
	  denUW[currCell] = den[currCell]*uVel[currCell]*wVel[currCell];
	  denVV[currCell] = den[currCell]*vVel[currCell]*vVel[currCell];
	  denVW[currCell] = den[currCell]*vVel[currCell]*wVel[currCell];
	  denWW[currCell] = den[currCell]*wVel[currCell]*wVel[currCell];
	  denPhiU[currCell] = den[currCell]*scalar[currCell]*uVel[currCell];
	  denPhiV[currCell] = den[currCell]*scalar[currCell]*vVel[currCell];
	  denPhiW[currCell] = den[currCell]*scalar[currCell]*wVel[currCell];
	}
      }
    }
    Array3<double> filterdenUU(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenUV(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenUW(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenVV(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenVW(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenWW(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterDen(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterUVel(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterVVel(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterWVel(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterPhi(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenPhiU(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenPhiV(patch->getLowIndex(), patch->getHighIndex());
    Array3<double> filterdenPhiW(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellFORTLowIndex();
    IntVector indexHigh = patch->getCellFORTHighIndex();
    for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	  IntVector currCell(colX, colY, colZ);
	  double cube_delta = (2.0*cellinfo->sew[colX])*(2.0*cellinfo->sns[colY])*
                 	    (2.0*cellinfo->stb[colZ]);
	  double invDelta = 1.0/cube_delta;
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
	  
	  for (int kk = -1; kk <= 1; kk ++) {
	    for (int jj = -1; jj <= 1; jj ++) {
	      for (int ii = -1; ii <= 1; ii ++) {
		IntVector filterCell = IntVector(colX+ii,colY+jj,colZ+kk);
		double vol = cellinfo->sew[colX+ii]*cellinfo->sns[colY+jj]*
		             cellinfo->stb[colZ+kk]*
		             (1.0-0.5*abs(ii))*
		             (1.0-0.5*abs(jj))*(1.0-0.5*abs(kk));
		//		filterDen[currCell] += den[filterCell]*vol; 
		filterDen[currCell] = den[currCell]*vol; 
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
	  
	  filterDen[currCell] *= invDelta;
	  filterUVel[currCell] *= invDelta;
	  filterVVel[currCell] *= invDelta;
	  filterWVel[currCell] *= invDelta;
	  filterdenUU[currCell] *= invDelta;
	  filterdenUV[currCell] *= invDelta;
	  filterdenUW[currCell] *= invDelta;
	  filterdenVV[currCell] *= invDelta;
	  filterdenVW[currCell] *= invDelta;
	  filterdenWW[currCell] *= invDelta;
	  filterPhi[currCell] *= invDelta;
	  filterdenPhiU[currCell] *= invDelta;
	  filterdenPhiV[currCell] *= invDelta;
	  filterdenPhiW[currCell] *= invDelta;
	  // compute stress tensor
	  // index 0: T11, 1:T12, 2:T13, 3:T21, 4:T22, 5:T23, 6:T31, 7:T32, 8:T33
	  (stressTensorCoeff[0])[currCell] = CF*(filterdenUU[currCell] -
						 filterDen[currCell]*
						 filterUVel[currCell]*
						 filterUVel[currCell]);
	  (stressTensorCoeff[1])[currCell] = CF*(filterdenUV[currCell] -
						 filterDen[currCell]*
						 filterUVel[currCell]*
						 filterVVel[currCell]);
	  (stressTensorCoeff[2])[currCell] = CF*(filterdenUW[currCell] -
						 filterDen[currCell]*
						 filterUVel[currCell]*
						 filterWVel[currCell]);
	  (stressTensorCoeff[3])[currCell] = (stressTensorCoeff[1])[currCell];
	  (stressTensorCoeff[4])[currCell] = CF*(filterdenVV[currCell] -
		                                 filterDen[currCell]*
						 filterVVel[currCell]*
						 filterVVel[currCell]);
	  (stressTensorCoeff[5])[currCell] = CF*(filterdenVW[currCell] -
						 filterDen[currCell]*
						 filterVVel[currCell]*
						 filterWVel[currCell]);
	  (stressTensorCoeff[6])[currCell] = (stressTensorCoeff[2])[currCell];
	  (stressTensorCoeff[7])[currCell] = (stressTensorCoeff[5])[currCell];
	  (stressTensorCoeff[8])[currCell] = CF*(filterdenWW[currCell] -
						 filterDen[currCell]*
						 filterWVel[currCell]*
						 filterWVel[currCell]);

	  // scalar fluxes uf, vf, wf
	  (scalarFluxCoeff[0])[currCell] = CF*(filterdenPhiU[currCell] -
						 filterDen[currCell]*
						 filterPhi[currCell]*
						 filterUVel[currCell]);
	  (scalarFluxCoeff[1])[currCell] = CF*(filterdenPhiV[currCell] -
						 filterDen[currCell]*
						 filterPhi[currCell]*
						 filterVVel[currCell]);
	  (scalarFluxCoeff[2])[currCell] = CF*(filterdenPhiW[currCell] -
						 filterDen[currCell]*
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
					      const MaterialSet* matls)
{
  Task* tsk = scinew Task("ScaleSimilarityModel::computeScalarVar",
			  this,
			  &ScaleSimilarityModel::computeScalarVariance);

  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
#ifdef correctorstep
  tsk->requires(Task::NewDW, d_lab->d_scalarPredLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#else  
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
#endif

  // Computes
  tsk->computes(d_lab->d_scalarVarSPLabel);

  sched->addTask(tsk, patches, matls);
}


void 
ScaleSimilarityModel::computeScalarVariance(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw)
{
  double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> scalar;
    CCVariable<double> scalarVar;
    // Get the velocity, density and viscosity from the old data warehouse
#ifdef correctorstep
    new_dw->get(scalar, d_lab->d_scalarPredLabel, matlIndex, patch, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
#else
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch, Ghost::AroundCells,
		Arches::ONEGHOSTCELL);
#endif
    new_dw->allocateAndPut(scalarVar, d_lab->d_scalarVarSPLabel, matlIndex, patch);
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
