//----- SmagorinksyModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
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

#include <Packages/Uintah/CCA/Components/Arches/fortran/smagmodel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalarvarmodel_fort.h>

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
  db->require("var_const",d_CFVar); // const reqd by variance eqn
  if (db->findBlock("turbulentPrandtlNumber")) 
    db->require("turbulentPrandtlNumber",d_turbPrNo);
  else
    d_turbPrNo = 0.4;

}

//****************************************************************************
// Schedule compute 
//****************************************************************************
void 
SmagorinskyModel::sched_computeTurbSubmodel(const LevelP&,
					    SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* tsk = scinew Task("SmagorinskyModel::TurbSubmodel",
			  this,
			  &SmagorinskyModel::computeTurbSubmodel);

  
  // Requires
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, Ghost::None,
		Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPLabel, 
		Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPLabel,
		Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPLabel,
		Ghost::AroundFaces,
		Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);


      // Computes
  tsk->computes(d_lab->d_viscosityCTSLabel);

  sched->addTask(tsk, patches, matls);

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
  tsk->requires(Task::NewDW, timelabels->density_out,
		Ghost::None, Arches::ZEROGHOSTCELLS);
  tsk->requires(Task::NewDW, timelabels->uvelocity_out,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, timelabels->vvelocity_out, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, timelabels->wvelocity_out, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);

  tsk->requires(Task::NewDW, d_lab->d_cellTypeLabel, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, 
		Ghost::None, Arches::ZEROGHOSTCELLS);
  // for multimaterial
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);

      // Computes
  tsk->computes(timelabels->viscosity_out);

  sched->addTask(tsk, patches, matls);
}

//****************************************************************************
// Actual compute 
//****************************************************************************
void 
SmagorinskyModel::computeTurbSubmodel(const ProcessorGroup*,
				      const PatchSubset* patches,
				      const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> density;
    CCVariable<double> viscosity;
    constCCVariable<int> cellType;

    // Get the velocity, density and viscosity from the old data warehouse
    
    new_dw->allocateAndPut(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);
    new_dw->copyOut(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVelocity,d_lab->d_wVelocitySPLabel, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    new_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch,
		  Ghost::AroundCells, Arches::ONEGHOSTCELL);


    PerPatch<CellInformationP> cellinfop;
    //if (old_dw->exists(d_cellInfoLabel, patch)) {
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) {
      new_dw->get(cellinfop, d_lab->d_cellInfoLabel, matlIndex, patch);
    } else {
      cellinfop.setData(scinew CellInformation(patch));
      new_dw->put(cellinfop, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    //  old_dw->get(cellinfop, d_lab->d_cellInfoLabel, matlIndex, patch);
    //} else {
    //  cellinfop.setData(scinew CellInformation(patch));
    //  old_dw->put(cellinfop, d_cellInfoLabel, matlIndex, patch);
    //}
    CellInformation* cellinfo = cellinfop.get().get_rep();
    
    // Get the patch details
    IntVector lowIndex = patch->getCellFORTLowIndex();
    IntVector highIndex = patch->getCellFORTHighIndex();

    // get physical constants
    double mol_viscos; // molecular viscosity
    mol_viscos = d_physicalConsts->getMolecularViscosity();
    double CF = d_CF;
#if 0
    if (time < 0.5 ) 
      CF *= (time+ 0.0001);
#endif
    fort_smagmodel(uVelocity, vVelocity, wVelocity, density, viscosity,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   mol_viscos, CF, d_factorMesh, d_filterl);

    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int wallID = d_boundaryCondition->wallCellType();
    if (xminus) {
      for (int colZ = lowIndex.z(); colZ <=  highIndex.z(); colZ ++) {
	for (int colY = lowIndex.y(); colY <=  highIndex.y(); colY ++) {
	  int colX = lowIndex.x();
	  IntVector currCell(colX-1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (xplus) {
      for (int colZ = lowIndex.z(); colZ <=  highIndex.z(); colZ ++) {
	for (int colY = lowIndex.y(); colY <=  highIndex.y(); colY ++) {
	  int colX =  highIndex.x();
	  IntVector currCell(colX+1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yminus) {
      for (int colZ = lowIndex.z(); colZ <=  highIndex.z(); colZ ++) {
	for (int colX = lowIndex.x(); colX <=  highIndex.x(); colX ++) {
	  int colY = lowIndex.y();
	  IntVector currCell(colX, colY-1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yplus) {
      for (int colZ = lowIndex.z(); colZ <=  highIndex.z(); colZ ++) {
	for (int colX = lowIndex.x(); colX <=  highIndex.x(); colX ++) {
	  int colY =  highIndex.y();
	  IntVector currCell(colX, colY+1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zminus) {
      for (int colY = lowIndex.y(); colY <=  highIndex.y(); colY ++) {
	for (int colX = lowIndex.x(); colX <=  highIndex.x(); colX ++) {
	  int colZ = lowIndex.z();
	  IntVector currCell(colX, colY, colZ-1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zplus) {
      for (int colY = lowIndex.y(); colY <=  highIndex.y(); colY ++) {
	for (int colX = lowIndex.x(); colX <=  highIndex.x(); colX ++) {
	  int colZ =  highIndex.z();
	  IntVector currCell(colX, colY, colZ+1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }

#ifdef multimaterialform
    if (d_mmInterface) {
      IntVector indexLow = patch->getCellLowIndex();
      IntVector indexHigh = patch->getCellHighIndex();
      MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
      for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
	for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	  for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	    // Store current cell
	    IntVector currCell(colX, colY, colZ);
	    viscosity[currCell] *=  mmVars->voidFraction[currCell];
	  }
	}
      }
    }
#endif

#ifdef ARCHES_DEBUG
      // Testing if correct values have been put
      cerr << " AFTER COMPUTE TURBULENCE SUBMODEL " << endl;
      for (int ii = domLoVis.x(); ii <= domHiVis.x(); ii++) {
	cerr << "Viscosity for ii = " << ii << endl;
	for (int jj = domLoVis.y(); jj <= domHiVis.y(); jj++) {
	  for (int kk = domLoVis.z(); kk <= domHiVis.z(); kk++) {
	    cerr.width(10);
	    cerr << viscosity[IntVector(ii,jj,kk)] << " " ; 
	  }
	  cerr << endl;
	}
      }
#endif
      // Create the new viscosity variable to write the result to 
      // and allocate space in the new data warehouse for this variable
      // Put the calculated viscosityvalue into the new data warehouse
      // allocateAndPut instead:
      /* new_dw->put(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch); */;
  }
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
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constSFCXVariable<double> uVelocity;
    constSFCYVariable<double> vVelocity;
    constSFCZVariable<double> wVelocity;
    constCCVariable<double> density;
    CCVariable<double> viscosity;
    constCCVariable<double> voidFraction;
    constCCVariable<int> cellType;
    // Get the velocity, density and viscosity from the old data warehouse

    new_dw->allocateAndPut(viscosity, timelabels->viscosity_out,
			   matlIndex, patch);
    new_dw->copyOut(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch,
		    Ghost::None, Arches::ZEROGHOSTCELLS);
    
    new_dw->get(uVelocity, timelabels->uvelocity_out, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(vVelocity, timelabels->vvelocity_out, matlIndex, patch,
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(wVelocity, timelabels->wvelocity_out, matlIndex, patch, 
		Ghost::AroundFaces, Arches::ONEGHOSTCELL);
    new_dw->get(density, timelabels->density_out, matlIndex, patch,
		Ghost::None, Arches::ZEROGHOSTCELLS);
    
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
    double mol_viscos; // molecular viscosity
    mol_viscos = d_physicalConsts->getMolecularViscosity();
    
    // Get the patch and variable details
    // compatible with fortran index
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    double CF = d_CF;
#if 0
    if (time < 2.0 ) 
      CF *= (time+ 0.0001)*0.5;
#endif      
    fort_smagmodel(uVelocity, vVelocity, wVelocity, density, viscosity,
		   idxLo, idxHi,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   mol_viscos, CF, d_factorMesh, d_filterl);

    // boundary conditions
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    int wallID = d_boundaryCondition->wallCellType();
    if (xminus) {
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	  int colX = idxLo.x();
	  IntVector currCell(colX-1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (xplus) {
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	  int colX = idxHi.x();
	  IntVector currCell(colX+1, colY, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yminus) {
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  int colY = idxLo.y();
	  IntVector currCell(colX, colY-1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (yplus) {
      for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  int colY = idxHi.y();
	  IntVector currCell(colX, colY+1, colZ);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zminus) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  int colZ = idxLo.z();
	  IntVector currCell(colX, colY, colZ-1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }
    if (zplus) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
	for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	  int colZ = idxHi.z();
	  IntVector currCell(colX, colY, colZ+1);
	  if (cellType[currCell] != wallID)
	    viscosity[currCell] = viscosity[IntVector(colX,colY,colZ)];
	}
      }
    }

    if (d_MAlab) {
      IntVector indexLow = patch->getCellLowIndex();
      IntVector indexHigh = patch->getCellHighIndex();
      for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
	for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	  for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	    // Store current cell
	    IntVector currCell(colX, colY, colZ);
	    viscosity[currCell] *=  voidFraction[currCell];
	  }
	}
      }
    }

#ifdef ARCHES_PRES_DEBUG
    // Testing if correct values have been put
    cerr << " AFTER COMPUTE TURBULENCE SUBMODEL " << endl;
    viscosity.print(cerr);
#endif
    
    // Put the calculated viscosityvalue into the new data warehouse
    // allocateAndPut instead:
    /* new_dw->put(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch); */;
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
  tsk->requires(Task::NewDW, timelabels->scalar_out, 
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
     tsk->computes(d_lab->d_scalarVarSPLabel);
  else
     tsk->modifies(d_lab->d_scalarVarSPLabel);

  sched->addTask(tsk, patches, matls);
}


void 
SmagorinskyModel::computeScalarVariance(const ProcessorGroup*,
					const PatchSubset* patches,
					const MaterialSubset*,
					DataWarehouse*,
					DataWarehouse* new_dw,
			    		const TimeIntegratorLabel* timelabels)
{
  //double time = d_lab->d_sharedState->getElapsedTime();
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> scalar;
    CCVariable<double> scalarVar;
    // Get the velocity, density and viscosity from the old data warehouse
    new_dw->get(scalar, timelabels->scalar_out, matlIndex, patch,
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
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  if (old_dw->exists(d_cellInfoLabel, patch)) 
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //else {
    //  cellInfoP.setData(scinew CellInformation(patch));
    //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //}
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // compatible with fortran index
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    double CFVar = d_CFVar;
#if 0
    if (time < 2.0 ) 
      CFVar *= (time+ 0.0001)*0.5;
#endif
    fort_scalarvarmodel(scalar, idxLo, idxHi, scalarVar, cellinfo->dxpw,
			cellinfo->dyps, cellinfo->dzpb, cellinfo->sew,
			cellinfo->sns, cellinfo->stb, CFVar, d_factorMesh,
			d_filterl);

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
  tsk->requires(Task::NewDW, timelabels->scalar_out,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
  tsk->requires(Task::NewDW, timelabels->viscosity_in,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

  // Computes
  if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
     tsk->computes(d_lab->d_scalarDissSPLabel);
  else
     tsk->modifies(d_lab->d_scalarDissSPLabel);

  sched->addTask(tsk, patches, matls);
}


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
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    constCCVariable<double> viscosity;
    constCCVariable<double> scalar;
    CCVariable<double> scalarDiss;  // dissipation..chi

    new_dw->get(scalar, timelabels->scalar_out, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);
    new_dw->get(viscosity, timelabels->viscosity_in, matlIndex, patch,
		Ghost::AroundCells, Arches::ONEGHOSTCELL);

    if (timelabels->integrator_step_number == TimeIntegratorStepNumber::First)
       new_dw->allocateAndPut(scalarDiss, d_lab->d_scalarDissSPLabel,
			      matlIndex, patch);
    else
       new_dw->getModifiable(scalarDiss, d_lab->d_scalarDissSPLabel,
			      matlIndex, patch);
    scalarDiss.initialize(0.0);
    
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  if (old_dw->exists(d_cellInfoLabel, patch)) 
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //else {
    //  cellInfoP.setData(scinew CellInformation(patch));
    //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //}
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
	  scalarDiss[currCell] = viscosity[currCell]/d_turbPrNo*
	                        (dfdx*dfdx + dfdy*dfdy + dfdz*dfdz); 
	}
      }
    }
  }
}




//****************************************************************************
// Calculate the Velocity BC at the Wall
//****************************************************************************
void SmagorinskyModel::calcVelocityWallBC(const ProcessorGroup*,
					  const Patch*,
					  DataWarehouseP&,
					  DataWarehouseP&,
					  int,
					  int)
{
#ifdef WONT_COMPILE_YET
  int matlIndex = 0;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  switch(eqnType) {
  case Arches::PRESSURE:
    old_dw->get(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    old_dw->get(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    old_dw->get(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    break;
  case Arches::MOMENTUM:
    old_dw->get(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    old_dw->get(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    old_dw->get(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		Arches::ZEROGHOSTCELLS);
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }

  CCVariable<double> density;
  old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
	      Arches::ZEROGHOSTCELLS);

  // Get the PerPatch CellInformation data
  PerPatch<CellInformationP> cellInfoP;
  //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  
  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  old_dw->get(cellType, d_lab->d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      Arches::ZEROGHOSTCELLS);

  //get Molecular Viscosity of the fluid
  double mol_viscos = d_physicalConsts->getMolecularViscosity();


  cellFieldType walltype = WALLTYPE;



  SFCXVariable<double> uVelLinearSrc; //SP term in Arches
  SFCXVariable<double> uVelNonLinearSrc; // SU in Arches
  SFCYVariable<double> vVelLinearSrc; //SP term in Arches
  SFCYVariable<double> vVelNonLinearSrc; // SU in Arches
  SFCZVariable<double> wVelLinearSrc; //SP term in Arches
  SFCZVariable<double> wVelNonLinearSrc; // SU in Arches

  switch(eqnType) {
  case Arches::PRESSURE:
    switch(index) {
    case 0:
      new_dw->get(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(uVelNonLinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    case 1:
      new_dw->get(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(vVelNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    case 2:
      new_dw->get(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(wVelNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  case Arches::MOMENTUM:
    switch(index) {
    case 0:
      new_dw->get(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(uVelNonLinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    case 1:
      new_dw->get(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(vVelNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    case 2:
      new_dw->get(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      new_dw->get(wVelNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, Arches::ZEROGHOSTCELLS);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }

  // Get the patch and variable details
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  switch(index) {
  case 1:
    {
      // Get the patch and variable details
      IntVector domLoU = uVelocity.getFortLowIndex();
      IntVector domHiU = uVelocity.getFortHighIndex();
      IntVector idxLoU = patch->getSFCXFORTLowIndex();
      IntVector idxHiU = patch->getSFCXFORTHighIndex();


      // compute momentum source because of turbulence
      FORT_BCUTURB(domLoU.get_pointer(), domHiU.get_pointer(),
		   idxLoU.get_pointer(), idxHiU.get_pointer(),
		   uVelLinearSrc.getPointer(), 
		   uVelNonLinearSrc.getPointer(), 
		   uVelocity.getPointer(), 
		   domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   density.getPointer(), 
		   &mol_viscos, 
		   cellType.getPointer(), 
		   cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);
    }
    break;
  case 2:  
    {
      // Get the patch and variable details
      IntVector domLoV = vVelocity.getFortLowIndex();
      IntVector domHiV = vVelocity.getFortHighIndex();
      IntVector idxLoV = patch->getSFCYFORTLowIndex();
      IntVector idxHiV = patch->getSFCYFORTHighIndex();


      // compute momentum source because of turbulence
      FORT_BCVTURB(domLoV.get_pointer(), domHiV.get_pointer(),
		   idxLoV.get_pointer(), idxHiV.get_pointer(),
		   vVelLinearSrc.getPointer(), 
		   vVelNonLinearSrc.getPointer(), 
		   vVelocity.getPointer(), 
		   domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   density.getPointer(), 
		   &mol_viscos, 
		   cellType.getPointer(), 
		   cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);

    }
    break;
  case 3:
    {
      // Get the patch and variable details
      IntVector domLoW = wVelocity.getFortLowIndex();
      IntVector domHiW = wVelocity.getFortHighIndex();
      IntVector idxLoW = patch->getSFCZFORTLowIndex();
      IntVector idxHiW = patch->getSFCZFORTHighIndex();


      // compute momentum source because of turbulence
      FORT_BCWTURB(domLoW.get_pointer(), domHiW.get_pointer(),
		   idxLoW.get_pointer(), idxHiW.get_pointer(),
		   wVelLinearSrc.getPointer(), 
		   wVelNonLinearSrc.getPointer(), 
		   wVelocity.getPointer(), 
		   domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   density.getPointer(), 
		   &mol_viscos, 
		   cellType.getPointer(), 
		   cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);

    }
    break;
  default:
    throw InvalidValue("Invalid Index value in CalcVelWallBC");
  }

  switch(eqnType) {
  case Arches::PRESSURE:
    switch(index) {
    case 0:
      new_dw->put(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(uVelNonLinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    case 1:
      new_dw->put(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(vVelNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    case 2:
      new_dw->put(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(wVelNonLinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  case Arches::MOMENTUM:
    switch(index) {
    case 0:
      new_dw->put(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(uVelNonLinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    case 1:
      new_dw->put(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(vVelNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    case 2:
      new_dw->put(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(wVelNonLinearSrc, d_wVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }
#endif
}


//****************************************************************************
// No source term for smagorinsky model
//****************************************************************************
void SmagorinskyModel::calcVelocitySource(const ProcessorGroup*,
					  const Patch*,
					  const DataWarehouseP&,
					  DataWarehouseP&,
					  int)
{
}
