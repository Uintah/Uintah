//----- SmagorinksyModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
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
				   PhysicalConstants* phyConsts):
                                    TurbulenceModel(), 
                                    d_lab(label), d_MAlab(MAlb),
				    d_physicalConsts(phyConsts)
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
}

//****************************************************************************
// Schedule compute 
//****************************************************************************
void 
SmagorinskyModel::sched_computeTurbSubmodel(SchedulerP& sched, const PatchSet* patches,
					    const MaterialSet* matls)
{
  Task* tsk = scinew Task("SmagorinskyModel::TurbSubmodel",
			  this,
			  &SmagorinskyModel::computeTurbSubmodel);

  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  // Requires
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, Ghost::None,
		zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPLabel, 
		Ghost::AroundCells,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPLabel,
		Ghost::AroundCells,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPLabel,
		Ghost::AroundCells,
		numGhostCells);

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
					      const MaterialSet* matls)
{
  Task* tsk = scinew Task("SmagorinskyModel::ReTurbSubmodel",
			  this,
			  &SmagorinskyModel::reComputeTurbSubmodel);

  int numGhostCells = 1;
  int zeroGhostCells = 0;
  // Requires
  tsk->requires(Task::NewDW, d_lab->d_densityCPLabel, Ghost::None,
		zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_viscosityINLabel, 
		Ghost::None,
		zeroGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_uVelocitySPBCLabel,
		Ghost::AroundCells,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_vVelocitySPBCLabel, 
		Ghost::AroundCells,
		numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_wVelocitySPBCLabel, 
		Ghost::AroundCells,
		numGhostCells);
  // for multimaterial
  if (d_MAlab)
    tsk->requires(Task::NewDW, d_lab->d_mmgasVolFracLabel, 
		  Ghost::None, zeroGhostCells);

      // Computes
  tsk->computes(d_lab->d_viscosityCTSLabel);

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
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    CCVariable<double> density;
    CCVariable<double> viscosity;

    // Get the velocity, density and viscosity from the old data warehouse
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    new_dw->get(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch, Ghost::None,
		zeroGhostCells);
    new_dw->get(uVelocity, d_lab->d_uVelocitySPLabel, matlIndex, patch,
		Ghost::AroundCells, numGhostCells);
    new_dw->get(vVelocity, d_lab->d_vVelocitySPLabel, matlIndex, patch, 
		Ghost::AroundCells, numGhostCells);
    new_dw->get(wVelocity,d_lab->d_wVelocitySPLabel, matlIndex, patch, 
		Ghost::AroundCells, numGhostCells);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		zeroGhostCells);

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
    fort_smagmodel(uVelocity, vVelocity, wVelocity, density, viscosity,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   mol_viscos, d_CF, d_factorMesh, d_filterl);

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
      new_dw->put(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);
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
					DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    // Variables
    SFCXVariable<double> uVelocity;
    SFCYVariable<double> vVelocity;
    SFCZVariable<double> wVelocity;
    CCVariable<double> density;
    CCVariable<double> viscosity;
    CCVariable<double> voidFraction;
    // Get the velocity, density and viscosity from the old data warehouse
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    new_dw->get(uVelocity,d_lab->d_uVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundCells, numGhostCells);
    new_dw->get(vVelocity,d_lab->d_vVelocitySPBCLabel, matlIndex, patch,
		Ghost::AroundCells, numGhostCells);
    new_dw->get(wVelocity, d_lab->d_wVelocitySPBCLabel, matlIndex, patch, 
		Ghost::AroundCells, numGhostCells);
    new_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
		zeroGhostCells);
    new_dw->get(viscosity, d_lab->d_viscosityINLabel, matlIndex, patch, Ghost::None,
		zeroGhostCells);
    if (d_MAlab)
      new_dw->get(voidFraction, d_lab->d_mmgasVolFracLabel, matlIndex, patch,
		  Ghost::None, zeroGhostCells);
    // Get the PerPatch CellInformation data
    PerPatch<CellInformationP> cellInfoP;
    //  old_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    if (new_dw->exists(d_lab->d_cellInfoLabel, matlIndex, patch)) 
      new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    else {
      cellInfoP.setData(scinew CellInformation(patch));
      new_dw->put(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    }
    //  new_dw->get(cellInfoP, d_lab->d_cellInfoLabel, matlIndex, patch);
    //  if (old_dw->exists(d_cellInfoLabel, patch)) 
    //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //else {
    //  cellInfoP.setData(scinew CellInformation(patch));
    //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
    //}
    CellInformation* cellinfo = cellInfoP.get().get_rep();
    
    // get physical constants
    double mol_viscos; // molecular viscosity
    mol_viscos = d_physicalConsts->getMolecularViscosity();
    
    // Get the patch and variable details
    // compatible with fortran index
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    fort_smagmodel(uVelocity, vVelocity, wVelocity, density, viscosity,
		   idxLo, idxHi,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   mol_viscos, d_CF, d_factorMesh, d_filterl);

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
    new_dw->put(viscosity, d_lab->d_viscosityCTSLabel, matlIndex, patch);
  }
}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
SmagorinskyModel::sched_computeScalarVariance(SchedulerP& sched, 
					      const PatchSet* patches,
					      const MaterialSet* matls)
{
  Task* tsk = scinew Task("SmagorinskyModel::computeScalarVar",
			  this,
			  &SmagorinskyModel::computeScalarVariance);

  int numGhostCells = 1;
  int zeroGhostCells = 0;
  
  // Requires, only the scalar corresponding to matlindex = 0 is
  //           required. For multiple scalars this will be put in a loop
  
  tsk->requires(Task::NewDW, d_lab->d_scalarSPLabel, 
		Ghost::AroundCells, numGhostCells);
  tsk->requires(Task::NewDW, d_lab->d_scalarVarINLabel, 
		Ghost::None, zeroGhostCells);

  // Computes
  tsk->computes(d_lab->d_scalarVarSPLabel);

  sched->addTask(tsk, patches, matls);
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("SmagorinskyModel::computeScalarVar",
			      patch, old_dw, new_dw, this,
			      &SmagorinskyModel::computeScalarVariance);

      int numGhostCells = 1;
      int zeroGhostCells = 0;
      int matlIndex = 0;

      // Requires, only the scalar corresponding to matlindex = 0 is
      //           required. For multiple scalars this will be put in a loop
      
      tsk->requires(new_dw, d_lab->d_scalarSPLabel, matlIndex, patch, 
		    Ghost::AroundCells, numGhostCells);
      tsk->requires(new_dw, d_lab->d_scalarVarINLabel, matlIndex, patch, 
		    Ghost::None, zeroGhostCells);

      // Computes
      tsk->computes(new_dw, d_lab->d_scalarVarSPLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
#endif
}


void 
SmagorinskyModel::computeScalarVariance(const ProcessorGroup*,
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
    CCVariable<double> scalar;
    CCVariable<double> scalarVar;
    // Get the velocity, density and viscosity from the old data warehouse
    int numGhostCells = 1;
    int zeroGhostCells = 0;
    new_dw->get(scalar, d_lab->d_scalarSPLabel, matlIndex, patch, Ghost::AroundCells,
		numGhostCells);
    new_dw->get(scalarVar, d_lab->d_scalarVarINLabel, matlIndex, patch,
		Ghost::None, zeroGhostCells);
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
    
    fort_scalarvarmodel(scalar, idxLo, idxHi, scalarVar, cellinfo->dxpw,
			cellinfo->dyps, cellinfo->dzpb, cellinfo->sew,
			cellinfo->sns, cellinfo->stb, d_CFVar, d_factorMesh,
			d_filterl);

    // Put the calculated viscosityvalue into the new data warehouse
    new_dw->put(scalarVar, d_lab->d_scalarVarSPLabel, matlIndex, patch);
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
  int numGhostCells = 0;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  switch(eqnType) {
  case Arches::PRESSURE:
    old_dw->get(uVelocity, d_lab->d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_lab->d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_lab->d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case Arches::MOMENTUM:
    old_dw->get(uVelocity, d_lab->d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_lab->d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_lab->d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }

  CCVariable<double> density;
  old_dw->get(density, d_lab->d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

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
	      numGhostCells);

  //get Molecular Viscosity of the fluid
  double mol_viscos = d_physicalConsts->getMolecularViscosity();


  cellFieldType walltype = WALLTYPE;


  numGhostCells = 0;

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
		  Ghost::None, numGhostCells);
      new_dw->get(uVelNonLinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 1:
      new_dw->get(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(vVelNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 2:
      new_dw->get(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(wVelNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  case Arches::MOMENTUM:
    switch(index) {
    case 0:
      new_dw->get(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(uVelNonLinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 1:
      new_dw->get(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(vVelNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 2:
      new_dw->get(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(wVelNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
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
// No source term for samgorinsky model
//****************************************************************************
void SmagorinskyModel::calcVelocitySource(const ProcessorGroup*,
					  const Patch*,
					  const DataWarehouseP&,
					  DataWarehouseP&,
					  int)
{
}
