//----- SmagorinksyModel.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/Stencil.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>

using namespace std;

using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Constructor that is private
//****************************************************************************
SmagorinskyModel::SmagorinskyModel(): TurbulenceModel()
{
}

//****************************************************************************
// Default constructor for SmagorinkyModel
//****************************************************************************
SmagorinskyModel::SmagorinskyModel(PhysicalConstants* phyConsts):
                                                 TurbulenceModel(), 
                                                 d_physicalConsts(phyConsts)
{
  // BB : (**WARNING**) velocity is set as CCVariable (should be FCVariable)
  // Changed all vel related vars to FCVariable and then delete this comment.

  // Inputs & Outputs (computeTurbSubModel (CTS)
  d_cellTypeLabel = scinew VarLabel("celltype",
				   CCVariable<int>::getTypeDescription() );
  d_uVelocitySPLabel = scinew VarLabel("uVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocitySPLabel = scinew VarLabel("vVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocitySPLabel = scinew VarLabel("wVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityINLabel = scinew VarLabel("viscosityIN",
				   CCVariable<double>::getTypeDescription() );

  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				   CCVariable<double>::getTypeDescription() );

  // Inputs & Outputs (calcVelocityWallBC)
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
				   CCVariable<double>::getTypeDescription() );

  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );

  // For the recomputaion of Turbulence SubModel (during solve)
  d_densityRCPLabel = scinew VarLabel("densityRCP",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				    CCVariable<double>::getTypeDescription() );
  d_viscosityRCTSLabel = scinew VarLabel("viscosityRCTS",
				    CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
SmagorinskyModel::~SmagorinskyModel()
{
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
}

//****************************************************************************
// Schedule compute 
//****************************************************************************
void 
SmagorinskyModel::sched_computeTurbSubmodel(const LevelP& level,
					    SchedulerP& sched,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("SmagorinskyModel::TurbSubmodel",
			      patch, old_dw, new_dw, this,
			      &SmagorinskyModel::computeTurbSubmodel);

      int numGhostCells = 0;
      int matlIndex = 0;

      // Requires
      tsk->requires(old_dw, d_densityCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityINLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // Computes
      tsk->computes(new_dw, d_viscosityCTSLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Schedule recomputation of the turbulence sub model 
//****************************************************************************
void 
SmagorinskyModel::sched_reComputeTurbSubmodel(const LevelP& level,
					    SchedulerP& sched,
					    DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("SmagorinskyModel::ReTurbSubmodel",
			      patch, old_dw, new_dw, this,
			      &SmagorinskyModel::reComputeTurbSubmodel);

      int numGhostCells = 0;
      int matlIndex = 0;

      // Requires
      tsk->requires(new_dw, d_densityRCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(new_dw, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);

      // Computes
      tsk->computes(new_dw, d_viscosityRCTSLabel, matlIndex, patch);

      sched->addTask(tsk);
    }
  }
}

//****************************************************************************
// Actual compute 
//****************************************************************************
void 
SmagorinskyModel::computeTurbSubmodel(const ProcessorGroup* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{

  // Get the velocity, density and viscosity from the old data warehouse
  // (tmp) velocity should be FCVariable
  int matlIndex = 0;
  int numGhostCells = 0;
  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  old_dw->get(vVelocity, d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  old_dw->get(wVelocity, d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityINLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  // Get the patch details
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();

  // Create the new viscosity variable to write the result to 
  // and allocate space in the new data warehouse for this variable
  CCVariable<double> new_viscosity;
  new_dw->allocate(new_viscosity, d_viscosityCTSLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
  FORT_SMAGMODEL(new_viscosity, velocity, viscosity, density, mol_viscos,
		 lowIndex, highIndex,
		 cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		 cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		 cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		 cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		 cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		 cellinfo->tfac, cellinfo->bfac);
#endif

  // Put the calculated viscosityvalue into the new data warehouse
  new_dw->put(new_viscosity, d_viscosityCTSLabel, matlIndex, patch);
}

//****************************************************************************
// Actual recompute 
//****************************************************************************
void 
SmagorinskyModel::reComputeTurbSubmodel(const ProcessorGroup* pc,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw)
{

  // Get the velocity, density and viscosity from the old data warehouse
  // (tmp) velocity should be FCVariable
  int matlIndex = 0;
  int numGhostCells = 0;
  CCVariable<double> uVelocity;
  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> density;
  new_dw->get(density, d_densityRCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  // Get the patch details
  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();

  // Create the new viscosity variable to write the result to 
  // and allocate space in the new data warehouse for this variable
  CCVariable<double> new_viscosity;
  new_dw->allocate(new_viscosity, d_viscosityRCTSLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
  FORT_SMAGMODEL(new_viscosity, velocity, viscosity, density, mol_viscos,
		 lowIndex, highIndex,
		 cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		 cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		 cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		 cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		 cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		 cellinfo->tfac, cellinfo->bfac);
#endif

  // Put the calculated viscosityvalue into the new data warehouse
  new_dw->put(new_viscosity, d_viscosityRCTSLabel, matlIndex, patch);
}

//****************************************************************************
// Calculate the Velocity BC at the Wall
//****************************************************************************
void SmagorinskyModel::calcVelocityWallBC(const ProcessorGroup* pc,
					  const Patch* patch,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index,
					  int eqnType)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;
  switch(eqnType) {
  case Discretization::PRESSURE:
    old_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case Discretization::MOMENTUM:
    old_dw->get(uVelocity, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  CCVariable<double> density;
  old_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(density, "density", patch, 1);

  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();

#ifdef WONT_COMPILE_YET
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //top_dw->get(cellType, "CellType", patch, 1);

  //get Molecular Viscosity of the fluid
  double mol_viscos = d_physicalConsts->getMolecularViscosity();

#ifdef WONT_COMPILE_YET
  cellFieldType walltype = WALLTYPE;
#endif

  numGhostCells = 0;

  // Variables used (** WARNING ** FCVars)
  CCVariable<double> velLinearSrc; //SP term in Arches
  CCVariable<double> velNonLinearSrc; // SU in Arches

  switch(eqnType) {
  case Discretization::PRESSURE:
    switch(index) {
    case 0:
      new_dw->get(velLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 1:
      new_dw->get(velLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 2:
      new_dw->get(velLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  case Discretization::MOMENTUM:
    switch(index) {
    case 0:
      new_dw->get(velLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 1:
      new_dw->get(velLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case 2:
      new_dw->get(velLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      new_dw->get(velNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }

  switch(index) {
  case 1:
#ifdef WONT_COMPILE_YET
    // compute momentum source because of turbulence
    FORT_BCUTURB(velLinearSrc, velNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif
    break;
  case 2:  
#ifdef WONT_COMPILE_YET
    // compute momentum source because of turbulence
    FORT_BCVTURB(vLinearSrc, vNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif
    break;
  case 3:
#ifdef WONT_COMPILE_YET
    // compute momentum source because of turbulence
    FORT_BCWTURB(wLinearSrc, wNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif
    break;
  default:
    throw InvalidValue("Invalid Index value in CalcVelWallBC");
  }

  switch(eqnType) {
  case Discretization::PRESSURE:
    switch(index) {
    case 0:
      new_dw->put(velLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    case 1:
      new_dw->put(velLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    case 2:
      new_dw->put(velLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  case Discretization::MOMENTUM:
    switch(index) {
    case 0:
      new_dw->put(velLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    case 1:
      new_dw->put(velLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    case 2:
      new_dw->put(velLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch);
      new_dw->put(velNonLinearSrc, d_wVelNonLinSrcMBLMLabel, matlIndex, patch);
      break;
    default:
      throw InvalidValue("Index can only be 0, 1 or 2");
    }
    break;
  default:
    throw InvalidValue("Equation type can only be pressure or momentum");
  }

}


//****************************************************************************
// No source term for samgorinsky model
//****************************************************************************
void SmagorinskyModel::calcVelocitySource(const ProcessorGroup* pc,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
}

//
// $Log$
// Revision 1.17  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.16  2000/06/21 06:12:12  bbanerje
// Added missing VarLabel* mallocs .
//
// Revision 1.15  2000/06/18 01:20:17  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.14  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.13  2000/06/16 21:50:48  bbanerje
// Changed the Varlabels so that sequence in understood in init stage.
// First cycle detected in task graph.
//
// Revision 1.12  2000/06/14 20:40:49  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.11  2000/06/12 21:30:00  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.10  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.9  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
// Revision 1.8  2000/05/31 20:11:30  bbanerje
// Cocoon stuff, tasks added to SmagorinskyModel, TurbulenceModel.
// Added schedule compute of properties and TurbModel to Arches.
//
//

