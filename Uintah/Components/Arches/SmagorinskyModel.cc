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
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
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
  // Inputs & Outputs (computeTurbSubModel (CTS)
  d_cellTypeLabel = scinew VarLabel("celltype",
				   CCVariable<int>::getTypeDescription() );
  d_uVelocitySPLabel = scinew VarLabel("uVelocitySP",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocitySPLabel = scinew VarLabel("vVelocitySP",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocitySPLabel = scinew VarLabel("wVelocitySP",
				    SFCZVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityINLabel = scinew VarLabel("viscosityIN",
				   CCVariable<double>::getTypeDescription() );

  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				   CCVariable<double>::getTypeDescription() );

  // Inputs & Outputs (calcVelocityWallBC)
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    SFCZVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
				   CCVariable<double>::getTypeDescription() );

  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );

  // For the recomputaion of Turbulence SubModel (during solve)
  d_densityRCPLabel = scinew VarLabel("densityRCP",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				    SFCZVariable<double>::getTypeDescription() );
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
  // Variables
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  // Get the velocity, density and viscosity from the old data warehouse
  int matlIndex = 0;
  int numGhostCells = 0;
  old_dw->get(uVelocity, d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(vVelocity, d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(wVelocity, d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
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

  // get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();

  // Create the new viscosity variable to write the result to 
  // and allocate space in the new data warehouse for this variable
  CCVariable<double> new_viscosity;
  new_dw->allocate(new_viscosity, d_viscosityCTSLabel, matlIndex, patch);

  // Get the patch and variable details
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_SMAGMODEL(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 new_viscosity.getPointer(), 
		 domLoU.get_pointer(), domHiU.get_pointer(),
		 idxLoU.get_pointer(), idxHiU.get_pointer(),
		 uVelocity.getPointer(), 
		 domLoV.get_pointer(), domHiV.get_pointer(),
		 idxLoV.get_pointer(), idxHiV.get_pointer(),
		 vVelocity.getPointer(), 
		 domLoW.get_pointer(), domHiW.get_pointer(),
		 idxLoW.get_pointer(), idxHiW.get_pointer(),
		 wVelocity.getPointer(), 
		 viscosity.getPointer(), 
		 density.getPointer(), 
		 &mol_viscos,
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
  // Variables
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  // Get the velocity, density and viscosity from the old data warehouse
  int matlIndex = 0;
  int numGhostCells = 0;
  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(density, d_densityRCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
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

  // get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();

  // Create the new viscosity variable to write the result to 
  // and allocate space in the new data warehouse for this variable
  CCVariable<double> new_viscosity;
  new_dw->allocate(new_viscosity, d_viscosityRCTSLabel, matlIndex, patch);

  // Get the patch and variable details
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_SMAGMODEL(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 new_viscosity.getPointer(), 
		 domLoU.get_pointer(), domHiU.get_pointer(),
		 idxLoU.get_pointer(), idxHiU.get_pointer(),
		 uVelocity.getPointer(), 
		 domLoV.get_pointer(), domHiV.get_pointer(),
		 idxLoV.get_pointer(), idxHiV.get_pointer(),
		 vVelocity.getPointer(), 
		 domLoW.get_pointer(), domHiW.get_pointer(),
		 idxLoW.get_pointer(), idxHiW.get_pointer(),
		 wVelocity.getPointer(), 
		 viscosity.getPointer(), 
		 density.getPointer(), 
		 &mol_viscos,
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
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
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

  CCVariable<double> density;
  old_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

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

  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, d_cellTypeLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  //get Molecular Viscosity of the fluid
  double mol_viscos = d_physicalConsts->getMolecularViscosity();

#ifdef WONT_COMPILE_YET
  cellFieldType walltype = WALLTYPE;
#endif

  numGhostCells = 0;

  SFCXVariable<double> uVelLinearSrc; //SP term in Arches
  SFCXVariable<double> uVelNonLinearSrc; // SU in Arches
  SFCYVariable<double> vVelLinearSrc; //SP term in Arches
  SFCYVariable<double> vVelNonLinearSrc; // SU in Arches
  SFCZVariable<double> wVelLinearSrc; //SP term in Arches
  SFCZVariable<double> wVelNonLinearSrc; // SU in Arches

  switch(eqnType) {
  case Discretization::PRESSURE:
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
  case Discretization::MOMENTUM:
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

#ifdef WONT_COMPILE_YET
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
#endif
    }
    break;
  case 2:  
    {
      // Get the patch and variable details
      IntVector domLoV = vVelocity.getFortLowIndex();
      IntVector domHiV = vVelocity.getFortHighIndex();
      IntVector idxLoV = patch->getSFCYFORTLowIndex();
      IntVector idxHiV = patch->getSFCYFORTHighIndex();

#ifdef WONT_COMPILE_YET
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
#endif
    }
    break;
  case 3:
    {
      // Get the patch and variable details
      IntVector domLoW = wVelocity.getFortLowIndex();
      IntVector domHiW = wVelocity.getFortHighIndex();
      IntVector idxLoW = patch->getSFCZFORTLowIndex();
      IntVector idxHiW = patch->getSFCZFORTHighIndex();

#ifdef WONT_COMPILE_YET
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
#endif
    }
    break;
  default:
    throw InvalidValue("Invalid Index value in CalcVelWallBC");
  }

  switch(eqnType) {
  case Discretization::PRESSURE:
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
  case Discretization::MOMENTUM:
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
// Revision 1.19  2000/06/29 23:37:12  bbanerje
// Changed FCVarsto SFC[X,Y,Z]Vars and added relevant getIndex() calls.
//
// Revision 1.18  2000/06/22 23:06:38  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
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

