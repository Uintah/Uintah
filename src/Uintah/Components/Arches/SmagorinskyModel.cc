//----- SmagorinksyModel.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
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
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_velocityLabel = scinew VarLabel("velocity",
				    CCVariable<Vector>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
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

      int numGhostCells = 1;
      int matlIndex = 0;
      tsk->requires(old_dw, d_velocityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      numGhostCells = 0;
      tsk->requires(old_dw, d_densityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->computes(new_dw, d_viscosityLabel, matlIndex, patch);
      sched->addTask(tsk);
    }
  }
}


//****************************************************************************
// Actual compute 
//****************************************************************************
void 
SmagorinskyModel::computeTurbSubmodel(const ProcessorContext* pc,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{

  // Get the velocity, density and viscosity from the old data warehouse
  // (tmp) velocity should be FCVariable
  CCVariable<Vector> velocity;
  int matlIndex = 0;
  int nofGhostCells = 1;
  old_dw->get(velocity, d_velocityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      nofGhostCells);

#ifdef NOT_SURE_WHAT_THIS_DOES
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
  new_dw->allocate(new_viscosity, d_viscosityLabel, matlIndex, patch);

#ifdef NOT_COMPILED_YET
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
  new_dw->put(new_viscosity, d_viscosityLabel, matlIndex, patch);
}

//****************************************************************************
// Calculate the Velocity BC at the Wall
//****************************************************************************
void SmagorinskyModel::calcVelocityWallBC(const ProcessorContext* pc,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 1);
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
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();
  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", patch, 1);
  //get Molecular Viscosity of the fluid
    //get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();
  cellFieldType walltype = WALLTYPE;
  if (Index == 1) {
    //SP term in Arches
    FCVariable<double> uLinearSrc;
    new_dw->get(uLinearSrc, "uLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> uNonlinearSource;
    new_dw->get(uNonlinearSrc, "uNonlinearSource", patch, 0);
    // compute momentum source because of turbulence
    FORT_BCUTURB(uLinearSrc, uNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(uLinearSrc, "uLinearSource", patch);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", patch);
  }

  else if (Index == 2) {
    //SP term in Arches
    FCVariable<double> vLinearSrc;
    new_dw->get(vLinearSrc, "vLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> vNonlinearSource;
    new_dw->get(vNonlinearSrc, "vNonlinearSource", patch, 0);
    // compute momentum source because of turbulence
    FORT_BCVTURB(vLinearSrc, vNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(vLinearSrc, "vLinearSource", patch);
    new_dw->put(vNonlinearSrc, "vNonlinearSource", patch);
  }
  else if (Index == 3) {
    //SP term in Arches
    FCVariable<double> wLinearSrc;
    new_dw->get(wLinearSrc, "wLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> wNonlinearSource;
    new_dw->get(wNonlinearSrc, "wNonlinearSource", patch, 0);
    // compute momentum source because of turbulence
    FORT_BCWTURB(wLinearSrc, wNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(wLinearSrc, "wLinearSource", patch);
    new_dw->put(wNonlinearSrc, "wNonlinearSource", patch);
  }
    else {
    cerr << "Invalid Index value" << endl;
  }
#endif
}


//****************************************************************************
// No source term for samgorinsky model
//****************************************************************************
void SmagorinskyModel::calcVelocitySource(const ProcessorContext* pc,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
}

//
// $Log$
// Revision 1.9  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
// Revision 1.8  2000/05/31 20:11:30  bbanerje
// Cocoon stuff, tasks added to SmagorinskyModel, TurbulenceModel.
// Added schedule compute of properties and TurbModel to Arches.
//
//

