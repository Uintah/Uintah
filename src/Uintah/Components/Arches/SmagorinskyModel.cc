//----- SmagorinksyModel.cc --------------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/SmagorinskyModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
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

  // Inputs that need to be changed later
  d_uVelocityLabel = scinew VarLabel("uVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityLabel = scinew VarLabel("vVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityLabel = scinew VarLabel("wVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  // Inputs
  d_uVelocitySPLabel = scinew VarLabel("uVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocitySPLabel = scinew VarLabel("vVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocitySPLabel = scinew VarLabel("wVelocitySP",
				    CCVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				   CCVariable<double>::getTypeDescription() );
  d_cellTypeLabel = scinew VarLabel("celltype",
				   CCVariable<int>::getTypeDescription() );

  // Outputs
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				   CCVariable<double>::getTypeDescription() );

  d_uLinSrcLabel = scinew VarLabel("uLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_vLinSrcLabel = scinew VarLabel("vLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_wLinSrcLabel = scinew VarLabel("wLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_uNonLinSrcLabel = scinew VarLabel("uNonLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_vNonLinSrcLabel = scinew VarLabel("vNonLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_wNonLinSrcLabel = scinew VarLabel("wNonLinearSrc",
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
      tsk->requires(old_dw, d_uVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_vVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_wVelocitySPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      numGhostCells = 0;
      tsk->requires(old_dw, d_densityCPLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->requires(old_dw, d_viscosityLabel, matlIndex, patch, Ghost::None,
		    numGhostCells);
      tsk->computes(new_dw, d_viscosityCTSLabel, matlIndex, patch);
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
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
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
// Calculate the Velocity BC at the Wall
//****************************************************************************
void SmagorinskyModel::calcVelocityWallBC(const ProcessorGroup* pc,
					  const Patch* patch,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
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

  switch(index) {
  case 1:
    {
      //SP term in Arches
      CCVariable<double> uLinearSrc;
      new_dw->get(uLinearSrc, d_uLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> uLinearSrc;
      //new_dw->get(uLinearSrc, "uLinearSource", patch, 0);

      // SU in Arches
      CCVariable<double> uNonLinearSrc;
      new_dw->get(uNonLinearSrc, d_uNonLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> uNonlinearSource;
      //new_dw->get(uNonlinearSrc, "uNonlinearSource", patch, 0);

#ifdef WONT_COMPILE_YET
      // compute momentum source because of turbulence
      FORT_BCUTURB(uLinearSrc, uNonlinearSrc, velocity,  
		   density, mol_viscos, lowIndex, highIndex,
		   cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif

      new_dw->put(uLinearSrc, d_uLinSrcLabel, matlIndex, patch);
      new_dw->put(uNonLinearSrc, d_uNonLinSrcLabel, matlIndex, patch);
      //new_dw->put(uLinearSrc, "uLinearSource", patch);
      //new_dw->put(uNonlinearSrc, "uNonlinearSource", patch);

      break;
    }
  case 2:  
    {
      //SP term in Arches
      CCVariable<double> vLinearSrc;
      new_dw->get(vLinearSrc, d_vLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> vLinearSrc;
      //new_dw->get(vLinearSrc, "vLinearSource", patch, 0);

      // SU in Arches
      CCVariable<double> vNonLinearSrc;
      new_dw->get(vNonLinearSrc, d_vNonLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> vNonlinearSource;
      //new_dw->get(vNonlinearSrc, "vNonlinearSource", patch, 0);

#ifdef WONT_COMPILE_YET
      // compute momentum source because of turbulence
      FORT_BCVTURB(vLinearSrc, vNonlinearSrc, velocity,  
		   density, mol_viscos, lowIndex, highIndex,
		   cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif

      new_dw->put(vLinearSrc, d_vLinSrcLabel, matlIndex, patch);
      new_dw->put(vNonLinearSrc, d_vNonLinSrcLabel, matlIndex, patch);
      //new_dw->put(vLinearSrc, "vLinearSource", patch);
      //new_dw->put(vNonlinearSrc, "vNonlinearSource", patch);

      break;
    }
  case 3:
    {
      //SP term in Arches
      CCVariable<double> wLinearSrc;
      new_dw->get(wLinearSrc, d_wLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> wLinearSrc;
      //new_dw->get(wLinearSrc, "wLinearSource", patch, 0);

      // SU in Arches
      CCVariable<double> wNonLinearSrc;
      new_dw->get(wNonLinearSrc, d_wNonLinSrcLabel, matlIndex, patch, Ghost::None,
		  numGhostCells);
      //FCVariable<double> wNonlinearSource;
      //new_dw->get(wNonlinearSrc, "wNonlinearSource", patch, 0);

#ifdef WONT_COMPILE_YET
      // compute momentum source because of turbulence
      FORT_BCWTURB(wLinearSrc, wNonlinearSrc, velocity,  
		   density, mol_viscos, lowIndex, highIndex,
		   cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		   cellinfo->xu, cellinfo->yv, cellinfo->zw);
#endif

      new_dw->put(wLinearSrc, d_wLinSrcLabel, matlIndex, patch);
      new_dw->put(wNonLinearSrc, d_wNonLinSrcLabel, matlIndex, patch);
      //new_dw->put(wLinearSrc, "wLinearSource", patch);
          //new_dw->put(wNonlinearSrc, "wNonlinearSource", patch);

      break;
    }
  default:
    {
      throw InvalidValue("Invalid Index value in CalcVelWallBC");
      break;
    }
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

