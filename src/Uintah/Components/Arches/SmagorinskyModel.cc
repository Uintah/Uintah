
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
#include <Uintah/Grid/PerRegion.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;

using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

SmagorinskyModel::SmagorinskyModel(PhysicalConstants* phyConsts):
TurbulenceModel(), d_physicalConsts(phyConsts)
{
}

SmagorinskyModel::~SmagorinskyModel()
{
}

void SmagorinskyModel::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("Turbulence");
  db->require("cf", d_CF);
  db->require("fac_mesh", d_factorMesh);
  db->require("filterl", d_filterl);
}

void SmagorinskyModel::sched_computeTurbSubmodel(const LevelP& level,
						 SchedulerP& sched,
						 const DataWarehouseP& old_dw,
						 DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("SmagorinskyModel::TurbSubmodel",
			   region, old_dw, new_dw, this,
			   SmagorinskyModel::computeTurbSubmodel);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "viscosity", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
#endif
}


void SmagorinskyModel::computeTurbSubmodel(const ProcessorContext* pc,
					   const Region* region,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  CCVariable<double> viscosity;
  old_dw->get(viscosity, "viscosity", region, 1);
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerRegion<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", region)){
    top_dw->get(cellinfop, "cellinfo", region);
  } else {
    cellinfop.setData(new CellInformation(region));
    top_dw->put(cellinfop, "cellinfo", region);
  } 
  CellInformation* cellinfo = cellinfop;
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  //get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();
  CCVariable<double> new_viscosity;
  new_dw->allocate(new_viscosity, "viscosity", region, 0);
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
  new_dw->put(new_viscosity, "viscosity", region);
#endif
}

void SmagorinskyModel::calcVelocityWallBC(const ProcessorContext* pc,
					  const Region* region,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerRegion<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", region)){
    top_dw->get(cellinfop, "cellinfo", region);
  } else {
    cellinfop.setData(new CellInformation(region));
    top_dw->put(cellinfop, "cellinfo", region);
  } 
  CellInformation* cellinfo = cellinfop;
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();
  // stores cell type info for the region with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", region, 1);
  //get Molecular Viscosity of the fluid
    //get physical constants
  double mol_viscos; // molecular viscosity
  mol_viscos = d_physicalConsts->getMolecularViscosity();
  cellFieldType walltype = WALLTYPE;
  if (Index == 1) {
    //SP term in Arches
    FCVariable<double> uLinearSrc;
    new_dw->get(uLinearSrc, "uLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> uNonlinearSource;
    new_dw->get(uNonlinearSrc, "uNonlinearSource", region, 0);
    // compute momentum source because of turbulence
    FORT_BCUTURB(uLinearSrc, uNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(uLinearSrc, "uLinearSource", region);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", region);
  }

  else if (Index == 2) {
    //SP term in Arches
    FCVariable<double> vLinearSrc;
    new_dw->get(vLinearSrc, "vLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> vNonlinearSource;
    new_dw->get(vNonlinearSrc, "vNonlinearSource", region, 0);
    // compute momentum source because of turbulence
    FORT_BCVTURB(vLinearSrc, vNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(vLinearSrc, "vLinearSource", region);
    new_dw->put(vNonlinearSrc, "vNonlinearSource", region);
  }
  else if (Index == 3) {
    //SP term in Arches
    FCVariable<double> wLinearSrc;
    new_dw->get(wLinearSrc, "wLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> wNonlinearSource;
    new_dw->get(wNonlinearSrc, "wNonlinearSource", region, 0);
    // compute momentum source because of turbulence
    FORT_BCWTURB(wLinearSrc, wNonlinearSrc, velocity,  
		 density, mol_viscos, lowIndex, highIndex,
		 cellType, cellinfo->x, cellinfo->y, cellinfo->z,
		 cellinfo->xu, cellinfo->yv, cellinfo->zw);
    new_dw->put(wLinearSrc, "wLinearSource", region);
    new_dw->put(wNonlinearSrc, "wNonlinearSource", region);
  }
    else {
    cerr << "Invalid Index value" << endl;
  }
#endif
}


// No source term for samgorinsky model
void SmagorinskyModel::calcVelocitySource(const ProcessorContext* pc,
					  const Region* region,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  int index)
{
}



