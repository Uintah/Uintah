#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Grid/Stencil.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerRegion.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;
using Uintah::Components::Discretization;
using namespace Uintah::Components;
using namespace Uintah::Grid;
using SCICore::Geometry::Vector;

BoundaryCondition::BoundaryCondition()
{
}

BoundaryCondition::BoundaryCondition(TurbulenceModel* turb_model)
  :d_turbModel(turb_model)
{
}

BoundaryCondition::~BoundaryCondition()
{
}

void BoundaryCondition::problemSetup(const ProblemSpecP& params,
				     DataWarehouseP& dw)
{
  ProblemSpecP db = params->findBlock("Boundary Conditions");
  int numFlowInlets;
  db->require("NumFlowInlets", numFlowInlets);
  dw->put(numFlowInlets, "NumFlowInlets");
  int numMixingScalars;
  // set number of scalars in properties based on
  // num of streams read
  dw->get(numMixingScalars, "NumMixingScalars");
  for (int i = 0; i < numFlowInlets; i++) {
    FlowInlet* flow_inlet = new FlowInlet(numMixingScalars);
    // put flow_inlet to the database, use celltype info to
    // differentiate between different inlets
    flow_inlet->problemSetup(db, dw);
  }
  bool pressureBC;
  // set the boolean in the section where cell type info is read
  dw->get(pressureBC, "bool_pressureBC");
  if (pressureBC) {
    PressureInlet* press_inlet = new PressureInlet(numMixingScalars);
    press_inlet->problemSetup(db, dw);
  }
  bool outletBC;
  dw->get(outletBC, "bool_outletBC");
  if (outletBC) {
    FlowOutlet* flow_outlet = new FlowOutlet(numMixingScalars);
    flow_outlet->problemSetup(db, dw);
  }
  
}

    
// assigns flat velocity profiles for primary and secondary inlets
// Also sets flat profiles for density
void BoundaryCondition::sched_setFlatProfile(const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("BoundaryCondition::setProfile",
			   region, old_dw, new_dw, this,
			   BoundaryCondition::setFlatProfile);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
}


void BoundaryCondition::setFlatProfile(const ProcessorContext* pc,
				       const Region* region,
				       const DataWarehouseP& old_dw)
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();
  // stores cell type info for the region with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", region, 1);
  int num_flow_inlets;
  old_dw->get(num_flow_inlets, "NumFlowInlets");
  for (int indx = 0; indx < num_flow_inlets; indx++) {
    FlowInlet* flowinletp;
    // return properties of indx inlet
    old_dw->get(flowinletp, "FlowInlet", indx);
    FORT_PROFV(velocity, density, lowIndex, highIndex, cellType,
	       flowinletp->flowrate, flowinletp->area,
	       flowinletp->density, flowinletp->inletType);
    old_dw->put(velocity, "velocity", region, 1);
    old_dw->put(density, "density", region, 1);
  }
}



void BoundaryCondition::sched_setInletVelocityBC(const LevelP& level,
						 SchedulerP& sched,
						 const DataWarehouseP& old_dw,
						 DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("BoundaryCondition::setProfile",
			   region, old_dw, new_dw, this,
			   BoundaryCondition::setInletVelocityBC);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
}


void BoundaryCondition::setInletVelocityBC(const ProcessorContext* pc,
					   const Region* region,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw) 
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();
  // stores cell type info for the region with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", region, 1);
  int num_flow_inlets;
  old_dw->get(num_flow_inlets, "NumFlowInlets");
  for (int indx = 0; indx < num_flow_inlets; indx++) {
    FlowInlet* flowinletp;
    // return properties of indx inlet
    old_dw->get(flowinletp, "FlowInlet", indx);
    // assign flowType the value that corresponds to flow
    CellTypeInfo flowType = FLOW;
    FORT_INLBCS(velocity, density, lowIndex, highIndex, cellType,
		flowinletp->flowrate, flowinletp->area,
		flowinletp->density, flowinletp->inletType,
		flowType);
    new_dw->put(velocity, "velocity", region, 1);
    new_dw->put(density, "density", region, 1);
  }
}

 

void BoundaryCondition::sched_computePressureBC(const LevelP& level,
						SchedulerP& sched,
						const DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("BoundaryCondition::setProfile",
			   region, old_dw, new_dw, this,
			   BoundaryCondition::calculatePressBC);
      tsk->requires(old_dw, "pressure", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
}


void BoundaryCondition::calculatePressBC(const ProcessorContext* pc,
					 const Region* region,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) 
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 1);
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();
  // stores cell type info for the region with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", region, 1);
  PressureInlet* pressinletp;
  old_dw->get(pressinletp, "PressureInlet");
  FORT_CALPBC(velocity, pressure, density, lowIndex, highIndex, cellType,
	      pressinletp->refPressure, pressinletp->area,
	      pressinletp->density, pressinletp->inletType);
  new_dw->put(velocity, "velocity", region, 1);
 
} 
// sets velocity bc
void BoundaryCondition::sched_velocityBC(const int index,
					 const LevelP& level,
					 SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("BoundaryCondition::VelocityBC",
			   region, old_dw, new_dw, this,
			   BoundaryCondition::velocityBC,
			   index);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      if (index == 1) {
	tsk->requires(new_dw, "uVelocityCoeff", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "uLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "uNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uVelocityCoeff", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());

      }
      else if (index == 2) {
	tsk->requires(new_dw, "vVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "vLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "vNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 3) {
	tsk->requires(new_dw, "wVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "wLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "wNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else {
	throw InvalidValue("Invalid component for velocity" +index);
      }
      sched->addTask(tsk);
    }

  }
}

void BoundaryCondition::velocityBC(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     const int index) 
{
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
  SoleVariable<double> VISCOS_CONST;
  top_dw->get(VISCOS_CONST, "viscosity"); 
  double VISCOS = VISCOS_CONST;

  if (Index == 1) {
    Stencil<double> uVelocityCoeff;
    new_dw->get(uVelocityCoeff, "uVelocityCoeff", region, 0);
    //SP term in Arches
    FCVariable<double> uLinearSrc;
    new_dw->get(uLinearSrc, "uLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> uNonlinearSource;
    new_dw->get(uNonlinearSrc, "uNonlinearSource", region, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCUVEL(uVelocityCoeff, uLinearSrc, uNonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
    new_dw->put(uVelocityCoeff, "uVelocityCoeff", region, 0);
    new_dw->put(uLinearSrc, "uLinearSource", region);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", region);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, region, old_dw, new_dw, Index);
  }

  else if (Index == 2) {
    Stencil<double> vVelocityCoeff;
    new_dw->get(vVelocityCoeff, "vVelocityCoeff", region, 0);
    //SP term in Arches
    FCVariable<double> vLinearSrc;
    new_dw->get(vLinearSrc, "vLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> vNonlinearSource;
    new_dw->get(vNonlinearSrc, "vNonlinearSource", region, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCVVEL(vVelocityCoeff, vLinearSrc, vNonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
    new_dw->put(vVelocityCoeff, "vVelocityCoeff", region, 0);
    new_dw->put(vLinearSrc, "vLinearSource", region);
    new_dw->put(vNonlinearSrc, "vNonlinearSource", region);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, region, old_dw, new_dw, Index);
  }
  else if (Index == 3) {
    Stencil<double> wVelocityCoeff;
    new_dw->get(wVelocityCoeff, "wVelocityCoeff", region, 0);
    //SP term in Arches
    FCVariable<double> wLinearSrc;
    new_dw->get(wLinearSrc, "wLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> wNonlinearSource;
    new_dw->get(wNonlinearSrc, "wNonlinearSource", region, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_BCWVEL(wVelocityCoeff, wLinearSrc, wNonlinearSrc, velocity,  
		density, VISCOS, ioff, joff, koff, lowIndex, highIndex,
		cellType,
		cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		cellinfo->cnn, cellinfo->csn, cellinfo->css,
		cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac, volume);
    new_dw->put(wVelocityCoeff, "wVelocityCoeff", region, 0);
    new_dw->put(wLinearSrc, "wLinearSource", region);
    new_dw->put(wNonlinearSrc, "wNonlinearSource", region);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, region, old_dw, new_dw, Index);
  }
    else {
    cerr << "Invalid Index value" << endl;
  }
}

void BoundaryCondition::sched_pressureBC(const LevelP& level,
					 const Region* region,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = new Task("BoundaryCondition::PressureBC",region,
			   old_dw, new_dw, this,
			   Discretization::pressureBC);
      tsk->requires(old_dw, "pressure", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureCoeff", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}



void BoundaryCondition::pressureBC(const ProcessorContext*,
				   const Region* region,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 1);
  CCVariable<Vector> pressCoeff; //7 point stencil
  new_dw->get(pressCoeff,"pressureCoeff",region, 0);
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
  //fortran call
  FORT_PRESSBC(pressCoeff, pressure, 
	       lowIndex, highIndex, cellType);
  new_dw->put(pressCoeff, "pressureCoeff", region, 0);
}

void BoundaryCondition::sched_scalarBC(const int index,
				       const LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
}

