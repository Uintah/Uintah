#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Grid/Stencil.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <iostream>
using namespace std;
using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

BoundaryCondition::BoundaryCondition()
{
  //construct 3d array for storing boundary type 
  //  IntVector lowIndex = (0,0,0);
  //  cellTypes = scinew Array3<int>(lowIndex, DOMAIN_HIGH);
}

BoundaryCondition::BoundaryCondition(TurbulenceModel* turb_model)
  :d_turbModel(turb_model)
{
  //  IntVector lowIndex = (0,0,0);
  //  cellTypes = scinew Array3<int>(lowIndex, DOMAIN_HIGH);
}

BoundaryCondition::~BoundaryCondition()
{
}

void BoundaryCondition::problemSetup(const ProblemSpecP& params)
{
#if 0
  ProblemSpecP db = params->findBlock("BoundaryConditions");
  int numFlowInlets;
  db->require("numFlowInlets", numFlowInlets);
  dw->put(numFlowInlets, "NumFlowInlets");
  int numMixingScalars;
  // set number of scalars in properties based on
  // num of streams read
  // change
  dw->get(numMixingScalars, "NumMixingScalars");
  for (int i = 0; i < numFlowInlets; i++) {
    FlowInlet* flow_inlet = scinew FlowInlet(numMixingScalars);
    // put flow_inlet to the database, use celltype info to
    // differentiate between different inlets
    flow_inlet->problemSetup(db, dw);
  }
  bool pressureBC;
  // set the boolean in the section where cell type info is read
  dw->get(pressureBC, "bool_pressureBC");
  if (pressureBC) {
    PressureInlet* press_inlet = scinew PressureInlet(numMixingScalars);
    press_inlet->problemSetup(db, dw);
  }
  bool outletBC;
  dw->get(outletBC, "bool_outletBC");
  if (outletBC) {
    FlowOutlet* flow_outlet = scinew FlowOutlet(numMixingScalars);
    flow_outlet->problemSetup(db, dw);
  }
#endif
}

    
// assigns flat velocity profiles for primary and secondary inlets
// Also sets flat profiles for density
void BoundaryCondition::sched_setFlatProfile(const LevelP& level,
					     SchedulerP& sched,
					     const DataWarehouseP& old_dw)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::setProfile",
			      patch, old_dw, new_dw, this,
			      BoundaryCondition::setFlatProfile);
      tsk->requires(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(old_dw, "density", patch, 1,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
#endif
}


void BoundaryCondition::setFlatProfile(const ProcessorContext* pc,
				       const Patch* patch,
				       const DataWarehouseP& old_dw)
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 1);
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();
  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", patch, 1);
  int num_flow_inlets;
  old_dw->get(num_flow_inlets, "NumFlowInlets");
  for (int indx = 0; indx < num_flow_inlets; indx++) {
    FlowInlet* flowinletp;
    // return properties of indx inlet
    old_dw->get(flowinletp, "FlowInlet", indx);
    FORT_PROFV(velocity, density, lowIndex, highIndex, cellType,
	       flowinletp->flowrate, flowinletp->area,
	       flowinletp->density, flowinletp->inletType);
    old_dw->put(velocity, "velocity", patch, 1);
    old_dw->put(density, "density", patch, 1);
  }
#endif
}



void BoundaryCondition::sched_setInletVelocityBC(const LevelP& level,
						 SchedulerP& sched,
						 const DataWarehouseP& old_dw,
						 DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::setProfile",
			      patch, old_dw, new_dw, this,
			      BoundaryCondition::setInletVelocityBC);
      tsk->requires(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
#endif
}


void BoundaryCondition::setInletVelocityBC(const ProcessorContext* pc,
					   const Patch* patch,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw) 
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();
  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", patch, 1);
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
    new_dw->put(velocity, "velocity", patch, 1);
    new_dw->put(density, "density", patch, 1);
  }
#endif
}

 

void BoundaryCondition::sched_computePressureBC(const LevelP& level,
						SchedulerP& sched,
						const DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::setProfile",
			      patch, old_dw, new_dw, this,
			      BoundaryCondition::calculatePressBC);
      tsk->requires(old_dw, "pressure", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }
  }
#endif
}


void BoundaryCondition::calculatePressBC(const ProcessorContext* pc,
					 const Patch* patch,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) 
{
#if 0
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 1);
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();
  // stores cell type info for the patch with the ghost cell type
  CCVariable<int> cellType;
  top_dw->get(cellType, "CellType", patch, 1);
  PressureInlet* pressinletp;
  old_dw->get(pressinletp, "PressureInlet");
  FORT_CALPBC(velocity, pressure, density, lowIndex, highIndex, cellType,
	      pressinletp->refPressure, pressinletp->area,
	      pressinletp->density, pressinletp->inletType);
  new_dw->put(velocity, "velocity", patch, 1);
#endif
} 
// sets velocity bc
void BoundaryCondition::sched_velocityBC(const int index,
					 const LevelP& level,
					 SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
#ifdef WONT_COMPILE_YET
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      Task* tsk = scinew Task("BoundaryCondition::VelocityBC",
			      patch, old_dw, new_dw, this,
			      BoundaryCondition::velocityBC,
			      index);
      tsk->requires(old_dw, "velocity", patch, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", patch, 1,
		    CCVariable<double>::getTypeDescription());
      if (index == 1) {
	tsk->requires(new_dw, "uVelocityCoeff", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "uLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "uNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uVelocityCoeff", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());

      }
      else if (index == 2) {
	tsk->requires(new_dw, "vVelocityCoeff", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "vLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "vNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vVelocityCoeff", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 3) {
	tsk->requires(new_dw, "wVelocityCoeff", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "wLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->requires(new_dw, "wNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wVelocityCoeff", patch, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wLinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wNonlinearSource", patch, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else {
	throw InvalidValue("Invalid component for velocity" +index);
      }
      sched->addTask(tsk);
    }

  }
#endif
}

void BoundaryCondition::velocityBC(const ProcessorContext* pc,
				   const Patch* patch,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   const int index) 
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
  SoleVariable<double> VISCOS_CONST;
  top_dw->get(VISCOS_CONST, "viscosity"); 
  double VISCOS = VISCOS_CONST;

  if (Index == 1) {
    Stencil<double> uVelocityCoeff;
    new_dw->get(uVelocityCoeff, "uVelocityCoeff", patch, 0);
    //SP term in Arches
    FCVariable<double> uLinearSrc;
    new_dw->get(uLinearSrc, "uLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> uNonlinearSource;
    new_dw->get(uNonlinearSrc, "uNonlinearSource", patch, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
    new_dw->put(uVelocityCoeff, "uVelocityCoeff", patch, 0);
    new_dw->put(uLinearSrc, "uLinearSource", patch);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", patch);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, patch, old_dw, new_dw, Index);
  }

  else if (Index == 2) {
    Stencil<double> vVelocityCoeff;
    new_dw->get(vVelocityCoeff, "vVelocityCoeff", patch, 0);
    //SP term in Arches
    FCVariable<double> vLinearSrc;
    new_dw->get(vLinearSrc, "vLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> vNonlinearSource;
    new_dw->get(vNonlinearSrc, "vNonlinearSource", patch, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
    new_dw->put(vVelocityCoeff, "vVelocityCoeff", patch, 0);
    new_dw->put(vLinearSrc, "vLinearSource", patch);
    new_dw->put(vNonlinearSrc, "vNonlinearSource", patch);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, patch, old_dw, new_dw, Index);
  }
  else if (Index == 3) {
    Stencil<double> wVelocityCoeff;
    new_dw->get(wVelocityCoeff, "wVelocityCoeff", patch, 0);
    //SP term in Arches
    FCVariable<double> wLinearSrc;
    new_dw->get(wLinearSrc, "wLinearSource", patch, 0);
    // SU in Arches
    FCVariable<double> wNonlinearSource;
    new_dw->get(wNonlinearSrc, "wNonlinearSource", patch, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
    new_dw->put(wVelocityCoeff, "wVelocityCoeff", patch, 0);
    new_dw->put(wLinearSrc, "wLinearSource", patch);
    new_dw->put(wNonlinearSrc, "wNonlinearSource", patch);

    // it computes the wall bc and updates ucoef and usource
    d_turbModel->calcVelocityWallBC(pc, patch, old_dw, new_dw, Index);
  }
    else {
    cerr << "Invalid Index value" << endl;
  }
#endif
}

void BoundaryCondition::sched_pressureBC(const LevelP& level,
					 const Patch* patch,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
#if 0
  for(Level::const_patchIterator iter=level->patchesBegin();
      iter != level->patchesEnd(); iter++){
    const Patch* patch=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = scinew Task("BoundaryCondition::PressureBC",patch,
			      old_dw, new_dw, this,
			      Discretization::pressureBC);
      tsk->requires(old_dw, "pressure", patch, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureCoeff", patch, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
#endif
}



void BoundaryCondition::pressureBC(const ProcessorContext*,
				   const Patch* patch,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw)
{
#if 0
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 1);
  CCVariable<Vector> pressCoeff; //7 point stencil
  new_dw->get(pressCoeff,"pressureCoeff",patch, 0);
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
  //fortran call
  FORT_PRESSBC(pressCoeff, pressure, 
	       lowIndex, highIndex, cellType);
  new_dw->put(pressCoeff, "pressureCoeff", patch, 0);
#endif
}

void BoundaryCondition::sched_scalarBC(const int index,
				       const LevelP& level,
				       SchedulerP& sched,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{
}

