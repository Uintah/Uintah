
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Stencil.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerRegion.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <iostream>
using std::cerr;

using Uintah::Components::Discretization;
using namespace Uintah::Components;
using namespace Uintah::Grid;
using SCICore::Geometry::Vector;

Discretization::Discretization()
{
}

Discretization::~Discretization()
{
}

void Discretization::sched_calculateVelocityCoeff(const int index,
						  const LevelP& level,
						  SchedulerP& sched,
						  const DataWarehouseP& old_dw,
						  DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("Discretization::VelocityCoeff",
			   region, old_dw, new_dw, this,
			   Discretization::calculateVelocityCoeff,
			   index);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      /// requires convection coeff because of the nodal
      // differencing
      if (index == 1){ 
	tsk->computes(new_dw, "uVelocityConvectCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uVelocityCoeff", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      } else if (index == 2) {
	tsk->computes(new_dw, "vVelocityConvectCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      } else if (index == 3) {
        tsk->computes(new_dw, "wVelocityConvectCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "wVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      } else {
	throw InvalidValue("Invalid component for velocity" +index);
      }
      sched->addTask(tsk);
    }

  }
}

void Discretization::calculateVelocityCoeff(const ProcessorContext* pc,
					    const Region* region,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw,
					    const int Index)
{
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
  Vector gravity; // 3 components of gravity
  top_dw->get(SoleVariable<Vector>(gravity), "gravity"); 

  if (Index == 1) {
    //7pt stencil declaration
    Stencil uVelocityCoeff(new_dw, "uVelocityCoeff", region);
    // convection coeffs
    Stencil uVelocityConvectCoeff(new_dw, "uVelocityConvectCoeff", region);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    Array3<double> volume(region); // 3-d array for volume
    FORT_VELCOEF(velocity, viscosity, density, gravity.x(), 
		 uVelocityConvectCoeff, uVelocityCoeff,
		 ioff, joff, koff, lowIndex, highIndex,
		 cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		 cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		 cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		 cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		 cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		 cellinfo->tfac, cellinfo->bfac, volume);
    new_dw->put(uVelocityCoeff, "uVelocityCoeff", region);
    new_dw->put(uVelocityConvectCoeff, "uVelocityConvectCoeff", region);
  }
  else if (Index == 2) {
    //7pt stencil declaration
    Stencil vVelocityCoeff(new_dw, "vVelocityCoeff", region);
    // convection coeffs
    Stencil vVelocityConvectCoeff(new_dw, "vVelocityConvectCoeff", region);
    int ioff = 0;
    int joff = 1;
    int koff = 0;
    Array3<double> volume(region); // 3-d array for volume
    // pass velocity as v,w, u
    FORT_VELCOEF(velocity, viscosity, density, gravity.y(), 
		 vVelocityConvectCoeff, vVelocityCoeff,
		 ioff, joff, koff, lowIndex, highIndex,
		 cellinfo->cnnv, cellinfo->csnv, cellinfo->cssv,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		 cellinfo->snsv, cellinfo->stb, cellinfo->sew,
		 cellinfo->dynpv, cellinfo->dztp, cellinfo->dxep,
		 cellinfo->dyps, cellinfo->fac1v, cellinfo->fac2v,
		 cellinfo->fac3v, cellinfo->fac4v,cellinfo->jnsdv,
		 cellinfo->jssdv, cellinfo->tfac, cellinfo->bfac,
		 cellinfo->efac, cellinfo->wfac, volume);
    new_dw->put(vVelocityCoeff, "vVelocityCoeff", region);
    new_dw->put(vVelocityConvectCoeff, "vVelocityConvectCoeff", region);
  }
  else if (Index == 3) {
    //7pt stencil declaration
    Stencil wVelocityCoeff(new_dw, "wVelocityCoeff", region);
    // convection coeffs
    Stencil wVelocityConvectCoeff(new_dw, "wVelocityConvectCoeff", region);
    int ioff = 0;
    int joff = 0;
    int koff = 1;
    Array3<double> volume(region); // 3-d array for volume
    FORT_VELCOEF(velocity, viscosity, density, gravity.z(), 
		 wVelocityConvectCoeff, wVelocityCoeff,
		 ioff, joff, koff, lowIndex, highIndex,
		 cellinfo->cttw, cellinfo->cbtw, cellinfo->cbbw,
		 cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->stbw, cellinfo->sew, cellinfo->sns,
		 cellinfo->dztpw, cellinfo->dxep, cellinfo->dynp,
		 cellinfo->dzpb, cellinfo->fac1w, cellinfo->fac2w,
		 cellinfo->fac3w, cellinfo->fac4w,cellinfo->ktsdw,
		 cellinfo->kbsdw, cellinfo->efac, cellinfo->wfac,
		 cellinfo->enfac, cellinfo->sfac, volume);
    new_dw->put(wVelocityCoeff, "wVelocityCoeff", region);
    new_dw->put(wVelocityConvectCoeff, "wVelocityConvectCoeff", region);
  }
  else {
    cerr << "Invalid Index value" << endl;
  }
}



void Discretization::sched_calculatePressureCoeff(const LevelP& level,
						  SchedulerP& sched,
						  const DataWarehouseP& old_dw,
						  DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      //copies old db to new_db and then uses non-linear
      //solver to compute new values
      Task* tsk = new Task("Discretization::PressureCoeff",region,
			   old_dw, new_dw, this,
			   Discretization::calculatePressureCoeff);
      tsk->requires(old_dw, "pressure", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "uVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "vVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(new_dw, "wVelocityCoeff", region, 0,
		    FCVariable<Vector>::getTypeDescription());
      tsk->computes(new_dw, "pressureCoeff", region, 0,
		    CCVariable<Vector>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}

void Discretization::calculatePressureCoeff(const ProcessorContext*,
					    const Region* region,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 1);
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  FCVariable<Vector> uVelCoeff;
  new_dw->get(uVelCoeff,"uVelocityCoeff",region, 0);
  FCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff,"vVelocityCoeff",region, 0);
  FCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff,"wVelocityCoeff",region, 0);

  // Create vars for new_dw
  CCVariable<Vector> pressCoeff; //7 point stencil
  new_dw->allocate(pressCoeff,"pressureCoeff",region, 0);
  // get high and low from region
  //
  FORT_PRESSCOEF(lowIndex, highIndex,geom.sew, geom.sns, geom.stb,
		 geom.dxep,geom.dxpw, geom.dynp, geom.dyps, geom.dztp,
		 geom.dzpb, pressure, velocity, density, uVelCoeff,
		 vVelCoeff, wVelCoeff, pressCoeff);
  new_dw->put(pressCoeff, "pressureCoeff", region, 0);
}
  
void  Discretization::calculateResidual(const ProcessorContext*,
				       SchedulerP& sched,
				       const DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw)
{

  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_computeResidual(level, sched,
					old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->get(d_residual, "PressResidual"); 


}

void  Discretization::calculateOrderMagnitude(const LevelP& level,
					      SchedulerP& sched,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{

  //pass a string to identify the eqn for which residual 
  // needs to be solved, in this case pressure string
  d_linearSolver->sched_calculateOrderMagnitude(level, sched,
						old_dw, new_dw);
  // reduces from all processors to compute L1 norm
  new_dw->get(d_ordermagnitude, "PressOMG"); 


}


