#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerRegion.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
using Uintah::Components::Discretization;
using namespace Uintah::Components;
using namespace Uintah::Grid;
using SCICore::Geometry::Vector;

Source::Source()
{
}

Source::Source(TurbulenceModel* turb_model)
  :d_turbModel(turb_model)
{
}

Source::~Source()
{
}

void Source::sched_calculateVelocitySource(const int index,
					  const LevelP& level,
					  SchedulerP& sched,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw)
{
  for(Level::const_regionIterator iter=level->regionsBegin();
      iter != level->regionsEnd(); iter++){
    const Region* region=*iter;
    {
      Task* tsk = new Task("Source::VelocitySource",
			   region, old_dw, new_dw, this,
			   Source::calculateVelocitySource,
			   index);
      tsk->requires(old_dw, "velocity", region, 1,
		    FCVariable<Vector>::getTypeDescription());
      tsk->requires(old_dw, "density", region, 1,
		    CCVariable<double>::getTypeDescription());
      tsk->requires(old_dw, "viscosity", region, 1,
		    CCVariable<double>::getTypeDescription());
      if (index == 1) {
	tsk->computes(new_dw, "uLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "uNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 2) {
	tsk->computes(new_dw, "vLinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
	tsk->computes(new_dw, "vNonlinearSource", region, 0,
		      FCVariable<Vector>::getTypeDescription());
      }
      else if (index == 3) {
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

void Source::calculateVelocitySource(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     const int index) 
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
  SoleVariable<Vector> gravity_var;
  top_dw->get(gravity_var, "gravity"); 
  Vector gravity = gravity_var;

  if (Index == 1) {
    //SP term in Arches
    FCVariable<double> uLinearSrc;
    new_dw->allocate(uLinearSrc, "uLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> uNonlinearSource;
    new_dw->allocate(uNonlinearSrc, "uNonlinearSource", region, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_VELSOURCE(uLinearSrc, uNonlinearSrc, velocity, viscosity, 
		   density, gravity.x(), 
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
    new_dw->put(uLinearSrc, "uLinearSource", region);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", region);

    // pass the pointer to turbulence model object and make 
    // it a data memeber of Source class
    // it computes the source in momentum eqn due to the turbulence
    // model used.
    d_turbModel->calcVelocitySource(pc, region, old_dw, new_dw, Index);
  }

  else if (Index == 2) {
    //SP term in Arches
    FCVariable<double> vLinearSrc;
    new_dw->allocate(vLinearSrc, "vLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> vNonlinearSource;
    new_dw->allocate(vNonlinearSrc, "vNonlinearSource", region, 0);
    int ioff = 0;
    int joff = 1;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_VELSOURCE(vLinearSrc, vNonlinearSrc, velocity, viscosity, 
		   density, gravity.y(), 
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
    new_dw->put(vLinearSrc, "vLinearSource", region);
    new_dw->put(vNonlinearSrc, "vNonlinearSource", region);

    // pass the pointer to turbulence model object and make 
    // it a data memeber of Source class
    // it computes the source in momentum eqn due to the turbulence
    // model used.
    d_turbModel->calcVelocitySource(pc, region, old_dw, new_dw, Index);
  }
  else if (Index == 3) {
    //SP term in Arches
    FCVariable<double> wLinearSrc;
    new_dw->allocate(wLinearSrc, "wLinearSource", region, 0);
    // SU in Arches
    FCVariable<double> wNonlinearSource;
    new_dw->allocate(wNonlinearSrc, "wNonlinearSource", region, 0);
    int ioff = 1;
    int joff = 0;
    int koff = 0;
    // 3-d array for volume - fortran uses it for temporary storage
    Array3<double> volume(region->getLowIndex(), region->getHighIndex());
    // computes remianing diffusion term and also computes source due to gravity
    FORT_VELSOURCE(wLinearSrc, wNonlinearSrc, velocity, viscosity, 
		   density, gravity.z(), 
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
    new_dw->put(uLinearSrc, "uLinearSource", region);
    new_dw->put(uNonlinearSrc, "uNonlinearSource", region);

    // pass the pointer to turbulence model object and make 
    // it a data memeber of Source class
    // it computes the source in momentum eqn due to the turbulence
    // model used.
    d_turbModel->calcVelocitySource();
    new_dw->put(uVelocityCoeff, "uVelocityCoeff", region);
    new_dw->put(uVelocityConvectCoeff, "uVelocityConvectCoeff", region);
  }
    else {
    cerr << "Invalid Index value" << endl;
  }
}

void Source::sched_calculatePressureSource(const LevelP& level,
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
      Task* tsk = new Task("Source::PressureSource",region,
			   old_dw, new_dw, this,
			   Discretization::calculatePressureSource);
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
      tsk->requires(new_dw, "uNonlinearSource", region, 0,
		    FCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "vNonlinearSource", region, 0,
		    FCVariable<double>::getTypeDescription());
      tsk->requires(new_dw, "wNonlinearSource", region, 0,
		    FCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "pressureLinearSource", region, 0,
		    CCVariable<double>::getTypeDescription());
      tsk->computes(new_dw, "pressureNonlinearSource", region, 0,
		    CCVariable<double>::getTypeDescription());
      sched->addTask(tsk);
    }

  }
}



void Source::calculatePressureSource(const ProcessorContext*,
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
  FCVariable<double> uNonlinearSrc;
  new_dw->get(uNonlinearSrc,"uNonlinearSource",region, 0);
  FCVariable<double> vNonlinearSrc;
  new_dw->get(vNonlinearSrc,"vNonlinearSource",region, 0);
  FCVariable<Vector> wNonlinearSrc;
  new_dw->get(wNonlinearSrc,"wNonlinearSource",region, 0);

  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  // move cell information to global space of Arches
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

  // Create vars for new_dw
  CCVariable<double> pressLinearSrc;
  new_dw->allocate(pressLinearSrc,"pressureLinearSource",region, 0);
  CCVariable<double> pressNonlinearSrc;
  new_dw->allocate(pressNonlinearSrc,"pressureNonlinearSource",region, 0);
  //fortran call
  FORT_PRESSSOURCE(pressLinearSrc, pressNonlinearSrc, pressure, velocity,
		   density, uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   uNonlinearSource, vNonlinearSource, wNonlinearSource,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
		   
  new_dw->put(pressLinearSrc, "pressureLinearSource", region, 0);
  new_dw->put(pressNonlinearSrc, "pressureNonlinearSource", region, 0);
}

void Source::sched_calculateScalarSource(const int index,
					 const LevelP& level,
					 SchedulerP& sched,
					 const DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw)
{
}

