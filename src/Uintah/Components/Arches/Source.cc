#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
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

Source::Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const)
  :d_turbModel(turb_model), d_physicalConsts(phys_const)
{
}

Source::~Source()
{
}

void Source::calculateVelocitySource(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t,
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

  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);
  //SP term in Arches
  FCVariable<double> uLinearSrc;
  new_dw->allocate(uLinearSrc, "VelLinearSrc", region, index, 0);
  // SU in Arches
  FCVariable<double> uNonlinearSource;
  new_dw->allocate(uNonlinearSrc, "VelNonlinearSource", region, index, 0);
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(region->getLowIndex(), region->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_VELSOURCE(uLinearSrc, uNonlinearSrc, velocity, viscosity, 
		 density, gravity, 
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
  new_dw->put(uLinearSrc, "velLinearSource", region, index, 0);
  new_dw->put(uNonlinearSrc, "velNonlinearSource", region, index, 0);

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  d_turbModel->calcVelocitySource(pc, region, old_dw, new_dw, index);
}




void Source::calculatePressureSource(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", region, 1);
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", region, 1);
  int index = 1;
  FCVariable<Vector> uVelCoeff;
  new_dw->get(uVelCoeff,"uVelocityCoeff",region, index, 0);
  FCVariable<double> uNonlinearSrc;
  new_dw->get(uNonlinearSrc,"uNonlinearSource",region, index, 0);
  ++index;
  FCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff,"vVelocityCoeff",region,index,  0);
  FCVariable<double> vNonlinearSrc;
  new_dw->get(vNonlinearSrc,"vNonlinearSource",region, index, 0);
  ++index;
  FCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff,"wVelocityCoeff",region, index, 0);
  FCVariable<Vector> wNonlinearSrc;
  new_dw->get(wNonlinearSrc,"wNonlinearSource",region, index, 0);
  

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


void Source::calculateScalarSource(const ProcessorContext* pc,
				   const Region* region,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t,
				   const int index) 
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", region, 1);
  CCVariable<Vector> scalar;
  old_dw->get(scalar, "scalar", region, 1);
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

  //SP term in Arches
  CCVariable<double> scalarLinearSrc;
  new_dw->allocate(scalarLinearSrc, "ScalarLinearSrc", region, index, 0);
  // SU in Arches
  CCVariable<double> scalarNonlinearSource;
  new_dw->allocate(scalarNonlinearSrc, "ScalarNonlinearSource", region, index, 0);
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(region->getLowIndex(), region->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_SCALARSOURCE(scalarLinearSrc, scalarNonlinearSrc, scalar, velocity,
		    viscosity, density, 
		    lowIndex, highIndex,
		    cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		    cellinfo->cnn, cellinfo->csn, cellinfo->css,
		    cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		    cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		    cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		    cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		    cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		    cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		    cellinfo->tfac, cellinfo->bfac, volume);
  new_dw->put(scalarLinearSrc, "scalarLinearSource", region, index, 0);
  new_dw->put(scalarNonlinearSrc, "scalarNonlinearSource", region, index, 0);

}

void Source::modifyVelMassSource(const ProcessorContext* pc,
				 const Region* region,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw,
				 double delta_t, const int index){
  // FORT_MASCAL

}
void Source::modifyScalarMassSource(const ProcessorContext* pc,
				    const Region* region,
				    const DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw,
				    double delta_t, const int index){
  //FORT_MASCAL
}

void Source::addPressureSource(const ProcessorContext* pc,
			       const Region* region,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw,
			       const int index){
  //FORT_ADDPRESSSOURCE
}
