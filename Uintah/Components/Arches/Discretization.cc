
#include <Uintah/Components/Arches/Discretization.h>
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
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;

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


void Discretization::calculateVelocityCoeff(const ProcessorContext* pc,
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

  //7pt stencil declaration
  Stencil<double> uVelocityCoeff(region);
  // convection coeffs
  Stencil<double> uVelocityConvectCoeff(region);
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(region->getLowIndex(), region->getHighIndex());
  FORT_VELCOEF(velocity, viscosity, density,
	       uVelocityConvectCoeff, uVelocityCoeff, delta_t,
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
  new_dw->put(uVelocityCoeff, "VelocityCoeff", region, index);
  new_dw->put(uVelocityConvectCoeff, "VelocityConvectCoeff", region, index);
}




void Discretization::calculatePressureCoeff(const ProcessorContext*,
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
  // need to be consistent, use Stencil
  FCVariable<Vector> uVelCoeff;
  int index = 1;
  new_dw->get(uVelCoeff,"VelocityCoeff",region, 0, index);
  index++;
  FCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff,"VelocityCoeff",region, 0, index);
  index++;
  FCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff,"VelocityCoeff",region, 0, index);

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

 // Create vars for new_dw
  CCVariable<Vector> pressCoeff; //7 point stencil
  new_dw->allocate(pressCoeff,"pressureCoeff",region, 0);

  FORT_PRESSSOURCE(pressCoeff, pressure, velocity, density
		   uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
  new_dw->put(pressCoeff, "pressureCoeff", region, 0);
}
  
void Discretization::calculateScalarCoeff(const ProcessorContext* pc,
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
  // ithe componenet of scalar vector
  CCVariable<double> scalar;
  old_dw->get(scalar, "scalar", region, 1, index);
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

  //7pt stencil declaration
  Stencil<double> scalarCoeff(region);
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(region->getLowIndex(), region->getHighIndex());
  FORT_SCALARCOEF(scalarCoeff, scalar, velocity, viscosity, density,
		  delta_t, lowIndex, highIndex,
		  cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		  cellinfo->cnn, cellinfo->csn, cellinfo->css,
		  cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		  cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		  cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		  cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		  cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		  cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		  cellinfo->tfac, cellinfo->bfac, volume);
  new_dw->put(scalarCoeff, "ScalarCoeff", region, index);
}

void Discretization::calculateVelDiagonal(const ProcessorContext*,
					  const Region* region,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  const int index){
  
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  Stencil<double> uVelCoeff;
  new_dw->get(uVelCoeff, "VelocityCoeff", region, index, 0);
  FCVariable<double> uVelLinearSrc;
  new_dw->get(uVelLinearSrc, "VelLinearSrc", region, index, 0);
  FORT_APCAL(uVelCoeffvelocity, uVelLinearSrc, lowIndex, highIndex);
  new_dw->put(uVelCoeff, "VelocityCoeff", region, index, 0);
}

void Discretization::calculatePressDiagonal(const ProcessorContext*,
					    const Region* region,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw) {
  
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  Stencil<double> pressCoeff;
  new_dw->get(pressCoeff, "PressureCoCoeff", region, 0);
  FCVariable<double> pressLinearSrc;
  new_dw->get(pressLinearSrc, "pressureLinearSource", region, 0);
  FORT_APCAL(pressCoeff, pressLinearSrc, lowIndex, highIndex);
  new_dw->put(pressCoeff, "pressureLinearSource", region, 0);
}

void Discretization::calculateScalarDiagonal(const ProcessorContext*,
					  const Region* region,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  const int index){
  
  Array3Index lowIndex = region->getLowIndex();
  Array3Index highIndex = region->getHighIndex();

  Stencil<double> scalarCoeff;
  new_dw->get(scalarCoeff, "ScalarCoeff", region, index, 0);
  FCVariable<double> scalarLinearSrc;
  new_dw->get(scalarLinearSrc, "ScalarLinearSource", region, index, 0);
  FORT_APCAL(scalarCoeff, scalarLinearSrc, lowIndex, highIndex);
  new_dw->put(scalarCoeff, "ScalarCoeff", region, index, 0);
}



