
#include <Uintah/Components/Arches/Discretization.h>
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
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;

using namespace Uintah::Arches;
using SCICore::Geometry::Vector;

Discretization::Discretization()
{
}

Discretization::~Discretization()
{
}


void Discretization::calculateVelocityCoeff(const ProcessorContext* pc,
					    const Patch* patch,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw,
					    double delta_t,
					    const int index)
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 1);
  CCVariable<double> viscosity;
  old_dw->get(viscosity, "viscosity", patch, 1);
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

  //7pt stencil declaration
  Stencil<double> uVelocityCoeff(patch);
  // convection coeffs
  Stencil<double> uVelocityConvectCoeff(patch);
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
  new_dw->put(uVelocityCoeff, "VelocityCoeff", patch, index);
  new_dw->put(uVelocityConvectCoeff, "VelocityConvectCoeff", patch, index);
}




void Discretization::calculatePressureCoeff(const ProcessorContext*,
					    const Patch* patch,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw,
					    double delta_t)
{
  CCVariable<double> pressure;
  old_dw->get(pressure, "pressure", patch, 1);
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 1);
  // need to be consistent, use Stencil
  FCVariable<Vector> uVelCoeff;
  int index = 1;
  new_dw->get(uVelCoeff,"VelocityCoeff",patch, 0, index);
  index++;
  FCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff,"VelocityCoeff",patch, 0, index);
  index++;
  FCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff,"VelocityCoeff",patch, 0, index);

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

 // Create vars for new_dw
  CCVariable<Vector> pressCoeff; //7 point stencil
  new_dw->allocate(pressCoeff,"pressureCoeff",patch, 0);

  FORT_PRESSSOURCE(pressCoeff, pressure, velocity, density
		   uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
  new_dw->put(pressCoeff, "pressureCoeff", patch, 0);
}
  
void Discretization::calculateScalarCoeff(const ProcessorContext* pc,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  double delta_t,
					  const int index)
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<double> density;
  old_dw->get(density, "density", patch, 1);
  CCVariable<double> viscosity;
  old_dw->get(viscosity, "viscosity", patch, 1);
  // ithe componenet of scalar vector
  CCVariable<double> scalar;
  old_dw->get(scalar, "scalar", patch, 1, index);
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

  //7pt stencil declaration
  Stencil<double> scalarCoeff(patch);
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
  new_dw->put(scalarCoeff, "ScalarCoeff", patch, index);
}

void Discretization::calculateVelDiagonal(const ProcessorContext*,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  const int index){
  
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();

  Stencil<double> uVelCoeff;
  new_dw->get(uVelCoeff, "VelocityCoeff", patch, index, 0);
  FCVariable<double> uVelLinearSrc;
  new_dw->get(uVelLinearSrc, "VelLinearSrc", patch, index, 0);
  FORT_APCAL(uVelCoeffvelocity, uVelLinearSrc, lowIndex, highIndex);
  new_dw->put(uVelCoeff, "VelocityCoeff", patch, index, 0);
}

void Discretization::calculatePressDiagonal(const ProcessorContext*,
					    const Patch* patch,
					    const DataWarehouseP& old_dw,
					    DataWarehouseP& new_dw) {
  
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();

  Stencil<double> pressCoeff;
  new_dw->get(pressCoeff, "PressureCoCoeff", patch, 0);
  FCVariable<double> pressLinearSrc;
  new_dw->get(pressLinearSrc, "pressureLinearSource", patch, 0);
  FORT_APCAL(pressCoeff, pressLinearSrc, lowIndex, highIndex);
  new_dw->put(pressCoeff, "pressureLinearSource", patch, 0);
}

void Discretization::calculateScalarDiagonal(const ProcessorContext*,
					  const Patch* patch,
					  const DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw,
					  const int index){
  
  Array3Index lowIndex = patch->getLowIndex();
  Array3Index highIndex = patch->getHighIndex();

  Stencil<double> scalarCoeff;
  new_dw->get(scalarCoeff, "ScalarCoeff", patch, index, 0);
  FCVariable<double> scalarLinearSrc;
  new_dw->get(scalarLinearSrc, "ScalarLinearSource", patch, index, 0);
  FORT_APCAL(scalarCoeff, scalarLinearSrc, lowIndex, highIndex);
  new_dw->put(scalarCoeff, "ScalarCoeff", patch, index, 0);
}



