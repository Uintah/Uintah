#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
using namespace Uintah::Arches;
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

  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);
  //SP term in Arches
  FCVariable<double> uLinearSrc;
  new_dw->allocate(uLinearSrc, "VelLinearSrc", patch, index, 0);
  // SU in Arches
  FCVariable<double> uNonlinearSource;
  new_dw->allocate(uNonlinearSrc, "VelNonlinearSource", patch, index, 0);
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
  new_dw->put(uLinearSrc, "velLinearSource", patch, index, 0);
  new_dw->put(uNonlinearSrc, "velNonlinearSource", patch, index, 0);

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  d_turbModel->calcVelocitySource(pc, patch, old_dw, new_dw, index);
}




void Source::calculatePressureSource(const ProcessorContext*,
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
  int index = 1;
  FCVariable<Vector> uVelCoeff;
  new_dw->get(uVelCoeff,"uVelocityCoeff",patch, index, 0);
  FCVariable<double> uNonlinearSrc;
  new_dw->get(uNonlinearSrc,"uNonlinearSource",patch, index, 0);
  ++index;
  FCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff,"vVelocityCoeff",patch,index,  0);
  FCVariable<double> vNonlinearSrc;
  new_dw->get(vNonlinearSrc,"vNonlinearSource",patch, index, 0);
  ++index;
  FCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff,"wVelocityCoeff",patch, index, 0);
  FCVariable<Vector> wNonlinearSrc;
  new_dw->get(wNonlinearSrc,"wNonlinearSource",patch, index, 0);
  

  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  // move cell information to global space of Arches
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
  CCVariable<double> pressLinearSrc;
  new_dw->allocate(pressLinearSrc,"pressureLinearSource",patch, 0);
  CCVariable<double> pressNonlinearSrc;
  new_dw->allocate(pressNonlinearSrc,"pressureNonlinearSource",patch, 0);
  //fortran call
  FORT_PRESSSOURCE(pressLinearSrc, pressNonlinearSrc, pressure, velocity,
		   density, uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   uNonlinearSource, vNonlinearSource, wNonlinearSource,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
		   
  new_dw->put(pressLinearSrc, "pressureLinearSource", patch, 0);
  new_dw->put(pressNonlinearSrc, "pressureNonlinearSource", patch, 0);
}


void Source::calculateScalarSource(const ProcessorContext* pc,
				   const Patch* patch,
				   const DataWarehouseP& old_dw,
				   DataWarehouseP& new_dw,
				   double delta_t,
				   const int index) 
{
  FCVariable<Vector> velocity;
  old_dw->get(velocity, "velocity", patch, 1);
  CCVariable<Vector> scalar;
  old_dw->get(scalar, "scalar", patch, 1);
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

  //SP term in Arches
  CCVariable<double> scalarLinearSrc;
  new_dw->allocate(scalarLinearSrc, "ScalarLinearSrc", patch, index, 0);
  // SU in Arches
  CCVariable<double> scalarNonlinearSource;
  new_dw->allocate(scalarNonlinearSrc, "ScalarNonlinearSource", patch, index, 0);
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
  new_dw->put(scalarLinearSrc, "scalarLinearSource", patch, index, 0);
  new_dw->put(scalarNonlinearSrc, "scalarNonlinearSource", patch, index, 0);

}

void Source::modifyVelMassSource(const ProcessorContext* pc,
				 const Patch* patch,
				 const DataWarehouseP& old_dw,
				 DataWarehouseP& new_dw,
				 double delta_t, const int index){
  // FORT_MASCAL

}
void Source::modifyScalarMassSource(const ProcessorContext* pc,
				    const Patch* patch,
				    const DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw,
				    double delta_t, const int index){
  //FORT_MASCAL
}

void Source::addPressureSource(const ProcessorContext* pc,
			       const Patch* patch,
			       const DataWarehouseP& old_dw,
			       DataWarehouseP& new_dw,
			       const int index){
  //FORT_ADDPRESSSOURCE
}
