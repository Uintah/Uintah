//----- Source.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>

using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Default constructor for Source
//****************************************************************************
Source::Source()
{
  d_densityLabel = scinew VarLabel("density",
				    CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				    CCVariable<double>::getTypeDescription() );
  d_pressureLabel = scinew VarLabel("pressure",
				    CCVariable<double>::getTypeDescription() );
  d_presLinearSrcLabel = scinew VarLabel("pressureLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_presNonlinearSrcLabel = scinew VarLabel("pressureNonlinearSrc",
				    CCVariable<double>::getTypeDescription() );
  // ** WARNING ** velocity is a FCVariable
  d_velocityLabel = scinew VarLabel("velocity",
				    CCVariable<Vector>::getTypeDescription() );
  // ** WARNING ** velLinearSrc is a FCVariable
  d_velLinearSrcLabel = scinew VarLabel("velLinearSrc",
				    CCVariable<double>::getTypeDescription() );
  // ** WARNING ** velNonlinearSrc is a FCVariable
  d_velNonlinearSrcLabel = scinew VarLabel("velNonlinearSrc",
				    CCVariable<double>::getTypeDescription() );
  // ** WARNING ** uVelCoeffLabel is a FCVariable
  d_uVelCoeffLabel = scinew VarLabel("uVelocityCoeff",
				    CCVariable<Vector>::getTypeDescription() );
  // ** WARNING ** uNonlinearSrc is a FCVariable
  d_uNonlinearSrcLabel = scinew VarLabel("uNonlinearSrc",
				    CCVariable<double>::getTypeDescription() );
  // ** WARNING ** vVelCoeffLabel is a FCVariable
  d_vVelCoeffLabel = scinew VarLabel("vVelocityCoeff",
				    CCVariable<Vector>::getTypeDescription() );
  // ** WARNING ** vNonlinearSrc is a FCVariable
  d_vNonlinearSrcLabel = scinew VarLabel("vNonlinearSrc",
				    CCVariable<double>::getTypeDescription() );
  // ** WARNING ** wVelCoeffLabel is a FCVariable
  d_wVelCoeffLabel = scinew VarLabel("wVelocityCoeff",
				    CCVariable<Vector>::getTypeDescription() );
  // ** WARNING ** wNonlinearSrc is a FCVariable
  d_wNonlinearSrcLabel = scinew VarLabel("wNonlinearSrc",
				    CCVariable<double>::getTypeDescription() );
  d_scalarLabel = scinew VarLabel("scalar",
				    CCVariable<Vector>::getTypeDescription() );
  d_scalarLinearSrcLabel = scinew VarLabel("scalarLinearSrc",
				    CCVariable<Vector>::getTypeDescription() );
  d_scalarNonlinearSrcLabel = scinew VarLabel("scalarNonlinearSrc",
				    CCVariable<Vector>::getTypeDescription() );
}

//****************************************************************************
// Another Constructor for Source
//****************************************************************************
Source::Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const)
                           :d_turbModel(turb_model), 
                            d_physicalConsts(phys_const)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
Source::~Source()
{
}

//****************************************************************************
// Velocity source calculation
//****************************************************************************
void 
Source::calculateVelocitySource(const ProcessorContext* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				int index) 
{
  int numGhostCells = 0;
  int matlIndex = 0;

  // ** WARNING ** velocity is a FCVariable
  CCVariable<Vector> velocity;
  old_dw->get(velocity, d_velocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
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

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);

  //SP term in Arches (** WARNING ** uLinearSrc is a FCVariable)
  CCVariable<double> uLinearSrc;
  new_dw->allocate(uLinearSrc, d_velLinearSrcLabel, matlIndex, patch);

  // SU in Arches (** WARNING ** uNonLinearSrc is a FCVariable)
  CCVariable<double> uNonlinearSrc;
  new_dw->allocate(uNonlinearSrc, d_velNonlinearSrcLabel, matlIndex, patch);

#ifdef WONT_COMPILE_YET
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
#endif

  new_dw->put(uLinearSrc, d_velLinearSrcLabel, matlIndex, patch);
  new_dw->put(uNonlinearSrc, d_velNonlinearSrcLabel, matlIndex, patch);
  //new_dw->put(uLinearSrc, "velLinearSource", patch, index, 0);
  //new_dw->put(uNonlinearSrc, "velNonlinearSource", patch, index, 0);

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  d_turbModel->calcVelocitySource(pc, patch, old_dw, new_dw, index);
}

//****************************************************************************
// Pressure source calculation
//****************************************************************************
void 
Source::calculatePressureSource(const ProcessorContext*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t)
{
  int numGhostCells = 0;
  int matlIndex = 0;

  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(pressure, "pressure", patch, 1);

  CCVariable<Vector> velocity;
  old_dw->get(velocity, d_velocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(density, "density", patch, 1);

  int index = 1;
  CCVariable<Vector> uVelCoeff;
  new_dw->get(uVelCoeff, d_uVelCoeffLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> uVelCoeff;
  //new_dw->get(uVelCoeff,"uVelocityCoeff",patch, index, 0);

  CCVariable<double> uNonlinearSrc;
  new_dw->get(uNonlinearSrc, d_uNonlinearSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<double> uNonlinearSrc;
  //new_dw->get(uNonlinearSrc,"uNonlinearSource",patch, index, 0);

  ++index;

  CCVariable<Vector> vVelCoeff;
  new_dw->get(vVelCoeff, d_vVelCoeffLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> vVelCoeff;
  //new_dw->get(vVelCoeff,"vVelocityCoeff",patch,index,  0);

  CCVariable<double> vNonlinearSrc;
  new_dw->get(vNonlinearSrc, d_vNonlinearSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<double> vNonlinearSrc;
  //new_dw->get(vNonlinearSrc,"vNonlinearSource",patch, index, 0);

  ++index;

  CCVariable<Vector> wVelCoeff;
  new_dw->get(wVelCoeff, d_wVelCoeffLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> wVelCoeff;
  //new_dw->get(wVelCoeff,"wVelocityCoeff",patch, index, 0);

  CCVariable<double> wNonlinearSrc;
  new_dw->get(wNonlinearSrc, d_wNonlinearSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> wNonlinearSrc;
  //new_dw->get(wNonlinearSrc,"wNonlinearSource",patch, index, 0);
  
#ifdef WONT_COMPILE_YET
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
#endif

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // Create vars for new_dw
  CCVariable<double> pressLinearSrc;
  new_dw->allocate(pressLinearSrc, d_presLinearSrcLabel, matlIndex, patch);
  //new_dw->allocate(pressLinearSrc,"pressureLinearSource",patch, 0);

  CCVariable<double> pressNonlinearSrc;
  new_dw->allocate(pressNonlinearSrc, d_presNonlinearSrcLabel, matlIndex, patch);
  //new_dw->allocate(pressNonlinearSrc,"pressureNonlinearSource",patch, 0);

#ifdef WONT_COMPILE_YET
  //fortran call
  FORT_PRESSSOURCE(pressLinearSrc, pressNonlinearSrc, pressure, velocity,
		   density, uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   uNonlinearSource, vNonlinearSource, wNonlinearSource,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#endif
		   
  new_dw->put(pressLinearSrc, d_presLinearSrcLabel, matlIndex, patch);
  //new_dw->put(pressLinearSrc, "pressureLinearSource", patch, 0);
  new_dw->put(pressNonlinearSrc, d_presNonlinearSrcLabel, matlIndex, patch);
  //new_dw->put(pressNonlinearSrc, "pressureNonlinearSource", patch, 0);
}

//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateScalarSource(const ProcessorContext*,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      double delta_t,
			      int index) 
{
  int numGhostCells = 0;
  int matlIndex = 0;

  CCVariable<Vector> velocity;
  old_dw->get(velocity, d_velocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  CCVariable<Vector> scalar;
  old_dw->get(scalar, d_scalarLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(scalar, "scalar", patch, 1);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(density, "density", patch, 1);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(viscosity, "viscosity", patch, 1);

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

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  //SP term in Arches
  CCVariable<double> scalarLinearSrc;
  new_dw->allocate(scalarLinearSrc, d_scalarLinearSrcLabel, matlIndex, patch);
  //new_dw->allocate(scalarLinearSrc, "ScalarLinearSrc", patch, index, 0);

  // SU in Arches
  CCVariable<double> scalarNonlinearSrc;
  new_dw->allocate(scalarNonlinearSrc, d_scalarNonlinearSrcLabel, 
		   matlIndex, patch);
  //new_dw->allocate(scalarNonlinearSrc, "ScalarNonlinearSource", patch, index, 0);

#ifdef WONT_COMPILE_YET
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
#endif

  new_dw->put(scalarLinearSrc, d_scalarLinearSrcLabel, matlIndex, patch);
  //new_dw->put(scalarLinearSrc, "scalarLinearSource", patch, index, 0);
  new_dw->put(scalarNonlinearSrc, d_scalarNonlinearSrcLabel, matlIndex, patch);
  //new_dw->put(scalarNonlinearSrc, "scalarNonlinearSource", patch, index, 0);

}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::modifyVelMassSource(const ProcessorContext* ,
			    const Patch* ,
			    DataWarehouseP& ,
			    DataWarehouseP& ,
			    double delta_t, 
			    int index)
{
  // FORT_MASCAL

}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::modifyScalarMassSource(const ProcessorContext* ,
			       const Patch* ,
			       DataWarehouseP& ,
			       DataWarehouseP& ,
			       double delta_t, 
			       int index)
{
  //FORT_MASCAL
}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::addPressureSource(const ProcessorContext* ,
			  const Patch* ,
			  DataWarehouseP& ,
			  DataWarehouseP& ,
			  int index)
{
  //FORT_ADDPRESSSOURCE
}

//
//$ Log: $
//
