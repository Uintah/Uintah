//----- Source.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/PhysicalConstants.h>
#include <Uintah/Components/Arches/CellInformation.h>
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
  // ** WARNING ** velocity is a FCVariable
  // Change all velocity related variables and then remove this comment

  // inputs (calcVelocitySource)
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
				    CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    CCVariable<double>::getTypeDescription() );

  // outputs (calcVelocitySource)
  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    CCVariable<double>::getTypeDescription() );

  // inputs/outputs for calculatePressureSource()
  d_uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
				  CCVariable<double>::getTypeDescription() );
  d_vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
				  CCVariable<double>::getTypeDescription() );
  d_wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
				  CCVariable<double>::getTypeDescription() );
  d_pressureINLabel = scinew VarLabel("pressureIN",
				  CCVariable<double>::getTypeDescription() );
  d_presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM",
				  CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM",
				  CCVariable<double>::getTypeDescription() );

  // inputs/outputs for calculateScalarSource()
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				  CCVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				  CCVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				  CCVariable<double>::getTypeDescription() );
  d_scalarSPLabel = scinew VarLabel("scalarSP",
				  CCVariable<double>::getTypeDescription() );
  d_scalLinSrcSBLMLabel = scinew VarLabel("scalLinSrcSBLM",
				CCVariable<double>::getTypeDescription() );
  d_scalNonLinSrcSBLMLabel = scinew VarLabel("scalNonLinSrcSBLM",
				CCVariable<double>::getTypeDescription() );
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
Source::calculateVelocitySource(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t,
				int index,
				int eqnType) 
{
  int numGhostCells = 0;
  int matlIndex = 0;

  // ** WARNING ** velocity is a FCVariable
  // Change all velocity related variables and then remove this comment
  CCVariable<double> uVelocity;
  CCVariable<double> vVelocity;
  CCVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  CCVariable<double> uVelLinearSrc; //SP term in Arches 
  CCVariable<double> uVelNonlinearSrc; // SU in Arches 
  CCVariable<double> vVelLinearSrc; //SP term in Arches 
  CCVariable<double> vVelNonlinearSrc; // SU in Arches 
  CCVariable<double> wVelLinearSrc; //SP term in Arches 
  CCVariable<double> wVelNonlinearSrc; // SU in Arches 

  // get data
  switch(eqnType) {
  case Discretization::PRESSURE:
    old_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case Discretization::MOMENTUM:
    old_dw->get(uVelocity, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(vVelocity, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(wVelocity, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type can be only PRESSURE or MOMENTUM");
  }
  old_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
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

  // allocate
  switch(eqnType) {
  case Discretization::PRESSURE:
    new_dw->allocate(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->allocate(uVelNonlinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, 
		     patch);
    new_dw->allocate(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->allocate(vVelNonlinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, 
		     patch);
    new_dw->allocate(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->allocate(wVelNonlinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, 
		     patch);
    break;
  case Discretization::MOMENTUM:
    new_dw->allocate(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(uVelNonlinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, 
		     patch);
    new_dw->allocate(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(vVelNonlinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, 
		     patch);
    new_dw->allocate(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->allocate(wVelNonlinearSrc, d_wVelNonLinSrcMBLMLabel, matlIndex, 
		     patch);
    break;
  default:
    break;
  }

#ifdef WONT_COMPILE_YET
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_VELSOURCE(velLinearSrc, velNonlinearSrc, velocity, viscosity, 
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

  switch(eqnType) {
  case Discretization::PRESSURE:
    new_dw->put(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(uVelNonlinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(vVelNonlinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch);
    new_dw->put(wVelNonlinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, patch);
    break;
  case Discretization::MOMENTUM:
    new_dw->put(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->put(uVelNonlinearSrc, d_uVelNonLinSrcMBLMLabel, matlIndex, patch);
    new_dw->put(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->put(vVelNonlinearSrc, d_vVelNonLinSrcMBLMLabel, matlIndex, patch);
    new_dw->put(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch);
    new_dw->put(wVelNonlinearSrc, d_wVelNonLinSrcMBLMLabel, matlIndex, patch);
    break;
  default:
    break;
  }
  //new_dw->put(uLinearSrc, "velLinearSource", patch, index, 0);
  //new_dw->put(uNonlinearSrc, "velNonlinearSource", patch, index, 0);

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  // inputs : 
  // outputs : 
  d_turbModel->calcVelocitySource(pc, patch, old_dw, new_dw, index);
}

//****************************************************************************
// Pressure source calculation
//****************************************************************************
void 
Source::calculatePressureSource(const ProcessorGroup*,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw,
				double delta_t)
{
  int numGhostCells = 0;
  int matlIndex = 0;
  int nofStencils = 7;

  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureINLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(pressure, "pressure", patch, 1);

  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  old_dw->get(uVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  old_dw->get(uVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  CCVariable<double> density;
  old_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(density, "density", patch, 1);

  StencilMatrix<CCVariable<double> > uVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(uVelCoeff[ii], d_uVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  //int index = 1;
  //FCVariable<Vector> uVelCoeff;
  //new_dw->get(uVelCoeff,"uVelocityCoeff",patch, index, 0);

  CCVariable<double> uNonlinearSrc;
  new_dw->get(uNonlinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  //FCVariable<double> uNonlinearSrc;
  //new_dw->get(uNonlinearSrc,"uNonlinearSource",patch, index, 0);
  //++index;

  StencilMatrix<CCVariable<double> > vVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(vVelCoeff[ii], d_vVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  //FCVariable<Vector> vVelCoeff;
  //new_dw->get(vVelCoeff,"vVelocityCoeff",patch,index,  0);

  CCVariable<double> vNonlinearSrc;
  new_dw->get(vNonlinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  //FCVariable<double> vNonlinearSrc;
  //new_dw->get(vNonlinearSrc,"vNonlinearSource",patch, index, 0);
  //++index;

  StencilMatrix<CCVariable<double> > wVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(wVelCoeff[ii], d_wVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  //FCVariable<Vector> wVelCoeff;
  //new_dw->get(wVelCoeff,"wVelocityCoeff",patch, index, 0);

  CCVariable<double> wNonlinearSrc;
  new_dw->get(wNonlinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
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
  new_dw->allocate(pressLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch);
  //new_dw->allocate(pressLinearSrc,"pressureLinearSource",patch, 0);

  CCVariable<double> pressNonlinearSrc;
  new_dw->allocate(pressNonlinearSrc, d_presNonLinSrcPBLMLabel, matlIndex, 
		   patch);
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
		   
  new_dw->put(pressLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch);
  //new_dw->put(pressLinearSrc, "pressureLinearSource", patch, 0);
  new_dw->put(pressNonlinearSrc, d_presNonLinSrcPBLMLabel, matlIndex, patch);
  //new_dw->put(pressNonlinearSrc, "pressureNonlinearSource", patch, 0);
}

//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateScalarSource(const ProcessorGroup*,
			      const Patch* patch,
			      DataWarehouseP& old_dw,
			      DataWarehouseP& new_dw,
			      double delta_t,
			      int index) 
{
  int numGhostCells = 0;
  int matlIndex = 0;

  CCVariable<double> uVelocity;
  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<Vector> velocity;
  //old_dw->get(velocity, "velocity", patch, 1);

  // ** WARNING ** The scalar is got based on the input index
  CCVariable<double> scalar;
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(scalar, "scalar", patch, 1);

  CCVariable<double> density;
  new_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //old_dw->get(density, "density", patch, 1);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
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
  new_dw->allocate(scalarLinearSrc, d_scalLinSrcSBLMLabel, index, patch);
  //new_dw->allocate(scalarLinearSrc, "ScalarLinearSrc", patch, index, 0);

  // SU in Arches
  CCVariable<double> scalarNonlinearSrc;
  new_dw->allocate(scalarNonlinearSrc, d_scalNonLinSrcSBLMLabel, 
		   index, patch);
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

  new_dw->put(scalarLinearSrc, d_scalLinSrcSBLMLabel, index, patch);
  //new_dw->put(scalarLinearSrc, "scalarLinearSource", patch, index, 0);

  new_dw->put(scalarNonlinearSrc, d_scalNonLinSrcSBLMLabel, index, patch);
  //new_dw->put(scalarNonlinearSrc, "scalarNonlinearSource", patch, index, 0);

}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::modifyVelMassSource(const ProcessorGroup* ,
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
Source::modifyScalarMassSource(const ProcessorGroup* ,
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
Source::addPressureSource(const ProcessorGroup* ,
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
