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
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
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
  // inputs (calcVelocitySource)
  d_cellInfoLabel = scinew VarLabel("cellInformation",
			    PerPatch<CellInformation*>::getTypeDescription());
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				    SFCZVariable<double>::getTypeDescription() );
  d_densityCPLabel = scinew VarLabel("densityCP",
				    CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				    CCVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    SFCZVariable<double>::getTypeDescription() );

  // outputs (calcVelocitySource)
  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcPBLMLabel = scinew VarLabel("uVelNonLinSrcPBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcPBLMLabel = scinew VarLabel("vVelNonLinSrcPBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcPBLMLabel = scinew VarLabel("wVelNonLinSrcPBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_uVelNonLinSrcMBLMLabel = scinew VarLabel("uVelNonLinSrcMBLM",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_vVelNonLinSrcMBLMLabel = scinew VarLabel("vVelNonLinSrcMBLM",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );
  d_wVelNonLinSrcMBLMLabel = scinew VarLabel("wVelNonLinSrcMBLM",
				    SFCZVariable<double>::getTypeDescription() );

  // inputs/outputs for calculatePressureSource()
  d_uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
				  SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
				  SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
				  SFCZVariable<double>::getTypeDescription() );
  d_pressureSPBCLabel = scinew VarLabel("pressureSPBC",
				  CCVariable<double>::getTypeDescription() );
  d_presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM",
				  CCVariable<double>::getTypeDescription() );
  d_presNonLinSrcPBLMLabel = scinew VarLabel("presNonLinSrcPBLM",
				  CCVariable<double>::getTypeDescription() );

  // inputs/outputs for calculateScalarSource()
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				  SFCXVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				  SFCYVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				  SFCZVariable<double>::getTypeDescription() );
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

  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  SFCXVariable<double> uVelLinearSrc; //SP term in Arches 
  SFCXVariable<double> uVelNonlinearSrc; // SU in Arches 
  SFCYVariable<double> vVelLinearSrc; //SP term in Arches 
  SFCYVariable<double> vVelNonlinearSrc; // SU in Arches 
  SFCZVariable<double> wVelLinearSrc; //SP term in Arches 
  SFCZVariable<double> wVelNonlinearSrc; // SU in Arches 

  // get data
  switch(eqnType) {
  case Discretization::PRESSURE:
    new_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case Discretization::MOMENTUM:
    new_dw->get(uVelocity, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type can be only PRESSURE or MOMENTUM");
  }
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  
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

  // Get the patch and variable indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = uVelocity.getFortLowIndex();
  IntVector domHiV = uVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = uVelocity.getFortLowIndex();
  IntVector domHiW = uVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_VELSOURCE(domLoU.get_pointer(), domHiU.get_pointer(),
		 idxLoU.get_pointer(), idxHiU.get_pointer(),
		 uVelLinearSrc.getPointer(), 
		 uVelNonlinearSrc.getPointer(), 
		 uVelocity.getPointer(), 
		 domLoV.get_pointer(), domHiV.get_pointer(),
		 idxLoV.get_pointer(), idxHiV.get_pointer(),
		 vVelLinearSrc.getPointer(), 
		 vVelNonlinearSrc.getPointer(), 
		 vVelocity.getPointer(), 
		 domLoW.get_pointer(), domHiW.get_pointer(),
		 idxLoW.get_pointer(), idxHiW.get_pointer(),
		 wVelLinearSrc.getPointer(), 
		 wVelNonlinearSrc.getPointer(), 
		 wVelocity.getPointer(), 
		 domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 density.getPointer(),
		 viscosity.getPointer(), 
		 &gravity, 
		 ioff, joff, koff, 
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
  CCVariable<double> density;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  StencilMatrix<SFCXVariable<double> > uVelCoeff;
  SFCXVariable<double> uNonlinearSrc;
  StencilMatrix<SFCYVariable<double> > vVelCoeff;
  SFCYVariable<double> vNonlinearSrc;
  StencilMatrix<SFCZVariable<double> > wVelCoeff;
  SFCZVariable<double> wNonlinearSrc;

  old_dw->get(pressure, d_pressureSPBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(uVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(uVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(uVelCoeff[ii], d_uVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  new_dw->get(uNonlinearSrc, d_uVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(vVelCoeff[ii], d_vVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  new_dw->get(vNonlinearSrc, d_vVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(wVelCoeff[ii], d_wVelCoefPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
  }
  new_dw->get(wNonlinearSrc, d_wVelNonLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  
  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  
  // Create vars for new_dw
  CCVariable<double> pressLinearSrc;
  CCVariable<double> pressNonlinearSrc;

  // Allocate space
  new_dw->allocate(pressLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch);
  new_dw->allocate(pressNonlinearSrc, d_presNonLinSrcPBLMLabel, matlIndex, 
		   patch);

  // Get the patch and variable indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = uVelocity.getFortLowIndex();
  IntVector domHiV = uVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = uVelocity.getFortLowIndex();
  IntVector domHiW = uVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call
  FORT_PRESSSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   pressLinearSrc.getPointer(),
		   pressNonlinearSrc.getPointer(),
		   pressure.getPointer(),
		   density.getPointer(),
		   domLoU.get_pointer(), domHiU.get_pointer(),
		   idxLoU.get_pointer(), idxHiU.get_pointer(),
		   uVelocity.getPointer(), 
		   uVelCoeff[StencilMatrix::AP].getPointer(),
		   uVelCoeff[StencilMatrix::AE].getPointer(),
		   uVelCoeff[StencilMatrix::AW].getPointer(),
		   uVelCoeff[StencilMatrix::AN].getPointer(),
		   uVelCoeff[StencilMatrix::AS].getPointer(),
		   uVelCoeff[StencilMatrix::AT].getPointer(),
		   uVelCoeff[StencilMatrix::AB].getPointer(),
		   uNonlinearSrc.getPointer(),
		   domLoV.get_pointer(), domHiV.get_pointer(),
		   idxLoV.get_pointer(), idxHiV.get_pointer(),
		   vVelocity.getPointer(), 
		   vVelCoeff[StencilMatrix::AP].getPointer(),
		   vVelCoeff[StencilMatrix::AE].getPointer(),
		   vVelCoeff[StencilMatrix::AW].getPointer(),
		   vVelCoeff[StencilMatrix::AN].getPointer(),
		   vVelCoeff[StencilMatrix::AS].getPointer(),
		   vVelCoeff[StencilMatrix::AT].getPointer(),
		   vVelCoeff[StencilMatrix::AB].getPointer(),
		   vNonlinearSrc.getPointer(),
		   domLoW.get_pointer(), domHiW.get_pointer(),
		   idxLoW.get_pointer(), idxHiW.get_pointer(),
		   wVelocity.getPointer(), 
		   wVelCoeff[StencilMatrix::AP].getPointer(),
		   wVelCoeff[StencilMatrix::AE].getPointer(),
		   wVelCoeff[StencilMatrix::AW].getPointer(),
		   wVelCoeff[StencilMatrix::AN].getPointer(),
		   wVelCoeff[StencilMatrix::AS].getPointer(),
		   wVelCoeff[StencilMatrix::AT].getPointer(),
		   wVelCoeff[StencilMatrix::AB].getPointer(),
		   wNonlinearSrc.getPointer(),
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#endif
		   
  new_dw->put(pressLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch);
  new_dw->put(pressNonlinearSrc, d_presNonLinSrcPBLMLabel, matlIndex, patch);
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

  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> scalar;
  CCVariable<double> density;
  CCVariable<double> viscosity;

  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // ** WARNING ** The scalar is got based on the input index
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // Get the PerPatch CellInformation data
  PerPatch<CellInformation*> cellInfoP;
  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //  if (old_dw->exists(d_cellInfoLabel, patch)) 
  //  old_dw->get(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //else {
  //  cellInfoP.setData(scinew CellInformation(patch));
  //  old_dw->put(cellInfoP, d_cellInfoLabel, matlIndex, patch);
  //}
  CellInformation* cellinfo = cellInfoP;
  
  CCVariable<double> scalarLinearSrc; //SP term in Arches
  CCVariable<double> scalarNonlinearSrc; // SU in Arches

  new_dw->allocate(scalarLinearSrc, d_scalLinSrcSBLMLabel, index, patch);
  new_dw->allocate(scalarNonlinearSrc, d_scalNonLinSrcSBLMLabel, 
		   index, patch);

  // Get the patch and variable indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = uVelocity.getFortLowIndex();
  IntVector domHiV = uVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = uVelocity.getFortLowIndex();
  IntVector domHiW = uVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = scalar.getFortLowIndex();
  IntVector domHi = scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_SCALARSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    scalarLinearSrc.getPointer(),
		    scalarNonlinearSrc.getPointer(),
		    scalar.getPointer(),
		    density.getPointer(),
		    viscosity.getPointer(),
		    domLoU.get_pointer(), domHiU.get_pointer(),
		    idxLoU.get_pointer(), idxHiU.get_pointer(),
		    uVelLinearSrc.getPointer(), 
		    uVelNonlinearSrc.getPointer(), 
		    uVelocity.getPointer(), 
		    domLoV.get_pointer(), domHiV.get_pointer(),
		    idxLoV.get_pointer(), idxHiV.get_pointer(),
		    vVelLinearSrc.getPointer(), 
		    vVelNonlinearSrc.getPointer(), 
		    vVelocity.getPointer(), 
		    domLoW.get_pointer(), domHiW.get_pointer(),
		    idxLoW.get_pointer(), idxHiW.get_pointer(),
		    wVelLinearSrc.getPointer(), 
		    wVelNonlinearSrc.getPointer(), 
		    wVelocity.getPointer(), 
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
  new_dw->put(scalarNonlinearSrc, d_scalNonLinSrcSBLMLabel, index, patch);

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
//$Log$
//Revision 1.19  2000/07/08 08:03:35  bbanerje
//Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
//made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
//to Arches::AE .. doesn't like enums in templates apparently.
//
//Revision 1.18  2000/07/03 05:30:16  bbanerje
//Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
//Revision 1.17  2000/07/02 05:47:31  bbanerje
//Uncommented all PerPatch and CellInformation stuff.
//Updated array sizes in inlbcs.F
//
//Revision 1.16  2000/06/30 04:36:47  bbanerje
//Changed FCVarsto SFC[X,Y,Z]Vars and added relevant getIndex() calls.
//
//
