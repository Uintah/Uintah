//----- Discretization.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/Stencil.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;

using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Default constructor for Discretization
//****************************************************************************
Discretization::Discretization()
{
  // inputs
  d_uVelocitySIVBCLabel = scinew VarLabel("uVelocitySIVBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocitySIVBCLabel = scinew VarLabel("vVelocitySIVBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocitySIVBCLabel = scinew VarLabel("wVelocitySIVBC",
				    SFCZVariable<double>::getTypeDescription() );
  d_densitySIVBCLabel = scinew VarLabel("densitySIVBC",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityCTSLabel = scinew VarLabel("viscosityCTS",
				   CCVariable<double>::getTypeDescription() );
  d_uVelocityCPBCLabel = scinew VarLabel("uVelocityCPBC",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityCPBCLabel = scinew VarLabel("vVelocityCPBC",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityCPBCLabel = scinew VarLabel("wVelocityCPBC",
				    SFCZVariable<double>::getTypeDescription() );

  // computes (calculateVelocityCoeff)
  d_uVelCoefPBLMLabel = scinew VarLabel("uVelCoefPBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefPBLMLabel = scinew VarLabel("vVelCoefPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefPBLMLabel = scinew VarLabel("wVelCoefPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelConvCoefPBLMLabel = scinew VarLabel("uVelConvCoefPBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelConvCoefPBLMLabel = scinew VarLabel("vVelConvCoefPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelConvCoefPBLMLabel = scinew VarLabel("wVelConvCoefPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelCoefMBLMLabel = scinew VarLabel("uVelCoefMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelCoefMBLMLabel = scinew VarLabel("vVelCoefMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelCoefMBLMLabel = scinew VarLabel("wVelCoefMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelConvCoefMBLMLabel = scinew VarLabel("uVelConvCoefMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelConvCoefMBLMLabel = scinew VarLabel("vVelConvCoefMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelConvCoefMBLMLabel = scinew VarLabel("wVelConvCoefMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  // calculateVelDiagonal
  d_uVelLinSrcPBLMLabel = scinew VarLabel("uVelLinSrcPBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcPBLMLabel = scinew VarLabel("vVelLinSrcPBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcPBLMLabel = scinew VarLabel("wVelLinSrcPBLM",
				   SFCZVariable<double>::getTypeDescription() );
  d_uVelLinSrcMBLMLabel = scinew VarLabel("uVelLinSrcMBLM",
				   SFCXVariable<double>::getTypeDescription() );
  d_vVelLinSrcMBLMLabel = scinew VarLabel("vVelLinSrcMBLM",
				   SFCYVariable<double>::getTypeDescription() );
  d_wVelLinSrcMBLMLabel = scinew VarLabel("wVelLinSrcMBLM",
				   SFCZVariable<double>::getTypeDescription() );
  // calculatePressureCoeff
  d_pressureINLabel = scinew VarLabel("pressureIN",
				   CCVariable<double>::getTypeDescription() );
  d_presCoefPBLMLabel = scinew VarLabel("presCoefPBLM",
				   CCVariable<double>::getTypeDescription() );

  // calculatePressDiagonal
  d_presLinSrcPBLMLabel = scinew VarLabel("presLinSrcPBLM",
				   CCVariable<double>::getTypeDescription() );

  // calculateScalarCoeff
  d_scalarSPLabel = scinew VarLabel("scalarSP",
				   CCVariable<double>::getTypeDescription() );
  d_uVelocityMSLabel = scinew VarLabel("uVelocityMS",
				    SFCXVariable<double>::getTypeDescription() );
  d_vVelocityMSLabel = scinew VarLabel("vVelocityMS",
				    SFCYVariable<double>::getTypeDescription() );
  d_wVelocityMSLabel = scinew VarLabel("wVelocityMS",
				    SFCZVariable<double>::getTypeDescription() );
  d_scalCoefSBLMLabel = scinew VarLabel("scalCoefSBLM",
				   CCVariable<double>::getTypeDescription() );

  // calculateScalarDiagonal
  d_scalLinSrcSBLMLabel = scinew VarLabel("scalLinSrcSBLM",
				   CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
Discretization::~Discretization()
{
}

//****************************************************************************
// Velocity stencil weights
//****************************************************************************
void 
Discretization::calculateVelocityCoeff(const ProcessorGroup* pc,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t,
				       int index,
				       int eqnType)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;
  StencilMatrix<SFCXVariable<double> > uVelocityCoeff;
  StencilMatrix<SFCYVariable<double> > vVelocityCoeff;
  StencilMatrix<SFCZVariable<double> > wVelocityCoeff;
  StencilMatrix<SFCXVariable<double> > uVelocityConvectCoeff;
  StencilMatrix<SFCYVariable<double> > vVelocityConvectCoeff;
  StencilMatrix<SFCZVariable<double> > wVelocityConvectCoeff;

  // Get the required data
  switch(eqnType) {
  case PRESSURE:
    new_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case MOMENTUM:
    new_dw->get(uVelocity, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type should be PRESSURE or MOMENTUM");
  }

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

  // Allocate space in new datawarehouse
  for (int ii = 0; ii < nofStencils; ii++) {
    switch(eqnType) {
    case PRESSURE:
      new_dw->allocate(uVelocityCoeff[ii], d_uVelCoefPBLMLabel, ii, patch);
      new_dw->allocate(vVelocityCoeff[ii], d_vVelCoefPBLMLabel, ii, patch);
      new_dw->allocate(wVelocityCoeff[ii], d_wVelCoefPBLMLabel, ii, patch);
      new_dw->allocate(uVelocityConvectCoeff[ii], d_uVelConvCoefPBLMLabel, ii, 
		       patch);
      new_dw->allocate(vVelocityConvectCoeff[ii], d_vVelConvCoefPBLMLabel, ii, 
		       patch);
      new_dw->allocate(wVelocityConvectCoeff[ii], d_wVelConvCoefPBLMLabel, ii, 
		       patch);
      break;
    case MOMENTUM:
      new_dw->allocate(uVelocityCoeff[ii], d_uVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(vVelocityCoeff[ii], d_vVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(wVelocityCoeff[ii], d_wVelCoefMBLMLabel, ii, patch);
      new_dw->allocate(uVelocityConvectCoeff[ii], d_uVelConvCoefMBLMLabel, ii, 
		       patch);
      new_dw->allocate(vVelocityConvectCoeff[ii], d_vVelConvCoefMBLMLabel, ii, 
		       patch);
      new_dw->allocate(wVelocityConvectCoeff[ii], d_wVelConvCoefMBLMLabel, ii, 
		       patch);
      break;
    default:
      throw InvalidValue("EqnType in calcVelCoef should be 0 or 1");
    }
  }

  // Get the domain size and the patch indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
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
  FORT_VELCOEF(domLoU.get_pointer(), domHiU.get_pointer(),
	       idxLoU.get_pointer(), idxHiU.get_pointer(),
	       uVelocity.getPointer(),
	       domLoV.get_pointer(), domHiV.get_pointer(),
	       idxLoV.get_pointer(), idxHiV.get_pointer(),
	       vVelocity.getPointer(),
	       domLoW.get_pointer(), domHiW.get_pointer(),
	       idxLoW.get_pointer(), idxHiW.get_pointer(),
	       wVelocity.getPointer(),
	       domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       density.getPointer(),
	       viscosity.getPointer(),
	       uVelocityConvectCoeff[StencilMatrix::AP].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AE].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AW].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AN].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AS].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AT].getPointer(), 
	       uVelocityConvectCoeff[StencilMatrix::AB].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AP].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AE].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AW].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AN].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AS].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AT].getPointer(), 
	       uVelocityCoeff[StencilMatrix::AB].getPointer(), 
	       delta_t,
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

  for (int ii = 0; ii < nofStencils; ii++) {
    switch(eqnType) {
    case PRESSURE:
      new_dw->put(uVelocityCoeff[ii], d_uVelCoefPBLMLabel, ii, patch);
      new_dw->put(vVelocityCoeff[ii], d_vVelCoefPBLMLabel, ii, patch);
      new_dw->put(wVelocityCoeff[ii], d_wVelCoefPBLMLabel, ii, patch);
      new_dw->put(uVelocityConvectCoeff[ii], d_uVelConvCoefPBLMLabel, ii, 
		  patch);
      new_dw->put(vVelocityConvectCoeff[ii], d_vVelConvCoefPBLMLabel, ii, 
		  patch);
      new_dw->put(wVelocityConvectCoeff[ii], d_wVelConvCoefPBLMLabel, ii,
		   patch);
      break;
    case MOMENTUM:
      new_dw->put(uVelocityCoeff[ii], d_uVelCoefMBLMLabel, ii, patch);
      new_dw->put(vVelocityCoeff[ii], d_vVelCoefMBLMLabel, ii, patch);
      new_dw->put(wVelocityCoeff[ii], d_wVelCoefMBLMLabel, ii, patch);
      new_dw->put(uVelocityConvectCoeff[ii], d_uVelConvCoefMBLMLabel, ii, 
		  patch);
      new_dw->put(vVelocityConvectCoeff[ii], d_vVelConvCoefMBLMLabel, ii, 
		  patch);
      new_dw->put(wVelocityConvectCoeff[ii], d_wVelConvCoefMBLMLabel, ii, 
		  patch);
      break;
    default:
      throw InvalidValue("EqnType in calcVelCoef should be 0 or 1");
    }
  }
  //new_dw->put(uVelocityCoeff, "VelocityCoeff", patch, index);
  //new_dw->put(uVelocityConvectCoeff, "VelocityConvectCoeff", patch, index);

}


//****************************************************************************
// Pressure stencil weights
//****************************************************************************
void 
Discretization::calculatePressureCoeff(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 0;

  CCVariable<double> pressure;
  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> viscosity;

  old_dw->get(pressure, d_pressureINLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  StencilMatrix<SFCXVariable<double> > uVelCoeff;
  StencilMatrix<SFCYVariable<double> > vVelCoeff;
  StencilMatrix<SFCZVariable<double> > wVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(uVelCoeff[ii], d_uVelCoefPBLMLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelCoeff[ii], d_vVelCoefPBLMLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelCoeff[ii], d_wVelCoefPBLMLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //int index = 1;
  //new_dw->get(uVelCoeff,"VelocityCoeff",patch, 0, index);
  //index++;
  //FCVariable<Vector> vVelCoeff;
  //new_dw->get(vVelCoeff,"VelocityCoeff",patch, 0, index);
  //index++;
  //FCVariable<Vector> wVelCoeff;
  //new_dw->get(wVelCoeff,"VelocityCoeff",patch, 0, index);

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

  // Create vars for new_dw
  StencilMatrix<CCVariable<double> > pressCoeff; //7 point stencil
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
  }
  //new_dw->allocate(pressCoeff,"pressureCoeff",patch, 0);

  // Get the domain size and the patch indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_PRESSSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   pressure.getPointer(),
		   pressCoeff[StencilMatrix::AP].getPointer(), 
		   pressCoeff[StencilMatrix::AE].getPointer(), 
		   pressCoeff[StencilMatrix::AW].getPointer(), 
		   pressCoeff[StencilMatrix::AN].getPointer(), 
		   pressCoeff[StencilMatrix::AS].getPointer(), 
		   pressCoeff[StencilMatrix::AT].getPointer(), 
		   pressCoeff[StencilMatrix::AB].getPointer(), 
		   domLoU.get_pointer(), domHiU.get_pointer(),
		   idxLoU.get_pointer(), idxHiU.get_pointer(),
		   uVelocity.getPointer(),
		   domLoV.get_pointer(), domHiV.get_pointer(),
		   idxLoV.get_pointer(), idxHiV.get_pointer(),
		   vVelocity.getPointer(),
		   domLoW.get_pointer(), domHiW.get_pointer(),
		   idxLoW.get_pointer(), idxHiW.get_pointer(),
		   wVelocity.getPointer(),
		   density.getPointer(),
		   uVelCoeff[StencilMatrix::AP].getPointer(), 
		   uVelCoeff[StencilMatrix::AE].getPointer(), 
		   uVelCoeff[StencilMatrix::AW].getPointer(), 
		   uVelCoeff[StencilMatrix::AN].getPointer(), 
		   uVelCoeff[StencilMatrix::AS].getPointer(), 
		   uVelCoeff[StencilMatrix::AT].getPointer(), 
		   uVelCoeff[StencilMatrix::AB].getPointer(), 
		   vVelCoeff[StencilMatrix::AP].getPointer(), 
		   vVelCoeff[StencilMatrix::AE].getPointer(), 
		   vVelCoeff[StencilMatrix::AW].getPointer(), 
		   vVelCoeff[StencilMatrix::AN].getPointer(), 
		   vVelCoeff[StencilMatrix::AS].getPointer(), 
		   vVelCoeff[StencilMatrix::AT].getPointer(), 
		   vVelCoeff[StencilMatrix::AB].getPointer(), 
		   wVelCoeff[StencilMatrix::AP].getPointer(), 
		   wVelCoeff[StencilMatrix::AE].getPointer(), 
		   wVelCoeff[StencilMatrix::AW].getPointer(), 
		   wVelCoeff[StencilMatrix::AN].getPointer(), 
		   wVelCoeff[StencilMatrix::AS].getPointer(), 
		   wVelCoeff[StencilMatrix::AT].getPointer(), 
		   wVelCoeff[StencilMatrix::AB].getPointer(), 
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
  }
  //new_dw->put(pressCoeff, "pressureCoeff", patch, 0);
}
  
//****************************************************************************
// Scalar stencil weights
//****************************************************************************
void 
Discretization::calculateScalarCoeff(const ProcessorGroup* pc,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t,
				     int index)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  SFCXVariable<double> uVelocity;
  SFCYVariable<double> vVelocity;
  SFCZVariable<double> wVelocity;
  CCVariable<double> density;
  CCVariable<double> viscosity;
  CCVariable<double> scalar;

  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(density, d_densitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // ithe componenet of scalar vector
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
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

  //7pt stencil declaration
  StencilMatrix<CCVariable<double> > scalarCoeff;

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(scalarCoeff[ii], d_scalCoefSBLMLabel, ii, patch);
  }

  // Get the domain size and the patch indices
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  IntVector domLo = scalar.getFortLowIndex();
  IntVector domHi = scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  FORT_SCALARCOEF(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  scalar.getPointer(),
		  scalarCoeff[StencilMatrix::AP].getPointer(), 
		  scalarCoeff[StencilMatrix::AE].getPointer(), 
		  scalarCoeff[StencilMatrix::AW].getPointer(), 
		  scalarCoeff[StencilMatrix::AN].getPointer(), 
		  scalarCoeff[StencilMatrix::AS].getPointer(), 
		  scalarCoeff[StencilMatrix::AT].getPointer(), 
		  scalarCoeff[StencilMatrix::AB].getPointer(), 
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  idxLoU.get_pointer(), idxHiU.get_pointer(),
		  uVelocity.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  idxLoV.get_pointer(), idxHiV.get_pointer(),
		  vVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  idxLoW.get_pointer(), idxHiW.get_pointer(),
		  wVelocity.getPointer(),
		  density.getPointer(),
		  viscosity.getPointer(), 
		  delta_t, 
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

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefSBLMLabel, ii, patch);
  }
  //new_dw->put(scalarCoeff, "ScalarCoeff", patch, index);
}

//****************************************************************************
// Calculate the diagonal terms (velocity)
//****************************************************************************
void 
Discretization::calculateVelDiagonal(const ProcessorGroup*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     int index,
				     int eqnType)
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  StencilMatrix<SFCXVariable<double> > uVelCoeff;
  StencilMatrix<SFCYVariable<double> > vVelCoeff;
  StencilMatrix<SFCZVariable<double> > wVelCoeff;
  SFCXVariable<double> uVelLinearSrc;
  SFCYVariable<double> vVelLinearSrc;
  SFCZVariable<double> wVelLinearSrc;

  switch(eqnType) {
  case PRESSURE:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->get(uVelCoeff[ii], d_uVelCoefPBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
      new_dw->get(vVelCoeff[ii], d_vVelCoefPBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
      new_dw->get(wVelCoeff[ii], d_wVelCoefPBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
    }
    new_dw->get(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    new_dw->get(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    new_dw->get(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    break;
  case MOMENTUM:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->get(uVelCoeff[ii], d_uVelCoefMBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
      new_dw->get(vVelCoeff[ii], d_vVelCoefMBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
      new_dw->get(wVelCoeff[ii], d_wVelCoefMBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
    }
    new_dw->get(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    new_dw->get(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    new_dw->get(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch, 
		Ghost::None, numGhostCells);
    break;
  default:
    break;
  }

  // Get the domain size and the patch indices
  IntVector domLoU = uVelLinearSrc.getFortLowIndex();
  IntVector domHiU = uVelLinearSrc.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector domLoV = vVelLinearSrc.getFortLowIndex();
  IntVector domHiV = vVelLinearSrc.getFortHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector domLoW = wVelLinearSrc.getFortLowIndex();
  IntVector domHiW = wVelLinearSrc.getFortHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_APCAL(domLoU.get_pointer(), domHiU.get_pointer(),
	     idxLoU.get_pointer(), idxHiU.get_pointer(),
	     uVelLinearSrc.getPointer(),
	     uVelCoeff[StencilMatrix::AP].getPointer(), 
	     uVelCoeff[StencilMatrix::AE].getPointer(), 
	     uVelCoeff[StencilMatrix::AW].getPointer(), 
	     uVelCoeff[StencilMatrix::AN].getPointer(), 
	     uVelCoeff[StencilMatrix::AS].getPointer(), 
	     uVelCoeff[StencilMatrix::AT].getPointer(), 
	     uVelCoeff[StencilMatrix::AB].getPointer());
  FORT_APCAL(domLoV.get_pointer(), domHiV.get_pointer(),
	     idxLoV.get_pointer(), idxHiV.get_pointer(),
	     vVelLinearSrc.getPointer(),
	     vVelCoeff[StencilMatrix::AP].getPointer(), 
	     vVelCoeff[StencilMatrix::AE].getPointer(), 
	     vVelCoeff[StencilMatrix::AW].getPointer(), 
	     vVelCoeff[StencilMatrix::AN].getPointer(), 
	     vVelCoeff[StencilMatrix::AS].getPointer(), 
	     vVelCoeff[StencilMatrix::AT].getPointer(), 
	     vVelCoeff[StencilMatrix::AB].getPointer());
  FORT_APCAL(domLoW.get_pointer(), domHiW.get_pointer(),
	     idxLoW.get_pointer(), idxHiW.get_pointer(),
	     wVelLinearSrc.getPointer(),
	     wVelCoeff[StencilMatrix::AP].getPointer(), 
	     wVelCoeff[StencilMatrix::AE].getPointer(), 
	     wVelCoeff[StencilMatrix::AW].getPointer(), 
	     wVelCoeff[StencilMatrix::AN].getPointer(), 
	     wVelCoeff[StencilMatrix::AS].getPointer(), 
	     wVelCoeff[StencilMatrix::AT].getPointer(), 
	     wVelCoeff[StencilMatrix::AB].getPointer());
#endif

  switch(eqnType) {
  case PRESSURE:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->put(uVelCoeff[ii], d_uVelCoefPBLMLabel, ii, patch);
      new_dw->put(vVelCoeff[ii], d_vVelCoefPBLMLabel, ii, patch);
      new_dw->put(wVelCoeff[ii], d_wVelCoefPBLMLabel, ii, patch);
    }
    break;
  case MOMENTUM:
    for (int ii = 0; ii < nofStencils; ii++) {
      new_dw->put(uVelCoeff[ii], d_uVelCoefMBLMLabel, ii, patch);
      new_dw->put(vVelCoeff[ii], d_vVelCoefMBLMLabel, ii, patch);
      new_dw->put(wVelCoeff[ii], d_wVelCoefMBLMLabel, ii, patch);
    }
    break;
  default:
    break;
  }

}

//****************************************************************************
// Pressure diagonal
//****************************************************************************
void 
Discretization::calculatePressDiagonal(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) 
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  StencilMatrix<CCVariable<double> > pressCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //Stencil<double> pressCoeff;
  //new_dw->get(pressCoeff, "PressureCoCoeff", patch, 0);

  CCVariable<double> presLinearSrc;
  new_dw->get(presLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  //FCVariable<double> pressLinearSrc;
  //new_dw->get(pressLinearSrc, "pressureLinearSource", patch, 0);

  // Get the domain size and the patch indices
  IntVector domLo = presLinearSrc.getFortLowIndex();
  IntVector domHi = presLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     presLinearSrc.getPointer(),
	     pressCoeff[StencilMatrix::AP].getPointer(), 
	     pressCoeff[StencilMatrix::AE].getPointer(), 
	     pressCoeff[StencilMatrix::AW].getPointer(), 
	     pressCoeff[StencilMatrix::AN].getPointer(), 
	     pressCoeff[StencilMatrix::AS].getPointer(), 
	     pressCoeff[StencilMatrix::AT].getPointer(), 
	     pressCoeff[StencilMatrix::AB].getPointer());
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
  }
  //new_dw->put(pressCoeff, "pressureLinearSource", patch, 0);
}

//****************************************************************************
// Scalar diagonal
//****************************************************************************
void 
Discretization::calculateScalarDiagonal(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					int index)
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // **WARNING** Don't know how to get the stencil data yet for scalars
  //             Currently overwriting scalarCoeff[ii] with the same data
  //             for the current scalar
  StencilMatrix<CCVariable<double> > scalarCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(scalarCoeff[ii], d_scalCoefSBLMLabel, index, patch, 
		Ghost::None, numGhostCells);
  }
  //Stencil<double> scalarCoeff;
  //new_dw->get(scalarCoeff, "ScalarCoeff", patch, index, 0);

  CCVariable<double> scalarLinearSrc;
  new_dw->get(scalarLinearSrc, d_scalLinSrcSBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);
  //FCVariable<double> scalarLinearSrc;
  //new_dw->get(scalarLinearSrc, "ScalarLinearSource", patch, index, 0);

  // Get the domain size and the patch indices
  IntVector domLo = scalarLinearSrc.getFortLowIndex();
  IntVector domHi = scalarLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     scalarLinearSrc.getPointer(),
	     scalarCoeff[StencilMatrix::AP].getPointer(), 
	     scalarCoeff[StencilMatrix::AE].getPointer(), 
	     scalarCoeff[StencilMatrix::AW].getPointer(), 
	     scalarCoeff[StencilMatrix::AN].getPointer(), 
	     scalarCoeff[StencilMatrix::AS].getPointer(), 
	     scalarCoeff[StencilMatrix::AT].getPointer(), 
	     scalarCoeff[StencilMatrix::AB].getPointer());
#endif

  // **WARNING** Don't know how to get the stencil data yet for scalars
  //             Currently overwriting scalarCoeff[ii] with the same data
  //             for the current scalar
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefSBLMLabel, index, patch);
  }
  //new_dw->put(scalarCoeff, "ScalarCoeff", patch, index, 0);
}

//
// $Log$
// Revision 1.19  2000/06/29 21:48:58  bbanerje
// Changed FC Vars to SFCX,Y,ZVars and added correct getIndex() to get domainhi/lo
// and index hi/lo
//
// Revision 1.18  2000/06/22 23:06:33  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.17  2000/06/21 07:50:59  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.16  2000/06/18 01:20:15  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.15  2000/06/17 07:06:23  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.14  2000/06/14 20:40:48  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.13  2000/06/13 06:02:31  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.12  2000/06/07 06:13:54  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.11  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
