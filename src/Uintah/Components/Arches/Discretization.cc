//----- Discretization.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/ArchesFort.h>
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

  // computes (calculateVelocityCoeff)
  d_DUPBLMLabel = scinew VarLabel("DUPBLM",
				   SFCXVariable<double>::getTypeDescription() );
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
  d_pressureSPBCLabel = scinew VarLabel("pressureSPBC",
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
  SFCXVariable<double> variableCalledDU;
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
  new_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  switch(eqnType) {
  case Arches::PRESSURE:
    new_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  case Arches::MOMENTUM:
    new_dw->get(uVelocity, d_uVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelocity, d_vVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelocity, d_wVelocityCPBCLabel, matlIndex, patch, Ghost::None,
		numGhostCells);
    break;
  default:
    throw InvalidValue("Equation type should be PRESSURE or MOMENTUM");
  }

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
  
  // Allocate space in new datawarehouse
  switch(eqnType) {
  case Arches::PRESSURE:
    if (index == Arches::XDIR) {
      new_dw->allocate(variableCalledDU, d_DUPBLMLabel, matlIndex, patch);
    } 
    break;
  case Arches::MOMENTUM:
    break;
  default:
    break;
  }
  for (int ii = 0; ii < nofStencils; ii++) {
    switch(eqnType) {
    case Arches::PRESSURE:
      if (index == Arches::XDIR) {
	new_dw->allocate(uVelocityCoeff[ii], d_uVelCoefPBLMLabel, ii, patch);
	new_dw->allocate(uVelocityConvectCoeff[ii], d_uVelConvCoefPBLMLabel, ii, 
			 patch);
      }
      else if (index == Arches::YDIR) {
	new_dw->allocate(vVelocityCoeff[ii], d_vVelCoefPBLMLabel, ii, patch);
	new_dw->allocate(vVelocityConvectCoeff[ii], d_vVelConvCoefPBLMLabel, ii, 
			 patch);
      }
      else if (index == Arches::ZDIR) {
	new_dw->allocate(wVelocityCoeff[ii], d_wVelCoefPBLMLabel, ii, patch);
	new_dw->allocate(wVelocityConvectCoeff[ii], d_wVelConvCoefPBLMLabel, ii, 
			 patch);
      }
      else
	throw InvalidValue("Invalid index, should lie between {1,3]");
      break;
    case Arches::MOMENTUM:
      if (index == Arches::XDIR) {
	new_dw->allocate(uVelocityCoeff[ii], d_uVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(uVelocityConvectCoeff[ii], d_uVelConvCoefMBLMLabel, ii, 
			 patch);
      }
      else if (index == Arches::YDIR) {
	new_dw->allocate(vVelocityCoeff[ii], d_vVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(vVelocityConvectCoeff[ii], d_vVelConvCoefMBLMLabel, ii, 
			 patch);
      }
      else {
	new_dw->allocate(wVelocityCoeff[ii], d_wVelCoefMBLMLabel, ii, patch);
	new_dw->allocate(wVelocityConvectCoeff[ii], d_wVelConvCoefMBLMLabel, ii, 
			 patch);
      }
      break;
    default:
      throw InvalidValue("EqnType in calcVelCoef should be 0 or 1");
    }
  }

  // Get the domain size 
  IntVector domLoU = uVelocity.getFortLowIndex();
  IntVector domHiU = uVelocity.getFortHighIndex();
  IntVector domLoV = vVelocity.getFortLowIndex();
  IntVector domHiV = vVelocity.getFortHighIndex();
  IntVector domLoW = wVelocity.getFortLowIndex();
  IntVector domHiW = wVelocity.getFortHighIndex();
  IntVector domLo = density.getFortLowIndex();
  IntVector domHi = density.getFortHighIndex();
  //IntVector idxLo = patch->getCellFORTLowIndex();
  //IntVector idxHi = patch->getCellFORTHighIndex();

  if (index == Arches::XDIR) {

    // Get the patch indices
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();

    // Calculate the coeffs
    FORT_UVELCOEF(domLoU.get_pointer(), domHiU.get_pointer(),
		  idxLoU.get_pointer(), idxHiU.get_pointer(),
		  uVelocity.getPointer(),
		  uVelocityConvectCoeff[Arches::AE].getPointer(), 
		  uVelocityConvectCoeff[Arches::AW].getPointer(), 
		  uVelocityConvectCoeff[Arches::AN].getPointer(), 
		  uVelocityConvectCoeff[Arches::AS].getPointer(), 
		  uVelocityConvectCoeff[Arches::AT].getPointer(), 
		  uVelocityConvectCoeff[Arches::AB].getPointer(), 
		  uVelocityCoeff[Arches::AP].getPointer(), 
		  uVelocityCoeff[Arches::AE].getPointer(), 
		  uVelocityCoeff[Arches::AW].getPointer(), 
		  uVelocityCoeff[Arches::AN].getPointer(), 
		  uVelocityCoeff[Arches::AS].getPointer(), 
		  uVelocityCoeff[Arches::AT].getPointer(), 
		  uVelocityCoeff[Arches::AB].getPointer(), 
		  variableCalledDU.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  vVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  wVelocity.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  density.getPointer(),
		  viscosity.getPointer(),
		  &delta_t,
		  cellinfo->ceeu.get_objs(), cellinfo->cweu.get_objs(),
		  cellinfo->cwwu.get_objs(),
		  cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		  cellinfo->css.get_objs(),
		  cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		  cellinfo->cbb.get_objs(),
		  cellinfo->sewu.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stb.get_objs(),
		  cellinfo->dxepu.get_objs(), cellinfo->dxpwu.get_objs(),
		  cellinfo->dynp.get_objs(), cellinfo->dyps.get_objs(),
		  cellinfo->dztp.get_objs(), cellinfo->dzpb.get_objs(),
		  cellinfo->fac1u.get_objs(), cellinfo->fac2u.get_objs(),
		  cellinfo->fac3u.get_objs(), cellinfo->fac4u.get_objs(),
		  cellinfo->iesdu.get_objs(), cellinfo->iwsdu.get_objs(), 
		  cellinfo->enfac.get_objs(), cellinfo->sfac.get_objs(),
		  cellinfo->tfac.get_objs(), cellinfo->bfac.get_objs());
  } else if (index == Arches::YDIR) {

    // Get the patch indices
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();

    // Calculate the coeffs
    FORT_VVELCOEF(domLoV.get_pointer(), domHiV.get_pointer(),
		  idxLoV.get_pointer(), idxHiV.get_pointer(),
		  vVelocity.getPointer(),
		  vVelocityConvectCoeff[Arches::AE].getPointer(), 
		  vVelocityConvectCoeff[Arches::AW].getPointer(), 
		  vVelocityConvectCoeff[Arches::AN].getPointer(), 
		  vVelocityConvectCoeff[Arches::AS].getPointer(), 
		  vVelocityConvectCoeff[Arches::AT].getPointer(), 
		  vVelocityConvectCoeff[Arches::AB].getPointer(), 
		  vVelocityCoeff[Arches::AP].getPointer(), 
		  vVelocityCoeff[Arches::AE].getPointer(), 
		  vVelocityCoeff[Arches::AW].getPointer(), 
		  vVelocityCoeff[Arches::AN].getPointer(), 
		  vVelocityCoeff[Arches::AS].getPointer(), 
		  vVelocityCoeff[Arches::AT].getPointer(), 
		  vVelocityCoeff[Arches::AB].getPointer(), 
		  variableCalledDU.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  uVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  wVelocity.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  density.getPointer(),
		  viscosity.getPointer(),
		  &delta_t,
		  cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(),
		  cellinfo->cww.get_objs(),
		  cellinfo->cnnv.get_objs(), cellinfo->csnv.get_objs(),
		  cellinfo->cssv.get_objs(),
		  cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		  cellinfo->cbb.get_objs(),
		  cellinfo->sew.get_objs(), cellinfo->snsv.get_objs(),
		  cellinfo->stb.get_objs(),
		  cellinfo->dxep.get_objs(), cellinfo->dxpw.get_objs(),
		  cellinfo->dynpv.get_objs(), cellinfo->dypsv.get_objs(),
		  cellinfo->dztp.get_objs(), cellinfo->dzpb.get_objs(),
		  cellinfo->fac1v.get_objs(), cellinfo->fac2v.get_objs(),
		  cellinfo->fac3v.get_objs(), cellinfo->fac4v.get_objs(),
		  cellinfo->jnsdv.get_objs(), cellinfo->jssdv.get_objs(), 
		  cellinfo->efac.get_objs(), cellinfo->wfac.get_objs(),
		  cellinfo->tfac.get_objs(), cellinfo->bfac.get_objs());
  } else if (index == Arches::ZDIR) {

    // Get the patch indices
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();

    // Calculate the coeffs
    FORT_WVELCOEF(domLoW.get_pointer(), domHiW.get_pointer(),
		  idxLoW.get_pointer(), idxHiW.get_pointer(),
		  wVelocity.getPointer(),
		  wVelocityConvectCoeff[Arches::AE].getPointer(), 
		  wVelocityConvectCoeff[Arches::AW].getPointer(), 
		  wVelocityConvectCoeff[Arches::AN].getPointer(), 
		  wVelocityConvectCoeff[Arches::AS].getPointer(), 
		  wVelocityConvectCoeff[Arches::AT].getPointer(), 
		  wVelocityConvectCoeff[Arches::AB].getPointer(), 
		  wVelocityCoeff[Arches::AP].getPointer(), 
		  wVelocityCoeff[Arches::AE].getPointer(), 
		  wVelocityCoeff[Arches::AW].getPointer(), 
		  wVelocityCoeff[Arches::AN].getPointer(), 
		  wVelocityCoeff[Arches::AS].getPointer(), 
		  wVelocityCoeff[Arches::AT].getPointer(), 
		  wVelocityCoeff[Arches::AB].getPointer(), 
		  variableCalledDU.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  uVelocity.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  vVelocity.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  density.getPointer(),
		  viscosity.getPointer(),
		  &delta_t,
		  cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(),
		  cellinfo->cww.get_objs(),
		  cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		  cellinfo->css.get_objs(),
		  cellinfo->cttw.get_objs(), cellinfo->cbtw.get_objs(),
		  cellinfo->cbbw.get_objs(),
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stbw.get_objs(),
		  cellinfo->dxep.get_objs(), cellinfo->dxpw.get_objs(),
		  cellinfo->dynp.get_objs(), cellinfo->dyps.get_objs(),
		  cellinfo->dztpw.get_objs(), cellinfo->dzpbw.get_objs(),
		  cellinfo->fac1w.get_objs(), cellinfo->fac2w.get_objs(),
		  cellinfo->fac3w.get_objs(), cellinfo->fac4w.get_objs(),
		  cellinfo->ktsdw.get_objs(), cellinfo->kbsdw.get_objs(), 
		  cellinfo->efac.get_objs(), cellinfo->wfac.get_objs(),
		  cellinfo->enfac.get_objs(), cellinfo->sfac.get_objs());
  }

#ifdef MAY_BE_USEFUL_LATER  
  // int ioff = 1;
  // int joff = 0;
  // int koff = 0;

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // FORT_VELCOEF(domLoU.get_pointer(), domHiU.get_pointer(),
  //       idxLoU.get_pointer(), idxHiU.get_pointer(),
  //       uVelocity.getPointer(),
  //       domLoV.get_pointer(), domHiV.get_pointer(),
  //       idxLoV.get_pointer(), idxHiV.get_pointer(),
  //       vVelocity.getPointer(),
  //       domLoW.get_pointer(), domHiW.get_pointer(),
  //       idxLoW.get_pointer(), idxHiW.get_pointer(),
  //       wVelocity.getPointer(),
  //       domLo.get_pointer(), domHi.get_pointer(),
  //       idxLo.get_pointer(), idxHi.get_pointer(),
  //       density.getPointer(),
  //       viscosity.getPointer(),
  //       uVelocityConvectCoeff[Arches::AP].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AE].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AW].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AN].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AS].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AT].getPointer(), 
  //       uVelocityConvectCoeff[Arches::AB].getPointer(), 
  //       uVelocityCoeff[Arches::AP].getPointer(), 
  //       uVelocityCoeff[Arches::AE].getPointer(), 
  //       uVelocityCoeff[Arches::AW].getPointer(), 
  //       uVelocityCoeff[Arches::AN].getPointer(), 
  //       uVelocityCoeff[Arches::AS].getPointer(), 
  //       uVelocityCoeff[Arches::AT].getPointer(), 
  //       uVelocityCoeff[Arches::AB].getPointer(), 
  //       delta_t,
  //       ioff, joff, koff, 
  //       cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
  //       cellinfo->cnn, cellinfo->csn, cellinfo->css,
  //       cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
  //       cellinfo->sewu, cellinfo->sns, cellinfo->stb,
  //       cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
  //       cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
  //       cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
  //       cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
  //       cellinfo->tfac, cellinfo->bfac, volume);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    switch(eqnType) {
    case Arches::PRESSURE:
      if (index == Arches::XDIR) {
	new_dw->put(uVelocityCoeff[ii], d_uVelCoefPBLMLabel, ii, patch);
	new_dw->put(uVelocityConvectCoeff[ii], d_uVelConvCoefPBLMLabel, ii, 
		    patch);
      } else if (index == Arches::YDIR) {
	new_dw->put(vVelocityCoeff[ii], d_vVelCoefPBLMLabel, ii, patch);
	new_dw->put(vVelocityConvectCoeff[ii], d_vVelConvCoefPBLMLabel, ii, 
		    patch);
      } else if (index == Arches::ZDIR) {
	new_dw->put(wVelocityCoeff[ii], d_wVelCoefPBLMLabel, ii, patch);
	new_dw->put(wVelocityConvectCoeff[ii], d_wVelConvCoefPBLMLabel, ii,
		    patch);
      }
      break;
    case Arches::MOMENTUM:
      if (index == Arches::XDIR) {
	new_dw->put(uVelocityCoeff[ii], d_uVelCoefMBLMLabel, ii, patch);
	new_dw->put(uVelocityConvectCoeff[ii], d_uVelConvCoefMBLMLabel, ii, 
		    patch);
      } else if (index == Arches::YDIR) {
	new_dw->put(vVelocityCoeff[ii], d_vVelCoefMBLMLabel, ii, patch);
	new_dw->put(vVelocityConvectCoeff[ii], d_vVelConvCoefMBLMLabel, ii, 
		    patch);
      } else if (index == Arches::ZDIR) {
	new_dw->put(wVelocityCoeff[ii], d_wVelCoefMBLMLabel, ii, patch);
	new_dw->put(wVelocityConvectCoeff[ii], d_wVelConvCoefMBLMLabel, ii, 
		    patch);
      }
      break;
    default:
      throw InvalidValue("EqnType in calcVelCoef should be 0 or 1");
    }
  }
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

  old_dw->get(pressure, d_pressureSPBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(uVelocity, d_uVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelocity, d_vVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelocity, d_wVelocitySIVBCLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
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
  StencilMatrix<CCVariable<double> > pressCoeff; //7 point stencil
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
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
  IntVector domLo = pressure.getFortLowIndex();
  IntVector domHi = pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_PRESSSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   pressure.getPointer(),
		   pressCoeff[Arches::AP].getPointer(), 
		   pressCoeff[Arches::AE].getPointer(), 
		   pressCoeff[Arches::AW].getPointer(), 
		   pressCoeff[Arches::AN].getPointer(), 
		   pressCoeff[Arches::AS].getPointer(), 
		   pressCoeff[Arches::AT].getPointer(), 
		   pressCoeff[Arches::AB].getPointer(), 
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
		   uVelCoeff[Arches::AP].getPointer(), 
		   uVelCoeff[Arches::AE].getPointer(), 
		   uVelCoeff[Arches::AW].getPointer(), 
		   uVelCoeff[Arches::AN].getPointer(), 
		   uVelCoeff[Arches::AS].getPointer(), 
		   uVelCoeff[Arches::AT].getPointer(), 
		   uVelCoeff[Arches::AB].getPointer(), 
		   vVelCoeff[Arches::AP].getPointer(), 
		   vVelCoeff[Arches::AE].getPointer(), 
		   vVelCoeff[Arches::AW].getPointer(), 
		   vVelCoeff[Arches::AN].getPointer(), 
		   vVelCoeff[Arches::AS].getPointer(), 
		   vVelCoeff[Arches::AT].getPointer(), 
		   vVelCoeff[Arches::AB].getPointer(), 
		   wVelCoeff[Arches::AP].getPointer(), 
		   wVelCoeff[Arches::AE].getPointer(), 
		   wVelCoeff[Arches::AW].getPointer(), 
		   wVelCoeff[Arches::AN].getPointer(), 
		   wVelCoeff[Arches::AS].getPointer(), 
		   wVelCoeff[Arches::AT].getPointer(), 
		   wVelCoeff[Arches::AB].getPointer(), 
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
  }
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

  old_dw->get(density, d_densityCPLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  old_dw->get(viscosity, d_viscosityCTSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // ithe componenet of scalar vector
  old_dw->get(scalar, d_scalarSPLabel, index, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(uVelocity, d_uVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelocity, d_vVelocityMSLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelocity, d_wVelocityMSLabel, matlIndex, patch, Ghost::None,
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
		  scalarCoeff[Arches::AP].getPointer(), 
		  scalarCoeff[Arches::AE].getPointer(), 
		  scalarCoeff[Arches::AW].getPointer(), 
		  scalarCoeff[Arches::AN].getPointer(), 
		  scalarCoeff[Arches::AS].getPointer(), 
		  scalarCoeff[Arches::AT].getPointer(), 
		  scalarCoeff[Arches::AB].getPointer(), 
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
  case Arches::PRESSURE:
    switch(index) {
    case Arches::XDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
	new_dw->get(uVelCoeff[ii], d_uVelCoefPBLMLabel, ii, patch, 
		    Ghost::None, numGhostCells);
      new_dw->get(uVelLinearSrc, d_uVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case Arches::YDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
	new_dw->get(vVelCoeff[ii], d_vVelCoefPBLMLabel, ii, patch, 
		    Ghost::None, numGhostCells);
      new_dw->get(vVelLinearSrc, d_vVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case Arches::ZDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
      new_dw->get(wVelCoeff[ii], d_wVelCoefPBLMLabel, ii, patch, Ghost::None,
		  numGhostCells);
      new_dw->get(wVelLinearSrc, d_wVelLinSrcPBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    default:
      throw InvalidValue("Invalid index in Pressure::calcVelDiagonal");
    }
    break;
  case Arches::MOMENTUM:
    switch(index) {
    case Arches::XDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
	new_dw->get(uVelCoeff[ii], d_uVelCoefMBLMLabel, ii, patch, 
		    Ghost::None, numGhostCells);
      new_dw->get(uVelLinearSrc, d_uVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case Arches::YDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
	new_dw->get(vVelCoeff[ii], d_vVelCoefMBLMLabel, ii, patch, 
		    Ghost::None, numGhostCells);
      new_dw->get(vVelLinearSrc, d_vVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    case Arches::ZDIR:
      for (int ii = 0; ii < nofStencils; ii++) 
	new_dw->get(wVelCoeff[ii], d_wVelCoefMBLMLabel, ii, patch, 
		    Ghost::None, numGhostCells);
      new_dw->get(wVelLinearSrc, d_wVelLinSrcMBLMLabel, matlIndex, patch, 
		  Ghost::None, numGhostCells);
      break;
    default:
      throw InvalidValue("Invalid index in Momentum::calcVelDiagonal");
    }
    break;
  default:
    throw InvalidValue("Invalid eqnType in Discretization::calcVelDiagonal");
  }

  // Get the patch and variable indices
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;
  switch(index) {
  case Arches::XDIR:
    domLo = uVelLinearSrc.getFortLowIndex();
    domHi = uVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       uVelCoeff[Arches::AP].getPointer(), 
	       uVelCoeff[Arches::AE].getPointer(), 
	       uVelCoeff[Arches::AW].getPointer(), 
	       uVelCoeff[Arches::AN].getPointer(), 
	       uVelCoeff[Arches::AS].getPointer(), 
	       uVelCoeff[Arches::AT].getPointer(), 
	       uVelCoeff[Arches::AB].getPointer(),
	       uVelLinearSrc.getPointer());
    break;
  case Arches::YDIR:
    domLo = vVelLinearSrc.getFortLowIndex();
    domHi = vVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       vVelCoeff[Arches::AP].getPointer(), 
	       vVelCoeff[Arches::AE].getPointer(), 
	       vVelCoeff[Arches::AW].getPointer(), 
	       vVelCoeff[Arches::AN].getPointer(), 
	       vVelCoeff[Arches::AS].getPointer(), 
	       vVelCoeff[Arches::AT].getPointer(), 
	       vVelCoeff[Arches::AB].getPointer(),
	       vVelLinearSrc.getPointer());
    break;
  case Arches::ZDIR:
    domLo = wVelLinearSrc.getFortLowIndex();
    domHi = wVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       wVelCoeff[Arches::AP].getPointer(), 
	       wVelCoeff[Arches::AE].getPointer(), 
	       wVelCoeff[Arches::AW].getPointer(), 
	       wVelCoeff[Arches::AN].getPointer(), 
	       wVelCoeff[Arches::AS].getPointer(), 
	       wVelCoeff[Arches::AT].getPointer(), 
	       wVelCoeff[Arches::AB].getPointer(),
	       wVelLinearSrc.getPointer());
    break;
  default:
    throw InvalidValue("Invalid index in Discretization::calcVelDiagonal");
  }

  switch(eqnType) {
  case Arches::PRESSURE:
    switch(index) {
    case Arches::XDIR:
      new_dw->put(uVelCoeff[Arches::AP], d_uVelCoefPBLMLabel, Arches::AP, patch);
      break;
    case Arches::YDIR:
      new_dw->put(vVelCoeff[Arches::AP], d_vVelCoefPBLMLabel, Arches::AP, patch);
      break;
    case Arches::ZDIR:
      new_dw->put(wVelCoeff[Arches::AP], d_wVelCoefPBLMLabel, Arches::AP, patch);
      break;
    default:
      throw InvalidValue("Invalid index in Pressure::calcVelDiagonal");
    }
    break;
  case Arches::MOMENTUM:
    switch(index) {
    case Arches::XDIR:
      new_dw->put(uVelCoeff[Arches::AP], d_uVelCoefMBLMLabel, Arches::AP, patch);
      break;
    case Arches::YDIR:
      new_dw->put(vVelCoeff[Arches::AP], d_vVelCoefMBLMLabel, Arches::AP, patch);
      break;
    case Arches::ZDIR:
      new_dw->put(wVelCoeff[Arches::AP], d_wVelCoefMBLMLabel, Arches::AP, patch);
      break;
    default:
      throw InvalidValue("Invalid index in Pressure::calcVelDiagonal");
    }
    break;
  default:
    throw InvalidValue("Invalid eqnType in Discretization::calcVelDiagonal");
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
  CCVariable<double> presLinearSrc;

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  new_dw->get(presLinearSrc, d_presLinSrcPBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);

  // Get the domain size and the patch indices
  IntVector domLo = presLinearSrc.getFortLowIndex();
  IntVector domHi = presLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     presLinearSrc.getPointer(),
	     pressCoeff[Arches::AP].getPointer(), 
	     pressCoeff[Arches::AE].getPointer(), 
	     pressCoeff[Arches::AW].getPointer(), 
	     pressCoeff[Arches::AN].getPointer(), 
	     pressCoeff[Arches::AS].getPointer(), 
	     pressCoeff[Arches::AT].getPointer(), 
	     pressCoeff[Arches::AB].getPointer());
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefPBLMLabel, ii, patch);
  }
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
  CCVariable<double> scalarLinearSrc;

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(scalarCoeff[ii], d_scalCoefSBLMLabel, index, patch, 
		Ghost::None, numGhostCells);
  }
  new_dw->get(scalarLinearSrc, d_scalLinSrcSBLMLabel, matlIndex, patch, 
	      Ghost::None, numGhostCells);

  // Get the domain size and the patch indices
  IntVector domLo = scalarLinearSrc.getFortLowIndex();
  IntVector domHi = scalarLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     scalarLinearSrc.getPointer(),
	     scalarCoeff[Arches::AP].getPointer(), 
	     scalarCoeff[Arches::AE].getPointer(), 
	     scalarCoeff[Arches::AW].getPointer(), 
	     scalarCoeff[Arches::AN].getPointer(), 
	     scalarCoeff[Arches::AS].getPointer(), 
	     scalarCoeff[Arches::AT].getPointer(), 
	     scalarCoeff[Arches::AB].getPointer());
#endif

  // **WARNING** Don't know how to get the stencil data yet for scalars
  //             Currently overwriting scalarCoeff[ii] with the same data
  //             for the current scalar
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefSBLMLabel, index, patch);
  }
}

//
// $Log$
// Revision 1.28  2000/07/12 19:55:43  bbanerje
// Added apcal stuff in calcVelDiagonal
//
// Revision 1.27  2000/07/11 15:46:27  rawat
// added setInitialGuess in PicardNonlinearSolver and also added uVelSrc
//
// Revision 1.26  2000/07/09 00:23:58  bbanerje
// Made changes to calcVelocitySource .. still getting seg violation here.
//
// Revision 1.25  2000/07/08 23:42:54  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.24  2000/07/08 23:08:54  bbanerje
// Added vvelcoef and wvelcoef ..
// Rawat check the ** WARNING ** tags in these files for possible problems.
//
// Revision 1.23  2000/07/08 08:03:33  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.22  2000/07/07 23:07:45  rawat
// added inlet bc's
//
// Revision 1.21  2000/07/03 05:30:14  bbanerje
// Minor changes for inlbcs dummy code to compile and work. densitySIVBC is no more.
//
// Revision 1.20  2000/07/02 05:47:30  bbanerje
// Uncommented all PerPatch and CellInformation stuff.
// Updated array sizes in inlbcs.F
//
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
