//----- Discretization.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>

#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_modify_prescoef_fort.h>
#ifdef divergenceconstraint
#include <Packages/Uintah/CCA/Components/Arches/fortran/prescoef_var_fort.h>
#else
#include <Packages/Uintah/CCA/Components/Arches/fortran/prescoef_fort.h>
#endif
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/vvelcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/wvelcoef_fort.h>

//****************************************************************************
// Default constructor for Discretization
//****************************************************************************
Discretization::Discretization()
{
#ifdef PetscFilter
  d_filter = 0;
#endif
}

//****************************************************************************
// Destructor
//****************************************************************************
Discretization::~Discretization()
{
}

//****************************************************************************
// compute vel hat for explicit projection
//****************************************************************************
void 
Discretization::calculateVelRhoHat(const ProcessorGroup* /*pc*/,
			    const Patch* patch,
			    int index, double delta_t,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)

{
  // Get the patch bounds and the variable bounds
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;

  switch (index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->uVelRhoHat,
		      constvars->uVelocity,
		      vars->uVelocityCoeff[Arches::AE], 
		      vars->uVelocityCoeff[Arches::AW], 
		      vars->uVelocityCoeff[Arches::AN], 
		      vars->uVelocityCoeff[Arches::AS], 
		      vars->uVelocityCoeff[Arches::AT], 
		      vars->uVelocityCoeff[Arches::AB], 
		      vars->uVelocityCoeff[Arches::AP], 
		      vars->uVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		      delta_t, ioff, joff, koff);
    break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->vVelRhoHat,
		      constvars->vVelocity,
		      vars->vVelocityCoeff[Arches::AE], 
		      vars->vVelocityCoeff[Arches::AW], 
		      vars->vVelocityCoeff[Arches::AN], 
		      vars->vVelocityCoeff[Arches::AS], 
		      vars->vVelocityCoeff[Arches::AT], 
		      vars->vVelocityCoeff[Arches::AB], 
		      vars->vVelocityCoeff[Arches::AP], 
		      vars->vVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
		      delta_t, ioff, joff, koff);

    break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->wVelRhoHat,
		      constvars->wVelocity,
		      vars->wVelocityCoeff[Arches::AE], 
		      vars->wVelocityCoeff[Arches::AW], 
		      vars->wVelocityCoeff[Arches::AN], 
		      vars->wVelocityCoeff[Arches::AS], 
		      vars->wVelocityCoeff[Arches::AT], 
		      vars->wVelocityCoeff[Arches::AB], 
		      vars->wVelocityCoeff[Arches::AP], 
		      vars->wVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		      delta_t, ioff, joff, koff);

    vars->residWVel = 1.0E-7;
    vars->truncWVel = 1.0;

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}

//****************************************************************************
// Velocity stencil weights
//****************************************************************************
void 
Discretization::calculateVelocityCoeff(const ProcessorGroup*,
				       const Patch* patch,
				       double delta_t,
				       int index, bool lcentral,
				       CellInformation* cellinfo,
				       ArchesVariables* coeff_vars,
				       ArchesConstVariables* coeff_constvars)
{
  if (index == Arches::XDIR) {

    // Get the patch indices
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();
    // Calculate the coeffs
    fort_uvelcoef(coeff_constvars->uVelocity,
		  coeff_vars->uVelocityConvectCoeff[Arches::AE],
		  coeff_vars->uVelocityConvectCoeff[Arches::AW],
		  coeff_vars->uVelocityConvectCoeff[Arches::AN],
		  coeff_vars->uVelocityConvectCoeff[Arches::AS],
		  coeff_vars->uVelocityConvectCoeff[Arches::AT],
		  coeff_vars->uVelocityConvectCoeff[Arches::AB],
		  coeff_vars->uVelocityCoeff[Arches::AP],
		  coeff_vars->uVelocityCoeff[Arches::AE],
		  coeff_vars->uVelocityCoeff[Arches::AW],
		  coeff_vars->uVelocityCoeff[Arches::AN],
		  coeff_vars->uVelocityCoeff[Arches::AS],
		  coeff_vars->uVelocityCoeff[Arches::AT],
		  coeff_vars->uVelocityCoeff[Arches::AB],
		  coeff_constvars->vVelocity, coeff_constvars->wVelocity,
		  coeff_constvars->density, coeff_constvars->viscosity,
		  delta_t, lcentral,
		  cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		  cellinfo->cnn, cellinfo->csn, cellinfo->css,
		  cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		  cellinfo->sewu, cellinfo->sew, cellinfo->sns,
		  cellinfo->stb, cellinfo->dxepu, cellinfo->dxpwu,
		  cellinfo->dxpw, cellinfo->dynp, cellinfo->dyps,
		  cellinfo->dztp, cellinfo->dzpb, cellinfo->fac1u,
		  cellinfo->fac2u, cellinfo->fac3u, cellinfo->fac4u,
		  cellinfo->iesdu, cellinfo->iwsdu, cellinfo->enfac,
		  cellinfo->sfac, cellinfo->tfac, cellinfo->bfac,
		  idxLoU, idxHiU);
  } else if (index == Arches::YDIR) {

    // Get the patch indices
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();

    // Calculate the coeffs
    fort_vvelcoef(coeff_constvars->vVelocity,
		  coeff_vars->vVelocityConvectCoeff[Arches::AE],
		  coeff_vars->vVelocityConvectCoeff[Arches::AW],
		  coeff_vars->vVelocityConvectCoeff[Arches::AN],
		  coeff_vars->vVelocityConvectCoeff[Arches::AS],
		  coeff_vars->vVelocityConvectCoeff[Arches::AT],
		  coeff_vars->vVelocityConvectCoeff[Arches::AB],
		  coeff_vars->vVelocityCoeff[Arches::AP],
		  coeff_vars->vVelocityCoeff[Arches::AE],
		  coeff_vars->vVelocityCoeff[Arches::AW],
		  coeff_vars->vVelocityCoeff[Arches::AN],
		  coeff_vars->vVelocityCoeff[Arches::AS],
		  coeff_vars->vVelocityCoeff[Arches::AT],
		  coeff_vars->vVelocityCoeff[Arches::AB],
		  coeff_constvars->uVelocity, coeff_constvars->wVelocity,
		  coeff_constvars->density, coeff_constvars->viscosity,
		  delta_t,lcentral,
		  cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		  cellinfo->cnnv, cellinfo->csnv, cellinfo->cssv,
		  cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		  cellinfo->sew, cellinfo->snsv, cellinfo->sns,
		  cellinfo->stb, cellinfo->dxep, cellinfo->dxpw,
		  cellinfo->dynpv, cellinfo->dypsv, cellinfo->dyps,
		  cellinfo->dztp, cellinfo->dzpb, cellinfo->fac1v,
		  cellinfo->fac2v, cellinfo->fac3v, cellinfo->fac4v,
		  cellinfo->jnsdv, cellinfo->jssdv, cellinfo->efac,
		  cellinfo->wfac, cellinfo->tfac, cellinfo->bfac,
		  idxLoV, idxHiV);
  } else if (index == Arches::ZDIR) {

    // Get the patch indices
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();

    // Calculate the coeffs
    fort_wvelcoef(coeff_constvars->wVelocity,
		  coeff_vars->wVelocityConvectCoeff[Arches::AE],
		  coeff_vars->wVelocityConvectCoeff[Arches::AW],
		  coeff_vars->wVelocityConvectCoeff[Arches::AN],
		  coeff_vars->wVelocityConvectCoeff[Arches::AS],
		  coeff_vars->wVelocityConvectCoeff[Arches::AT],
		  coeff_vars->wVelocityConvectCoeff[Arches::AB],
		  coeff_vars->wVelocityCoeff[Arches::AP],
		  coeff_vars->wVelocityCoeff[Arches::AE],
		  coeff_vars->wVelocityCoeff[Arches::AW],
		  coeff_vars->wVelocityCoeff[Arches::AN],
		  coeff_vars->wVelocityCoeff[Arches::AS],
		  coeff_vars->wVelocityCoeff[Arches::AT],
		  coeff_vars->wVelocityCoeff[Arches::AB],
		  coeff_constvars->uVelocity, coeff_constvars->vVelocity,
		  coeff_constvars->density, coeff_constvars->viscosity,
		  delta_t,lcentral,
		  cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		  cellinfo->cnn, cellinfo->csn, cellinfo->css,
		  cellinfo->cttw, cellinfo->cbtw, cellinfo->cbbw,
		  cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		  cellinfo->stb, cellinfo->dxep, cellinfo->dxpw,
		  cellinfo->dynp, cellinfo->dyps, cellinfo->dztpw,
		  cellinfo->dzpbw, cellinfo->dzpb, cellinfo->fac1w,
		  cellinfo->fac2w, cellinfo->fac3w, cellinfo->fac4w,
		  cellinfo->ktsdw, cellinfo->kbsdw, cellinfo->efac,
		  cellinfo->wfac, cellinfo->enfac, cellinfo->sfac,
		  idxLoW, idxHiW);
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

}

void 
Discretization::computeDivergence(const ProcessorGroup* pc,
				  const Patch* patch,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector indexLow = patch->getCellFORTLowIndex();
  IntVector indexHigh = patch->getCellFORTHighIndex();
  CCVariable<double> unfiltered_divergence;
  unfiltered_divergence.resize(patch->getLowIndex(), patch->getHighIndex());
  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	unfiltered_divergence[currCell] = -constvars->drhodf[currCell]*
	  ((constvars->scalarDiffusionCoeff[Arches::AE])[currCell]*
	   constvars->scalar[IntVector(colX+1,colY,colZ)]+
	   (constvars->scalarDiffusionCoeff[Arches::AW])[currCell]*
	   constvars->scalar[IntVector(colX-1,colY,colZ)]+
	   (constvars->scalarDiffusionCoeff[Arches::AS])[currCell]*
	   constvars->scalar[IntVector(colX,colY-1,colZ)]+
	   (constvars->scalarDiffusionCoeff[Arches::AN])[currCell]*
	   constvars->scalar[IntVector(colX,colY+1,colZ)]+
	   (constvars->scalarDiffusionCoeff[Arches::AB])[currCell]*
	   constvars->scalar[IntVector(colX,colY,colZ-1)]+
	   (constvars->scalarDiffusionCoeff[Arches::AT])[currCell]*
	   constvars->scalar[IntVector(colX,colY,colZ+1)]+
	   constvars->scalarDiffNonlinearSrc[currCell] -
	   (constvars->scalarDiffusionCoeff[Arches::AP])[currCell]*
	   constvars->scalar[currCell])/(constvars->new_density[currCell]*
				    constvars->new_density[currCell]);
      }
    }
  }
#define FILTER_DIVERGENCE
#ifdef FILTER_DIVERGENCE
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch, unfiltered_divergence, vars->divergence);
#else
    vars->divergence.copy(unfiltered_divergence,
			  unfiltered_divergence.getLowIndex(),
		          unfiltered_divergence.getHighIndex());
#endif
#else
    vars->divergence.copy(unfiltered_divergence,
			  unfiltered_divergence.getLowIndex(),
		          unfiltered_divergence.getHighIndex());
#endif
}


//****************************************************************************
// Pressure stencil weights
//****************************************************************************
void 
Discretization::calculatePressureCoeff(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouse*,
				       DataWarehouse*,
				       double, 
				       CellInformation* cellinfo,
				       ArchesVariables* coeff_vars,
				       ArchesConstVariables* constcoeff_vars)
{
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef divergenceconstraint
  fort_prescoef_var(idxLo, idxHi, constcoeff_vars->density,
		    coeff_vars->pressCoeff[Arches::AE],
		    coeff_vars->pressCoeff[Arches::AW],
		    coeff_vars->pressCoeff[Arches::AN],
		    coeff_vars->pressCoeff[Arches::AS],
		    coeff_vars->pressCoeff[Arches::AT],
		    coeff_vars->pressCoeff[Arches::AB],
		    cellinfo->sew, cellinfo->sns, cellinfo->stb,
		    cellinfo->sewu, cellinfo->dxep, cellinfo->dxpw, 
		    cellinfo->snsv, cellinfo->dynp, cellinfo->dyps, 
		    cellinfo->stbw, cellinfo->dztp, cellinfo->dzpb);
#else
  fort_prescoef(idxLo, idxHi, 
		coeff_vars->pressCoeff[Arches::AE],
		coeff_vars->pressCoeff[Arches::AW],
		coeff_vars->pressCoeff[Arches::AN],
		coeff_vars->pressCoeff[Arches::AS],
		coeff_vars->pressCoeff[Arches::AT],
		coeff_vars->pressCoeff[Arches::AB],
		cellinfo->sew, cellinfo->sns, cellinfo->stb,
		cellinfo->sewu, cellinfo->dxep, cellinfo->dxpw, 
		cellinfo->snsv, cellinfo->dynp, cellinfo->dyps, 
		cellinfo->stbw, cellinfo->dztp, cellinfo->dzpb);
#endif
}

//****************************************************************************
// Modify Pressure Stencil for Multimaterial
//****************************************************************************

void
Discretization::mmModifyPressureCoeffs(const ProcessorGroup*,
				      const Patch* patch,
				      ArchesVariables* coeff_vars,
				      ArchesConstVariables* constcoeff_vars)

{
  // Get the domain size and the patch indices

  IntVector valid_lo = patch->getCellFORTLowIndex();
  IntVector valid_hi = patch->getCellFORTHighIndex();

  fort_mm_modify_prescoef(coeff_vars->pressCoeff[Arches::AE],
			  coeff_vars->pressCoeff[Arches::AW],
			  coeff_vars->pressCoeff[Arches::AN],
			  coeff_vars->pressCoeff[Arches::AS],
			  coeff_vars->pressCoeff[Arches::AT],
			  coeff_vars->pressCoeff[Arches::AB],
			  constcoeff_vars->voidFraction, valid_lo, valid_hi);
}
  
//****************************************************************************
// Scalar stencil weights
//****************************************************************************
void 
Discretization::calculateScalarCoeff(const ProcessorGroup*,
				     const Patch* patch,
				     double,
				     int, 
				     CellInformation* cellinfo,
				     ArchesVariables* coeff_vars,
				     ArchesConstVariables* constcoeff_vars,
				     int conv_scheme)
{
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  
  fort_scalcoef(idxLo, idxHi,
		constcoeff_vars->density, constcoeff_vars->viscosity,
		coeff_vars->scalarCoeff[Arches::AE],
		coeff_vars->scalarCoeff[Arches::AW],
		coeff_vars->scalarCoeff[Arches::AN],
		coeff_vars->scalarCoeff[Arches::AS],
		coeff_vars->scalarCoeff[Arches::AT],
		coeff_vars->scalarCoeff[Arches::AB],
		coeff_vars->scalarConvectCoeff[Arches::AE],
		coeff_vars->scalarConvectCoeff[Arches::AW],
		coeff_vars->scalarConvectCoeff[Arches::AN],
		coeff_vars->scalarConvectCoeff[Arches::AS],
		coeff_vars->scalarConvectCoeff[Arches::AT],
		coeff_vars->scalarConvectCoeff[Arches::AB],
		coeff_vars->scalarDiffusionCoeff[Arches::AE],
		coeff_vars->scalarDiffusionCoeff[Arches::AW],
		coeff_vars->scalarDiffusionCoeff[Arches::AN],
		coeff_vars->scalarDiffusionCoeff[Arches::AS],
		coeff_vars->scalarDiffusionCoeff[Arches::AT],
		coeff_vars->scalarDiffusionCoeff[Arches::AB],
		constcoeff_vars->uVelocity, constcoeff_vars->vVelocity,
		constcoeff_vars->wVelocity, cellinfo->sew, cellinfo->sns,
		cellinfo->stb, cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		cellinfo->cnn, cellinfo->csn, cellinfo->css, cellinfo->ctt,
		cellinfo->cbt, cellinfo->cbb, cellinfo->efac,
		cellinfo->wfac,	cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac,
		cellinfo->dxpw, cellinfo->dxep, cellinfo->dyps,
		cellinfo->dynp, cellinfo->dzpb, cellinfo->dztp,
		conv_scheme);

}

//****************************************************************************
// Calculate the diagonal terms (velocity)
//****************************************************************************
void 
Discretization::calculateVelDiagonal(const ProcessorGroup*,
				     const Patch* patch,
				     int index,
				     ArchesVariables* coeff_vars)
{
  
  // Get the patch and variable indices
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;
  switch(index) {
  case Arches::XDIR:
    domLo = coeff_vars->uVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->uVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();

    fort_apcalvel(idxLo, idxHi, coeff_vars->uVelocityCoeff[Arches::AP],
		  coeff_vars->uVelocityCoeff[Arches::AE],
		  coeff_vars->uVelocityCoeff[Arches::AW],
		  coeff_vars->uVelocityCoeff[Arches::AN],
		  coeff_vars->uVelocityCoeff[Arches::AS],
		  coeff_vars->uVelocityCoeff[Arches::AT],
		  coeff_vars->uVelocityCoeff[Arches::AB],
		  coeff_vars->uVelLinearSrc);

    break;
  case Arches::YDIR:
    domLo = coeff_vars->vVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->vVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

    fort_apcalvel(idxLo, idxHi, coeff_vars->vVelocityCoeff[Arches::AP],
		  coeff_vars->vVelocityCoeff[Arches::AE],
		  coeff_vars->vVelocityCoeff[Arches::AW],
		  coeff_vars->vVelocityCoeff[Arches::AN],
		  coeff_vars->vVelocityCoeff[Arches::AS],
		  coeff_vars->vVelocityCoeff[Arches::AT],
		  coeff_vars->vVelocityCoeff[Arches::AB],
		  coeff_vars->vVelLinearSrc);

    break;
  case Arches::ZDIR:
    domLo = coeff_vars->wVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->wVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();

    fort_apcalvel(idxLo, idxHi, coeff_vars->wVelocityCoeff[Arches::AP],
		  coeff_vars->wVelocityCoeff[Arches::AE],
		  coeff_vars->wVelocityCoeff[Arches::AW],
		  coeff_vars->wVelocityCoeff[Arches::AN],
		  coeff_vars->wVelocityCoeff[Arches::AS],
		  coeff_vars->wVelocityCoeff[Arches::AT],
		  coeff_vars->wVelocityCoeff[Arches::AB],
		  coeff_vars->wVelLinearSrc);

    break;
  default:
    throw InvalidValue("Invalid index in Discretization::calcVelDiagonal");
  }

}

//****************************************************************************
// Pressure diagonal
//****************************************************************************
void 
Discretization::calculatePressDiagonal(const ProcessorGroup*,
				       const Patch* patch,
				       DataWarehouse*,
				       DataWarehouse*,
				       ArchesVariables* coeff_vars) 
{
  
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // Calculate the diagonal terms (AP)
  fort_apcal(idxLo, idxHi, coeff_vars->pressCoeff[Arches::AP],
	     coeff_vars->pressCoeff[Arches::AE],
	     coeff_vars->pressCoeff[Arches::AW],
	     coeff_vars->pressCoeff[Arches::AN],
	     coeff_vars->pressCoeff[Arches::AS],
	     coeff_vars->pressCoeff[Arches::AT],
	     coeff_vars->pressCoeff[Arches::AB],
	     coeff_vars->pressLinearSrc);
}

//****************************************************************************
// Scalar diagonal
//****************************************************************************
void 
Discretization::calculateScalarDiagonal(const ProcessorGroup*,
					const Patch* patch,
					int,
					ArchesVariables* coeff_vars)
{
  
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  fort_apcal(idxLo, idxHi, coeff_vars->scalarCoeff[Arches::AP],
	     coeff_vars->scalarCoeff[Arches::AE],
	     coeff_vars->scalarCoeff[Arches::AW],
	     coeff_vars->scalarCoeff[Arches::AN],
	     coeff_vars->scalarCoeff[Arches::AS],
	     coeff_vars->scalarCoeff[Arches::AT],
	     coeff_vars->scalarCoeff[Arches::AB],
	     coeff_vars->scalarLinearSrc);
  coeff_vars->scalarLinearSrc.initialize(0.0);
  // for computing divergence constraint
  fort_apcal(idxLo, idxHi, coeff_vars->scalarDiffusionCoeff[Arches::AP],
	     coeff_vars->scalarDiffusionCoeff[Arches::AE],
	     coeff_vars->scalarDiffusionCoeff[Arches::AW],
	     coeff_vars->scalarDiffusionCoeff[Arches::AN],
	     coeff_vars->scalarDiffusionCoeff[Arches::AS],
	     coeff_vars->scalarDiffusionCoeff[Arches::AT],
	     coeff_vars->scalarDiffusionCoeff[Arches::AB],
	     coeff_vars->scalarLinearSrc);

}
//****************************************************************************
// Scalar ENO scheme (for convection part only)
//****************************************************************************
void 
Discretization::calculateScalarENOscheme(const ProcessorGroup*,
					const Patch* patch,
					int,
					CellInformation* cellinfo,
					const double maxAbsU,
					const double maxAbsV,
					const double maxAbsW,
					ArchesVariables*  scal_vars,
					ArchesConstVariables*  constscal_vars,
					const int wall_celltypeval)
{
  // ONLY SECOND ORDER ENO
  // ONLY FOR UNIFORM GRID
  Array3<double> x_undivided_difference;
  Array3<double> y_undivided_difference;
  Array3<double> z_undivided_difference;
  Array3<int> x_stencil_begin;
  Array3<int> y_stencil_begin;
  Array3<int> z_stencil_begin;
  Array3<double> x_flux;
  Array3<double> y_flux;
  Array3<double> z_flux;

// Assuming number of ghost cells = number of boundary cells = 1.
  IntVector domLo = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector domHi = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);
//  IntVector idxLo = patch->getCellFORTLowIndex();
//  IntVector idxHi = patch->getCellFORTHighIndex();
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";

//  cout << "maxAbsVelocity" << maxAbsU << maxAbsV << maxAbsW << "\n";

  IntVector idxLo = domLo + IntVector(1,1,1);
  IntVector idxHi = domHi - IntVector(1,1,1);
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";
  x_flux.resize(idxLo,idxHi+IntVector(1,0,0));
  y_flux.resize(idxLo,idxHi+IntVector(0,1,0));
  z_flux.resize(idxLo,idxHi+IntVector(0,0,1));
  x_flux.initialize(0.0);
  y_flux.initialize(0.0);
  z_flux.initialize(0.0);
//  cout << "domLo" << domLo.x() << domLo.y() << domLo.z() << "\n";
//  cout << "domHi" << domHi.x() << domHi.y() << domHi.z() << "\n";
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";
  
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int x_start = idxLo.x();
  int x_end = idxHi.x();
  int y_start = idxLo.y();
  int y_end = idxHi.y();
  int z_start = idxLo.z();
  int z_end = idxHi.z();
  int x_domstart = domLo.x();
  int x_domend = domHi.x();
  int y_domstart = domLo.y();
  int y_domend = domHi.y();
  int z_domstart = domLo.z();
  int z_domend = domHi.z();
  
  if (xminus) {
    int colX = x_start;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

/*	x_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[xminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[xminusCell]) *
			   constscal_vars->uVelocity[currCell];*/
	x_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[xminusCell] *
			   constscal_vars->scalar[xminusCell]) *
			   constscal_vars->uVelocity[currCell];
        if ((constscal_vars->cellType[xminusCell] == wall_celltypeval)
	    && (!(constscal_vars->cellType[currCell] == wall_celltypeval))) {
                     x_flux[currCell] = 0.0;
        }
      }
    }
    x_start ++;
    x_domstart ++; 
  }

  if (xplus) {
    int colX = x_end;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

/*	x_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[xminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[xminusCell]) *
			   constscal_vars->uVelocity[currCell];*/
	x_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[xminusCell] *
			   constscal_vars->scalar[xminusCell]) *
			   constscal_vars->uVelocity[currCell];
      }
    }
    x_end --;
    x_domend --; 
  }
  
  if (yminus) {
    int colY = y_start;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

/*	y_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[yminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[yminusCell]) *
			   constscal_vars->vVelocity[currCell];*/
	y_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[yminusCell] *
			   constscal_vars->scalar[yminusCell]) *
			   constscal_vars->vVelocity[currCell];
      }
    }
    y_start ++;
    y_domstart ++; 
  }

  if (yplus) {
    int colY = y_end;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

/*	y_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[yminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[yminusCell]) *
			   constscal_vars->vVelocity[currCell];*/
	y_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[yminusCell] *
			   constscal_vars->scalar[yminusCell]) *
			   constscal_vars->vVelocity[currCell];
      }
    }
    y_end --;
    y_domend --; 
  }
  
  if (zminus) {
    int colZ = z_start;
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

/*	z_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[zminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[zminusCell]) *
			   constscal_vars->wVelocity[currCell];*/
	z_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[zminusCell] *
			   constscal_vars->scalar[zminusCell]) *
			   constscal_vars->wVelocity[currCell];
      }
    }
    z_start ++;
    z_domstart ++; 
  }

  if (zplus) {
    int colZ = z_end;
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

/*	z_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[zminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[zminusCell]) *
			   constscal_vars->wVelocity[currCell];*/
	z_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[zminusCell] *
			   constscal_vars->scalar[zminusCell]) *
			   constscal_vars->wVelocity[currCell];
      }
    }
    z_end --;
    z_domend --; 
  }

// cout << "start " << x_start << "  " << y_start << "  " << z_start << "\n";
// cout << "end " << x_end << "  " << y_end << "  " << z_end << "\n";
//  cout << x_start << y_start << z_start << "\n";
//  cout << x_end << y_end << z_end << "\n";
  int x_domstartminus=x_domstart-1;
  int y_domstartminus=y_domstart-1;
  int z_domstartminus=z_domstart-1;
//  IntVector domstart(x_domstart, y_domstart, z_domstart);
//  IntVector domstartminus(x_domstartminus, y_domstartminus, z_domstartminus);
//  IntVector domend(x_domend, y_domend, z_domend);

  x_undivided_difference.resize(idxLo+IntVector(x_domstartminus-idxLo.x(),0,0),
		  		  idxHi+IntVector(x_domend-idxHi.x(),0,0));
  y_undivided_difference.resize(idxLo+IntVector(0,y_domstartminus-idxLo.y(),0),
		  		  idxHi+IntVector(0,y_domend-idxHi.y(),0));
  z_undivided_difference.resize(idxLo+IntVector(0,0,z_domstartminus-idxLo.z()),
		  		  idxHi+IntVector(0,0,z_domend-idxHi.z()));
  x_undivided_difference.initialize(0.0);
  y_undivided_difference.initialize(0.0);
  z_undivided_difference.initialize(0.0);
  x_stencil_begin.resize(idxLo+IntVector(x_domstart-idxLo.x(),0,0),
		  	   idxHi+IntVector(x_domend-idxHi.x(),0,0));
  y_stencil_begin.resize(idxLo+IntVector(0,y_domstart-idxLo.y(),0),
		  	   idxHi+IntVector(0,y_domend-idxHi.y(),0));
  z_stencil_begin.resize(idxLo+IntVector(0,0,z_domstart-idxLo.z()),
		  	   idxHi+IntVector(0,0,z_domend-idxHi.z()));
  x_stencil_begin.initialize(0);
  y_stencil_begin.initialize(0);
  z_stencil_begin.initialize(0);

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = x_domstartminus; colX < x_domend; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);

            x_undivided_difference[currCell] =
		constscal_vars->scalar[xplusCell] * constscal_vars->density[xplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colY = y_domstartminus; colY < y_domend; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);

            y_undivided_difference[currCell] =
		constscal_vars->scalar[yplusCell] * constscal_vars->density[yplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }

  for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colZ = z_domstartminus; colZ < z_domend; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zplusCell(colX, colY, colZ+1);

            z_undivided_difference[currCell] =
		constscal_vars->scalar[zplusCell] * constscal_vars->density[zplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = x_domstart; colX < x_domend; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);

	    x_stencil_begin[currCell] = colX;

	    if (Abs(x_undivided_difference[xminusCell]) <
		Abs(x_undivided_difference[currCell])) 
		   x_stencil_begin[currCell] --;
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colY = y_domstart; colY < y_domend; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);

	    y_stencil_begin[currCell] = colY;

	    if (Abs(y_undivided_difference[yminusCell]) <
		Abs(y_undivided_difference[currCell])) 
		   y_stencil_begin[currCell] --;
      }
    }
  }

  for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colZ = z_domstart; colZ < z_domend; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);

	    z_stencil_begin[currCell] = colZ;

	    if (Abs(z_undivided_difference[zminusCell]) <
		Abs(z_undivided_difference[currCell])) 
		   z_stencil_begin[currCell] --;
      }
    }
  }


  double uplus, uminus, vplus, vminus, wplus, wminus;
  double x_mass_flux, y_mass_flux, z_mass_flux;

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = x_start; colX < x_end+1; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector xminusminusCell(colX-2, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
	  
	  if (x_stencil_begin[currCell] == colX)
	     uplus = 1.5 * constscal_vars->scalar[currCell] * 
			   constscal_vars->density[currCell] -
		     0.5 * constscal_vars->scalar[xplusCell] *
			   constscal_vars->density[xplusCell];
	  else if (x_stencil_begin[currCell] == colX-1)
		  uplus = 0.5 * (constscal_vars->scalar[currCell] *
				 constscal_vars->density[currCell] +
				 constscal_vars->scalar[xminusCell] *
				 constscal_vars->density[xminusCell]);
	  else throw InternalError("x stensil choice error");

	  if (x_stencil_begin[xminusCell] == colX-1)
	     uminus = 0.5 * (constscal_vars->scalar[currCell] *
			     constscal_vars->density[currCell] +
		             constscal_vars->scalar[xminusCell] *
			     constscal_vars->density[xminusCell]);
	  else if (x_stencil_begin[xminusCell] == colX-2)
		  uminus = 1.5 * constscal_vars->scalar[xminusCell] *
				 constscal_vars->density[xminusCell] -
		           0.5 * constscal_vars->scalar[xminusminusCell] *
				 constscal_vars->density[xminusminusCell];
	  else throw InternalError("x stensil choice error");
	  
	  x_mass_flux = constscal_vars->uVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (uminus <= uplus)
//	     x_flux[currCell] = min(x_mass_flux * uminus, x_mass_flux * uplus);
//	  else
//	     x_flux[currCell] = max(x_mass_flux * uminus, x_mass_flux * uplus);

// Lax-Friedrichs flux computation
	  x_flux[currCell] = 0.5 * (x_mass_flux * (uminus + uplus) -
				    maxAbsU * (uplus - uminus));
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colY = y_start; colY < y_end+1; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector yminusminusCell(colX, colY-2, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
	  
	  if (y_stencil_begin[currCell] == colY)
	     vplus = 1.5 * constscal_vars->scalar[currCell] *
			   constscal_vars->density[currCell] -
		     0.5 * constscal_vars->scalar[yplusCell] *
			   constscal_vars->density[yplusCell];
	  else if (y_stencil_begin[currCell] == colY-1)
		  vplus = 0.5 * (constscal_vars->scalar[currCell] *
				 constscal_vars->density[currCell] +
				 constscal_vars->scalar[yminusCell] *
				 constscal_vars->density[yminusCell]);
	  else throw InternalError("y stensil choice error");

	  if (y_stencil_begin[yminusCell] == colY-1)
	     vminus = 0.5 * (constscal_vars->scalar[currCell] *
			     constscal_vars->density[currCell] +
			     constscal_vars->scalar[yminusCell] *
			     constscal_vars->density[yminusCell]);
	  else if (y_stencil_begin[yminusCell] == colY-2)
		  vminus = 1.5 * constscal_vars->scalar[yminusCell] *
				 constscal_vars->density[yminusCell] -
		           0.5 * constscal_vars->scalar[yminusminusCell] *
				 constscal_vars->density[yminusminusCell];
	  else throw InternalError("y stensil choice error");

	  y_mass_flux = constscal_vars->vVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (vminus <= vplus)
//	     y_flux[currCell] = min(y_mass_flux * vminus, y_mass_flux * vplus);
//	  else
//	     y_flux[currCell] = max(y_mass_flux * vminus, y_mass_flux * vplus);

// Lax-Friedrichs flux computation
	  y_flux[currCell] = 0.5 * (y_mass_flux * (vminus + vplus) -
				    maxAbsV * (vplus - vminus));
      }
    }
  }

  for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colZ = z_start; colZ < z_end+1; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          IntVector zminusminusCell(colX, colY, colZ-2);
          IntVector zplusCell(colX, colY, colZ+1);
	  
	  if (z_stencil_begin[currCell] == colZ)
	     wplus = 1.5 * constscal_vars->scalar[currCell] *
			   constscal_vars->density[currCell] -
		     0.5 * constscal_vars->scalar[zplusCell] *
			   constscal_vars->density[zplusCell];
	  else if (z_stencil_begin[currCell] == colZ-1)
		  wplus = 0.5 * (constscal_vars->scalar[currCell] *
				 constscal_vars->density[currCell] +
				 constscal_vars->scalar[zminusCell] *
				 constscal_vars->density[zminusCell]);
	  else throw InternalError("z stensil choice error");

	  if (z_stencil_begin[zminusCell] == colZ-1)
	     wminus = 0.5 * (constscal_vars->scalar[currCell] *
			     constscal_vars->density[currCell] +
			     constscal_vars->scalar[zminusCell] *
			     constscal_vars->density[zminusCell]);
	  else if (z_stencil_begin[zminusCell] == colZ-2)
		  wminus = 1.5 * constscal_vars->scalar[zminusCell] *
				 constscal_vars->density[zminusCell] -
		           0.5 * constscal_vars->scalar[zminusminusCell] *
				 constscal_vars->density[zminusminusCell];
	  else throw InternalError("z stensil choice error");

	  z_mass_flux = constscal_vars->wVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (wminus <= wplus)
//	     z_flux[currCell] = min(z_mass_flux * wminus, z_mass_flux * wplus);
//	  else
//	     z_flux[currCell] = max(z_mass_flux * wminus, z_mass_flux * wplus);

// Lax-Friedrichs flux computation
	  z_flux[currCell] = 0.5 * (z_mass_flux * (wminus + wplus) -
				    maxAbsW * (wplus - wminus));
      }
    }
  }


  double areaew;
  double areans;
  double areatb;
  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
          IntVector zplusCell(colX, colY, colZ+1);
	  areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
	  areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
	  areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

	  scal_vars->scalarNonlinearSrc[currCell] -=
		(x_flux[xplusCell]-x_flux[currCell]) * areaew +
		(y_flux[yplusCell]-y_flux[currCell]) * areans +
		(z_flux[zplusCell]-z_flux[currCell]) * areatb;
      }
    }
  }
}
//****************************************************************************
// Scalar WENO scheme (for convection part only)
//****************************************************************************
void 
Discretization::calculateScalarWENOscheme(const ProcessorGroup*,
					const Patch* patch,
					int,
					CellInformation* cellinfo,
					const double maxAbsU,
					const double maxAbsV,
					const double maxAbsW,
					ArchesVariables*  scal_vars,
					ArchesConstVariables*  constscal_vars,
					const int wall_celltypeval)
{
  // ONLY SECOND ORDER WENO
  // ONLY FOR UNIFORM GRID
  Array3<double> x_undivided_difference;
  Array3<double> y_undivided_difference;
  Array3<double> z_undivided_difference;
  Array3<double> x_flux;
  Array3<double> y_flux;
  Array3<double> z_flux;

// Assuming number of ghost cells = number of boundary cells = 1.
  IntVector domLo = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector domHi = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);
//  IntVector idxLo = patch->getCellFORTLowIndex();
//  IntVector idxHi = patch->getCellFORTHighIndex();
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";

//  cout << "maxAbsVelocity" << maxAbsU << maxAbsV << maxAbsW << "\n";

  IntVector idxLo = domLo + IntVector(1,1,1);
  IntVector idxHi = domHi - IntVector(1,1,1);
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";
  x_flux.resize(idxLo,idxHi+IntVector(1,0,0));
  y_flux.resize(idxLo,idxHi+IntVector(0,1,0));
  z_flux.resize(idxLo,idxHi+IntVector(0,0,1));
  x_flux.initialize(0.0);
  y_flux.initialize(0.0);
  z_flux.initialize(0.0);
//  cout << "domLo" << domLo.x() << domLo.y() << domLo.z() << "\n";
//  cout << "domHi" << domHi.x() << domHi.y() << domHi.z() << "\n";
//  cout << "idxLo" << idxLo.x() << idxLo.y() << idxLo.z() << "\n";
//  cout << "idxHi" << idxHi.x() << idxHi.y() << idxHi.z() << "\n";
  
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int x_start = idxLo.x();
  int x_end = idxHi.x();
  int y_start = idxLo.y();
  int y_end = idxHi.y();
  int z_start = idxLo.z();
  int z_end = idxHi.z();
  int x_domstart = domLo.x();
  int x_domend = domHi.x();
  int y_domstart = domLo.y();
  int y_domend = domHi.y();
  int z_domstart = domLo.z();
  int z_domend = domHi.z();
  
  if (xminus) {
    int colX = x_start;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

	x_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[xminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[xminusCell]) *
			   constscal_vars->uVelocity[currCell];
/*	x_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[xminusCell] *
			   constscal_vars->scalar[xminusCell]) *
			   constscal_vars->uVelocity[currCell];*/
        if ((constscal_vars->cellType[xminusCell] == wall_celltypeval)
	    && (!(constscal_vars->cellType[currCell] == wall_celltypeval))) {
                     x_flux[currCell] = 0.0;
        }
      }
    }
    x_start ++;
    x_domstart ++; 
  }

  if (xplus) {
    int colX = x_end;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

	x_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[xminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[xminusCell]) *
			   constscal_vars->uVelocity[currCell];
/*	x_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[xminusCell] *
			   constscal_vars->scalar[xminusCell]) *
			   constscal_vars->uVelocity[currCell];*/
      }
    }
    x_end --;
    x_domend --; 
  }
  
  if (yminus) {
    int colY = y_start;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

	y_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[yminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[yminusCell]) *
			   constscal_vars->vVelocity[currCell];
/*	y_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[yminusCell] *
			   constscal_vars->scalar[yminusCell]) *
			   constscal_vars->vVelocity[currCell];*/
      }
    }
    y_start ++;
    y_domstart ++; 
  }

  if (yplus) {
    int colY = y_end;
    for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

	y_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[yminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[yminusCell]) *
			   constscal_vars->vVelocity[currCell];
/*	y_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[yminusCell] *
			   constscal_vars->scalar[yminusCell]) *
			   constscal_vars->vVelocity[currCell];*/
      }
    }
    y_end --;
    y_domend --; 
  }
  
  if (zminus) {
    int colZ = z_start;
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

	z_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[zminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[zminusCell]) *
			   constscal_vars->wVelocity[currCell];
/*	z_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[zminusCell] *
			   constscal_vars->scalar[zminusCell]) *
			   constscal_vars->wVelocity[currCell];*/
      }
    }
    z_start ++;
    z_domstart ++; 
  }

  if (zplus) {
    int colZ = z_end;
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

	z_flux[currCell] = 0.25 * (constscal_vars->scalar[currCell]+
			   constscal_vars->scalar[zminusCell]) *
			   (constscal_vars->density[currCell]+
			   constscal_vars->density[zminusCell]) *
			   constscal_vars->wVelocity[currCell];
/*	z_flux[currCell] = 0.5 * (constscal_vars->density[currCell] *
			   constscal_vars->scalar[currCell] +
			   constscal_vars->density[zminusCell] *
			   constscal_vars->scalar[zminusCell]) *
			   constscal_vars->wVelocity[currCell];*/
      }
    }
    z_end --;
    z_domend --; 
  }

// cout << "start " << x_start << "  " << y_start << "  " << z_start << "\n";
// cout << "end " << x_end << "  " << y_end << "  " << z_end << "\n";
//  cout << x_start << y_start << z_start << "\n";
//  cout << x_end << y_end << z_end << "\n";
  int x_domstartminus=x_domstart-1;
  int y_domstartminus=y_domstart-1;
  int z_domstartminus=z_domstart-1;
//  IntVector domstart(x_domstart, y_domstart, z_domstart);
//  IntVector domstartminus(x_domstartminus, y_domstartminus, z_domstartminus);
//  IntVector domend(x_domend, y_domend, z_domend);

  x_undivided_difference.resize(idxLo+IntVector(x_domstartminus-idxLo.x(),0,0),
		  		  idxHi+IntVector(x_domend-idxHi.x(),0,0));
  y_undivided_difference.resize(idxLo+IntVector(0,y_domstartminus-idxLo.y(),0),
		  		  idxHi+IntVector(0,y_domend-idxHi.y(),0));
  z_undivided_difference.resize(idxLo+IntVector(0,0,z_domstartminus-idxLo.z()),
		  		  idxHi+IntVector(0,0,z_domend-idxHi.z()));
  x_undivided_difference.initialize(0.0);
  y_undivided_difference.initialize(0.0);
  z_undivided_difference.initialize(0.0);

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = x_domstartminus; colX < x_domend; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);

            x_undivided_difference[currCell] =
		constscal_vars->scalar[xplusCell] * constscal_vars->density[xplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colY = y_domstartminus; colY < y_domend; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);

            y_undivided_difference[currCell] =
		constscal_vars->scalar[yplusCell] * constscal_vars->density[yplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }

  for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colZ = z_domstartminus; colZ < z_domend; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zplusCell(colX, colY, colZ+1);

            z_undivided_difference[currCell] =
		constscal_vars->scalar[zplusCell] * constscal_vars->density[zplusCell] -
		constscal_vars->scalar[currCell] * constscal_vars->density[currCell];
      }
    }
  }



  double uplus, uminus, vplus, vminus, wplus, wminus;
  double x_mass_flux, y_mass_flux, z_mass_flux;
  double epsilon = 1e-8;
// Make the following variables arrays in general case
  double d0 = 2.0/3.0, d1 = 1.0/3.0;
  double uplus0, uminus0, vplus0, vminus0, wplus0, wminus0;
  double uplus1, uminus1, vplus1, vminus1, wplus1, wminus1;
  double uplus_alpha0, uplus_alpha1, uminus_alpha0, uminus_alpha1;
  double vplus_alpha0, vplus_alpha1, vminus_alpha0, vminus_alpha1;
  double wplus_alpha0, wplus_alpha1, wminus_alpha0, wminus_alpha1;
  double uplus_beta0, uplus_beta1, uminus_beta0, uminus_beta1;
  double vplus_beta0, vplus_beta1, vminus_beta0, vminus_beta1;
  double wplus_beta0, wplus_beta1, wminus_beta0, wminus_beta1;
  double uplus_weight0, uplus_weight1, uminus_weight0, uminus_weight1;
  double vplus_weight0, vplus_weight1, vminus_weight0, vminus_weight1;
  double wplus_weight0, wplus_weight1, wminus_weight0, wminus_weight1;

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = x_start; colX < x_end+1; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector xminusminusCell(colX-2, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);

	  uplus_beta0 = Sqr(x_undivided_difference[xminusCell]);
	  uplus_beta1 = Sqr(x_undivided_difference[currCell]);
	  uminus_beta0 = uplus_beta0;
	  uminus_beta1 = Sqr(x_undivided_difference[xminusminusCell]);

	  uplus_alpha0 = d0/Sqr(epsilon+uplus_beta0);	  
	  uplus_alpha1 = d1/Sqr(epsilon+uplus_beta1);	  
	  uminus_alpha0 = d0/Sqr(epsilon+uminus_beta0);
	  uminus_alpha1 = d1/Sqr(epsilon+uminus_beta1);

	  uplus_weight0 = uplus_alpha0/(uplus_alpha0 + uplus_alpha1);
	  uplus_weight1 = uplus_alpha1/(uplus_alpha0 + uplus_alpha1);
	  uminus_weight0 = uminus_alpha0/(uminus_alpha0 + uminus_alpha1);
	  uminus_weight1 = uminus_alpha1/(uminus_alpha0 + uminus_alpha1);

	  uplus1 = 1.5 * constscal_vars->scalar[currCell] * 
			 constscal_vars->density[currCell] -
		   0.5 * constscal_vars->scalar[xplusCell] *
			 constscal_vars->density[xplusCell];
	  uplus0 = 0.5 * (constscal_vars->scalar[currCell] *
			  constscal_vars->density[currCell] +
			  constscal_vars->scalar[xminusCell] *
			  constscal_vars->density[xminusCell]);

	  uminus0 = uplus0;
	  uminus1 = 1.5 * constscal_vars->scalar[xminusCell] *
			  constscal_vars->density[xminusCell] -
	            0.5 * constscal_vars->scalar[xminusminusCell] *
			  constscal_vars->density[xminusminusCell];
	  
	  uplus = uplus_weight0 * uplus0 + uplus_weight1 * uplus1;
	  uminus = uminus_weight0 * uminus0 + uminus_weight1 * uminus1;

	  x_mass_flux = constscal_vars->uVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (uminus <= uplus)
//	     x_flux[currCell] = min(x_mass_flux * uminus, x_mass_flux * uplus);
//	  else
//	     x_flux[currCell] = max(x_mass_flux * uminus, x_mass_flux * uplus);

// Lax-Friedrichs flux computation
	  x_flux[currCell] = 0.5 * (x_mass_flux * (uminus + uplus) -
				    maxAbsU * (uplus - uminus));
      }
    }
  }

  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colY = y_start; colY < y_end+1; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector yminusminusCell(colX, colY-2, colZ);
          IntVector yplusCell(colX, colY+1, colZ);

	  vplus_beta0 = Sqr(y_undivided_difference[yminusCell]);
	  vplus_beta1 = Sqr(y_undivided_difference[currCell]);
	  vminus_beta0 = vplus_beta0;
	  vminus_beta1 = Sqr(y_undivided_difference[yminusminusCell]);

	  vplus_alpha0 = d0/Sqr(epsilon+vplus_beta0);	  
	  vplus_alpha1 = d1/Sqr(epsilon+vplus_beta1);	  
	  vminus_alpha0 = d0/Sqr(epsilon+vminus_beta0);
	  vminus_alpha1 = d1/Sqr(epsilon+vminus_beta1);

	  vplus_weight0 = vplus_alpha0/(vplus_alpha0 + vplus_alpha1);
	  vplus_weight1 = vplus_alpha1/(vplus_alpha0 + vplus_alpha1);
	  vminus_weight0 = vminus_alpha0/(vminus_alpha0 + vminus_alpha1);
	  vminus_weight1 = vminus_alpha1/(vminus_alpha0 + vminus_alpha1);
	  
	  vplus1 = 1.5 * constscal_vars->scalar[currCell] *
			 constscal_vars->density[currCell] -
		   0.5 * constscal_vars->scalar[yplusCell] *
			 constscal_vars->density[yplusCell];
	  vplus0 = 0.5 * (constscal_vars->scalar[currCell] *
			  constscal_vars->density[currCell] +
			  constscal_vars->scalar[yminusCell] *
			  constscal_vars->density[yminusCell]);

	  vminus0 = vplus0;
	  vminus1 = 1.5 * constscal_vars->scalar[yminusCell] *
			  constscal_vars->density[yminusCell] -
		    0.5 * constscal_vars->scalar[yminusminusCell] *
			  constscal_vars->density[yminusminusCell];
	  
	  vplus = vplus_weight0 * vplus0 + vplus_weight1 * vplus1;
	  vminus = vminus_weight0 * vminus0 + vminus_weight1 * vminus1;

	  y_mass_flux = constscal_vars->vVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (vminus <= vplus)
//	     y_flux[currCell] = min(y_mass_flux * vminus, y_mass_flux * vplus);
//	  else
//	     y_flux[currCell] = max(y_mass_flux * vminus, y_mass_flux * vplus);

// Lax-Friedrichs flux computation
	  y_flux[currCell] = 0.5 * (y_mass_flux * (vminus + vplus) -
				    maxAbsV * (vplus - vminus));
      }
    }
  }

  for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {
      for (int colZ = z_start; colZ < z_end+1; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          IntVector zminusminusCell(colX, colY, colZ-2);
          IntVector zplusCell(colX, colY, colZ+1);

	  wplus_beta0 = Sqr(z_undivided_difference[zminusCell]);
	  wplus_beta1 = Sqr(z_undivided_difference[currCell]);
	  wminus_beta0 = wplus_beta0;
	  wminus_beta1 = Sqr(z_undivided_difference[zminusminusCell]);

	  wplus_alpha0 = d0/Sqr(epsilon+wplus_beta0);	  
	  wplus_alpha1 = d1/Sqr(epsilon+wplus_beta1);	  
	  wminus_alpha0 = d0/Sqr(epsilon+wminus_beta0);
	  wminus_alpha1 = d1/Sqr(epsilon+wminus_beta1);

	  wplus_weight0 = wplus_alpha0/(wplus_alpha0 + wplus_alpha1);
	  wplus_weight1 = wplus_alpha1/(wplus_alpha0 + wplus_alpha1);
	  wminus_weight0 = wminus_alpha0/(wminus_alpha0 + wminus_alpha1);
	  wminus_weight1 = wminus_alpha1/(wminus_alpha0 + wminus_alpha1);
	  
	  wplus1 = 1.5 * constscal_vars->scalar[currCell] *
			 constscal_vars->density[currCell] -
		   0.5 * constscal_vars->scalar[zplusCell] *
			 constscal_vars->density[zplusCell];
	  wplus0 = 0.5 * (constscal_vars->scalar[currCell] *
			  constscal_vars->density[currCell] +
			  constscal_vars->scalar[zminusCell] *
			  constscal_vars->density[zminusCell]);

	  wminus0 = wplus0;
	  wminus1 = 1.5 * constscal_vars->scalar[zminusCell] *
			  constscal_vars->density[zminusCell] -
		    0.5 * constscal_vars->scalar[zminusminusCell] *
			  constscal_vars->density[zminusminusCell];
	  
	  wplus = wplus_weight0 * wplus0 + wplus_weight1 * wplus1;
	  wminus = wminus_weight0 * wminus0 + wminus_weight1 * wminus1;

	  z_mass_flux = constscal_vars->wVelocity[currCell];
// The following if statement is for oversimplified version of Godunov flux
//	  if (wminus <= wplus)
//	     z_flux[currCell] = min(z_mass_flux * wminus, z_mass_flux * wplus);
//	  else
//	     z_flux[currCell] = max(z_mass_flux * wminus, z_mass_flux * wplus);

// Lax-Friedrichs flux computation
	  z_flux[currCell] = 0.5 * (z_mass_flux * (wminus + wplus) -
				    maxAbsW * (wplus - wminus));
      }
    }
  }


  double areaew;
  double areans;
  double areatb;
  for (int colZ = idxLo.z(); colZ < idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY < idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX < idxHi.x(); colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);
          IntVector yplusCell(colX, colY+1, colZ);
          IntVector zplusCell(colX, colY, colZ+1);
	  areaew = cellinfo->sns[colY] * cellinfo->stb[colZ];
	  areans = cellinfo->sew[colX] * cellinfo->stb[colZ];
	  areatb = cellinfo->sew[colX] * cellinfo->sns[colY];

	  scal_vars->scalarNonlinearSrc[currCell] -=
		(x_flux[xplusCell]-x_flux[currCell]) * areaew +
		(y_flux[yplusCell]-y_flux[currCell]) * areans +
		(z_flux[zplusCell]-z_flux[currCell]) * areatb;
      }
    }
  }
}
