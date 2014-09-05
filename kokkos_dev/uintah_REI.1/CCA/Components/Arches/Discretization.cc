//----- Discretization.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil.h>
#include <Packages/Uintah/Core/Grid/Task.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <math.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_all_fort.h>
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
		  cellinfo->iesdu, cellinfo->iwsdu, cellinfo->nfac,
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
		  cellinfo->wfac, cellinfo->nfac, cellinfo->sfac,
		  idxLoW, idxHiW);
  }
}

void 
Discretization::computeDivergence(const ProcessorGroup* pc,
				  const Patch* patch,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars,
				  const bool filter_divergence,
			     	  const bool periodic) 
{

  // Get the patch and variable indices
  IntVector indexLow = patch->getCellFORTLowIndex();
  IntVector indexHigh = patch->getCellFORTHighIndex();

  CCVariable<double> unfiltered_divergence;
  unfiltered_divergence.allocate(patch->getLowIndex(), patch->getHighIndex());
  unfiltered_divergence.initialize(0.0);

  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	if (constvars->new_density[currCell] > 0.0)
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
	else
	unfiltered_divergence[currCell] = 0.0;
      }
    }
  }

    if ((filter_divergence)&&(!(periodic))) {
    // filtering for periodic case is not implemented 
    // if it needs to be then unfiltered_divergence will require 1 layer of boundary cells to be computed
#ifdef PetscFilter
    d_filter->applyFilter(pc, patch, unfiltered_divergence, vars->divergence);
#else
    // filtering without petsc is not implemented
    // if it needs to be then unfiltered_divergence will have to be computed with ghostcells
    vars->divergence.copy(unfiltered_divergence,
			  unfiltered_divergence.getLowIndex(),
		          unfiltered_divergence.getHighIndex());
#endif
    }
    else
    vars->divergence.copy(unfiltered_divergence,
			  unfiltered_divergence.getLowIndex(),
		          unfiltered_divergence.getHighIndex());
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
		cellinfo->wfac,	cellinfo->nfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac,
		cellinfo->dxpw, cellinfo->dxep, cellinfo->dyps,
		cellinfo->dynp, cellinfo->dzpb, cellinfo->dztp,
		conv_scheme, d_turbPrNo);

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
  IntVector idxLo;
  IntVector idxHi;
  switch(index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();

    fort_apcal_all(idxLo, idxHi, coeff_vars->uVelocityCoeff[Arches::AP],
		  coeff_vars->uVelocityCoeff[Arches::AE],
		  coeff_vars->uVelocityCoeff[Arches::AW],
		  coeff_vars->uVelocityCoeff[Arches::AN],
		  coeff_vars->uVelocityCoeff[Arches::AS],
		  coeff_vars->uVelocityCoeff[Arches::AT],
		  coeff_vars->uVelocityCoeff[Arches::AB],
		  coeff_vars->uVelLinearSrc);

    break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

    fort_apcal_all(idxLo, idxHi, coeff_vars->vVelocityCoeff[Arches::AP],
		  coeff_vars->vVelocityCoeff[Arches::AE],
		  coeff_vars->vVelocityCoeff[Arches::AW],
		  coeff_vars->vVelocityCoeff[Arches::AN],
		  coeff_vars->vVelocityCoeff[Arches::AS],
		  coeff_vars->vVelocityCoeff[Arches::AT],
		  coeff_vars->vVelocityCoeff[Arches::AB],
		  coeff_vars->vVelLinearSrc);

    break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();

    fort_apcal_all(idxLo, idxHi, coeff_vars->wVelocityCoeff[Arches::AP],
		  coeff_vars->wVelocityCoeff[Arches::AE],
		  coeff_vars->wVelocityCoeff[Arches::AW],
		  coeff_vars->wVelocityCoeff[Arches::AN],
		  coeff_vars->wVelocityCoeff[Arches::AS],
		  coeff_vars->wVelocityCoeff[Arches::AT],
		  coeff_vars->wVelocityCoeff[Arches::AB],
		  coeff_vars->wVelLinearSrc);

    break;
  default:
    throw InvalidValue("Invalid index in Discretization::calcVelDiagonal", __FILE__, __LINE__);
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
  fort_apcal_all(idxLo, idxHi, coeff_vars->pressCoeff[Arches::AP],
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
					ArchesVariables* coeff_vars)
{
  
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  fort_apcal_all(idxLo, idxHi, coeff_vars->scalarCoeff[Arches::AP],
	     coeff_vars->scalarCoeff[Arches::AE],
	     coeff_vars->scalarCoeff[Arches::AW],
	     coeff_vars->scalarCoeff[Arches::AN],
	     coeff_vars->scalarCoeff[Arches::AS],
	     coeff_vars->scalarCoeff[Arches::AT],
	     coeff_vars->scalarCoeff[Arches::AB],
	     coeff_vars->scalarLinearSrc);
  coeff_vars->scalarLinearSrc.initialize(0.0);
  // for computing divergence constraint
  fort_apcal_all(idxLo, idxHi, coeff_vars->scalarDiffusionCoeff[Arches::AP],
	     coeff_vars->scalarDiffusionCoeff[Arches::AE],
	     coeff_vars->scalarDiffusionCoeff[Arches::AW],
	     coeff_vars->scalarDiffusionCoeff[Arches::AN],
	     coeff_vars->scalarDiffusionCoeff[Arches::AS],
	     coeff_vars->scalarDiffusionCoeff[Arches::AT],
	     coeff_vars->scalarDiffusionCoeff[Arches::AB],
	     coeff_vars->scalarLinearSrc);

}
//****************************************************************************
// Scalar central scheme with flux limiter (Superbee or Van Leer) 
// (for convection part only)
//****************************************************************************
void 
Discretization::calculateScalarFluxLimitedConvection(const ProcessorGroup*,
					const Patch* patch,
					CellInformation* cellinfo,
					ArchesVariables*  scal_vars,
					ArchesConstVariables*  constscal_vars,
					const int wall_celltypeval,
					int limiter_type,
					int boundary_limiter_type,
					bool central_limiter)
{
  Array3<double> x_flux;
  Array3<double> y_flux;
  Array3<double> z_flux;

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  x_flux.resize(idxLo,idxHi+IntVector(2,1,1));
  y_flux.resize(idxLo,idxHi+IntVector(1,2,1));
  z_flux.resize(idxLo,idxHi+IntVector(1,1,2));
  x_flux.initialize(0.0);
  y_flux.initialize(0.0);
  z_flux.initialize(0.0);
  
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

  int x_start = idxLo.x();
  int x_end = idxHi.x()+1;
  int y_start = idxLo.y();
  int y_end = idxHi.y()+1;
  int z_start = idxLo.z();
  int z_end = idxHi.z()+1;
  
// Need BC's for superbee and vanLeer flux limiter
  double c, Zup, Zdwn, psi, test;
  if (limiter_type < 3) {
  if (xminus) {
    int colX = x_start;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

	  c = constscal_vars->uVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[xminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[xminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[xminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[xminusCell])/
		   d_turbPrNo/cellinfo->sew[colX];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;

	  x_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));

        if ((constscal_vars->cellType[xminusCell] == wall_celltypeval)
	    && (!(constscal_vars->cellType[currCell] == wall_celltypeval))) {
                     x_flux[currCell] = 0.0;
        }
      }
    }
    x_start ++;
  }

  if (xplus) {
    int colX = x_end;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector xminusCell(colX-1, colY, colZ);

	  c = constscal_vars->uVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[xminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[xminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[xminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[xminusCell])/
		   d_turbPrNo/cellinfo->sew[colX];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;

	  x_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
    x_end --;
  }
  
  if (yminus) {
    int colY = y_start;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

	  c = constscal_vars->vVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[yminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[yminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[yminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[yminusCell])/
		   d_turbPrNo/cellinfo->sns[colY];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  y_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
    y_start ++;
  }

  if (yplus) {
    int colY = y_end;
    for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector yminusCell(colX, colY-1, colZ);

	  c = constscal_vars->vVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[yminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[yminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[yminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[yminusCell])/
		   d_turbPrNo/cellinfo->sns[colY];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  y_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
    y_end --;
  }
  
  if (zminus) {
    int colZ = z_start;
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

	  c = constscal_vars->wVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[zminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[zminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[zminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[zminusCell])/
		   d_turbPrNo/cellinfo->stb[colZ];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  z_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
    z_start ++;
  }

  if (zplus) {
    int colZ = z_end;
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        IntVector zminusCell(colX, colY, colZ-1);

	  c = constscal_vars->wVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[zminusCell]);

	  if (c > 0.0) {

	    Zup =   constscal_vars->scalar[zminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[zminusCell];
	  }

	  if (boundary_limiter_type == 2)
	    psi = 1.0;
	  else if (boundary_limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[zminusCell])/
		   d_turbPrNo/cellinfo->stb[colZ];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  z_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
    z_end --;
  }
  }
  double dZloc, dZup, r, temp1, temp2;

  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = x_start; colX <= x_end; colX ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector xminusCell(colX-1, colY, colZ);
          IntVector xminusminusCell(colX-2, colY, colZ);
          IntVector xplusCell(colX+1, colY, colZ);

	  c = constscal_vars->uVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[xminusCell]);

	  if (c > 0.0) {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[xminusCell];

	      dZup  = constscal_vars->scalar[xminusCell] -
		      constscal_vars->scalar[xminusminusCell];
	    }

	    Zup =   constscal_vars->scalar[xminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[xminusCell];

	      dZup  = constscal_vars->scalar[xplusCell] -
		      constscal_vars->scalar[currCell];
	    }

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[xminusCell];
	  }

          if (limiter_type < 3) {
	    if (!(dZloc == 0.0)) r = dZup/dZloc;
	    else if (dZup == dZloc) r = 1.0;
	    else r = 1.0e10; // doesn't take sign of dZup into accout, doesn't mattter since Zup = Zdwn in this case
	  }

	  if (limiter_type == 0) {
	    temp1 = min(2.0 * r,1.0);
	    temp2 = min(r,2.0);
	    temp1 = max(temp1, temp2);
	    psi = max(0.0,temp1);
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
          else if (limiter_type == 1) {
	    psi = (Abs(r)+r)/(1.0+Abs(r));
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
	  else if (limiter_type == 2)
	    psi = 1.0;
	  else if (limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[xminusCell])/
		   d_turbPrNo/cellinfo->sew[colX];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;

	  x_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
  }

  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
      for (int colY = y_start; colY <= y_end; colY ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector yminusCell(colX, colY-1, colZ);
          IntVector yminusminusCell(colX, colY-2, colZ);
          IntVector yplusCell(colX, colY+1, colZ);

	  c = constscal_vars->vVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[yminusCell]);

	  if (c > 0.0) {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[yminusCell];

	      dZup  = constscal_vars->scalar[yminusCell] -
		      constscal_vars->scalar[yminusminusCell];
	    }

	    Zup =   constscal_vars->scalar[yminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[yminusCell];

	      dZup  = constscal_vars->scalar[yplusCell] -
		      constscal_vars->scalar[currCell];
	    }

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[yminusCell];
	  }

          if (limiter_type < 3) {
	    if (!(dZloc == 0.0)) r = dZup/dZloc;
	    else if (dZup == dZloc) r = 1.0;
	    else r = 1.0e10; // doesn't take sign of dZup into accout, doesn't mattter since Zup = Zdwn in this case
	  }

	  if (limiter_type == 0) {
	    temp1 = min(2.0 * r,1.0);
	    temp2 = min(r,2.0);
	    temp1 = max(temp1, temp2);
	    psi = max(0.0,temp1);
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
          else if (limiter_type == 1) {
	    psi = (Abs(r)+r)/(1.0+Abs(r));
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
	  else if (limiter_type == 2)
	    psi = 1.0;
	  else if (limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[yminusCell])/
		   d_turbPrNo/cellinfo->sns[colY];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  y_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
  }

  for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
    for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
      for (int colZ = z_start; colZ <= z_end; colZ ++) {

          IntVector currCell(colX, colY, colZ);
          IntVector zminusCell(colX, colY, colZ-1);
          IntVector zminusminusCell(colX, colY, colZ-2);
          IntVector zplusCell(colX, colY, colZ+1);

	  c = constscal_vars->wVelocity[currCell] * 0.5 *
              (constscal_vars->density[currCell] +
	       constscal_vars->density[zminusCell]);

	  if (c > 0.0) {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[zminusCell];

	      dZup  = constscal_vars->scalar[zminusCell] -
		      constscal_vars->scalar[zminusminusCell];
	    }

	    Zup =   constscal_vars->scalar[zminusCell];

	    Zdwn =  constscal_vars->scalar[currCell];
	  }
	  else {
            if (limiter_type < 3) {
	      dZloc = constscal_vars->scalar[currCell] -
		      constscal_vars->scalar[zminusCell];

	      dZup  = constscal_vars->scalar[zplusCell] -
		      constscal_vars->scalar[currCell];
	    }

	    Zup =   constscal_vars->scalar[currCell];

	    Zdwn =  constscal_vars->scalar[zminusCell];
	  }

          if (limiter_type < 3) {
	    if (!(dZloc == 0.0)) r = dZup/dZloc;
	    else if (dZup == dZloc) r = 1.0;
	    else r = 1.0e10; // doesn't take sign of dZup into accout, doesn't mattter since Zup = Zdwn in this case
	  }

	  if (limiter_type == 0) {
	    temp1 = min(2.0 * r,1.0);
	    temp2 = min(r,2.0);
	    temp1 = max(temp1, temp2);
	    psi = max(0.0,temp1);
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
          else if (limiter_type == 1) {
	    psi = (Abs(r)+r)/(1.0+Abs(r));
	    if (central_limiter)
	      if (psi > 1.0) psi = 2.0 - psi;
	  }
	  else if (limiter_type == 2)
	    psi = 1.0;
	  else if (limiter_type == 3) {
	    test = 0.5*(constscal_vars->viscosity[currCell]+
			constscal_vars->viscosity[zminusCell])/
		   d_turbPrNo/cellinfo->stb[colZ];
	    if (test - 0.5 * Abs(c) > 0.0)
	      psi = 1.0;
	    else
	      psi = 0.0;
	  }
	  else
	    psi=0.0;
	  
	  z_flux[currCell] = c * (Zup + 0.5 * psi * (Zdwn - Zup));
      }
    }
  }


  double areaew;
  double areans;
  double areatb;
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {

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
