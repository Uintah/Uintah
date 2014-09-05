//----- Discretization.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
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
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/apcal_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mm_modify_prescoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/prescoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/vvelcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/wvelcoef_fort.h>

//****************************************************************************
// Default constructor for Discretization
//****************************************************************************
Discretization::Discretization()
{
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
			    ArchesVariables* vars)

{
  // Get the patch bounds and the variable bounds
  IntVector domLoU;
  IntVector domHiU;
  IntVector domLoUO;
  IntVector domHiUO;
  IntVector domLong;
  IntVector domHing;
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;

  switch (index) {
  case Arches::XDIR:
    domLoU = vars->uVelRhoHat.getFortLowIndex();
    domHiU = vars->uVelRhoHat.getFortHighIndex();
    domLoUO = vars->uVelocity.getFortLowIndex();
    domHiUO = vars->uVelocity.getFortHighIndex();
    domLong = vars->uVelNonlinearSrc.getFortLowIndex();
    domHing = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->uVelRhoHat,
		      vars->uVelocity,
		      vars->uVelocityCoeff[Arches::AE], 
		      vars->uVelocityCoeff[Arches::AW], 
		      vars->uVelocityCoeff[Arches::AN], 
		      vars->uVelocityCoeff[Arches::AS], 
		      vars->uVelocityCoeff[Arches::AT], 
		      vars->uVelocityCoeff[Arches::AB], 
		      vars->uVelocityCoeff[Arches::AP], 
		      vars->uVelNonlinearSrc,
		      vars->density,
		      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		      delta_t, ioff, joff, koff);
    break;
  case Arches::YDIR:
    domLoU = vars->vVelRhoHat.getFortLowIndex();
    domHiU = vars->vVelRhoHat.getFortHighIndex();
    domLoUO = vars->vVelocity.getFortLowIndex();
    domHiUO = vars->vVelocity.getFortHighIndex();
    domLong = vars->vVelNonlinearSrc.getFortLowIndex();
    domHing = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->vVelRhoHat,
		      vars->vVelocity,
		      vars->vVelocityCoeff[Arches::AE], 
		      vars->vVelocityCoeff[Arches::AW], 
		      vars->vVelocityCoeff[Arches::AN], 
		      vars->vVelocityCoeff[Arches::AS], 
		      vars->vVelocityCoeff[Arches::AT], 
		      vars->vVelocityCoeff[Arches::AB], 
		      vars->vVelocityCoeff[Arches::AP], 
		      vars->vVelNonlinearSrc,
		      vars->density,
		      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
		      delta_t, ioff, joff, koff);

    break;
  case Arches::ZDIR:
    domLoU = vars->wVelRhoHat.getFortLowIndex();
    domHiU = vars->wVelRhoHat.getFortHighIndex();
    domLoUO = vars->wVelocity.getFortLowIndex();
    domHiUO = vars->wVelocity.getFortHighIndex();
    domLong = vars->wVelNonlinearSrc.getFortLowIndex();
    domHing = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->wVelRhoHat,
		      vars->wVelocity,
		      vars->wVelocityCoeff[Arches::AE], 
		      vars->wVelocityCoeff[Arches::AW], 
		      vars->wVelocityCoeff[Arches::AN], 
		      vars->wVelocityCoeff[Arches::AS], 
		      vars->wVelocityCoeff[Arches::AT], 
		      vars->wVelocityCoeff[Arches::AB], 
		      vars->wVelocityCoeff[Arches::AP], 
		      vars->wVelNonlinearSrc,
		      vars->density,
		      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		      delta_t, ioff, joff, koff);

#ifdef ARCHES_VEL_DEBUG
    cerr << "Print wvelhat" << endl;
    vars->wVelRhoHat.print(cerr);

    cerr << " After W Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "W Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

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
				       int index,
				       CellInformation* cellinfo,
				       ArchesVariables* coeff_vars)
{
#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE VELCOEF" << endl;
  cerr << "Print Density" << endl;
  coeff_vars->density.print(cerr);
  cerr << "Print uVelocity" << endl;
  coeff_vars->uVelocity.print(cerr);
  cerr << "Print vVelocity" << endl;
  coeff_vars->vVelocity.print(cerr);
  cerr << "Print wVelocity" << endl;
  coeff_vars->wVelocity.print(cerr);
#endif

  if (index == Arches::XDIR) {

    // Get the patch indices
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();
#ifdef ARCHES_COEF_DEBUG
    cerr << "idxLou, idxHiU" << idxLoU << " " << idxHiU << endl;
#endif
    // Calculate the coeffs
    fort_uvelcoef(coeff_vars->uVelocity,
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
		  coeff_vars->vVelocity, coeff_vars->wVelocity,
		  coeff_vars->density, coeff_vars->viscosity, delta_t,
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
#ifdef ARCHES_COEF_DEBUG
    cerr << "After UVELCOEF" << endl;
    cerr << "Print UAW" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AW].print(cerr);
    cerr << "Print UAE" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AE].print(cerr);
    cerr << "Print UAN" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AN].print(cerr);
    cerr << "Print UAS" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AS].print(cerr);
    cerr << "Print UAT" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AT].print(cerr);
    cerr << "Print UAB" << endl;
    coeff_vars->uVelocityConvectCoeff[Arches::AB].print(cerr);
    cerr << "Print UAW" << endl;
    coeff_vars->uVelocityCoeff[Arches::AW].print(cerr);
    cerr << "Print UAE" << endl;
    coeff_vars->uVelocityCoeff[Arches::AE].print(cerr);
    cerr << "Print UAN" << endl;
    coeff_vars->uVelocityCoeff[Arches::AN].print(cerr);
    cerr << "Print UAS" << endl;
    coeff_vars->uVelocityCoeff[Arches::AS].print(cerr);
    cerr << "Print UAT" << endl;
    coeff_vars->uVelocityCoeff[Arches::AT].print(cerr);
    cerr << "Print UAB" << endl;
    coeff_vars->uVelocityCoeff[Arches::AB].print(cerr);
#endif
  } else if (index == Arches::YDIR) {

    // Get the patch indices
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();

    // Calculate the coeffs
    fort_vvelcoef(coeff_vars->vVelocity,
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
		  coeff_vars->uVelocity, coeff_vars->wVelocity,
		  coeff_vars->density, coeff_vars->viscosity, delta_t,
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
#ifdef ARCHES_COEF_DEBUG
    cerr << "After VVELCOEF" << endl;
    cerr << "Print VAW" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AW].print(cerr);
    cerr << "Print VAE" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AE].print(cerr);
    cerr << "Print VAN" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AN].print(cerr);
    cerr << "Print VAS" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AS].print(cerr);
    cerr << "Print VAT" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AT].print(cerr);
    cerr << "Print VAB" << endl;
    coeff_vars->vVelocityConvectCoeff[Arches::AB].print(cerr);
    cerr << "Print VAW" << endl;
    coeff_vars->vVelocityCoeff[Arches::AW].print(cerr);
    cerr << "Print VAE" << endl;
    coeff_vars->vVelocityCoeff[Arches::AE].print(cerr);
    cerr << "Print VAN" << endl;
    coeff_vars->vVelocityCoeff[Arches::AN].print(cerr);
    cerr << "Print VAS" << endl;
    coeff_vars->vVelocityCoeff[Arches::AS].print(cerr);
    cerr << "Print VAT" << endl;
    coeff_vars->vVelocityCoeff[Arches::AT].print(cerr);
    cerr << "Print VAB" << endl;
    coeff_vars->vVelocityCoeff[Arches::AB].print(cerr);
#endif
  } else if (index == Arches::ZDIR) {

    // Get the patch indices
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();

    // Calculate the coeffs
    fort_wvelcoef(coeff_vars->wVelocity,
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
		  coeff_vars->uVelocity, coeff_vars->vVelocity,
		  coeff_vars->density, coeff_vars->viscosity, delta_t,
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
#ifdef ARCHES_COEF_DEBUG
    cerr << "After WVELCOEF" << endl;
    cerr << "Print WAW" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AW].print(cerr);
    cerr << "Print WAE" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AE].print(cerr);
    cerr << "Print WAN" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AN].print(cerr);
    cerr << "Print WAS" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AS].print(cerr);
    cerr << "Print WAT" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AT].print(cerr);
    cerr << "Print WAB" << endl;
    coeff_vars->wVelocityConvectCoeff[Arches::AB].print(cerr);
    cerr << "Print WAW" << endl;
    coeff_vars->wVelocityCoeff[Arches::AW].print(cerr);
    cerr << "Print WAE" << endl;
    coeff_vars->wVelocityCoeff[Arches::AE].print(cerr);
    cerr << "Print WAN" << endl;
    coeff_vars->wVelocityCoeff[Arches::AN].print(cerr);
    cerr << "Print WAS" << endl;
    coeff_vars->wVelocityCoeff[Arches::AS].print(cerr);
    cerr << "Print WAT" << endl;
    coeff_vars->wVelocityCoeff[Arches::AT].print(cerr);
    cerr << "Print WAB" << endl;
    coeff_vars->wVelocityCoeff[Arches::AB].print(cerr);
#endif
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
				       ArchesVariables* coeff_vars)
{
  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE FORT_PRESSCOEFF" << endl;
  cerr << "Print density:" << endl;
  coeff_vars->density.print(cerr);
  cerr << "Print uVelocity AP:" << endl;
  coeff_vars->uVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print AP - V Vel Coeff: " << endl;
  coeff_vars->vVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print AP - W Vel Coeff: " << endl;
  coeff_vars->wVelocityCoeff[Arches::AP].print(cerr);
  cerr << "Print uVelocity:" << endl;
  coeff_vars->uVelocity.print(cerr);
  cerr << "Print vVelocity: " << endl;
  coeff_vars->vVelocity.print(cerr);
  cerr << "Print wVelocity: " << endl;
  coeff_vars->wVelocity.print(cerr);
#endif

  fort_prescoef(idxLo, idxHi, coeff_vars->density,
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

#ifdef ARCHES_COEF_DEBUG
  cerr << "After PRESSCOEFF" << endl;
  cerr << "Print PAW" << endl;
  coeff_vars->pressCoeff[Arches::AW].print(cerr);
  cerr << "Print PAE" << endl;
  coeff_vars->pressCoeff[Arches::AE].print(cerr);
  cerr << "Print PAN" << endl;
  coeff_vars->pressCoeff[Arches::AN].print(cerr);
  cerr << "Print PAS" << endl;
  coeff_vars->pressCoeff[Arches::AS].print(cerr);
  cerr << "Print PAT" << endl;
  coeff_vars->pressCoeff[Arches::AT].print(cerr);
  cerr << "Print PAB" << endl;
  coeff_vars->pressCoeff[Arches::AB].print(cerr);
#endif
}

//****************************************************************************
// Modify Pressure Stencil for Multimaterial
//****************************************************************************

void
Discretization::mmModifyPressureCoeffs(const ProcessorGroup*,
				      const Patch* patch,
				      ArchesVariables* coeff_vars)

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
			  coeff_vars->voidFraction, valid_lo, valid_hi);
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
				     ArchesVariables* coeff_vars)
{

  // Get the domain size and the patch indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  
#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE SCALARCOEFF for scalar " << index <<" " << endl;
  cerr << "Print Density: " << endl;
  coeff_vars->density.print(cerr);
  cerr << "Print Viscosity: " << endl;
  coeff_vars->viscosity.print(cerr);
  cerr << "Print uVelocity: " << endl;
  coeff_vars->uVelocity.print(cerr);
  cerr << "Print vVelocity: " << endl;
  coeff_vars->vVelocity.print(cerr);
  cerr << "Print wVelocity: " << endl;
  coeff_vars->wVelocity.print(cerr);
#endif

  fort_scalcoef(idxLo, idxHi, coeff_vars->density, coeff_vars->viscosity,
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
		coeff_vars->uVelocity, coeff_vars->vVelocity,
		coeff_vars->wVelocity, cellinfo->sew, cellinfo->sns,
		cellinfo->stb, cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		cellinfo->cnn, cellinfo->csn, cellinfo->css, cellinfo->ctt,
		cellinfo->cbt, cellinfo->cbb, cellinfo->efac,
		cellinfo->wfac,	cellinfo->enfac, cellinfo->sfac,
		cellinfo->tfac, cellinfo->bfac,
		cellinfo->dxpw, cellinfo->dxep, cellinfo->dyps,
		cellinfo->dynp, cellinfo->dzpb, cellinfo->dztp);

#ifdef ARCHES_COEF_DEBUG
  cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
  cerr << "SAE Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AE].print(cerr);
  cerr << "SAW Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AW].print(cerr);
  cerr << "SAN Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AN].print(cerr);
  cerr << "SAS Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AS].print(cerr);
  cerr << "SAT Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AT].print(cerr);
  cerr << "SAB Convect Coeff: " << endl;
  coeff_vars->scalarConvectCoeff[Arches::AB].print(cerr);
  cerr << "SAE Convect Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AE].print(cerr);
  cerr << "SAW  Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AW].print(cerr);
  cerr << "SAN  Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AN].print(cerr);
  cerr << "SAS  Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AS].print(cerr);
  cerr << "SAT  Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AT].print(cerr);
  cerr << "SAB  Coeff: " << endl;
  coeff_vars->scalarCoeff[Arches::AB].print(cerr);
#endif
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

#ifdef ARCHES_COEF_DEBUG
    cerr << "BEFORE Calculate U Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP - U Vel Linear Source for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelLinearSrc)[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    fort_apcalvel(idxLo, idxHi, coeff_vars->uVelocityCoeff[Arches::AP],
		  coeff_vars->uVelocityCoeff[Arches::AE],
		  coeff_vars->uVelocityCoeff[Arches::AW],
		  coeff_vars->uVelocityCoeff[Arches::AN],
		  coeff_vars->uVelocityCoeff[Arches::AS],
		  coeff_vars->uVelocityCoeff[Arches::AT],
		  coeff_vars->uVelocityCoeff[Arches::AB],
		  coeff_vars->uVelLinearSrc);

#ifdef ARCHES_COEF_DEBUG
    cerr << "After UVELCOEF" << endl;
    for(CellIterator iter = patch->getCellIterator();
	!iter.done(); iter++){
      cerr.width(10);
      cerr <<"AP"<< *iter << ": " << (coeff_vars->uVelocityCoeff[Arches::AP])[*iter] << "\n" ; 
    }
    cerr << "AFTER Calculate U Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "AP - U Vel Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AP])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  case Arches::YDIR:
    domLo = coeff_vars->vVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->vVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
    cerr << "BEFORE Calculate V Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP - V Vel Linear Source for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelLinearSrc)[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    fort_apcalvel(idxLo, idxHi, coeff_vars->vVelocityCoeff[Arches::AP],
		  coeff_vars->vVelocityCoeff[Arches::AE],
		  coeff_vars->vVelocityCoeff[Arches::AW],
		  coeff_vars->vVelocityCoeff[Arches::AN],
		  coeff_vars->vVelocityCoeff[Arches::AS],
		  coeff_vars->vVelocityCoeff[Arches::AT],
		  coeff_vars->vVelocityCoeff[Arches::AB],
		  coeff_vars->vVelLinearSrc);

#ifdef ARCHES_COEF_DEBUG
    cerr << "AFTER Calculate V Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "AP - V Vel Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AP])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  case Arches::ZDIR:
    domLo = coeff_vars->wVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->wVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
    cerr << "BEFORE Calculate W Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP - W Vel Linear Source for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelLinearSrc)[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    fort_apcalvel(idxLo, idxHi, coeff_vars->wVelocityCoeff[Arches::AP],
		  coeff_vars->wVelocityCoeff[Arches::AE],
		  coeff_vars->wVelocityCoeff[Arches::AW],
		  coeff_vars->wVelocityCoeff[Arches::AN],
		  coeff_vars->wVelocityCoeff[Arches::AS],
		  coeff_vars->wVelocityCoeff[Arches::AT],
		  coeff_vars->wVelocityCoeff[Arches::AB],
		  coeff_vars->wVelLinearSrc);

#ifdef ARCHES_COEF_DEBUG
    cerr << "AFTER Calculate W Velocity Diagonal :" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "AP - W Vel Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AP])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

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

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE Calculate Pressure Diagonal :" << endl;
  cerr << "Print SP - Pressure Linear source: " << endl;
  coeff_vars->pressLinearSrc.print(cerr);
#endif

  // Calculate the diagonal terms (AP)
  fort_apcal(idxLo, idxHi, coeff_vars->pressCoeff[Arches::AP],
	     coeff_vars->pressCoeff[Arches::AE],
	     coeff_vars->pressCoeff[Arches::AW],
	     coeff_vars->pressCoeff[Arches::AN],
	     coeff_vars->pressCoeff[Arches::AS],
	     coeff_vars->pressCoeff[Arches::AT],
	     coeff_vars->pressCoeff[Arches::AB],
	     coeff_vars->pressLinearSrc);
#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Calculate Pressure Diagonal :" << endl;
  cerr << "Print AP - Pressure Linear source: " << endl;
  coeff_vars->pressCoeff[Arches::AP].print(cerr);
  cerr << "Print nonlinear source: " << endl;
  coeff_vars->pressNonlinearSrc.print(cerr);
#endif

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

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE Calculate Scalar" << index << " Diagonal :" << endl;
  cerr << "Print SSP - Scalar Linear Source" << endl;
  coeff_vars->scalarLinearSrc.print(cerr);
#endif

  fort_apcal(idxLo, idxHi, coeff_vars->scalarCoeff[Arches::AP],
	     coeff_vars->scalarCoeff[Arches::AE],
	     coeff_vars->scalarCoeff[Arches::AW],
	     coeff_vars->scalarCoeff[Arches::AN],
	     coeff_vars->scalarCoeff[Arches::AS],
	     coeff_vars->scalarCoeff[Arches::AT],
	     coeff_vars->scalarCoeff[Arches::AB],
	     coeff_vars->scalarLinearSrc);

#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Calculate Scalar" << index << " Diagonal :" << endl;
  cerr << "SAP - Scalar Coeff " << endl;
  coeff_vars->scalarCoeff[Arches::AP].print(cerr);
#endif

}
