//----- Discretization.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Stencil.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

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
// Velocity stencil weights
//****************************************************************************
void 
Discretization::calculateVelocityCoeff(const ProcessorGroup* pc,
				       const Patch* patch,
				       double delta_t,
				       int index,
				       CellInformation* cellinfo,
				       ArchesVariables* coeff_vars)
{
  // Get the domain size with ghost cells
  IntVector domLoU = coeff_vars->uVelocity.getFortLowIndex();
  IntVector domHiU = coeff_vars->uVelocity.getFortHighIndex();
  IntVector domLoV = coeff_vars->vVelocity.getFortLowIndex();
  IntVector domHiV = coeff_vars->vVelocity.getFortHighIndex();
  IntVector domLoW = coeff_vars->wVelocity.getFortLowIndex();
  IntVector domHiW = coeff_vars->wVelocity.getFortHighIndex();
  IntVector domLo = coeff_vars->viscosity.getFortLowIndex();
  IntVector domHi = coeff_vars->viscosity.getFortHighIndex();
  IntVector domLoeg = coeff_vars->density.getFortLowIndex();
  IntVector domHieg = coeff_vars->density.getFortHighIndex();
  // get domain size without ghost cells
  // using ng for no ghost cell

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

    IntVector domLoUng = coeff_vars->uVelocityCoeff[Arches::AP].
                                              getFortLowIndex();
    IntVector domHiUng = coeff_vars->uVelocityCoeff[Arches::AP].
                                             getFortHighIndex();
    // Get the patch indices
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();
#ifdef ARCHES_COEF_DEBUG
    cerr << "idxLou, idxHiU" << idxLoU << " " << idxHiU << endl;
#endif
    // Calculate the coeffs
    FORT_UVELCOEF(domLoU.get_pointer(), domHiU.get_pointer(),
		  domLoUng.get_pointer(), domHiUng.get_pointer(),
		  idxLoU.get_pointer(), idxHiU.get_pointer(),
		  coeff_vars->uVelocity.getPointer(),
		  coeff_vars->uVelocityConvectCoeff[Arches::AE].getPointer(), 
		  coeff_vars->uVelocityConvectCoeff[Arches::AW].getPointer(), 
		  coeff_vars->uVelocityConvectCoeff[Arches::AN].getPointer(), 
		  coeff_vars->uVelocityConvectCoeff[Arches::AS].getPointer(), 
		  coeff_vars->uVelocityConvectCoeff[Arches::AT].getPointer(), 
		  coeff_vars->uVelocityConvectCoeff[Arches::AB].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AP].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AE].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AW].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AN].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AS].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AT].getPointer(), 
		  coeff_vars->uVelocityCoeff[Arches::AB].getPointer(), 
		  //		  coeff_vars->variableCalledDU.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  coeff_vars->vVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  coeff_vars->wVelocity.getPointer(),
		  domLoeg.get_pointer(), domHieg.get_pointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  coeff_vars->density.getPointer(),
		  coeff_vars->viscosity.getPointer(),
		  &delta_t,
		  cellinfo->ceeu.get_objs(), cellinfo->cweu.get_objs(),
		  cellinfo->cwwu.get_objs(),
		  cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		  cellinfo->css.get_objs(),
		  cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		  cellinfo->cbb.get_objs(),
		  cellinfo->sewu.get_objs(), cellinfo->sew.get_objs(),
		  cellinfo->sns.get_objs(),
		  cellinfo->stb.get_objs(),
		  cellinfo->dxepu.get_objs(), cellinfo->dxpwu.get_objs(),
		  cellinfo->dxpw.get_objs(),
		  cellinfo->dynp.get_objs(), cellinfo->dyps.get_objs(),
		  cellinfo->dztp.get_objs(), cellinfo->dzpb.get_objs(),
		  cellinfo->fac1u.get_objs(), cellinfo->fac2u.get_objs(),
		  cellinfo->fac3u.get_objs(), cellinfo->fac4u.get_objs(),
		  cellinfo->iesdu.get_objs(), cellinfo->iwsdu.get_objs(), 
		  cellinfo->enfac.get_objs(), cellinfo->sfac.get_objs(),
		  cellinfo->tfac.get_objs(), cellinfo->bfac.get_objs());
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

    IntVector domLoVng = coeff_vars->vVelocityCoeff[Arches::AP].
                                             getFortLowIndex();
    IntVector domHiVng = coeff_vars->vVelocityCoeff[Arches::AP].
                                             getFortHighIndex();
    // Get the patch indices
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();

    // Calculate the coeffs
    FORT_VVELCOEF(domLoV.get_pointer(), domHiV.get_pointer(),
		  domLoVng.get_pointer(), domHiVng.get_pointer(),
		  idxLoV.get_pointer(), idxHiV.get_pointer(),
		  coeff_vars->vVelocity.getPointer(),
		  coeff_vars->vVelocityConvectCoeff[Arches::AE].getPointer(), 
		  coeff_vars->vVelocityConvectCoeff[Arches::AW].getPointer(), 
		  coeff_vars->vVelocityConvectCoeff[Arches::AN].getPointer(), 
		  coeff_vars->vVelocityConvectCoeff[Arches::AS].getPointer(), 
		  coeff_vars->vVelocityConvectCoeff[Arches::AT].getPointer(), 
		  coeff_vars->vVelocityConvectCoeff[Arches::AB].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AP].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AE].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AW].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AN].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AS].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AT].getPointer(), 
		  coeff_vars->vVelocityCoeff[Arches::AB].getPointer(), 
		  //		  coeff_vars->variableCalledDV.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  coeff_vars->uVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  coeff_vars->wVelocity.getPointer(),
		  domLoeg.get_pointer(), domHieg.get_pointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  coeff_vars->density.getPointer(),
		  coeff_vars->viscosity.getPointer(),
		  &delta_t,
		  cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(),
		  cellinfo->cww.get_objs(),
		  cellinfo->cnnv.get_objs(), cellinfo->csnv.get_objs(),
		  cellinfo->cssv.get_objs(),
		  cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		  cellinfo->cbb.get_objs(),
		  cellinfo->sew.get_objs(), cellinfo->snsv.get_objs(),
		  cellinfo->sns.get_objs(),
		  cellinfo->stb.get_objs(),
		  cellinfo->dxep.get_objs(), cellinfo->dxpw.get_objs(),
		  cellinfo->dynpv.get_objs(), cellinfo->dypsv.get_objs(),
		  cellinfo->dyps.get_objs(),
		  cellinfo->dztp.get_objs(), cellinfo->dzpb.get_objs(),
		  cellinfo->fac1v.get_objs(), cellinfo->fac2v.get_objs(),
		  cellinfo->fac3v.get_objs(), cellinfo->fac4v.get_objs(),
		  cellinfo->jnsdv.get_objs(), cellinfo->jssdv.get_objs(), 
		  cellinfo->efac.get_objs(), cellinfo->wfac.get_objs(),
		  cellinfo->tfac.get_objs(), cellinfo->bfac.get_objs());
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

    IntVector domLoWng = coeff_vars->wVelocityCoeff[Arches::AP].
                                             getFortLowIndex();
    IntVector domHiWng = coeff_vars->wVelocityCoeff[Arches::AP].
                                             getFortHighIndex();
    // Get the patch indices
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();

    // Calculate the coeffs
    FORT_WVELCOEF(domLoW.get_pointer(), domHiW.get_pointer(),
		  domLoWng.get_pointer(), domHiWng.get_pointer(),
		  idxLoW.get_pointer(), idxHiW.get_pointer(),
		  coeff_vars->wVelocity.getPointer(),
		  coeff_vars->wVelocityConvectCoeff[Arches::AE].getPointer(), 
		  coeff_vars->wVelocityConvectCoeff[Arches::AW].getPointer(), 
		  coeff_vars->wVelocityConvectCoeff[Arches::AN].getPointer(), 
		  coeff_vars->wVelocityConvectCoeff[Arches::AS].getPointer(), 
		  coeff_vars->wVelocityConvectCoeff[Arches::AT].getPointer(), 
		  coeff_vars->wVelocityConvectCoeff[Arches::AB].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AP].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AE].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AW].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AN].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AS].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AT].getPointer(), 
		  coeff_vars->wVelocityCoeff[Arches::AB].getPointer(), 
		  //		  coeff_vars->variableCalledDW.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  coeff_vars->uVelocity.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  coeff_vars->vVelocity.getPointer(),
		  domLoeg.get_pointer(), domHieg.get_pointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  coeff_vars->density.getPointer(),
		  coeff_vars->viscosity.getPointer(),
		  &delta_t,
		  cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(),
		  cellinfo->cww.get_objs(),
		  cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		  cellinfo->css.get_objs(),
		  cellinfo->cttw.get_objs(), cellinfo->cbtw.get_objs(),
		  cellinfo->cbbw.get_objs(),
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stbw.get_objs(), cellinfo->stb.get_objs(),
		  cellinfo->dxep.get_objs(), cellinfo->dxpw.get_objs(),
		  cellinfo->dynp.get_objs(), cellinfo->dyps.get_objs(),
		  cellinfo->dztpw.get_objs(), cellinfo->dzpbw.get_objs(),
		  cellinfo->dzpb.get_objs(),
		  cellinfo->fac1w.get_objs(), cellinfo->fac2w.get_objs(),
		  cellinfo->fac3w.get_objs(), cellinfo->fac4w.get_objs(),
		  cellinfo->ktsdw.get_objs(), cellinfo->kbsdw.get_objs(), 
		  cellinfo->efac.get_objs(), cellinfo->wfac.get_objs(),
		  cellinfo->enfac.get_objs(), cellinfo->sfac.get_objs());
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
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw,
				       double delta_t, 
				       CellInformation* cellinfo,
				       ArchesVariables* coeff_vars)
{
  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->density.getFortLowIndex();
  IntVector domHi = coeff_vars->density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLoU = coeff_vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiU = coeff_vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
  IntVector domLoV = coeff_vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiV = coeff_vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
  IntVector domLoW = coeff_vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiW = coeff_vars->wVelocityCoeff[Arches::AP].getFortHighIndex();
  // no ghost cells
  IntVector domLong = coeff_vars->pressCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = coeff_vars->pressCoeff[Arches::AP].getFortHighIndex();

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

  FORT_PRESSCOEFF(domLo.get_pointer(), domHi.get_pointer(),
		  domLong.get_pointer(), domHing.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  coeff_vars->density.getPointer(),
		  coeff_vars->pressCoeff[Arches::AE].getPointer(), 
		  coeff_vars->pressCoeff[Arches::AW].getPointer(), 
		  coeff_vars->pressCoeff[Arches::AN].getPointer(), 
		  coeff_vars->pressCoeff[Arches::AS].getPointer(), 
		  coeff_vars->pressCoeff[Arches::AT].getPointer(), 
		  coeff_vars->pressCoeff[Arches::AB].getPointer(), 
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  coeff_vars->uVelocityCoeff[Arches::AP].getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  coeff_vars->vVelocityCoeff[Arches::AP].getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  coeff_vars->wVelocityCoeff[Arches::AP].getPointer(),
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
		  cellinfo->stb.get_objs(),
		  cellinfo->sewu.get_objs(), cellinfo->dxep.get_objs(), 
		  cellinfo->dxpw.get_objs(), 
		  cellinfo->snsv.get_objs(), cellinfo->dynp.get_objs(), 
		  cellinfo->dyps.get_objs(), 
		  cellinfo->stbw.get_objs(), cellinfo->dztp.get_objs(), 
		  cellinfo->dzpb.get_objs());

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
// Scalar stencil weights
//****************************************************************************
void 
Discretization::calculateScalarCoeff(const ProcessorGroup* pc,
				     const Patch* patch,
				     double delta_t,
				     int index, 
				     CellInformation* cellinfo,
				     ArchesVariables* coeff_vars)
{

  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->density.getFortLowIndex();
  IntVector domHi = coeff_vars->density.getFortHighIndex();
  IntVector domLoVis = coeff_vars->viscosity.getFortLowIndex();
  IntVector domHiVis = coeff_vars->viscosity.getFortHighIndex();
  IntVector domLong = coeff_vars->scalarNonlinearSrc.getFortLowIndex();
  IntVector domHing = coeff_vars->scalarNonlinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLoU = coeff_vars->uVelocity.getFortLowIndex();
  IntVector domHiU = coeff_vars->uVelocity.getFortHighIndex();
  IntVector domLoV = coeff_vars->vVelocity.getFortLowIndex();
  IntVector domHiV = coeff_vars->vVelocity.getFortHighIndex();
  IntVector domLoW = coeff_vars->wVelocity.getFortLowIndex();
  IntVector domHiW = coeff_vars->wVelocity.getFortHighIndex();
  
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

  FORT_SCALARCOEFF(domLo.get_pointer(), domHi.get_pointer(),
		   domLoVis.get_pointer(), domHiVis.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   coeff_vars->density.getPointer(),
		   coeff_vars->viscosity.getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AE].getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AW].getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AN].getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AS].getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AT].getPointer(), 
		   coeff_vars->scalarCoeff[Arches::AB].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AE].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AW].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AN].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AS].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AT].getPointer(), 
		   coeff_vars->scalarConvectCoeff[Arches::AB].getPointer(), 
		   domLoU.get_pointer(), domHiU.get_pointer(),
		   coeff_vars->uVelocity.getPointer(),
		   domLoV.get_pointer(), domHiV.get_pointer(),
		   coeff_vars->vVelocity.getPointer(),
		   domLoW.get_pointer(), domHiW.get_pointer(),
		   coeff_vars->wVelocity.getPointer(),
		   cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
		   cellinfo->stb.get_objs(),
		   cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(), 
		   cellinfo->cww.get_objs(),
		   cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(), 
		   cellinfo->css.get_objs(),
		   cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(), 
		   cellinfo->cbb.get_objs(),
		   cellinfo->efac.get_objs(), cellinfo->wfac.get_objs(),
		   cellinfo->enfac.get_objs(), cellinfo->sfac.get_objs(),
		   cellinfo->tfac.get_objs(), cellinfo->bfac.get_objs(),
		   cellinfo->dxpw.get_objs(), cellinfo->dxep.get_objs(),
		   cellinfo->dyps.get_objs(), cellinfo->dynp.get_objs(),
		   cellinfo->dzpb.get_objs(), cellinfo->dztp.get_objs());

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

    FORT_APCAL_VEL(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   coeff_vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AE].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AW].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AN].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AS].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AT].getPointer(), 
		   coeff_vars->uVelocityCoeff[Arches::AB].getPointer(),
		   coeff_vars->uVelLinearSrc.getPointer());

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

    FORT_APCAL_VEL(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   coeff_vars->vVelocityCoeff[Arches::AP].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AE].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AW].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AN].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AS].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AT].getPointer(), 
		   coeff_vars->vVelocityCoeff[Arches::AB].getPointer(),
		   coeff_vars->vVelLinearSrc.getPointer());

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

    FORT_APCAL_VEL(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   coeff_vars->wVelocityCoeff[Arches::AP].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AE].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AW].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AN].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AS].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AT].getPointer(), 
		   coeff_vars->wVelocityCoeff[Arches::AB].getPointer(),
		   coeff_vars->wVelLinearSrc.getPointer());

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
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw,
				       ArchesVariables* coeff_vars) 
{
  
  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->pressLinearSrc.getFortLowIndex();
  IntVector domHi = coeff_vars->pressLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE Calculate Pressure Diagonal :" << endl;
  cerr << "Print SP - Pressure Linear source: " << endl;
  coeff_vars->pressLinearSrc.print(cerr);
#endif

  // Calculate the diagonal terms (AP)
  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     coeff_vars->pressCoeff[Arches::AP].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AE].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AW].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AN].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AS].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AT].getPointer(), 
	     coeff_vars->pressCoeff[Arches::AB].getPointer(),
	     coeff_vars->pressLinearSrc.getPointer());

#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Calculate Pressure Diagonal :" << endl;
  cerr << "Print AP - Pressure Linear source: " << endl;
  coeff_vars->pressCoeff[Arches::AP].print(cerr);
#endif

}

//****************************************************************************
// Scalar diagonal
//****************************************************************************
void 
Discretization::calculateScalarDiagonal(const ProcessorGroup*,
					const Patch* patch,
					int index,
					ArchesVariables* coeff_vars)
{
  
  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->scalarLinearSrc.getFortLowIndex();
  IntVector domHi = coeff_vars->scalarLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE Calculate Scalar" << index << " Diagonal :" << endl;
  cerr << "Print SSP - Scalar Linear Source" << endl;
  coeff_vars->scalarLinearSrc.print(cerr);
#endif

  FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     coeff_vars->scalarCoeff[Arches::AP].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AE].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AW].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AN].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AS].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AT].getPointer(), 
	     coeff_vars->scalarCoeff[Arches::AB].getPointer(),
	     coeff_vars->scalarLinearSrc.getPointer());

#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Calculate Scalar" << index << " Diagonal :" << endl;
  cerr << "SAP - Scalar Coeff " << endl;
  coeff_vars->scalarCoeff[Arches::AP].print(cerr);
#endif

}
