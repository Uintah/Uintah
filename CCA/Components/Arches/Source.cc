//----- Source.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;

//****************************************************************************
// Constructor for Source
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
Source::calculateVelocitySource(const ProcessorGroup* ,
				const Patch* patch,
				double delta_t,
				int index,
				CellInformation* cellinfo,
				ArchesVariables* vars)
{
  
  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);
  // get iref, jref, kref and ref density by broadcasting from a patch that contains
  // iref, jref and kref

  //  double den_ref = vars->density[IntVector(3,3,3)]; // change it!!! use ipref, jpref and kpref
  //  double den_ref = 1.184344; // change it!!! use ipref, jpref and kpref
#ifdef ARCHES_MOM_DEBUG
  cerr << " ref_ density" << den_ref << endl;
#endif
  // Get the patch and variable indices
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();
  IntVector domLoeg = vars->density.getFortLowIndex();
  IntVector domHieg = vars->density.getFortHighIndex();
  IntVector domLodref = vars->denRefArray.getFortLowIndex();
  IntVector domHidref = vars->denRefArray.getFortHighIndex();
  IntVector domLo = vars->viscosity.getFortLowIndex();
  IntVector domHi = vars->viscosity.getFortHighIndex();
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  // get domain size without ghost cells
  // using ng for no ghost cell
  IntVector domLoUng;
  IntVector domHiUng;
  IntVector domLoVng;
  IntVector domHiVng;
  IntVector domLoWng;
  IntVector domHiWng;
  IntVector domLong = vars->old_density.getFortLowIndex();
  IntVector domHing = vars->old_density.getFortHighIndex();

#ifdef ARCHES_MOM_DEBUG
  for (int iii = 0; iii < (domHi.z()-domLo.z()); iii++)
    std::cerr << cellinfo->ktsdw[iii] << " " << cellinfo->kbsdw[iii] << endl;
  for (int iii = 0; iii < (domHi.y()-domLo.y()); iii++)
    std::cerr << cellinfo->jnsdv[iii] << " " << cellinfo->jssdv[iii] << endl;
#endif
  
  switch(index) {
  case Arches::XDIR:
    domLoUng = vars->uVelLinearSrc.getFortLowIndex();
    domHiUng = vars->uVelLinearSrc.getFortHighIndex();

    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_UVELSOURCE(domLoU.get_pointer(), domHiU.get_pointer(),
		    domLoUng.get_pointer(), domHiUng.get_pointer(),
		    idxLoU.get_pointer(), idxHiU.get_pointer(),
		    vars->uVelocity.getPointer(),
		    vars->old_uVelocity.getPointer(),
		    vars->uVelNonlinearSrc.getPointer(), 
		    vars->uVelLinearSrc.getPointer(), 
		    domLoV.get_pointer(), domHiV.get_pointer(),
		    vars->vVelocity.getPointer(), 
		    domLoW.get_pointer(), domHiW.get_pointer(),
		    vars->wVelocity.getPointer(), 
		    domLoeg.get_pointer(), domHieg.get_pointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->density.getPointer(),
		    vars->viscosity.getPointer(), 
		    domLong.get_pointer(), domHing.get_pointer(),
		    vars->old_density.getPointer(),
		    domLodref.get_pointer(), domHidref.get_pointer(),
		    vars->denRefArray.getPointer(),
		    &gravity, &delta_t, 
		    cellinfo->ceeu.get_objs(), cellinfo->cweu.get_objs(), 
		    cellinfo->cwwu.get_objs(),
		    cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		    cellinfo->css.get_objs(),
		    cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		    cellinfo->cbb.get_objs(),
		    cellinfo->sewu.get_objs(), cellinfo->sew.get_objs(),
		    cellinfo->sns.get_objs(),
		    cellinfo->stb.get_objs(),
		    cellinfo->dxpw.get_objs(),
		    cellinfo->fac1u.get_objs(), cellinfo->fac2u.get_objs(),
		    cellinfo->fac3u.get_objs(), 
		    cellinfo->fac4u.get_objs(),
		    cellinfo->iesdu.get_objs(), cellinfo->iwsdu.get_objs());

#ifdef ARCHES_SRC_DEBUG
    cerr << "patch: " << *patch << '\n';
    cerr << "dom: " << domLoU << ", " << domHiU << '\n';
    Array3Window<double>* win = vars->uVelNonlinearSrc.getWindow();
    cerr << "usrc: " << win->getLowIndex() << ", " << win->getHighIndex() << ", " << win->getOffset() << '\n';
    cerr << "AFTER U Velocity Source" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "SU for U velocity for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->uVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER U Velocity Source" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "SP for U velocity for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->uVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  case Arches::YDIR:
    domLoVng = vars->vVelLinearSrc.getFortLowIndex();
    domHiVng = vars->vVelLinearSrc.getFortHighIndex();

    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_VVELSOURCE(domLoV.get_pointer(), domHiV.get_pointer(),
		    domLoVng.get_pointer(), domHiVng.get_pointer(),
		    idxLoV.get_pointer(), idxHiV.get_pointer(),
		    vars->vVelocity.getPointer(),
		    vars->old_vVelocity.getPointer(),
		    vars->vVelNonlinearSrc.getPointer(), 
		    vars->vVelLinearSrc.getPointer(), 
		    domLoU.get_pointer(), domHiU.get_pointer(),
		    vars->uVelocity.getPointer(), 
		    domLoW.get_pointer(), domHiW.get_pointer(),
		    vars->wVelocity.getPointer(), 
		    domLoeg.get_pointer(), domHieg.get_pointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->density.getPointer(),
		    vars->viscosity.getPointer(), 
		    domLong.get_pointer(), domHing.get_pointer(),
		    vars->old_density.getPointer(),
		    domLodref.get_pointer(), domHidref.get_pointer(),
		    vars->denRefArray.getPointer(),
		    &gravity, &delta_t,  
		    cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(), 
		    cellinfo->cww.get_objs(),
		    cellinfo->cnnv.get_objs(), cellinfo->csnv.get_objs(),
		    cellinfo->cssv.get_objs(),
		    cellinfo->ctt.get_objs(), cellinfo->cbt.get_objs(),
		    cellinfo->cbb.get_objs(),
		    cellinfo->sew.get_objs(), 
		    cellinfo->snsv.get_objs(), cellinfo->sns.get_objs(),
		    cellinfo->stb.get_objs(),
		    cellinfo->dyps.get_objs(),
		    cellinfo->fac1v.get_objs(), cellinfo->fac2v.get_objs(),
		    cellinfo->fac3v.get_objs(), 
		    cellinfo->fac4v.get_objs(),
		    cellinfo->jnsdv.get_objs(), cellinfo->jssdv.get_objs()); 

#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER V Velocity Source" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "SU for V velocity for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->vVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER V Velocity Source" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "SP for V velocity for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->vVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  case Arches::ZDIR:

    domLoWng = vars->wVelLinearSrc.getFortLowIndex();
    domHiWng = vars->wVelLinearSrc.getFortHighIndex();
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_WVELSOURCE(domLoW.get_pointer(), domHiW.get_pointer(),
		    domLoWng.get_pointer(), domHiWng.get_pointer(),
		    idxLoW.get_pointer(), idxHiW.get_pointer(),
		    vars->wVelocity.getPointer(),
		    vars->old_wVelocity.getPointer(),
		    vars->wVelNonlinearSrc.getPointer(), 
		    vars->wVelLinearSrc.getPointer(), 
		    domLoU.get_pointer(), domHiU.get_pointer(),
		    vars->uVelocity.getPointer(), 
		    domLoV.get_pointer(), domHiV.get_pointer(),
		    vars->vVelocity.getPointer(), 
		    domLoeg.get_pointer(), domHieg.get_pointer(),
		    domLo.get_pointer(), domHi.get_pointer(),
		    vars->density.getPointer(),
		    vars->viscosity.getPointer(), 
		    domLong.get_pointer(), domHing.get_pointer(),
		    vars->old_density.getPointer(),
		    domLodref.get_pointer(), domHidref.get_pointer(),
		    vars->denRefArray.getPointer(),
		    &gravity, &delta_t,  
		    cellinfo->cee.get_objs(), cellinfo->cwe.get_objs(), 
		    cellinfo->cww.get_objs(),
		    cellinfo->cnn.get_objs(), cellinfo->csn.get_objs(),
		    cellinfo->css.get_objs(),
		    cellinfo->cttw.get_objs(), cellinfo->cbtw.get_objs(),
		    cellinfo->cbbw.get_objs(),
		    cellinfo->sew.get_objs(), 
		    cellinfo->sns.get_objs(),
		    cellinfo->stbw.get_objs(), cellinfo->stb.get_objs(),
		    cellinfo->dzpb.get_objs(), 
		    cellinfo->fac1w.get_objs(), cellinfo->fac2w.get_objs(),
		    cellinfo->fac3w.get_objs(), 
		    cellinfo->fac4w.get_objs(),
		    cellinfo->ktsdw.get_objs(), cellinfo->kbsdw.get_objs()); 

#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER W Velocity Source" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "SU for W velocity for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->wVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER W Velocity Source" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "SP for W velocity for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->wVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelSrc");
  }


#ifdef MAY_BE_USEFUL_LATER  
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

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  // inputs : 
  // outputs : 
  //  d_turbModel->calcVelocitySource(pc, patch, old_dw, new_dw, index);
}

//****************************************************************************
// Pressure source calculation
//****************************************************************************
void 
Source::calculatePressureSource(const ProcessorGroup*,
				const Patch* patch,
				double delta_t,
				CellInformation* cellinfo,
				ArchesVariables* vars)
{

  // Get the patch and variable indices
  IntVector domLo = vars->density.getFortLowIndex();
  IntVector domHi = vars->density.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLoU = vars->uVelocity.getFortLowIndex();
  IntVector domHiU = vars->uVelocity.getFortHighIndex();
  IntVector domLoV = vars->vVelocity.getFortLowIndex();
  IntVector domHiV = vars->vVelocity.getFortHighIndex();
  IntVector domLoW = vars->wVelocity.getFortLowIndex();
  IntVector domHiW = vars->wVelocity.getFortHighIndex();
  // Get the patch and variable indices
  IntVector domLong = vars->pressCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = vars->pressCoeff[Arches::AP].getFortHighIndex();
  IntVector domLoUng = vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiUng = vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
  IntVector domLoVng = vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiVng = vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
  IntVector domLoWng = vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
  IntVector domHiWng = vars->wVelocityCoeff[Arches::AP].getFortHighIndex();

  //fortran call ** WARNING ** ffield = -1
  int ffield = -1;
  FORT_PRESSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		  domLong.get_pointer(), domHing.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->pressNonlinearSrc.getPointer(),
		  vars->pressLinearSrc.getPointer(),
		  vars->density.getPointer(), vars->old_density.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  domLoUng.get_pointer(), domHiUng.get_pointer(),
		  vars->uVelocity.getPointer(), 
		  vars->uVelocityCoeff[Arches::AP].getPointer(),
		  vars->uVelocityCoeff[Arches::AE].getPointer(),
		  vars->uVelocityCoeff[Arches::AW].getPointer(),
		  vars->uVelocityCoeff[Arches::AN].getPointer(),
		  vars->uVelocityCoeff[Arches::AS].getPointer(),
		  vars->uVelocityCoeff[Arches::AT].getPointer(),
		  vars->uVelocityCoeff[Arches::AB].getPointer(),
		  vars->uVelNonlinearSrc.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  domLoVng.get_pointer(), domHiVng.get_pointer(),
		  vars->vVelocity.getPointer(), 
		  vars->vVelocityCoeff[Arches::AP].getPointer(),
		  vars->vVelocityCoeff[Arches::AE].getPointer(),
		  vars->vVelocityCoeff[Arches::AW].getPointer(),
		  vars->vVelocityCoeff[Arches::AN].getPointer(),
		  vars->vVelocityCoeff[Arches::AS].getPointer(),
		  vars->vVelocityCoeff[Arches::AT].getPointer(),
		  vars->vVelocityCoeff[Arches::AB].getPointer(),
		  vars->vVelNonlinearSrc.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  domLoWng.get_pointer(), domHiWng.get_pointer(),
		  vars->wVelocity.getPointer(), 
		  vars->wVelocityCoeff[Arches::AP].getPointer(),
		  vars->wVelocityCoeff[Arches::AE].getPointer(),
		  vars->wVelocityCoeff[Arches::AW].getPointer(),
		  vars->wVelocityCoeff[Arches::AN].getPointer(),
		  vars->wVelocityCoeff[Arches::AS].getPointer(),
		  vars->wVelocityCoeff[Arches::AT].getPointer(),
		  vars->wVelocityCoeff[Arches::AB].getPointer(),
		  vars->wVelNonlinearSrc.getPointer(),
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
		  cellinfo->stb.get_objs(),
		  vars->cellType.getPointer(), &ffield, &delta_t);

#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER Calculate Pressure Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SU for Pressure for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->pressNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER Calculate Pressure Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP for Pressure for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->pressLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

}

//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateScalarSource(const ProcessorGroup*,
			      const Patch* patch,
			      double delta_t,
			      int index, 
			      CellInformation* cellinfo,
			      ArchesVariables* vars) 
{

  // Get the patch and variable indices
  int numGhost = 1;
  IntVector domLo = patch->getGhostCellLowIndex(numGhost);
  IntVector domHi = patch->getGhostCellHighIndex(numGhost) -
                                               IntVector(1,1,1);
  IntVector domLong = vars->old_scalar.getFortLowIndex();
  IntVector domHing = vars->old_scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_SCALARSOURCE(domLo.get_pointer(), domHi.get_pointer(),
		    domLong.get_pointer(), domHing.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->scalarLinearSrc.getPointer(),
		    vars->scalarNonlinearSrc.getPointer(),
		    vars->old_density.getPointer(),
		    vars->old_scalar.getPointer(),
		    cellinfo->sew.get_objs(), cellinfo->sns.get_objs(), 
		    cellinfo->stb.get_objs(),
		    &delta_t);

#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER Calculate Scalar Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SU for Scalar " << index << " for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER Calculate Scalar Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP for Scalar " << index << " for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->scalarLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
}

//****************************************************************************
// Calls Fortran MASCAL
//****************************************************************************
void 
Source::modifyVelMassSource(const ProcessorGroup* ,
			    const Patch* patch,
			    double delta_t, 
			    int index,
			    ArchesVariables* vars)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;
  IntVector domLong;
  IntVector domHing;
  switch(index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    // no ghost cell
    domLong = vars->uVelLinearSrc.getFortLowIndex();
    domHing = vars->uVelLinearSrc.getFortHighIndex();
    FORT_MASCAL(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->uVelocity.getPointer(),
		vars->uVelocityCoeff[Arches::AE].getPointer(),
		vars->uVelocityCoeff[Arches::AW].getPointer(),
		vars->uVelocityCoeff[Arches::AN].getPointer(),
		vars->uVelocityCoeff[Arches::AS].getPointer(),
		vars->uVelocityCoeff[Arches::AT].getPointer(),
		vars->uVelocityCoeff[Arches::AB].getPointer(),
		vars->uVelNonlinearSrc.getPointer(), 
		vars->uVelLinearSrc.getPointer(), 
		vars->uVelocityConvectCoeff[Arches::AE].getPointer(),
		vars->uVelocityConvectCoeff[Arches::AW].getPointer(),
		vars->uVelocityConvectCoeff[Arches::AN].getPointer(),
		vars->uVelocityConvectCoeff[Arches::AS].getPointer(),
		vars->uVelocityConvectCoeff[Arches::AT].getPointer(),
		vars->uVelocityConvectCoeff[Arches::AB].getPointer());
#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SU for U velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->uVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP for U velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->uVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
    break;
  case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    // no ghost cell
    domLong = vars->vVelLinearSrc.getFortLowIndex();
    domHing = vars->vVelLinearSrc.getFortHighIndex();
    FORT_MASCAL(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->vVelocity.getPointer(),
		vars->vVelocityCoeff[Arches::AE].getPointer(),
		vars->vVelocityCoeff[Arches::AW].getPointer(),
		vars->vVelocityCoeff[Arches::AN].getPointer(),
		vars->vVelocityCoeff[Arches::AS].getPointer(),
		vars->vVelocityCoeff[Arches::AT].getPointer(),
		vars->vVelocityCoeff[Arches::AB].getPointer(),
		vars->vVelNonlinearSrc.getPointer(), 
		vars->vVelLinearSrc.getPointer(), 
		vars->vVelocityConvectCoeff[Arches::AE].getPointer(),
		vars->vVelocityConvectCoeff[Arches::AW].getPointer(),
		vars->vVelocityConvectCoeff[Arches::AN].getPointer(),
		vars->vVelocityConvectCoeff[Arches::AS].getPointer(),
		vars->vVelocityConvectCoeff[Arches::AT].getPointer(),
		vars->vVelocityConvectCoeff[Arches::AB].getPointer());
#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SU for V velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->vVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP for V velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->vVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
    break;
  case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    // no ghost cell
    domLong = vars->wVelLinearSrc.getFortLowIndex();
    domHing = vars->wVelLinearSrc.getFortHighIndex();
    FORT_MASCAL(domLo.get_pointer(), domHi.get_pointer(),
		domLong.get_pointer(), domHing.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->wVelocity.getPointer(),
		vars->wVelocityCoeff[Arches::AE].getPointer(),
		vars->wVelocityCoeff[Arches::AW].getPointer(),
		vars->wVelocityCoeff[Arches::AN].getPointer(),
		vars->wVelocityCoeff[Arches::AS].getPointer(),
		vars->wVelocityCoeff[Arches::AT].getPointer(),
		vars->wVelocityCoeff[Arches::AB].getPointer(),
		vars->wVelNonlinearSrc.getPointer(), 
		vars->wVelLinearSrc.getPointer(), 
		vars->wVelocityConvectCoeff[Arches::AE].getPointer(),
		vars->wVelocityConvectCoeff[Arches::AW].getPointer(),
		vars->wVelocityConvectCoeff[Arches::AN].getPointer(),
		vars->wVelocityConvectCoeff[Arches::AS].getPointer(),
		vars->wVelocityConvectCoeff[Arches::AT].getPointer(),
		vars->wVelocityConvectCoeff[Arches::AB].getPointer());
#ifdef ARCHES_SRC_DEBUG
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SU for W velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->wVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "AFTER Modify Velocity Mass Source" << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP for W velocity for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << vars->wVelLinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelMassSrc");
  }
}

//****************************************************************************
// Documentation here
// FORT_MASCAL
//****************************************************************************
void 
Source::modifyScalarMassSource(const ProcessorGroup* ,
			       const Patch* patch,
			       double delta_t, 
			       int index, ArchesVariables* vars)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  FORT_MASCALSCALAR(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->scalar.getPointer(),
		    vars->scalarCoeff[Arches::AE].getPointer(),
		    vars->scalarCoeff[Arches::AW].getPointer(),
		    vars->scalarCoeff[Arches::AN].getPointer(),
		    vars->scalarCoeff[Arches::AS].getPointer(),
		    vars->scalarCoeff[Arches::AT].getPointer(),
		    vars->scalarCoeff[Arches::AB].getPointer(),
		    vars->scalarNonlinearSrc.getPointer(), 
		    vars->scalarLinearSrc.getPointer(), 
		    vars->scalarConvectCoeff[Arches::AE].getPointer(),
		    vars->scalarConvectCoeff[Arches::AW].getPointer(),
		    vars->scalarConvectCoeff[Arches::AN].getPointer(),
		    vars->scalarConvectCoeff[Arches::AS].getPointer(),
		    vars->scalarConvectCoeff[Arches::AT].getPointer(),
		    vars->scalarConvectCoeff[Arches::AB].getPointer());
}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::addPressureSource(const ProcessorGroup* ,
			  const Patch* patch ,
			  double delta_t,
			  int index,
			  CellInformation* cellinfo,			  
			  ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector domLong = vars->old_density.getFortLowIndex();
  IntVector domHing = vars->old_density.getFortHighIndex();
#ifdef ARCHES_SOURCE_DEBUG
  if (patch->containsCell(IntVector(2,3,3))) {
    cerr << "[2,3,3] press" << vars->pressure[IntVector(2,3,3)] << " " <<
      vars->pressure[IntVector(1,3,3)] << endl;
  }
#endif
  int ioff, joff, koff;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->uVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();

    ioff = 1;
    joff = 0;
    koff = 0;
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sewu[colX]*cellInfo->sns[colY]*
	                         cellInfo->stb[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->uVelNonlinearSrc.getPointer(), 
		      vars->uVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer(),
		      cellinfo->dxpw.get_objs());
#endif    
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->uVelNonlinearSrc.getPointer(), 
		      vars->uVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      cellinfo->sewu.get_objs(), 
		      cellinfo->sns.get_objs(),
		      cellinfo->stb.get_objs(),
		      cellinfo->dxpw.get_objs());
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->vVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0;
    joff = 1;
    koff = 0;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sew[colX]*cellInfo->snsv[colY]*
	                         cellInfo->stb[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->vVelNonlinearSrc.getPointer(), 
		      vars->vVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer(),
		      cellinfo->dyps.get_objs());
#endif    

    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->vVelNonlinearSrc.getPointer(), 
		      vars->vVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      cellinfo->sew.get_objs(), 
		      cellinfo->snsv.get_objs(), 
		      cellinfo->stb.get_objs(),
		      cellinfo->dyps.get_objs());
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->wVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0;
    joff = 0;
    koff = 1;
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sew[colX]*cellInfo->sns[colY]*
	                         cellInfo->stbw[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    // set bulkvolume to zero if i,j,k or i-1,j,k equals to multimaterial wall
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->wVelNonlinearSrc.getPointer(), 
		      vars->wVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer();
		      cellinfo->dzpb.get_objs());

#endif    
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->wVelNonlinearSrc.getPointer(), 
		      vars->wVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      cellinfo->sew.get_objs(), 
		      cellinfo->sns.get_objs(),
		      cellinfo->stbw.get_objs(),
		      cellinfo->dzpb.get_objs());
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcPressGrad");
  }
}


//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::addTransMomSource(const ProcessorGroup* ,
			  const Patch* patch ,
			  double delta_t,
			  int index,
			  CellInformation* cellinfo,			  
			  ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector domLong = vars->old_density.getFortLowIndex();
  IntVector domHing = vars->old_density.getFortHighIndex();
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->uVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    FORT_ADDTRANSSRC(domLoU.get_pointer(), domHiU.get_pointer(),
		     domLoUng.get_pointer(), domHiUng.get_pointer(),
		     idxLoU.get_pointer(), idxHiU.get_pointer(),
		     vars->uVelocity.getPointer(),
		     vars->uVelNonlinearSrc.getPointer(), 
		     vars->uVelocityCoeff[Arches::AP].getPointer(),
		     domLo.get_pointer(), domHi.get_pointer(),
		     domLong.get_pointer(), domHing.get_pointer(),
		     vars->old_density.getPointer(),
		     &delta_t,
		     cellinfo->sewu.get_objs(), 
		     cellinfo->sns.get_objs(),
		     cellinfo->stb.get_objs());
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->vVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    FORT_ADDTRANSSRC(domLoU.get_pointer(), domHiU.get_pointer(),
		     domLoUng.get_pointer(), domHiUng.get_pointer(),
		     idxLoU.get_pointer(), idxHiU.get_pointer(),
		     vars->vVelocity.getPointer(),
		     vars->vVelNonlinearSrc.getPointer(), 
		     vars->vVelocityCoeff[Arches::AP].getPointer(),
		     domLo.get_pointer(), domHi.get_pointer(),
		     domLong.get_pointer(), domHing.get_pointer(),
		     vars->old_density.getPointer(),
		     &delta_t, 
		     cellinfo->sew.get_objs(), 
		     cellinfo->snsv.get_objs(), 
		     cellinfo->stb.get_objs());
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->wVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    FORT_ADDTRANSSRC(domLoU.get_pointer(), domHiU.get_pointer(),
		     domLoUng.get_pointer(), domHiUng.get_pointer(),
		     idxLoU.get_pointer(), idxHiU.get_pointer(),
		     vars->wVelocity.getPointer(),
		     vars->wVelNonlinearSrc.getPointer(), 
		     vars->wVelocityCoeff[Arches::AP].getPointer(),
		     domLo.get_pointer(), domHi.get_pointer(),
		     domLong.get_pointer(), domHing.get_pointer(),
		     vars->old_density.getPointer(),
		     &delta_t, 
		     cellinfo->sew.get_objs(), 
		     cellinfo->sns.get_objs(),
		     cellinfo->stbw.get_objs());
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcTrabsSource");
  }
}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::computePressureSource(const ProcessorGroup* ,
			      const Patch* patch ,
			      int index,
			      CellInformation* cellinfo,			  
			      ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
#ifdef ARCHES_SOURCE_DEBUG
  if (patch->containsCell(IntVector(2,3,3))) {
    cerr << "[2,3,3] press" << vars->pressure[IntVector(2,3,3)] << " " <<
      vars->pressure[IntVector(1,3,3)] << endl;
  }
#endif
  int ioff, joff, koff;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->pressGradUSu.getFortLowIndex();
    domHiUng = vars->pressGradUSu.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();

    ioff = 1;
    joff = 0;
    koff = 0;
    FORT_CALCPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		       domLoUng.get_pointer(), domHiUng.get_pointer(),
		       idxLoU.get_pointer(), idxHiU.get_pointer(),
		       vars->uVelocity.getPointer(),
		       vars->pressGradUSu.getPointer(), 
		       domLo.get_pointer(), domHi.get_pointer(),
		       vars->pressure.getPointer(),
		       &ioff, &joff, &koff,
		       cellinfo->sewu.get_objs(), 
		       cellinfo->sns.get_objs(),
		       cellinfo->stb.get_objs(),
		       cellinfo->dxpw.get_objs());
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->pressGradVSu.getFortLowIndex();
    domHiUng = vars->pressGradVSu.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0;
    joff = 1;
    koff = 0;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref

    FORT_CALCPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		       domLoUng.get_pointer(), domHiUng.get_pointer(),
		       idxLoU.get_pointer(), idxHiU.get_pointer(),
		       vars->vVelocity.getPointer(),
		       vars->pressGradVSu.getPointer(), 
		       domLo.get_pointer(), domHi.get_pointer(),
		       vars->pressure.getPointer(),
		       &ioff, &joff, &koff,
		       cellinfo->sew.get_objs(), 
		       cellinfo->snsv.get_objs(), 
		       cellinfo->stb.get_objs(),
		       cellinfo->dyps.get_objs());
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->pressGradWSu.getFortLowIndex();
    domHiUng = vars->pressGradWSu.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0;
    joff = 0;
    koff = 1;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    FORT_CALCPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		       domLoUng.get_pointer(), domHiUng.get_pointer(),
		       idxLoU.get_pointer(), idxHiU.get_pointer(),
		       vars->wVelocity.getPointer(),
		       vars->pressGradWSu.getPointer(), 
		       domLo.get_pointer(), domHi.get_pointer(),
		       vars->pressure.getPointer(),
		       &ioff, &joff, &koff,
		       cellinfo->sew.get_objs(), 
		       cellinfo->sns.get_objs(),
		       cellinfo->stbw.get_objs(),
		       cellinfo->dzpb.get_objs());
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcPressGrad");
  }
}


////////////////////////////////////////////////////////////////////////
// Add multimaterial source term
void 
Source::computemmMomentumSource(const ProcessorGroup* pc,
				const Patch* patch,
				int index,
				CellInformation* cellinfo,
				ArchesVariables* vars) {
  IntVector idxLoU;
  IntVector idxHiU;
  // for no ghost cells
  IntVector domLoUng;
  IntVector domHiUng;
  IntVector domLo;
  IntVector domHi;
  
  switch(index) {
  case 1:
  // Get the low and high index for the patch and the variables
  idxLoU = patch->getSFCXFORTLowIndex();
  idxHiU = patch->getSFCXFORTHighIndex();
  // for no ghost cells
  domLoUng = vars->uVelLinearSrc.getFortLowIndex();
  domHiUng = vars->uVelLinearSrc.getFortHighIndex();
  domLo = vars->mmuVelSu.getFortLowIndex();
  domHi = vars->mmuVelSu.getFortHighIndex();
  

  FORT_MMMOMSRC(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		idxLoU.get_pointer(), idxHiU.get_pointer(),
		vars->uVelNonlinearSrc.getPointer(), 
		vars->uVelLinearSrc.getPointer(),
		domLo.get_pointer(), domHi.get_pointer(),
		vars->mmuVelSu.getPointer(),
		vars->mmuVelSp.getPointer());
  break;
  case 2:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
  // for no ghost cells
    domLoUng = vars->vVelLinearSrc.getFortLowIndex();
    domHiUng = vars->vVelLinearSrc.getFortHighIndex();
    domLo = vars->mmvVelSu.getFortLowIndex();
    domHi = vars->mmvVelSu.getFortHighIndex();
  
    FORT_MMMOMSRC(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		  idxLoU.get_pointer(), idxHiU.get_pointer(),
		  vars->vVelNonlinearSrc.getPointer(), 
		  vars->vVelLinearSrc.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  vars->mmvVelSu.getPointer(),
		  vars->mmvVelSp.getPointer());
    break;
  case 3:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
  // for no ghost cells
    domLoUng = vars->wVelLinearSrc.getFortLowIndex();
    domHiUng = vars->wVelLinearSrc.getFortHighIndex();
    domLo = vars->mmwVelSu.getFortLowIndex();
    domHi = vars->mmwVelSu.getFortHighIndex();
  
    FORT_MMMOMSRC(domLoUng.get_pointer(), domHiUng.get_pointer(), 
		  idxLoU.get_pointer(), idxHiU.get_pointer(),
		  vars->wVelNonlinearSrc.getPointer(), 
		  vars->wVelLinearSrc.getPointer(),
		  domLo.get_pointer(), domHi.get_pointer(),
		  vars->mmwVelSu.getPointer(),
		  vars->mmwVelSp.getPointer());
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }

  // add in su and sp terms from multimaterial based on index
}
