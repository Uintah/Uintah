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
				       DataWarehouseP&,
				       DataWarehouseP&,
				       double delta_t,
				       int index,
				       int eqnType, CellInformation* cellinfo,
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Density for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->density[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE VELCOEF" << endl;

  cerr << "BEFORE VELCOEF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Density for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->density[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE VELCOEF" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "U Velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE VELCOEF" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "V Velocity for ii = " << ii << endl;
    for (int jj = domLoV.x(); jj <= domHiV.x(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE VELCOEF" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "W Velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
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
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AE Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AW Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AN Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AS Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AT Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AB Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityConvectCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AE Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AW Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AN Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AS Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AT Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After UVELCOEF" << endl;
    for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
      cerr << "U Vel AB Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
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
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AE Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AW Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AN Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AS Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AT Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AB Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityConvectCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AE Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AW Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AN Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AS Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AT Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After VVELCOEF" << endl;
    for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
      cerr << "V Vel AB Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
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
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AE Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AW Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AN Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AS Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AT Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AB Convection Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityConvectCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AE Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AW Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AN Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AS Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AT Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After WVELCOEF" << endl;
    for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
      cerr << "W Vel AB Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
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
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Density for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->density[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE FORT_PRESSCOEFF" << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "AP - U Vel Coeff for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->uVelocityCoeff[Arches::AP])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE FORT_PRESSCOEFF" << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "AP - V Vel Coeff for ii = " << ii << endl;
    for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->vVelocityCoeff[Arches::AP])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE FORT_PRESSCOEFF" << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "AP - W Vel Coeff for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->wVelocityCoeff[Arches::AP])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AE Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AE])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "After PRESSCOEFF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AW Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AW])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "After PRESSCOEFF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AN Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AN])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "After PRESSCOEFF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AS Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AS])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "After PRESSCOEFF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AT Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AT])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "After PRESSCOEFF" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << " Pressure AB Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AB])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
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
				     int index, 
				     CellInformation* cellinfo,
				     ArchesVariables* coeff_vars)
{

  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->density.getFortLowIndex();
  IntVector domHi = coeff_vars->density.getFortHighIndex();
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Density for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->density[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE SCALARCOEFF for scalar " << index <<" " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "Viscosity for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->viscosity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE SCALARCOEFF for scalar " << index <<" " << endl;
  for (int ii = domLoU.x(); ii <= domHiU.x(); ii++) {
    cerr << "U Velocity for ii = " << ii << endl;
    for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
      for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE SCALARCOEFF for scalar " << index <<" " << endl;
  for (int ii = domLoV.x(); ii <= domHiV.x(); ii++) {
    cerr << "V Velocity for ii = " << ii << endl;
    for (int jj = domLoV.x(); jj <= domHiV.x(); jj++) {
      for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "BEFORE SCALARCOEFF for scalar " << index <<" " << endl;
  for (int ii = domLoW.x(); ii <= domHiW.x(); ii++) {
    cerr << "W Velocity for ii = " << ii << endl;
    for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
      for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	cerr.width(10);
	cerr << coeff_vars->wVelocity[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << endl;
#endif

  FORT_SCALARCOEFF(domLo.get_pointer(), domHi.get_pointer(),
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
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AE Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AW Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AN Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AS Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AT Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AB Convection Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarConvectCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AE Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AE])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AW Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AW])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AN Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AN])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AS Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AS])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AT Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AT])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "After SCALARCOEFF for scalar " << index <<" " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "Scalar AB Coeff for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarCoeff[Arches::AB])
	    [IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif
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
				     int eqnType,
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
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       ArchesVariables* coeff_vars) 
{
  
  // Get the domain size and the patch indices
  IntVector domLo = coeff_vars->pressLinearSrc.getFortLowIndex();
  IntVector domHi = coeff_vars->pressLinearSrc.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef ARCHES_COEF_DEBUG
  cerr << "BEFORE Calculate Pressure Diagonal :" << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "SP - Pressure Linear Source for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressLinearSrc)[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AP - Pressure Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->pressCoeff[Arches::AP])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

}

//****************************************************************************
// Scalar diagonal
//****************************************************************************
void 
Discretization::calculateScalarDiagonal(const ProcessorGroup*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
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
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "SP - Scalar Linear Source for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->scalarLinearSrc)[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
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
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "AP - Scalar Coeff for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(10);
	cerr << (coeff_vars->scalarCoeff[Arches::AP])
	  [IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

}
