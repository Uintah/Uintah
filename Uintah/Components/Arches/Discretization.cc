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
  // Get the domain size 
  IntVector domLoU = coeff_vars->uVelocity.getFortLowIndex();
  IntVector domHiU = coeff_vars->uVelocity.getFortHighIndex();
  IntVector domLoV = coeff_vars->vVelocity.getFortLowIndex();
  IntVector domHiV = coeff_vars->vVelocity.getFortHighIndex();
  IntVector domLoW = coeff_vars->wVelocity.getFortLowIndex();
  IntVector domHiW = coeff_vars->wVelocity.getFortHighIndex();
  IntVector domLo = coeff_vars->density.getFortLowIndex();
  IntVector domHi = coeff_vars->density.getFortHighIndex();

#ifdef ARCHES_DEBUG
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

    // Get the patch indices
    IntVector idxLoU = patch->getSFCXFORTLowIndex();
    IntVector idxHiU = patch->getSFCXFORTHighIndex();

    // Calculate the coeffs
    FORT_UVELCOEF(domLoU.get_pointer(), domHiU.get_pointer(),
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
		  coeff_vars->variableCalledDU.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  coeff_vars->vVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  coeff_vars->wVelocity.getPointer(),
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
#ifdef ARCHES_DEBUG
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
      cerr << "U Vel AP Coeff for ii = " << ii << endl;
      for (int jj = domLoU.y(); jj <= domHiU.y(); jj++) {
	for (int kk = domLoU.z(); kk <= domHiU.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->uVelocityCoeff[Arches::AP])
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

    // Get the patch indices
    IntVector idxLoV = patch->getSFCYFORTLowIndex();
    IntVector idxHiV = patch->getSFCYFORTHighIndex();

    // Calculate the coeffs
    FORT_VVELCOEF(domLoV.get_pointer(), domHiV.get_pointer(),
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
		  coeff_vars->variableCalledDV.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  coeff_vars->uVelocity.getPointer(),
		  domLoW.get_pointer(), domHiW.get_pointer(),
		  coeff_vars->wVelocity.getPointer(),
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
#ifdef ARCHES_DEBUG
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
      cerr << "V Vel AP Coeff for ii = " << ii << endl;
      for (int jj = domLoV.y(); jj <= domHiV.y(); jj++) {
	for (int kk = domLoV.z(); kk <= domHiV.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->vVelocityCoeff[Arches::AP])
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

    // Get the patch indices
    IntVector idxLoW = patch->getSFCZFORTLowIndex();
    IntVector idxHiW = patch->getSFCZFORTHighIndex();

    // Calculate the coeffs
    FORT_WVELCOEF(domLoW.get_pointer(), domHiW.get_pointer(),
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
		  coeff_vars->variableCalledDW.getPointer(),
		  domLoU.get_pointer(), domHiU.get_pointer(),
		  coeff_vars->uVelocity.getPointer(),
		  domLoV.get_pointer(), domHiV.get_pointer(),
		  coeff_vars->vVelocity.getPointer(),
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
#ifdef ARCHES_DEBUG
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
      cerr << "W Vel AP Coeff for ii = " << ii << endl;
      for (int jj = domLoW.y(); jj <= domHiW.y(); jj++) {
	for (int kk = domLoW.z(); kk <= domHiW.z(); kk++) {
	  cerr.width(10);
	  cerr << (coeff_vars->wVelocityCoeff[Arches::AP])
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

  FORT_PRESSCOEFF(domLo.get_pointer(), domHi.get_pointer(),
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
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLoU = coeff_vars->uVelocity.getFortLowIndex();
  IntVector domHiU = coeff_vars->uVelocity.getFortHighIndex();
  IntVector domLoV = coeff_vars->vVelocity.getFortLowIndex();
  IntVector domHiV = coeff_vars->vVelocity.getFortHighIndex();
  IntVector domLoW = coeff_vars->wVelocity.getFortLowIndex();
  IntVector domHiW = coeff_vars->wVelocity.getFortHighIndex();
#ifdef ARCHES_DEBUG  
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dxep[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpw = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dxpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sew = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->sew[ii] << " " ; 
  }
  cerr << endl;
  cerr << " xu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->xu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpwu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dxpwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxepu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dxepu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sewu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->sewu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cee = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cee[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cww = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cww[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwe = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cwe[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ceeu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->ceeu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwwu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cwwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cweu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cweu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " efac = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->efac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " wfac = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->wfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1u = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac1u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2u = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac2u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iesdu = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->iesdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3u = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac3u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4u = " ;
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac4u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iwsdu = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->iwsdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dynp = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dynp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dyps = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dyps[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sns = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->sns[ii] << " " ; 
  }
  cerr << endl;
  cerr << " yv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->yv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dypsv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dypsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dynpv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dynpv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " snsv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->snsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnn = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cnn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " css = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->css[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csn = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->csn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnnv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cnnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cssv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cssv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csnv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->csnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " enfac = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->enfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sfac = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->sfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1v = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac1v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2v = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac2v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jnsdv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->jnsdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3v = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac3v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4v = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac4v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jssdv = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->jssdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dztp = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dztp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpb = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dzpb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stb = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->stb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " zw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->zw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpbw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dzpbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dztpw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->dztpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stbw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->stbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ctt = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->ctt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbb = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cbb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbt = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cbt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cttw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cttw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbbw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cbbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbtw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->cbtw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " tfac = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->tfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " bfac = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->bfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1w = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac1w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2w = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac2w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ktsdw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->ktsdw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3w = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac3w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4w = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->fac4w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " kbsdw = " ;
  for (int ii = idxLo.y(); ii <= idxHi.y(); ii++) {
    cerr.width(10);
    cerr << cellinfo->kbsdw[ii] << " " ; 
  }
  cerr << endl;
#endif
  FORT_SCALARCOEFF(domLo.get_pointer(), domHi.get_pointer(),
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
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       coeff_vars->uVelocityCoeff[Arches::AP].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AE].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AW].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AN].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AS].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AT].getPointer(), 
	       coeff_vars->uVelocityCoeff[Arches::AB].getPointer(),
	       coeff_vars->uVelLinearSrc.getPointer());
    break;
  case Arches::YDIR:
    domLo = coeff_vars->vVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->vVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       coeff_vars->vVelocityCoeff[Arches::AP].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AE].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AW].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AN].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AS].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AT].getPointer(), 
	       coeff_vars->vVelocityCoeff[Arches::AB].getPointer(),
	       coeff_vars->vVelLinearSrc.getPointer());
    break;
  case Arches::ZDIR:
    domLo = coeff_vars->wVelLinearSrc.getFortLowIndex();
    domHi = coeff_vars->wVelLinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    FORT_APCAL(domLo.get_pointer(), domHi.get_pointer(),
	       idxLo.get_pointer(), idxHi.get_pointer(),
	       coeff_vars->wVelocityCoeff[Arches::AP].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AE].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AW].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AN].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AS].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AT].getPointer(), 
	       coeff_vars->wVelocityCoeff[Arches::AB].getPointer(),
	       coeff_vars->wVelLinearSrc.getPointer());
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
}

//
// $Log$
// Revision 1.40  2000/08/09 20:19:25  rawat
// modified scalcoef.F
//
// Revision 1.39  2000/08/08 23:34:18  rawat
// fixed some bugs in profv.F and Properties.cc
//
// Revision 1.38  2000/08/04 03:02:01  bbanerje
// Add some inits.
//
// Revision 1.37  2000/08/04 02:14:32  bbanerje
// Added debug statements.
//
// Revision 1.36  2000/08/02 16:27:38  bbanerje
// Added -DDEBUG to sub.mk and Discretization
//
// Revision 1.35  2000/07/28 02:30:59  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.31  2000/07/14 03:45:45  rawat
// completed velocity bc and fixed some bugs
//
// Revision 1.30  2000/07/13 04:58:45  bbanerje
// Updated pressureDiagonal calc.
//
// Revision 1.29  2000/07/12 22:15:02  bbanerje
// Added pressure Coef .. will do until Kumar's code is up and running
//
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
