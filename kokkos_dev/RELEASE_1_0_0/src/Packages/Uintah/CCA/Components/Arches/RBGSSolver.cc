//----- RBGSSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesVariables.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Core/Containers/Array1.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

//****************************************************************************
// Default constructor for RBGSSolver
//****************************************************************************
RBGSSolver::RBGSSolver()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
RBGSSolver::~RBGSSolver()
{
}

//****************************************************************************
// Problem setup
//****************************************************************************
void 
RBGSSolver::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params->findBlock("LinearSolver");
  db->require("max_iter", d_maxSweeps);
  db->require("res_tol", d_residual);
  db->require("underrelax", d_underrelax);
}


//****************************************************************************
// Actual compute of pressure residual
//****************************************************************************
void 
RBGSSolver::computePressResidual(const ProcessorGroup*,
				 const Patch* patch,
				 DataWarehouseP&,
				 DataWarehouseP&,
				 ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call

  FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->pressure.getPointer(),
		    vars->residualPressure.getPointer(),
		    vars->pressCoeff[Arches::AE].getPointer(), 
		    vars->pressCoeff[Arches::AW].getPointer(), 
		    vars->pressCoeff[Arches::AN].getPointer(), 
		    vars->pressCoeff[Arches::AS].getPointer(), 
		    vars->pressCoeff[Arches::AT].getPointer(), 
		    vars->pressCoeff[Arches::AB].getPointer(), 
		    vars->pressCoeff[Arches::AP].getPointer(), 
		    vars->pressNonlinearSrc.getPointer(),
		    &vars->residPress, &vars->truncPress);

#ifdef ARCHES_PRES_DEBUG
  cerr << " After Pressure Compute Residual : " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "residual for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << vars->residualPressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << "Resid Press = " << vars->residPress << " Trunc Press = " <<
    vars->truncPress << endl;
#endif
}


//****************************************************************************
// Actual calculation of order of magnitude term for pressure equation
//****************************************************************************
void 
RBGSSolver::computePressOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

//&vars->truncPress

}

//****************************************************************************
// Actual compute of pressure underrelaxation
//****************************************************************************
void 
RBGSSolver::computePressUnderrelax(const ProcessorGroup*,
				   const Patch* patch,
				   DataWarehouseP&,
				   DataWarehouseP&, 
				   ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector domLong = vars->pressCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = vars->pressCoeff[Arches::AP].getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 domLong.get_pointer(), domHing.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->pressure.getPointer(),
		 vars->pressCoeff[Arches::AP].getPointer(), 
		 vars->pressNonlinearSrc.getPointer(), 
		 &d_underrelax);

#ifdef ARCHES_PRES_DEBUG
  cerr << " After Pressure Underrelax : " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << vars->pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << " After Pressure Underrelax : " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "pressure AP for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << (vars->pressCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
  cerr << " After Pressure Underrelax : " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "pressure SU for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << vars->pressNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif
}

//****************************************************************************
// Actual linear solve for pressure
//****************************************************************************
void 
RBGSSolver::pressLisolve(const ProcessorGroup* pc,
			 const Patch* patch,
			 DataWarehouseP& old_dw,
			 DataWarehouseP& new_dw,
			 ArchesVariables* vars,
			 const ArchesLabel* lab)
{
 
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->pressure.getFortLowIndex();
  IntVector domHi = vars->pressure.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //bool lswpwe = true;
  //bool lswpsn = true;
  //bool lswpbt = true;
  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;
  IntVector Size = domHi - domLo + IntVector(1,1,1);
  e1.resize(Size.x());
  f1.resize(Size.x());
  e2.resize(Size.y());
  f2.resize(Size.y());
  e3.resize(Size.z());
  f3.resize(Size.z());
  sum_vartype residP;
  sum_vartype truncP;
  old_dw->get(residP, lab->d_presResidPSLabel);
  old_dw->get(truncP, lab->d_presTruncPSLabel);
  double nlResid = residP;
  double trunc_conv = truncP*1.0E-7;
  //  double theta = 0.5;
  double theta = 0.0;
  int pressIter = 0;
  double pressResid = 0.0;
  do {
  //fortran call for lineGS solver
    FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->pressure.getPointer(),
		vars->pressCoeff[Arches::AE].getPointer(), 
		vars->pressCoeff[Arches::AW].getPointer(), 
		vars->pressCoeff[Arches::AN].getPointer(), 
		vars->pressCoeff[Arches::AS].getPointer(), 
		vars->pressCoeff[Arches::AT].getPointer(), 
		vars->pressCoeff[Arches::AB].getPointer(), 
		vars->pressCoeff[Arches::AP].getPointer(), 
		vars->pressNonlinearSrc.getPointer(),
		e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		e3.get_objs(), f3.get_objs(), &theta);
      //, &lswpwe, &lswpsn, &lswpbt);
    computePressResidual(pc, patch, old_dw, new_dw, vars);
    pressResid = vars->residPress;
    ++pressIter;
#ifdef ARCHES_PRES_DEBUG
    cerr << "Iter # = " << pressIter << " Max Iters = " << d_maxSweeps 
	 << " Press. Resid = " << pressResid << " d_residual = " << d_residual
	 << " nlResid = " << nlResid << endl;
#endif
  } while((pressIter < d_maxSweeps)&&((pressResid > d_residual*nlResid)));
  // while((pressIter < d_maxSweeps)&&((pressResid > d_residual*nlResid)||
  //			      (pressResid > trunc_conv)));
#ifdef ARCHES_PRES_DEBUG
  cerr << "After pressure solve " << pressIter << " " << pressResid << endl;
  cerr << "After pressure solve " << nlResid << " " << trunc_conv <<  endl;
  cerr << " After Pressure solve : " << endl;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr << "pressure for ii = " << ii << endl;
    for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
      for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	cerr.width(14);
	cerr << vars->pressure[IntVector(ii,jj,kk)] << " " ; 
      }
      cerr << endl;
    }
  }
#endif

  

}

//****************************************************************************
// Actual compute of Velocity residual
//****************************************************************************

void 
RBGSSolver::computeVelResidual(const ProcessorGroup* ,
			       const Patch* patch,
			       DataWarehouseP& ,
			       DataWarehouseP& , 
			       int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    //fortran call

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->residualUVelocity.getPointer(),
		      vars->uVelocityCoeff[Arches::AE].getPointer(), 
		      vars->uVelocityCoeff[Arches::AW].getPointer(), 
		      vars->uVelocityCoeff[Arches::AN].getPointer(), 
		      vars->uVelocityCoeff[Arches::AS].getPointer(), 
		      vars->uVelocityCoeff[Arches::AT].getPointer(), 
		      vars->uVelocityCoeff[Arches::AB].getPointer(), 
		      vars->uVelocityCoeff[Arches::AP].getPointer(), 
		      vars->uVelNonlinearSrc.getPointer(),
		      &vars->residUVel, &vars->truncUVel);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "u residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualUVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid U Vel = " << vars->residUVel << " Trunc U Vel = " <<
      vars->truncUVel << endl;
#endif

    break;
  case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    //fortran call

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->residualVVelocity.getPointer(),
		      vars->vVelocityCoeff[Arches::AE].getPointer(), 
		      vars->vVelocityCoeff[Arches::AW].getPointer(), 
		      vars->vVelocityCoeff[Arches::AN].getPointer(), 
		      vars->vVelocityCoeff[Arches::AS].getPointer(), 
		      vars->vVelocityCoeff[Arches::AT].getPointer(), 
		      vars->vVelocityCoeff[Arches::AB].getPointer(), 
		      vars->vVelocityCoeff[Arches::AP].getPointer(), 
		      vars->vVelNonlinearSrc.getPointer(),
		      &vars->residVVel, &vars->truncVVel);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "v residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualVVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid V Vel = " << vars->residVVel << " Trunc V Vel = " <<
      vars->truncVVel << endl;
#endif

    break;
  case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    //fortran call

    FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		      idxLo.get_pointer(), idxHi.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->residualWVelocity.getPointer(),
		      vars->wVelocityCoeff[Arches::AE].getPointer(), 
		      vars->wVelocityCoeff[Arches::AW].getPointer(), 
		      vars->wVelocityCoeff[Arches::AN].getPointer(), 
		      vars->wVelocityCoeff[Arches::AS].getPointer(), 
		      vars->wVelocityCoeff[Arches::AT].getPointer(), 
		      vars->wVelocityCoeff[Arches::AB].getPointer(), 
		      vars->wVelocityCoeff[Arches::AP].getPointer(), 
		      vars->wVelNonlinearSrc.getPointer(),
		      &vars->residWVel, &vars->truncWVel);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After W Velocity Compute Residual : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "w residual for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->residualWVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << "Resid W Vel = " << vars->residWVel << " Trunc W Vel = " <<
      vars->truncWVel << endl;
#endif

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Actual calculation of order of magnitude term for Velocity equation
//****************************************************************************
void 
RBGSSolver::computeVelOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

  //&vars->truncUVel
  //&vars->truncVVel
  //&vars->truncWVel

}



//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
RBGSSolver::computeVelUnderrelax(const ProcessorGroup* ,
				 const Patch* patch,
				 DataWarehouseP& ,
				 DataWarehouseP& , 
				 int index, ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector domLong;
  IntVector domHing;
  IntVector idxLo;
  IntVector idxHi;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    domLong = vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonlinearSrc.getPointer(),
		   &d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "U Vel AP for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->uVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After U Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "U Vel SU for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    domLong = vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->vVelocity.getPointer(),
		   vars->vVelocityCoeff[Arches::AP].getPointer(), 
		   vars->vVelNonlinearSrc.getPointer(),
		   &d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "V Vel AP for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->vVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After V Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "V Vel SU for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    domLong = vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->wVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->wVelocity.getPointer(),
		   vars->wVelocityCoeff[Arches::AP].getPointer(), 
		   vars->wVelNonlinearSrc.getPointer(),
		   &d_underrelax);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After W Vel Underrelax : " << endl;
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
    cerr << " After W Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "W Vel AP for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << (vars->wVelocityCoeff[Arches::AP])[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
    cerr << " After W Vel Underrelax : " << endl;
    for (int ii = domLong.x(); ii <= domHing.x(); ii++) {
      cerr << "W Vel SU for ii = " << ii << endl;
      for (int jj = domLong.y(); jj <= domHing.y(); jj++) {
	for (int kk = domLong.z(); kk <= domHing.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->wVelNonlinearSrc[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
RBGSSolver::velocityLisolve(const ProcessorGroup* pc,
			    const Patch* patch,
			    DataWarehouseP& old_dw ,
			    DataWarehouseP& new_dw, 
			    int index, double delta_t,
			    ArchesVariables* vars,
			    CellInformation* cellinfo,
			    const ArchesLabel* lab)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector domLong;
  IntVector domHing;
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();
  int numGhostCells = 1;
  IntVector domLoDenwg = patch->getGhostCellLowIndex(numGhostCells);
  IntVector domHiDenwg = patch->getGhostCellHighIndex(numGhostCells) -
                                                        IntVector(1,1,1);
  
  IntVector Size;

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  sum_vartype resid;
  sum_vartype trunc;

  double nlResid;
  double trunc_conv;

  int velIter = 0;
  double velResid = 0.0;
  double theta = 0.5;

  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    domLong = vars->uVelNonlinearSrc.getFortLowIndex();
    domHing = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();

#if implicit_defined
    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_uVelResidPSLabel);
    old_dw->get(trunc, lab->d_uVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;
    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->uVelocity.getPointer(),
		  vars->uVelocityCoeff[Arches::AE].getPointer(), 
		  vars->uVelocityCoeff[Arches::AW].getPointer(), 
		  vars->uVelocityCoeff[Arches::AN].getPointer(), 
		  vars->uVelocityCoeff[Arches::AS].getPointer(), 
		  vars->uVelocityCoeff[Arches::AT].getPointer(), 
		  vars->uVelocityCoeff[Arches::AB].getPointer(), 
		  vars->uVelocityCoeff[Arches::AP].getPointer(), 
		  vars->uVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residUVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After u Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After u Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  domLong.get_pointer(), domHing.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->uVelocity.getPointer(),
		  vars->old_uVelocity.getPointer(),
		  vars->uVelocityCoeff[Arches::AE].getPointer(), 
		  vars->uVelocityCoeff[Arches::AW].getPointer(), 
		  vars->uVelocityCoeff[Arches::AN].getPointer(), 
		  vars->uVelocityCoeff[Arches::AS].getPointer(), 
		  vars->uVelocityCoeff[Arches::AT].getPointer(), 
		  vars->uVelocityCoeff[Arches::AB].getPointer(), 
		  vars->uVelocityCoeff[Arches::AP].getPointer(), 
		  vars->uVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  domLoDenwg.get_pointer(), domHiDenwg.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sewu.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stb.get_objs(), &delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After U Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "U Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->uVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    vars->residUVel = 1.0E-7;
    vars->truncUVel = 1.0;
#endif
    break;
  case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    domLong = vars->vVelNonlinearSrc.getFortLowIndex();
    domHing = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();

#if implicit_defined
    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_vVelResidPSLabel);
    old_dw->get(trunc, lab->d_vVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;

    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->vVelocity.getPointer(),
		  vars->vVelocityCoeff[Arches::AE].getPointer(), 
		  vars->vVelocityCoeff[Arches::AW].getPointer(), 
		  vars->vVelocityCoeff[Arches::AN].getPointer(), 
		  vars->vVelocityCoeff[Arches::AS].getPointer(), 
		  vars->vVelocityCoeff[Arches::AT].getPointer(), 
		  vars->vVelocityCoeff[Arches::AB].getPointer(), 
		  vars->vVelocityCoeff[Arches::AP].getPointer(), 
		  vars->vVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residVVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After v Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After v Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  domLong.get_pointer(), domHing.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->vVelocity.getPointer(),
		  vars->old_vVelocity.getPointer(),
		  vars->vVelocityCoeff[Arches::AE].getPointer(), 
		  vars->vVelocityCoeff[Arches::AW].getPointer(), 
		  vars->vVelocityCoeff[Arches::AN].getPointer(), 
		  vars->vVelocityCoeff[Arches::AS].getPointer(), 
		  vars->vVelocityCoeff[Arches::AT].getPointer(), 
		  vars->vVelocityCoeff[Arches::AB].getPointer(), 
		  vars->vVelocityCoeff[Arches::AP].getPointer(), 
		  vars->vVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  domLoDenwg.get_pointer(), domHiDenwg.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sew.get_objs(), cellinfo->snsv.get_objs(),
		  cellinfo->stb.get_objs(), &delta_t);

#ifdef ARCHES_VEL_DEBUG
    cerr << " After V Vel Explicit solve : " << endl;
    for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
      cerr << "V Vel for ii = " << ii << endl;
      for (int jj = domLo.y(); jj <= domHi.y(); jj++) {
	for (int kk = domLo.z(); kk <= domHi.z(); kk++) {
	  cerr.width(14);
	  cerr << vars->vVelocity[IntVector(ii,jj,kk)] << " " ; 
	}
	cerr << endl;
      }
    }
#endif

    vars->residVVel = 1.0E-7;
    vars->truncVVel = 1.0;
#endif
    break;
  case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    domLong = vars->wVelNonlinearSrc.getFortLowIndex();
    domHing = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();

#if implicit_defined
    Size = domHi - domLo + IntVector(1,1,1);

    e1.resize(Size.x());
    f1.resize(Size.x());
    e2.resize(Size.y());
    f2.resize(Size.y());
    e3.resize(Size.z());
    f3.resize(Size.z());

    old_dw->get(resid, lab->d_wVelResidPSLabel);
    old_dw->get(trunc, lab->d_wVelTruncPSLabel);

    nlResid = resid;
    trunc_conv = trunc*1.0E-7;
    do {
      //fortran call for lineGS solver
      FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->wVelocity.getPointer(),
		  vars->wVelocityCoeff[Arches::AE].getPointer(), 
		  vars->wVelocityCoeff[Arches::AW].getPointer(), 
		  vars->wVelocityCoeff[Arches::AN].getPointer(), 
		  vars->wVelocityCoeff[Arches::AS].getPointer(), 
		  vars->wVelocityCoeff[Arches::AT].getPointer(), 
		  vars->wVelocityCoeff[Arches::AB].getPointer(), 
		  vars->wVelocityCoeff[Arches::AP].getPointer(), 
		  vars->wVelNonlinearSrc.getPointer(),
		  e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		  e3.get_objs(), f3.get_objs(), &theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residWVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After w Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After w Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		  domLong.get_pointer(), domHing.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->wVelocity.getPointer(),
		  vars->old_wVelocity.getPointer(),
		  vars->wVelocityCoeff[Arches::AE].getPointer(), 
		  vars->wVelocityCoeff[Arches::AW].getPointer(), 
		  vars->wVelocityCoeff[Arches::AN].getPointer(), 
		  vars->wVelocityCoeff[Arches::AS].getPointer(), 
		  vars->wVelocityCoeff[Arches::AT].getPointer(), 
		  vars->wVelocityCoeff[Arches::AB].getPointer(), 
		  vars->wVelocityCoeff[Arches::AP].getPointer(), 
		  vars->wVelNonlinearSrc.getPointer(),
		  domLoDen.get_pointer(), domHiDen.get_pointer(),
		  domLoDenwg.get_pointer(), domHiDenwg.get_pointer(),
		  vars->old_density.getPointer(), 
		  cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		  cellinfo->stbw.get_objs(), &delta_t);

#ifdef ARCHES_VEL_DEBUG
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
#endif
    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}

//****************************************************************************
// Calculate Scalar residuals
//****************************************************************************
void 
RBGSSolver::computeScalarResidual(const ProcessorGroup* ,
				  const Patch* patch,
				  DataWarehouseP& ,
				  DataWarehouseP& , 
				  int index,
				  ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call

  FORT_COMPUTERESID(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->scalar.getPointer(),
		    vars->residualScalar.getPointer(),
		    vars->scalarCoeff[Arches::AE].getPointer(), 
		    vars->scalarCoeff[Arches::AW].getPointer(), 
		    vars->scalarCoeff[Arches::AN].getPointer(), 
		    vars->scalarCoeff[Arches::AS].getPointer(), 
		    vars->scalarCoeff[Arches::AT].getPointer(), 
		    vars->scalarCoeff[Arches::AB].getPointer(), 
		    vars->scalarCoeff[Arches::AP].getPointer(), 
		    vars->scalarNonlinearSrc.getPointer(),
		    &vars->residScalar, &vars->truncScalar);
}


//****************************************************************************
// Actual calculation of order of magnitude term for Scalar equation
//****************************************************************************
void 
RBGSSolver::computeScalarOrderOfMagnitude(const ProcessorGroup* ,
				const Patch* ,
				DataWarehouseP& ,
				DataWarehouseP& , ArchesVariables* )
{

  //&vars->truncScalar

}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
RBGSSolver::computeScalarUnderrelax(const ProcessorGroup* ,
				    const Patch* patch,
				    DataWarehouseP& ,
				    DataWarehouseP& , 
				    int index,
				    ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector domLong = vars->scalarCoeff[Arches::AP].getFortLowIndex();
  IntVector domHing = vars->scalarCoeff[Arches::AP].getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 domLong.get_pointer(), domHing.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->scalar.getPointer(),
		 vars->scalarCoeff[Arches::AP].getPointer(), 
		 vars->scalarNonlinearSrc.getPointer(),
		 &d_underrelax);
#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Underrelaxation Scalar" << endl;
  cerr << "SAP - Scalar Coeff " << endl;
  vars->scalarCoeff[Arches::AP].print(cerr);
  cerr << "SSU - Scalar Source " << endl;
  vars->scalarNonlinearSrc.print(cerr);
#endif

}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RBGSSolver::scalarLisolve(const ProcessorGroup* pc,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw, 
			  int index, double delta_t,
			  ArchesVariables* vars,
			  CellInformation* cellinfo,
			  const ArchesLabel* lab)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLong = vars->scalarNonlinearSrc.getFortLowIndex();
  IntVector domHing = vars->scalarNonlinearSrc.getFortHighIndex();
  // for explicit solver
  IntVector domLoDen = vars->old_density.getFortLowIndex();
  IntVector domHiDen = vars->old_density.getFortHighIndex();
  int numGhostCells = 1;
  IntVector domLoDenwg = patch->getGhostCellLowIndex(numGhostCells);
  // to make it compatible with fortran
  IntVector domHiDenwg = patch->getGhostCellHighIndex(numGhostCells) -
                                                      IntVector(1,1,1);

#if implict_defined
  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  IntVector Size = domHi - domLo + IntVector(1,1,1);

  e1.resize(Size.x());
  f1.resize(Size.x());
  e2.resize(Size.y());
  f2.resize(Size.y());
  e3.resize(Size.z());
  f3.resize(Size.z());

  sum_vartype resid;
  sum_vartype trunc;

  old_dw->get(resid, lab->d_scalarResidLabel);
  old_dw->get(trunc, lab->d_scalarTruncLabel);

  double nlResid = resid;
  double trunc_conv = trunc*1.0E-7;
  double theta = 0.5;
  int scalarIter = 0;
  double scalarResid = 0.0;
  do {
    //fortran call for lineGS solver
    FORT_LINEGS(domLo.get_pointer(), domHi.get_pointer(),
		idxLo.get_pointer(), idxHi.get_pointer(),
		vars->scalar.getPointer(),
		vars->scalarCoeff[Arches::AE].getPointer(), 
		vars->scalarCoeff[Arches::AW].getPointer(), 
		vars->scalarCoeff[Arches::AN].getPointer(), 
		vars->scalarCoeff[Arches::AS].getPointer(), 
		vars->scalarCoeff[Arches::AT].getPointer(), 
		vars->scalarCoeff[Arches::AB].getPointer(), 
		vars->scalarCoeff[Arches::AP].getPointer(), 
		vars->scalarNonlinearSrc.getPointer(),
		e1.get_objs(), f1.get_objs(), e2.get_objs(), f2.get_objs(),
		e3.get_objs(), f3.get_objs(), &theta);
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif
     FORT_EXPLICIT(domLo.get_pointer(), domHi.get_pointer(),
		   domLong.get_pointer(), domHing.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->scalar.getPointer(), vars->old_scalar.getPointer(),
		   vars->scalarCoeff[Arches::AE].getPointer(), 
		   vars->scalarCoeff[Arches::AW].getPointer(), 
		   vars->scalarCoeff[Arches::AN].getPointer(), 
		   vars->scalarCoeff[Arches::AS].getPointer(), 
		   vars->scalarCoeff[Arches::AT].getPointer(), 
		   vars->scalarCoeff[Arches::AB].getPointer(), 
		   vars->scalarCoeff[Arches::AP].getPointer(), 
		   vars->scalarNonlinearSrc.getPointer(),
		   domLoDen.get_pointer(), domHiDen.get_pointer(),
		   domLoDenwg.get_pointer(), domHiDenwg.get_pointer(),
		   vars->old_density.getPointer(), 
		   cellinfo->sew.get_objs(), cellinfo->sns.get_objs(),
		   cellinfo->stb.get_objs(), &delta_t);
     for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
       for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
	for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
	  IntVector currCell(ii,jj,kk);
	  if (vars->scalar[currCell] > 1.0)
	    vars->scalar[currCell] = 1.0;
	  else if (vars->scalar[currCell] < 0.0)
	    vars->scalar[currCell] = 0.0;
	}
      }
    }
#ifdef ARCHES_DEBUG
    cerr << " After Scalar Explicit solve : " << endl;
    cerr << "Print Scalar: " << endl;
    vars->scalar.print(cerr);
#endif


    vars->residScalar = 1.0E-7;
    vars->truncScalar = 1.0;
   
}

void 
RBGSSolver::matrixCreate(const LevelP& level, LoadBalancer* lb)
{
}

bool
RBGSSolver::pressLinearSolve()
{
  cerr << "pressure linear solve not implemented for RBGS " << endl;
  return 0;
}

void 
RBGSSolver::copyPressSoln(const Patch* patch, ArchesVariables* vars)
{
}

void
RBGSSolver::destroyMatrix()
{
}

void 
RBGSSolver::setPressMatrix(const ProcessorGroup* ,
			    const Patch* patch,
			    DataWarehouseP&,
			    DataWarehouseP&,
			    ArchesVariables* vars,
			    const ArchesLabel* lab)
{
}


