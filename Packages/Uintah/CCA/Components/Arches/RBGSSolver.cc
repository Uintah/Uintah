//----- RBGSSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/debug.h>
#include <Packages/Uintah/CCA/Components/Arches/RBGSSolver.h>
#include <Core/Containers/Array1.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PressureSolver.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_velocity_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/linegs_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/rescal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/underelax_fort.h>

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
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_rescal(idxLo, idxHi, vars->pressure, vars->residualPressure,
	      vars->pressCoeff[Arches::AE], vars->pressCoeff[Arches::AW], 
	      vars->pressCoeff[Arches::AN], vars->pressCoeff[Arches::AS], 
	      vars->pressCoeff[Arches::AT], vars->pressCoeff[Arches::AB], 
	      vars->pressCoeff[Arches::AP], vars->pressNonlinearSrc,
	      vars->residPress, vars->truncPress);

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
				   ArchesVariables* vars,
				   ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_underelax(idxLo, idxHi, constvars->pressure,
		 vars->pressCoeff[Arches::AP],
		 vars->pressNonlinearSrc, d_underrelax);

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
  OffsetArray1<double> e1;
  OffsetArray1<double> f1;
  OffsetArray1<double> e2;
  OffsetArray1<double> f2;
  OffsetArray1<double> e3;
  OffsetArray1<double> f3;
  e1.resize(domLo.x(), domHi.x());
  f1.resize(domLo.x(), domHi.x());
  e2.resize(domLo.y(), domHi.y());
  f2.resize(domLo.y(), domHi.y());
  e3.resize(domLo.z(), domHi.z());
  f3.resize(domLo.z(), domHi.z());
  sum_vartype residP;
  sum_vartype truncP;
  old_dw->get(residP, lab->d_presResidPSLabel);
  old_dw->get(truncP, lab->d_presTruncPSLabel);
  double nlResid = residP;
  // double trunc_conv = truncP*1.0E-7;
  //  double theta = 0.5;
  double theta = 0.0;
  int pressIter = 0;
  double pressResid = 0.0;
  do {
  //fortran call for lineGS solver
    fort_linegs(idxLo, idxHi, vars->pressure,
		vars->pressCoeff[Arches::AE],
		vars->pressCoeff[Arches::AW],
		vars->pressCoeff[Arches::AN],
		vars->pressCoeff[Arches::AS],
		vars->pressCoeff[Arches::AT],
		vars->pressCoeff[Arches::AB],
		vars->pressCoeff[Arches::AP],
		vars->pressNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);
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

    fort_rescal(idxLo, idxHi, vars->uVelocity, vars->residualUVelocity,
		vars->uVelocityCoeff[Arches::AE], 
		vars->uVelocityCoeff[Arches::AW], 
		vars->uVelocityCoeff[Arches::AN], 
		vars->uVelocityCoeff[Arches::AS], 
		vars->uVelocityCoeff[Arches::AT], 
		vars->uVelocityCoeff[Arches::AB], 
		vars->uVelocityCoeff[Arches::AP], 
		vars->uVelNonlinearSrc, vars->residUVel,
		vars->truncUVel);

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

    fort_rescal(idxLo, idxHi, vars->vVelocity, vars->residualVVelocity,
		vars->vVelocityCoeff[Arches::AE], 
		vars->vVelocityCoeff[Arches::AW], 
		vars->vVelocityCoeff[Arches::AN], 
		vars->vVelocityCoeff[Arches::AS], 
		vars->vVelocityCoeff[Arches::AT], 
		vars->vVelocityCoeff[Arches::AB], 
		vars->vVelocityCoeff[Arches::AP], 
		vars->vVelNonlinearSrc, vars->residVVel,
		vars->truncVVel);

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

    fort_rescal(idxLo, idxHi, vars->wVelocity, vars->residualWVelocity,
		vars->wVelocityCoeff[Arches::AE], 
		vars->wVelocityCoeff[Arches::AW], 
		vars->wVelocityCoeff[Arches::AN], 
		vars->wVelocityCoeff[Arches::AS], 
		vars->wVelocityCoeff[Arches::AT], 
		vars->wVelocityCoeff[Arches::AB], 
		vars->wVelocityCoeff[Arches::AP], 
		vars->wVelNonlinearSrc, vars->residWVel,
		vars->truncWVel);

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
				 int index, ArchesVariables* vars,
				 ArchesConstVariables* constvars)
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
    domLo = constvars->uVelocity.getFortLowIndex();
    domHi = constvars->uVelocity.getFortHighIndex();
    domLong = vars->uVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->uVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->uVelocity,
		   vars->uVelocityCoeff[Arches::AP], vars->uVelNonlinearSrc,
		   d_underrelax);

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
    domLo = constvars->vVelocity.getFortLowIndex();
    domHi = constvars->vVelocity.getFortHighIndex();
    domLong = vars->vVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->vVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->vVelocity,
		   vars->vVelocityCoeff[Arches::AP], vars->vVelNonlinearSrc,
		   d_underrelax);

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
    domLo = constvars->wVelocity.getFortLowIndex();
    domHi = constvars->wVelocity.getFortHighIndex();
    domLong = vars->wVelocityCoeff[Arches::AP].getFortLowIndex();
    domHing = vars->wVelocityCoeff[Arches::AP].getFortHighIndex();
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    fort_underelax(idxLo, idxHi, constvars->wVelocity,
		   vars->wVelocityCoeff[Arches::AP], vars->wVelNonlinearSrc,
		   d_underrelax);

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
RBGSSolver::velocityLisolve(const ProcessorGroup* /*pc*/,
			    const Patch* patch,
			    int index, double delta_t,
			    ArchesVariables* vars,
			    CellInformation* cellinfo,
			    const ArchesLabel* /*lab*/)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo;
  IntVector domHi;
  IntVector domLong;
  IntVector domHing;
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  IntVector Size;

  Array1<double> e1;
  Array1<double> f1;
  Array1<double> e2;
  Array1<double> f2;
  Array1<double> e3;
  Array1<double> f3;

  sum_vartype resid;
  sum_vartype trunc;

  // int velIter = 0;
  int ioff, joff, koff;
  switch (index) {
  case Arches::XDIR:
    domLo = vars->uVelocity.getFortLowIndex();
    domHi = vars->uVelocity.getFortHighIndex();
    domLong = vars->uVelNonlinearSrc.getFortLowIndex();
    domHing = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;
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
      fort_linegs(idxLo, idxHi, vars->uVelocity,
		  vars->uVelocityCoeff[Arches::AE],
		  vars->uVelocityCoeff[Arches::AW],
		  vars->uVelocityCoeff[Arches::AN],
		  vars->uVelocityCoeff[Arches::AS],
		  vars->uVelocityCoeff[Arches::AT],
		  vars->uVelocityCoeff[Arches::AB],
		  vars->uVelocityCoeff[Arches::AP],
		  vars->uVelNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residUVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After u Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After u Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    fort_explicit_velocity(idxLo, idxHi, vars->uVelocity,
			   vars->old_uVelocity,
			   vars->uVelocityCoeff[Arches::AE],
			   vars->uVelocityCoeff[Arches::AW],
			   vars->uVelocityCoeff[Arches::AN],
			   vars->uVelocityCoeff[Arches::AS],
			   vars->uVelocityCoeff[Arches::AT],
			   vars->uVelocityCoeff[Arches::AB],
			   vars->uVelocityCoeff[Arches::AP],
			   vars->uVelNonlinearSrc,
			   vars->old_density,
			   cellinfo->sewu, cellinfo->sns, cellinfo->stb,
			   delta_t, ioff, joff, koff);


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
    ioff = 0; joff = 1; koff = 0;
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
      fort_linegs(idxLo, idxHi, vars->vVelocity,
		  vars->vVelocityCoeff[Arches::AE],
		  vars->vVelocityCoeff[Arches::AW],
		  vars->vVelocityCoeff[Arches::AN],
		  vars->vVelocityCoeff[Arches::AS],
		  vars->vVelocityCoeff[Arches::AT],
		  vars->vVelocityCoeff[Arches::AB],
		  vars->vVelocityCoeff[Arches::AP],
		  vars->vVelNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residVVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After v Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After v Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    fort_explicit_velocity(idxLo, idxHi, vars->vVelocity,
			   vars->old_vVelocity,
			   vars->vVelocityCoeff[Arches::AE],
			   vars->vVelocityCoeff[Arches::AW],
			   vars->vVelocityCoeff[Arches::AN],
			   vars->vVelocityCoeff[Arches::AS],
			   vars->vVelocityCoeff[Arches::AT],
			   vars->vVelocityCoeff[Arches::AB],
			   vars->vVelocityCoeff[Arches::AP],
			   vars->vVelNonlinearSrc,
			   vars->old_density,
			   cellinfo->sew, cellinfo->snsv, cellinfo->stb,
			   delta_t, ioff, joff, koff);

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
    ioff = 0; joff = 0; koff = 1;
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
      fort_linegs(idxLo, idxHi, vars->wVelocity,
		  vars->wVelocityCoeff[Arches::AE],
		  vars->wVelocityCoeff[Arches::AW],
		  vars->wVelocityCoeff[Arches::AN],
		  vars->wVelocityCoeff[Arches::AS],
		  vars->wVelocityCoeff[Arches::AT],
		  vars->wVelocityCoeff[Arches::AB],
		  vars->wVelocityCoeff[Arches::AP],
		  vars->wVelNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);

      computeVelResidual(pc, patch, old_dw, new_dw, index, vars);
      velResid = vars->residWVel;
      ++velIter;
    } while((velIter < d_maxSweeps)&&((velResid > d_residual*nlResid)||
				      (velResid > trunc_conv)));
    cerr << "After w Velocity solve " << velIter << " " << velResid << endl;
    cerr << "After w Velocity solve " << nlResid << " " << trunc_conv <<  endl;
#else
    fort_explicit_velocity(idxLo, idxHi, vars->wVelocity,
			   vars->old_wVelocity,
			   vars->wVelocityCoeff[Arches::AE],
			   vars->wVelocityCoeff[Arches::AW],
			   vars->wVelocityCoeff[Arches::AN],
			   vars->wVelocityCoeff[Arches::AS],
			   vars->wVelocityCoeff[Arches::AT],
			   vars->wVelocityCoeff[Arches::AB],
			   vars->wVelocityCoeff[Arches::AP],
			   vars->wVelNonlinearSrc,
			   vars->old_density,
			   cellinfo->sew, cellinfo->sns,  cellinfo->stbw,
			   delta_t, ioff, joff, koff);

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
				  int,
				  ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call

  fort_rescal(idxLo, idxHi, vars->scalar, vars->residualScalar,
	      vars->scalarCoeff[Arches::AE], vars->scalarCoeff[Arches::AW], 
	      vars->scalarCoeff[Arches::AN], vars->scalarCoeff[Arches::AS], 
	      vars->scalarCoeff[Arches::AT], vars->scalarCoeff[Arches::AB], 
	      vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
	      vars->residScalar,vars->truncScalar);
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
				    int,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  fort_underelax(idxLo, idxHi, constvars->scalar,
		 vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
		 d_underrelax);
#ifdef ARCHES_COEF_DEBUG
  cerr << "AFTER Underrelaxation Scalar" << endl;
  cerr << "SAP - Scalar Coeff " << endl;
  vars->scalarCoeff[Arches::AP].print(cerr);
  cerr << "SSU - Scalar Source " << endl;
  vars->scalarNonlinearSrc.print(cerr);
#endif

}

void 
RBGSSolver::computeEnthalpyUnderrelax(const ProcessorGroup* ,
				      const Patch* patch,
				      ArchesVariables* vars,
				      ArchesConstVariables* constvars)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
//#define enthalpySolve_debug
#ifdef enthalpySolve_debug

  // code to print all values of any variable within
  // a box, for multi-patch case

  IntVector indexLow = patch->getCellLowIndex();
  IntVector indexHigh = patch->getCellHighIndex();

  int ibot = 0;
  int itop = 0;
  int jbot = 8;
  int jtop = 8;
  int kbot = 8;
  int ktop = 8;

  // values above can be changed for each case as desired

  bool printvalues = true;
  int idloX = Max(indexLow.x(),ibot);
  int idhiX = Min(indexHigh.x()-1,itop);
  int idloY = Max(indexLow.y(),jbot);
  int idhiY = Min(indexHigh.y()-1,jtop);
  int idloZ = Max(indexLow.z(),kbot);
  int idhiZ = Min(indexHigh.z()-1,ktop);
  if ((idloX > idhiX) || (idloY > idhiY) || (idloZ > idhiZ))
    printvalues = false;
  printvalues = false;

  if (printvalues) {
    for (int ii = idloX; ii <= idhiX; ii++) {
      for (int jj = idloY; jj <= idhiY; jj++) {
	for (int kk = idloZ; kk <= idhiZ; kk++) {
	  cerr.width(14);
	  cerr << " point coordinates "<< ii << " " << jj << " " << kk << endl;
	  cerr << "Before Enthalpy Under-relaxation" << endl;
	  //	  cerr << "Diagonal coefficient = " << vars->scalarCoeff[Arches::AP][IntVector(ii,jj,kk)] << endl; 
	  cerr << "Nonlinear source     = " << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << endl; 
	}
      }
    }
  }

#endif
  fort_underelax(idxLo, idxHi, constvars->enthalpy,
		 vars->scalarCoeff[Arches::AP], vars->scalarNonlinearSrc,
		 d_underrelax);

#ifdef enthalpySolve_debug
  if (printvalues) {
    for (int ii = idloX; ii <= idhiX; ii++) {
      for (int jj = idloY; jj <= idhiY; jj++) {
	for (int kk = idloZ; kk <= idhiZ; kk++) {
	  cerr.width(14);
	  cerr << " point coordinates "<< ii << " " << jj << " " << kk << endl;
	  cerr << "After Enthalpy Under-relaxation" << endl;
	  //	  cerr << "Diagonal coefficient = " << constvars->scalarCoeff[Arches::AP][IntVector(ii,jj,kk)] << endl; 
	  cerr << "Nonlinear source     = " << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << endl; 
	}
      }
    }
  }
#endif

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
RBGSSolver::scalarLisolve(const ProcessorGroup*,
			  const Patch* patch,
			  int, double delta_t,
			  ArchesVariables* vars,
			  ArchesConstVariables* constvars,
			  CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

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
    fort_linegs(idxLo, idxHi, vars->scalar,
		vars->scalarCoeff[Arches::AE],
		vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN],
		vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT],
		vars->scalarCoeff[Arches::AB],
		vars->scalarCoeff[Arches::AP],
		vars->scalarNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif
    fort_explicit_func(idxLo, idxHi, vars->scalar, constvars->old_scalar,
		  constvars->scalarCoeff[Arches::AE], 
		  constvars->scalarCoeff[Arches::AW], 
		  constvars->scalarCoeff[Arches::AN], 
		  constvars->scalarCoeff[Arches::AS], 
		  constvars->scalarCoeff[Arches::AT], 
		  constvars->scalarCoeff[Arches::AB], 
		  constvars->scalarCoeff[Arches::AP], 
		  constvars->scalarNonlinearSrc, constvars->density_guess,
		  cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

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

//****************************************************************************
// Enthalpy Solve for Multimaterial
//****************************************************************************

void 
RBGSSolver::enthalpyLisolve(const ProcessorGroup*,
			  const Patch* patch,
			  double delta_t,
			  ArchesVariables* vars,
			  ArchesConstVariables* constvars,
			  CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

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
    fort_linegs(idxLo, idxHi, vars->enthalpy,
		vars->scalarCoeff[Arches::AE],
		vars->scalarCoeff[Arches::AW],
		vars->scalarCoeff[Arches::AN],
		vars->scalarCoeff[Arches::AS],
		vars->scalarCoeff[Arches::AT],
		vars->scalarCoeff[Arches::AB],
		vars->scalarCoeff[Arches::AP],
		vars->scalarNonlinearSrc, e1, f1, e2, f2, e3, f3, theta);
    computeScalarResidual(pc, patch, old_dw, new_dw, index, vars);
    scalarResid = vars->residScalar;
    ++scalarIter;
  } while((scalarIter < d_maxSweeps)&&((scalarResid > d_residual*nlResid)||
				      (scalarResid > trunc_conv)));
  cerr << "After scalar " << index <<" solve " << scalarIter << " " << scalarResid << endl;
  cerr << "After scalar " << index <<" solve " << nlResid << " " << trunc_conv <<  endl;
#endif
    fort_explicit_func(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
		  constvars->scalarCoeff[Arches::AE], 
		  constvars->scalarCoeff[Arches::AW], 
		  constvars->scalarCoeff[Arches::AN], 
		  constvars->scalarCoeff[Arches::AS], 
		  constvars->scalarCoeff[Arches::AT], 
		  constvars->scalarCoeff[Arches::AB], 
		  constvars->scalarCoeff[Arches::AP], 
		  constvars->scalarNonlinearSrc, constvars->density_guess,
		  cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);
     
#ifdef ARCHES_DEBUG
    cerr << " After Scalar Explicit solve : " << endl;
    cerr << "Print Enthalpy: " << endl;
    vars->enthalpy.print(cerr);
#endif


    vars->residScalar = 1.0E-7;
    vars->truncScalar = 1.0;
   
}

void 
RBGSSolver::matrixCreate(const PatchSet*,
			 const PatchSubset*)
{
}

bool
RBGSSolver::pressLinearSolve()
{
  cerr << "pressure linear solve not implemented for RBGS " << endl;
  return 0;
}

void 
RBGSSolver::copyPressSoln(const Patch*, ArchesVariables*)
{
}

void
RBGSSolver::destroyMatrix()
{
}

void 
RBGSSolver::setPressMatrix(const ProcessorGroup* ,
			    const Patch*,
			    ArchesVariables*,
			    ArchesConstVariables*,
			    const ArchesLabel*)
{
}


