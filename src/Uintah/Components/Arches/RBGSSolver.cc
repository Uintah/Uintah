//----- RBGSSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/RBGSSolver.h>
#include <Uintah/Components/Arches/PressureSolver.h>
#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/Source.h>
#include <Uintah/Components/Arches/BoundaryCondition.h>
#include <Uintah/Components/Arches/TurbulenceModel.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/SFCXVariable.h>
#include <Uintah/Grid/SFCYVariable.h>
#include <Uintah/Grid/SFCZVariable.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Components/Arches/Arches.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Components/Arches/ArchesVariables.h>
#include <Uintah/Components/Arches/ArchesLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <SCICore/Containers/Array1.h>
using namespace Uintah::ArchesSpace;
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

}


//****************************************************************************
// Actual calculation of order of magnitude term for pressure equation
//****************************************************************************
void 
RBGSSolver::computePressOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars)
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
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->pressure.getPointer(),
		 vars->pressCoeff[Arches::AP].getPointer(), 
		 vars->pressNonlinearSrc.getPointer(), 
		 &d_underrelax);

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
  bool lswpwe = true;
  bool lswpsn = true;
  bool lswpbt = true;
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
  double theta = 0.5;
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
  } while((pressIter < d_maxSweeps)&&((pressResid > d_residual*nlResid)||
				      (pressResid > trunc_conv)));
  cerr << "After pressure solve " << pressIter << " " << pressResid << endl;
  cerr << "After pressure solve " << nlResid << " " << trunc_conv <<  endl;
  

}

//****************************************************************************
// Actual compute of Velocity residual
//****************************************************************************

void 
RBGSSolver::computeVelResidual(const ProcessorGroup* ,
			       const Patch* patch,
			       DataWarehouseP& old_dw ,
			       DataWarehouseP& new_dw, 
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

    break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity");
  }
}


//****************************************************************************
// Actual calculation of order of magnitude term for Velocity equation
//****************************************************************************
void 
RBGSSolver::computeVelOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars)
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
				 DataWarehouseP& old_dw ,
				 DataWarehouseP& new_dw, 
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
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonLinSrc.getPointer(),
		   &d_underrelax);


#endif
    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->vVelocity.getPointer(),
		   vars->vVelocityCoeff[Arches::AP].getPointer(), 
		   vars->vVelNonLinSrc.getPointer(),
		   &d_underrelax);


#endif
    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		   idxLo.get_pointer(), idxHi.get_pointer(),
		   vars->uVelocity.getPointer(),
		   vars->uVelocityCoeff[Arches::AP].getPointer(), 
		   vars->uVelNonLinSrc.getPointer(),
		   &d_underrelax);

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
RBGSSolver::velocityLisolve(const ProcessorGroup* ,
			    const Patch* patch,
			    DataWarehouseP& old_dw ,
			    DataWarehouseP& new_dw, 
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
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->uVelocity.getPointer(),
		    vars->uVelocityCoeff[Arches::AP].getPointer(), 
		    vars->uVelocityCoeff[Arches::AE].getPointer(), 
		    vars->uVelocityCoeff[Arches::AW].getPointer(), 
		    vars->uVelocityCoeff[Arches::AN].getPointer(), 
		    vars->uVelocityCoeff[Arches::AS].getPointer(), 
		    vars->uVelocityCoeff[Arches::AT].getPointer(), 
		    vars->uVelocityCoeff[Arches::AB].getPointer(), 
		    vars->uVelNonLinSrc.getPointer());


#endif
    break;
    case Arches::YDIR:
    domLo = vars->vVelocity.getFortLowIndex();
    domHi = vars->vVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->vVelocity.getPointer(),
		    vars->vVelocityCoeff[Arches::AP].getPointer(), 
		    vars->vVelocityCoeff[Arches::AE].getPointer(), 
		    vars->vVelocityCoeff[Arches::AW].getPointer(), 
		    vars->vVelocityCoeff[Arches::AN].getPointer(), 
		    vars->vVelocityCoeff[Arches::AS].getPointer(), 
		    vars->vVelocityCoeff[Arches::AT].getPointer(), 
		    vars->vVelocityCoeff[Arches::AB].getPointer(), 
		    vars->vVelNonLinSrc.getPointer());


#endif
    break;
    case Arches::ZDIR:
    domLo = vars->wVelocity.getFortLowIndex();
    domHi = vars->wVelocity.getFortHighIndex();
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
  //fortran call
#ifdef WONT_COMPILE_YET
    FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		    idxLo.get_pointer(), idxHi.get_pointer(),
		    vars->wVelocity.getPointer(),
		    vars->wVelocityCoeff[Arches::AP].getPointer(), 
		    vars->wVelocityCoeff[Arches::AE].getPointer(), 
		    vars->wVelocityCoeff[Arches::AW].getPointer(), 
		    vars->wVelocityCoeff[Arches::AN].getPointer(), 
		    vars->wVelocityCoeff[Arches::AS].getPointer(), 
		    vars->wVelocityCoeff[Arches::AT].getPointer(), 
		    vars->wVelocityCoeff[Arches::AB].getPointer(), 
		    vars->wVelNonLinSrc.getPointer());



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
RBGSSolver::computeScalarOrderOfMagnitude(const ProcessorGroup* pc,
				const Patch* patch,
				DataWarehouseP& old_dw,
				DataWarehouseP& new_dw, ArchesVariables* vars)
{

  //&vars->truncScalar

}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
RBGSSolver::computeScalarUnderrelax(const ProcessorGroup* ,
				    const Patch* patch,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw, 
				    int index,
				    ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  //fortran call
#ifdef WONT_COMPILE_YET
  FORT_UNDERELAX(domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 vars->scalar.getPointer(),
		 vars->scalarCoeff[Arches::AP].getPointer(), 
		 vars->scalarCoeff[Arches::AE].getPointer(), 
		 vars->scalarCoeff[Arches::AW].getPointer(), 
		 vars->scalarCoeff[Arches::AN].getPointer(), 
		 vars->scalarCoeff[Arches::AS].getPointer(), 
		 vars->scalarCoeff[Arches::AT].getPointer(), 
		 vars->scalarCoeff[Arches::AB].getPointer(), 
		 vars->scalarNonLinSrc.getPointer(), 
		 &d_underrelax);
#endif
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RBGSSolver::scalarLisolve(const ProcessorGroup* ,
			  const Patch* patch,
			  DataWarehouseP& old_dw,
			  DataWarehouseP& new_dw, 
			  int index,
			  ArchesVariables* vars)
{
  // Get the patch bounds and the variable bounds
  IntVector domLo = vars->scalar.getFortLowIndex();
  IntVector domHi = vars->scalar.getFortHighIndex();
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

#ifdef WONT_COMPILE_YET
  //fortran call for red-black GS solver
  FORT_RBGSLISOLV(domLo.get_pointer(), domHi.get_pointer(),
		  idxLo.get_pointer(), idxHi.get_pointer(),
		  vars->scalar.getPointer(),
		  vars->scalarCoeff[Arches::AP].getPointer(), 
		  vars->scalarCoeff[Arches::AE].getPointer(), 
		  vars->scalarCoeff[Arches::AW].getPointer(), 
		  vars->scalarCoeff[Arches::AN].getPointer(), 
		  vars->scalarCoeff[Arches::AS].getPointer(), 
		  vars->scalarCoeff[Arches::AT].getPointer(), 
		  vars->scalarCoeff[Arches::AB].getPointer(), 
		  vars->scalarNonLinSrc.getPointer());
		  &d_underrelax);
#endif
}

//
// $Log$
// Revision 1.17  2000/08/11 21:26:36  rawat
// added linear solver for pressure eqn
//
// Revision 1.16  2000/08/01 23:28:43  skumar
// Added residual calculation procedure and modified templates in linear
// solver.  Added template for order-of-magnitude term calculation.
//
// Revision 1.15  2000/08/01 06:18:38  bbanerje
// Made ScalarSolver similar to PressureSolver and MomentumSolver.
//
// Revision 1.14  2000/07/28 02:31:00  rawat
// moved all the labels in ArchesLabel. fixed some bugs and added matrix_dw to store matrix
// coeffecients
//
// Revision 1.13  2000/07/08 23:42:55  bbanerje
// Moved all enums to Arches.h and made corresponding changes.
//
// Revision 1.12  2000/07/08 08:03:34  bbanerje
// Readjusted the labels upto uvelcoef, removed bugs in CellInformation,
// made needed changes to uvelcoef.  Changed from StencilMatrix::AE etc
// to Arches::AE .. doesn't like enums in templates apparently.
//
// Revision 1.11  2000/06/29 22:56:43  bbanerje
// Changed FCVars to SFC[X,Y,Z]Vars, and added the neceesary getIndex calls.
//
// Revision 1.10  2000/06/22 23:06:37  bbanerje
// Changed velocity related variables to FCVariable type.
// ** NOTE ** We may need 3 types of FCVariables (one for each direction)
//
// Revision 1.9  2000/06/21 07:51:01  bbanerje
// Corrected new_dw, old_dw problems, commented out intermediate dw (for now)
// and made the stuff go through schedule_time_advance.
//
// Revision 1.8  2000/06/18 01:20:16  bbanerje
// Changed names of varlabels in source to reflect the sequence of tasks.
// Result : Seg Violation in addTask in MomentumSolver
//
// Revision 1.7  2000/06/17 07:06:26  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.6  2000/06/12 21:29:59  bbanerje
// Added first Fortran routines, added Stencil Matrix where needed,
// removed unnecessary CCVariables (e.g., sources etc.)
//
// Revision 1.5  2000/06/07 06:13:56  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.4  2000/06/04 22:40:15  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//

