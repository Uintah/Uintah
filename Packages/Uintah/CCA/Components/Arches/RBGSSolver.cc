//----- RBGSSolver.cc ----------------------------------------------

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
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_velocity_fort.h>

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
  IntVector idxLo;
  IntVector idxHi;
  int ioff, joff, koff;
  switch (index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;
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

  break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;
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

  break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;
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

  break;
  default:
    throw InvalidValue("Invalid index in LinearSolver for velocity", __FILE__, __LINE__);
  }
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


