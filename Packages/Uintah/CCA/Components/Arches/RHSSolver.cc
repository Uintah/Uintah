//----- RHSSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/RHSSolver.h>
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

#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_scalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/explicit_vel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/computeVel_fort.h>

//****************************************************************************
// Default constructor for RHSSolver
//****************************************************************************
RHSSolver::RHSSolver()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
RHSSolver::~RHSSolver()
{
}

//****************************************************************************
// compute hat velocity for explicit projection
//****************************************************************************
void 
RHSSolver::calculateHatVelocity(const ProcessorGroup* /*pc*/,
			    const Patch* patch,
			    int index, double delta_t,
			    CellInformation* cellinfo,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)

{
  // Get the patch bounds and the variable bounds
  IntVector idxLo;
  IntVector idxHi;
  // for explicit solver
  int ioff, joff, koff;

  switch (index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->uVelRhoHat,
		      constvars->uVelocity,
		      vars->uVelocityCoeff[Arches::AE], 
		      vars->uVelocityCoeff[Arches::AW], 
		      vars->uVelocityCoeff[Arches::AN], 
		      vars->uVelocityCoeff[Arches::AS], 
		      vars->uVelocityCoeff[Arches::AT], 
		      vars->uVelocityCoeff[Arches::AB], 
		      vars->uVelocityCoeff[Arches::AP], 
		      vars->uVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		      delta_t, ioff, joff, koff);
    break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->vVelRhoHat,
		      constvars->vVelocity,
		      vars->vVelocityCoeff[Arches::AE], 
		      vars->vVelocityCoeff[Arches::AW], 
		      vars->vVelocityCoeff[Arches::AN], 
		      vars->vVelocityCoeff[Arches::AS], 
		      vars->vVelocityCoeff[Arches::AT], 
		      vars->vVelocityCoeff[Arches::AB], 
		      vars->vVelocityCoeff[Arches::AP], 
		      vars->vVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
		      delta_t, ioff, joff, koff);

    break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;

    fort_explicit_vel(idxLo, idxHi, 
		      vars->wVelRhoHat,
		      constvars->wVelocity,
		      vars->wVelocityCoeff[Arches::AE], 
		      vars->wVelocityCoeff[Arches::AW], 
		      vars->wVelocityCoeff[Arches::AN], 
		      vars->wVelocityCoeff[Arches::AS], 
		      vars->wVelocityCoeff[Arches::AT], 
		      vars->wVelocityCoeff[Arches::AB], 
		      vars->wVelocityCoeff[Arches::AP], 
		      vars->wVelNonlinearSrc,
		      constvars->new_density,
		      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		      delta_t, ioff, joff, koff);


    break;
  default:
    throw InvalidValue("Invalid index in RHSSolver for hat velocity", __FILE__, __LINE__);
  }
}

//****************************************************************************
// compute velocity from hat velocity and pressure gradient
//****************************************************************************

void 
RHSSolver::calculateVelocity(const ProcessorGroup* ,
			      const Patch* patch,
			      double delta_t,
			      int index,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars)
{
  
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;
  IntVector domLoU;
  IntVector domHiU;
  switch(index) {
  case Arches::XDIR:
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;

    fort_computevel(idxLoU, idxHiU, vars->uVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dxpw);
    break;
  case Arches::YDIR:
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;

    fort_computevel(idxLoU, idxHiU, vars->vVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dyps);

    break;
  case Arches::ZDIR:
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0; joff = 0; koff = 1;

    fort_computevel(idxLoU, idxHiU, vars->wVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dzpb);

    break;
  default:
    throw InvalidValue("Invalid index in RHSSolver::calculateVelocity", __FILE__, __LINE__);
  }

}


//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
RHSSolver::scalarLisolve(const ProcessorGroup*,
			  const Patch* patch,
			  double delta_t,
			  ArchesVariables* vars,
			  ArchesConstVariables* constvars,
			  CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

    fort_explicit_scalar(idxLo, idxHi, vars->scalar, constvars->old_scalar,
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
// Enthalpy Solve
//****************************************************************************

void 
RHSSolver::enthalpyLisolve(const ProcessorGroup*,
			  const Patch* patch,
			  double delta_t,
			  ArchesVariables* vars,
			  ArchesConstVariables* constvars,
			  CellInformation* cellinfo)
{
  // Get the patch bounds and the variable bounds
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

    fort_explicit_scalar(idxLo, idxHi, vars->enthalpy, constvars->old_enthalpy,
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

