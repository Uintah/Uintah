//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadLinearSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;


// ****************************************************************************
// Default constructor for RadLinearSolver
// ****************************************************************************
RadLinearSolver::RadLinearSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured");
}

// ****************************************************************************
// Destructor
// ****************************************************************************
RadLinearSolver::~RadLinearSolver()
{
// Comment out the following till better place to finalize PETSC is found
//  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
RadLinearSolver::problemSetup(const ProblemSpecP& )
{
  throw InternalError("PetscSolver not configured");
}


void 
RadLinearSolver::matrixCreate(const PatchSet* ,
			  const PatchSubset* )
{
  throw InternalError("PetscSolver not configured");
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
RadLinearSolver::setMatrix(const ProcessorGroup* ,
			   const Patch* ,
			   ArchesVariables* ,
			   bool , bool , bool ,
			   CCVariable<double>& ,
			   CCVariable<double>& ,
			   CCVariable<double>& ,
			   CCVariable<double>& ,
			   CCVariable<double>& )
{
  throw InternalError("PetscSolver not configured");
}


bool
RadLinearSolver::radLinearSolve()
{
  throw InternalError("PetscSolver not configured");
}


void
RadLinearSolver::copyRadSoln(const Patch* , ArchesVariables* )
{
  throw InternalError("PetscSolver not configured");
}

void
RadLinearSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured");
}


// Shutdown PETSc
void RadLinearSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured");
}













