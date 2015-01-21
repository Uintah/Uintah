//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/Radiation/RadLinearSolver.h>
#include <SCIRun/Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;


// ****************************************************************************
// Default constructor for RadLinearSolver
// ****************************************************************************
RadLinearSolver::RadLinearSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
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
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
RadLinearSolver::matrixCreate(const PatchSet* ,
			  const PatchSubset* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
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
			   CCVariable<double>& ,
			   CCVariable<double>& ,
			   CCVariable<double>& ,
			   CCVariable<double>& )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


bool
RadLinearSolver::radLinearSolve()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void
RadLinearSolver::copyRadSoln(const Patch* , ArchesVariables* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

void
RadLinearSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


// Shutdown PETSc
void RadLinearSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}













