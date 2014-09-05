//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for PetscSolver
// ****************************************************************************
PetscSolver::PetscSolver(const ProcessorGroup* myworld) :
  d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Destructor
// ****************************************************************************
PetscSolver::~PetscSolver()
{
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
PetscSolver::problemSetup(const ProblemSpecP&)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
PetscSolver::matrixCreate(const PatchSet*,
                          const PatchSubset*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
PetscSolver::setPressMatrix(const ProcessorGroup* ,
                            const Patch*,
                            ArchesVariables*,
                            ArchesConstVariables*,
                            const ArchesLabel*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


bool
PetscSolver::pressLinearSolve()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
  return false;
}


void
PetscSolver::copyPressSoln(const Patch*, ArchesVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}
  
void
PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}



// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

