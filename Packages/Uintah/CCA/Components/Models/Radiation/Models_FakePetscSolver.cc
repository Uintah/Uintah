//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_PetscSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for Models_PetscSolver
// ****************************************************************************
Models_PetscSolver::Models_PetscSolver(const ProcessorGroup* myworld) :
  d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured");
}

// ****************************************************************************
// Destructor
// ****************************************************************************
Models_PetscSolver::~Models_PetscSolver()
{
// Comment out the following till better place to finalize PETSC is found
//  finalizeSolver();
}

// ****************************************************************************
// Problem setup
// ****************************************************************************
void 
Models_PetscSolver::problemSetup(const ProblemSpecP& /*params*/, 
                                 bool /* shradiation */)
{
  throw InternalError("PetscSolver not configured");
}


void 
Models_PetscSolver::matrixCreate(const PatchSet* ,
                                 const PatchSubset* )
{
  throw InternalError("PetscSolver not configured");
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
Models_PetscSolver::setMatrix(const ProcessorGroup* ,
                           const Patch* ,
                           RadiationVariables* ,
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
  throw InternalError("PetscSolver not configured");
}


bool
Models_PetscSolver::radLinearSolve()
{
  throw InternalError("PetscSolver not configured");
}


void
Models_PetscSolver::copyRadSoln(const Patch* , RadiationVariables* )
{
  throw InternalError("PetscSolver not configured");
}

void
Models_PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured");
}


// Shutdown PETSc
void
Models_PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured");
}

