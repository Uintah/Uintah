//----- PetscSolver.cc ----------------------------------------------

#include <CCA/Components/Models/Radiation/Models_PetscSolver.h>
#include <SCIRun/Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for Models_PetscSolver
// ****************************************************************************
Models_PetscSolver::Models_PetscSolver(const ProcessorGroup* myworld) :
  d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
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
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
Models_PetscSolver::outputProblemSpec(ProblemSpecP& /*params*/)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void 
Models_PetscSolver::matrixCreate(const PatchSet* ,
                                 const PatchSubset* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
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
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


bool
Models_PetscSolver::radLinearSolve()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


void
Models_PetscSolver::copyRadSoln(const Patch* , RadiationVariables* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

void
Models_PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


// Shutdown PETSc
void
Models_PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

