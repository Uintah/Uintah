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


// ****************************************************************************
// Actual compute of pressure residual
// ****************************************************************************
void 
PetscSolver::computePressResidual(const ProcessorGroup*,
                                  const Patch*,
                                  DataWarehouseP&,
                                  DataWarehouseP&,
                                  ArchesVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


// ****************************************************************************
// Actual calculation of order of magnitude term for pressure equation
// ****************************************************************************
void 
PetscSolver::computePressOrderOfMagnitude(const ProcessorGroup* ,
                                          const Patch* ,
                                          DataWarehouseP& ,
                                          DataWarehouseP& , ArchesVariables* )
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
// Actual compute of pressure underrelaxation
// ****************************************************************************
void 
PetscSolver::computePressUnderrelax(const ProcessorGroup*,
                                    const Patch*,
                                    ArchesVariables*,
                                    ArchesConstVariables*)
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

// ****************************************************************************
// Actual linear solve for pressure
// ****************************************************************************
void 
PetscSolver::pressLisolve(const ProcessorGroup*,
                          const Patch*,
                          DataWarehouseP&,
                          DataWarehouseP&,
                          ArchesVariables*,
                          const ArchesLabel*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

//****************************************************************************
// Actual compute of Velocity residual
//****************************************************************************

void 
PetscSolver::computeVelResidual(const ProcessorGroup* ,
                                const Patch*,
                                DataWarehouseP&,
                                DataWarehouseP&, int,
                                ArchesVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


//****************************************************************************
// Actual calculation of order of magnitude term for Velocity equation
//****************************************************************************
void 
PetscSolver::computeVelOrderOfMagnitude(const ProcessorGroup* ,
                                        const Patch* ,
                                        DataWarehouseP& ,
                                        DataWarehouseP& , ArchesVariables* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}



//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
PetscSolver::computeVelUnderrelax(const ProcessorGroup* ,
                                  const Patch*,
                                  int,
                                  ArchesVariables*,
                                  ArchesConstVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


//****************************************************************************
// Velocity Solve
//****************************************************************************
void 
PetscSolver::velocityLisolve(const ProcessorGroup*,
                             const Patch*,
                             int,
                             double,
                             ArchesVariables*,
                             CellInformation*,
                             const ArchesLabel*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

//****************************************************************************
// Calculate Scalar residuals
//****************************************************************************
void 
PetscSolver::computeScalarResidual(const ProcessorGroup* ,
                                   const Patch*,
                                   DataWarehouseP& ,
                                   DataWarehouseP& , 
                                   int,
                                   ArchesVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


//****************************************************************************
// Actual calculation of order of magnitude term for Scalar equation
//****************************************************************************
void 
PetscSolver::computeScalarOrderOfMagnitude(const ProcessorGroup* ,
                                           const Patch* ,
                                           DataWarehouseP& ,
                                           DataWarehouseP& , ArchesVariables* )
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
PetscSolver::computeScalarUnderrelax(const ProcessorGroup* ,
                                     const Patch*,
                                     int,
                                     ArchesVariables*,
                                     ArchesConstVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
PetscSolver::scalarLisolve(const ProcessorGroup*,
                           const Patch*,
                           int, double,
                           ArchesVariables*,
                           ArchesConstVariables*,
                           CellInformation*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}

void 
PetscSolver::computeEnthalpyUnderrelax(const ProcessorGroup* ,
                                       const Patch*,
                                       ArchesVariables*,
                                       ArchesConstVariables*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}


//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
PetscSolver::enthalpyLisolve(const ProcessorGroup*,
                             const Patch*,
                             double ,
                             ArchesVariables*,
                             ArchesConstVariables*,
                             CellInformation*)
{
  throw InternalError("PetscSolver not configured", __FILE__, __LINE__);
}
