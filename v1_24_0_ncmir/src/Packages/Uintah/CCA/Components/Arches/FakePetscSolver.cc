//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>
#include <Core/Exceptions/InternalError.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for PetscSolver
// ****************************************************************************
PetscSolver::PetscSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
}

void 
PetscSolver::matrixCreate(const PatchSet*,
			  const PatchSubset*)
{
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
}


bool
PetscSolver::pressLinearSolve()
{
  throw InternalError("PetscSolver not configured");
  return false;
}


void
PetscSolver::copyPressSoln(const Patch*, ArchesVariables*)
{
  throw InternalError("PetscSolver not configured");
}
  
void
PetscSolver::destroyMatrix() 
{
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
}


// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
}

void 
PetscSolver::computeEnthalpyUnderrelax(const ProcessorGroup* ,
				       const Patch*,
				       ArchesVariables*,
				       ArchesConstVariables*)
{
  throw InternalError("PetscSolver not configured");
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
  throw InternalError("PetscSolver not configured");
}
