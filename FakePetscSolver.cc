//----- PetscSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/PetscSolver.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// ****************************************************************************
// Default constructor for PetscSolver
// ****************************************************************************
PetscSolver::PetscSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
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

//&vars->truncPress

}

void 
PetscSolver::matrixCreate(const LevelP&, LoadBalancer*)
{
}

// ****************************************************************************
// Actual compute of pressure underrelaxation
// ****************************************************************************
void 
PetscSolver::computePressUnderrelax(const ProcessorGroup*,
				  const Patch*,
				  ArchesVariables*)
{
}

// ****************************************************************************
// Fill linear parallel matrix
// ****************************************************************************
void 
PetscSolver::setPressMatrix(const ProcessorGroup* ,
			    const Patch*,
			    ArchesVariables*,
			    const ArchesLabel*)
{
}


bool
PetscSolver::pressLinearSolve()
{
  return false;
}


void
PetscSolver::copyPressSoln(const Patch*, ArchesVariables*)
{
}
  
void
PetscSolver::destroyMatrix() 
{
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
}


// Shutdown PETSc
void PetscSolver::finalizeSolver()
{
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
}



//****************************************************************************
// Velocity Underrelaxation
//****************************************************************************
void 
PetscSolver::computeVelUnderrelax(const ProcessorGroup* ,
				  const Patch*,
				  int,
				  ArchesVariables*)
{
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
}

//****************************************************************************
// Scalar Underrelaxation
//****************************************************************************
void 
PetscSolver::computeScalarUnderrelax(const ProcessorGroup* ,
				    const Patch*,
				    int,
				    ArchesVariables*)
{
}

//****************************************************************************
// Scalar Solve
//****************************************************************************
void 
PetscSolver::scalarLisolve(const ProcessorGroup*,
			  const Patch*,
			  int, double,
			  ArchesVariables*,
			  CellInformation*,
			  const ArchesLabel*)
{
}
