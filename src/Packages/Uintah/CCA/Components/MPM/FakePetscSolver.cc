#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Core/Exceptions/InternalError.h>


using namespace Uintah;
using namespace SCIRun;
using namespace std;

//#define LOG
#undef LOG
#undef DEBUG_PETSC

MPMPetscSolver::MPMPetscSolver()
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::MPMPetscSolver(), specify simple instead of petsc in input file!", __FILE__, __LINE__ );

}

MPMPetscSolver::~MPMPetscSolver()
{
}


void MPMPetscSolver::initialize()
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::initialize()!", __FILE__, __LINE__ );

}

void 
MPMPetscSolver::createLocalToGlobalMapping(const ProcessorGroup* d_myworld,
					   const PatchSet* perproc_patches,
					   const PatchSubset* patches,
                                           const int DOFsPerNode)
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::createLocalToGlobalMapping()!", __FILE__, __LINE__ );
}

void MPMPetscSolver::solve()
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::solve()!", __FILE__, __LINE__ );
}

void MPMPetscSolver::createMatrix(const ProcessorGroup* d_myworld,
				  const map<int,int>& dof_diag)
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::createMatrix()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::destroyMatrix(bool recursion)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::destroyMatrix()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::flushMatrix()
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::flushMatrix()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::fillVector(int i,double v,bool add)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::fillVector()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::fillTemporaryVector(int i,double v)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::fillTemporaryVector()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::assembleVector()
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::assembleVector()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::assembleTemporaryVector()
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::assembleTemporaryVector()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::applyBCSToRHS()
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::applyBCSToRHS()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::copyL2G()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::removeFixedDOF(int num_nodes)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::removeFixedDOF()!", __FILE__, __LINE__ );
}

void MPMPetscSolver::removeFixedDOFHeat(int num_nodes)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::removeFixedDOFHeat()!", __FILE__, __LINE__ );
}


void MPMPetscSolver::finalizeMatrix()
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::finalizeMatrix()!", __FILE__, __LINE__ );
}


int MPMPetscSolver::getSolution(vector<double>& xPetsc)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::getSolution()!", __FILE__, __LINE__ );
}


int MPMPetscSolver::getRHS(vector<double>& QPetsc)
{
 throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::getRHS()!", __FILE__, __LINE__ );
}

void MPMPetscSolver::fillMatrix(int numi,int i[],int numj,
                                       int j[],double value[])
{
  throw InternalError( "Don't have PETSc so shouldn't be calling MPMPetscSolver::fillMatrix()!", __FILE__, __LINE__ );

}
