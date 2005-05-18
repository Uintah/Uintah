#ifndef MPM_PETSC_SOLVER_H
#define MPM_PETSC_SOLVER_H

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h>  // Petsc uses mpi, so need to include mpi_defs.h

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>

#include <sgi_stl_warnings_off.h>
#include <set>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <iostream>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
#include "petscmat.h"
}
#endif

using std::set;
using std::map;
using std::vector;
using namespace std;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class MPMPetscSolver {

  public:
    MPMPetscSolver();
    ~MPMPetscSolver();

    void initialize();

    void createLocalToGlobalMapping(const ProcessorGroup* pg,
                                    const PatchSet* perproc_patches,
                                    const PatchSubset* patches,
                                    const int DOFsPerNode);

    void solve();

    void createMatrix(const ProcessorGroup* pg,
			      const map<int,int>& dof_diag);

    void destroyMatrix(bool recursion);

#ifdef HAVE_PETSC
    inline void fillMatrix(int,int[],int,int j[],PetscScalar v[]);
#endif

    void fillVector(int, double);

    void fillTemporaryVector(int, double);

    void copyL2G(Array3<int>& l2g, const Patch* patch);

    void removeFixedDOF(int num_nodes);

    void finalizeMatrix();

    void flushMatrix();

    int getSolution(vector<double>& xPetsc);

    int getRHS(vector<double>& QPetsc);

    void assembleVector();

    void assembleTemporaryVector();

    void applyBCSToRHS();

    set<int> d_DOF;
  private:

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    int d_DOFsPerNode;

    // Petsc matrix and vectors
#ifdef HAVE_PETSC
    Mat d_A;
    Vec d_B;
    Vec d_diagonal;
    Vec d_x;
    Vec d_t;
#endif
    inline bool compare(double num1, double num2)
      {
	double EPSILON=1.e-16;
	
	return (fabs(num1-num2) <= EPSILON);
      };

  };

#ifdef HAVE_PETSC
inline void MPMPetscSolver::fillMatrix(int numi,int i[],int numj,
                                       int j[],PetscScalar value[])
{
    MatSetValues(d_A,numi,i,numj,j,value,ADD_VALUES);
}
#endif

}
#endif
