#ifndef MPM_PETSC_SOLVER_H
#define MPM_PETSC_SOLVER_H

#include "Solver.h"
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscsles.h"
}
#endif
using std::map;
using std::vector;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class MPMPetscSolver : public Solver {

  public:
    MPMPetscSolver();
    virtual ~MPMPetscSolver();

    virtual void initialize();

    virtual void createLocalToGlobalMapping(const ProcessorGroup* pg,
					    const PatchSet* perproc_patches,
					    const PatchSubset* patches);

    virtual void solve();

    virtual void createMatrix(const ProcessorGroup* pg,
			      const map<int,int>& dof_diag);

    virtual void destroyMatrix(bool recursion);

    virtual void fillMatrix(int, int, double);
    
    virtual void fillVector(int, double);
    
    virtual void copyL2G(Array3<int>& l2g, const Patch* patch);

    virtual void removeFixedDOF(int num_nodes);

    virtual void finalizeMatrix();

    virtual void flushMatrix();

    virtual int getSolution(vector<double>& xPetsc);

    virtual int getRHS(vector<double>& QPetsc);

    virtual void assembleVector();
  private:

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    
    // Petsc matrix and vectors
#ifdef HAVE_PETSC
    Mat d_A;
    Vec d_B;
    Vec d_diagonal;
    Vec d_x;
    SLES sles;
#endif
    inline bool compare(double num1, double num2)
      {
	double EPSILON=1.e-16;
	
	return (fabs(num1-num2) <= EPSILON);
      };

  };

}


#endif
