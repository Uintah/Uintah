#ifndef PETSC_SOLVER_H
#define PETSC_SOLVER_H

#include "Solver.h"
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <map>
#include <vector>

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

  class PetscSolver : public Solver {

  public:
    PetscSolver();
    virtual ~PetscSolver();

    virtual void initialize();

    virtual void createLocalToGlobalMapping(const ProcessorGroup* pg,
					    const PatchSet* perproc_patches,
					    const PatchSubset* patches);

    virtual void solve();

    virtual void createMatrix(const ProcessorGroup* pg);

    virtual void destroyMatrix(bool recursion);

    virtual void setRhs();

    virtual void fillMatrix(int, int, double);
    
    virtual void fillVector(int, double);
    
    virtual void zeroRow(int);

    virtual void zeroColumn(int);

    virtual void copyL2G(Array3<int>& l2g, const Patch* patch);

    virtual void removeFixedDOF(set<int>& fixedDOF, int num_nodes);

    virtual void finalizeMatrix();

    virtual void flushMatrix();

    virtual int getSolution(vector<double>& xPetsc);

    virtual void assembleVector();
  private:

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    
    // Petsc matrix and vectors

    Mat d_A;
    Vec d_B;
    Vec d_diagonal;
    Vec d_x;
    SLES sles;

    inline bool compare(double num1, double num2)
      {
	double EPSILON=1.e-16;
	
	return (fabs(num1-num2) <= EPSILON);
      };

  };

}


#endif
