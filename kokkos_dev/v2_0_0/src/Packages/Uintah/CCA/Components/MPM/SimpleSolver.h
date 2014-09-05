#ifndef SIMPLE_SOLVER_H
#define SIMPLE_SOLVER_H

#include "Solver.h"
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Math/Sparse.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>

using std::map;
using std::vector;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class SimpleSolver : public Solver {

  public:
    SimpleSolver();
    virtual ~SimpleSolver();

    virtual void initialize();

    virtual void createLocalToGlobalMapping(const ProcessorGroup* pg,
					    const PatchSet* perproc_patches,
					    const PatchSubset* patches);

    virtual void solve();

    virtual void createMatrix(const ProcessorGroup* pg,
			      const map<int,int>& diag);

    virtual void destroyMatrix(bool recursion);

    virtual void fillMatrix(int, int, double);
    
    virtual void fillVector(int, double);
    
    virtual void copyL2G(Array3<int>& l2g, const Patch* patch);

    virtual void removeFixedDOF(int num_nodes);

    virtual void finalizeMatrix();

    virtual void flushMatrix();

    virtual int getSolution(vector<double>& xSimple);

    virtual int getRHS(vector<double>& QSimple);

    virtual void assembleVector();
  private:

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    
    // Simple matrix and vectors

    SparseMatrix<double,int> KK;
    valarray<double> Q;
    valarray<double> d_x;

    inline bool compare(double num1, double num2)
      {
	double EPSILON=1.e-16;
	
	return (fabs(num1-num2) <= EPSILON);
      };

  };

}


#endif
