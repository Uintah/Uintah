#ifndef SIMPLE_SOLVER_H
#define SIMPLE_SOLVER_H

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Math/Sparse.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <set>
#include <vector>
#include <sgi_stl_warnings_on.h>

using std::map;
using std::set;
using std::vector;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class SimpleSolver {

  public:
    SimpleSolver();
    ~SimpleSolver();

    void initialize();

    void createLocalToGlobalMapping(const ProcessorGroup* pg,
					    const PatchSet* perproc_patches,
					    const PatchSubset* patches);

    void solve();

    void createMatrix(const ProcessorGroup* pg,
			      const map<int,int>& diag);

    void destroyMatrix(bool recursion);

    void fillMatrix(int,int[],int,int j[],double v[]);
    
    void fillVector(int, double);
    
    void copyL2G(Array3<int>& l2g, const Patch* patch);

    void removeFixedDOF(int num_nodes);

    void finalizeMatrix();

    void flushMatrix();

    int getSolution(vector<double>& xSimple);

    int getRHS(vector<double>& QSimple);

    void assembleVector();

    set<int> d_DOF;
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
