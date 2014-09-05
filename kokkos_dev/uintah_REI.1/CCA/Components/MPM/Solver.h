#ifndef SOLVER_H
#define SOLVER_H

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <sgi_stl_warnings_off.h>
#include <set>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>

using std::set;
using std::vector;
using std::map;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class Solver {
    public:
    
    Solver();
    virtual ~Solver();

    virtual void initialize() = 0;

    virtual void createLocalToGlobalMapping(const ProcessorGroup* pg,
                                            const PatchSet* perproc_patches,
                                            const PatchSubset* patches,
                                            const int DOFsPerNode) = 0;

    virtual void solve(vector<double>& guess) = 0;

    virtual void createMatrix(const ProcessorGroup* pg, 
			      const map<int,int>& diag) = 0;

    virtual void destroyMatrix(bool recursion) = 0;
    
    virtual void fillMatrix(int, int[], int, int j[],double v[]) = 0;

    virtual void fillVector(int, double,bool add = false) = 0;

    virtual void fillTemporaryVector(int, double) = 0;

    virtual void fillFluxVector(int, double) = 0;
    
    virtual void copyL2G(Array3<int>& l2g, const Patch* patch) = 0;

    virtual void removeFixedDOF(int num_nodes) = 0;

    virtual void removeFixedDOFHeat(int num_nodes) = 0;

    virtual void flushMatrix() = 0;

    virtual void finalizeMatrix() = 0;

    virtual int getSolution(vector<double>& xPetsc) = 0;

    virtual int getRHS(vector<double>& QPetsc) = 0;

    virtual void assembleVector() = 0;

    virtual void assembleTemporaryVector() = 0;

    virtual void applyBCSToRHS() = 0;

    set<int> d_DOF,d_DOFFlux,d_DOFZero;
    map<int,vector<int> > d_DOFNeighbors;
  };

}


#endif //end of SOLVER_H
