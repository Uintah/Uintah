#ifndef SOLVER_H
#define SOLVER_H

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
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
					    const PatchSubset* patches) = 0;

    virtual void solve() = 0;

    virtual void createMatrix(const ProcessorGroup* pg, 
			      const map<int,int>& diag) = 0;

    virtual void destroyMatrix(bool recursion) = 0;
    
    virtual void fillMatrix(int, int, double) = 0;
    
    virtual void fillVector(int, double) = 0;
    
    virtual void copyL2G(Array3<int>& l2g, const Patch* patch) = 0;

    virtual void removeFixedDOF(int num_nodes) = 0;

    virtual void flushMatrix() = 0;

    virtual void finalizeMatrix() = 0;

    virtual int getSolution(vector<double>& xPetsc) = 0;

    virtual int getRHS(vector<double>& QPetsc) = 0;

    virtual void assembleVector() = 0;

    set<int> d_DOF;
  };

}


#endif //end of SOLVER_H
