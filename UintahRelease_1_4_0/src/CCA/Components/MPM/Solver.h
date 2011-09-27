/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef SOLVER_H
#define SOLVER_H

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/Array3.h>
#include <set>
#include <vector>
#include <map>

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
                                            const int DOFsPerNode,
                                            const int n8or27) = 0;

    virtual void solve(vector<double>& guess) = 0;

    virtual void createMatrix(const ProcessorGroup* pg, 
                              const map<int,int>& diag) = 0;

    virtual void destroyMatrix(bool recursion) = 0;
    
    virtual void fillMatrix(int, int[], int, int j[],double v[]) = 0;

    virtual void fillVector(int, double,bool add = false) = 0;

    virtual void fillTemporaryVector(int, double) = 0;

    virtual void fillFluxVector(int, double) = 0;
    
    virtual void copyL2G(Array3<int>& l2g, const Patch* patch) = 0;

    virtual void removeFixedDOF() = 0;

    virtual void removeFixedDOFHeat() = 0;

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
