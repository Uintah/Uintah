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


#ifndef MPM_PETSC_SOLVER_H
#define MPM_PETSC_SOLVER_H

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h>  // Petsc uses mpi, so need to include mpi_defs.h

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/Array3.h>
#include <CCA/Components/MPM/Solver.h>

#include <set>
#include <map>
#include <vector>
#include <iostream>

#ifdef HAVE_PETSC
extern "C" {
#include "petscksp.h"
#include "petscmat.h"
}
#endif


namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class MPMPetscSolver : public Solver {
    
  public:
    MPMPetscSolver();
    ~MPMPetscSolver();
    
    void initialize();
    
    void createLocalToGlobalMapping(const ProcessorGroup* pg,
                                    const PatchSet* perproc_patches,
                                    const PatchSubset* patches,
                                    const int DOFsPerNode,
                                    const int n8or27);
    
    void solve(vector<double>& guess);

    void createMatrix(const ProcessorGroup* pg, const map<int,int>& dof_diag);

    void destroyMatrix(bool recursion);

    inline void fillMatrix(int,int[],int,int j[],double v[]);

    void fillVector(int, double,bool add = false);

    void fillTemporaryVector(int, double);

    void fillFluxVector(int, double);

    void copyL2G(Array3<int>& l2g, const Patch* patch);

    void removeFixedDOF();

    void removeFixedDOFHeat();

    void finalizeMatrix();

    void flushMatrix();

    int getSolution(vector<double>& xPetsc);

    int getRHS(vector<double>& QPetsc);

    void assembleVector();

    void assembleTemporaryVector();

    void applyBCSToRHS();

  private:

    void assembleFluxVector();

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    int d_DOFsPerNode;
    int d_iteration;

    // Petsc matrix and vectors
#ifdef HAVE_PETSC
    Mat d_A,d_C;
    Vec d_B;
    Vec d_diagonal;
    Vec d_x;
    Vec d_t,d_flux;
#endif

    inline bool compare(double num1, double num2)
      {
        double EPSILON=1.e-16;
        
        return (fabs(num1-num2) <= EPSILON);
      };

  };

#ifdef HAVE_PETSC
  inline void MPMPetscSolver::fillMatrix(int numi,int i[],int numj,
                                         int j[],double value[])
  {
    MatSetValues(d_A,numi,i,numj,j,value,ADD_VALUES);
  }

#endif

}
#endif
