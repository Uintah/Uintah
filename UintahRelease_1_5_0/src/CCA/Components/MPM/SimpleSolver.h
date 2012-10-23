/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef SIMPLE_SOLVER_H
#define SIMPLE_SOLVER_H

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Math/Sparse.h>
#include <CCA/Components/MPM/Solver.h>
#include <map>
#include <set>
#include <vector>

using std::map;
using std::set;
using std::vector;

namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class SimpleSolver : public Solver {

  public:
    SimpleSolver();
    ~SimpleSolver();

    void initialize();

    void createLocalToGlobalMapping(const ProcessorGroup* pg,
                                    const PatchSet* perproc_patches,
                                    const PatchSubset* patches,
                                    const int DOFsPerNode,
                                    const int n8or27);

    void solve(vector<double>& guess);

    void createMatrix(const ProcessorGroup* pg, const map<int,int>& diag);

    void destroyMatrix(bool recursion);

    void fillMatrix(int,int[],int,int j[],double v[]);

    void fillVector(int, double,bool add = false);

    void fillTemporaryVector(int, double);

    void fillFluxVector(int, double);
    
    void copyL2G(Array3<int>& l2g, const Patch* patch);

    void removeFixedDOF();

    void removeFixedDOFHeat();

    void finalizeMatrix();

    void flushMatrix();

    int getSolution(vector<double>& xSimple);

    int getRHS(vector<double>& QSimple);

    void assembleVector();

    void assembleTemporaryVector();

    void applyBCSToRHS();

    void printMatrix();

    void printRHS();

    map<int,double> d_BC;
  private:

    // Needed for the local to global mappings
    map<const Patch*, int> d_petscGlobalStart;
    map<const Patch*, Array3<int> > d_petscLocalToGlobal;
    vector<int> d_numNodes,d_startIndex;
    int d_totalNodes;
    
    // Simple matrix and vectors

    SparseMatrix<double,int> KK;
    std::valarray<double> Q;
    std::valarray<double> d_x;
    std::valarray<double> d_t,d_flux;

    inline bool compare(double num1, double num2)
      {
        double EPSILON=1.e-16;
        
        return (fabs(num1-num2) <= EPSILON);
      };

  };

}


#endif
