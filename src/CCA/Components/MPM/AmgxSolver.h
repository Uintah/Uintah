
/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef MPM_AMGX_SOLVER_H
#define MPM_AMGX_SOLVER_H

#include <sci_defs/mpi_defs.h>
#include <sci_defs/petsc_defs.h> 
#include <sci_defs/amgx_defs.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/Array3.h>
#include <CCA/Components/MPM/Solver.h>

#include <set>
#include <map>
#include <vector>
#include <iostream>
#include <string>

#ifdef HAVE_AMGX
#include <amgx_c.h>
#endif



namespace Uintah {

  class ProcessorGroup;
  class Patch;

  class MPMAmgxSolver : public Solver {
    
  public:
    
    MPMAmgxSolver(std::string& config_name);
    ~MPMAmgxSolver();
    
    void initialize();
    
    void createLocalToGlobalMapping(const ProcessorGroup* pg,
                                    const PatchSet* perproc_patches,
                                    const PatchSubset* patches,
                                    const int DOFsPerNode,
                                    const int n8or27);
    
    void solve(vector<double>& guess);

    void createMatrix(const ProcessorGroup* pg, const map<int,int>& dof_diag);

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

    int numlcolumns;
    int numlrows;
    void fromCOOtoCSR(vector<double>&, vector<int>&, vector<int>&);
    
#ifdef HAVE_AMGX
    //Use device memory with data in the form of double floating point
    //on both the device and the host.
    AMGX_Mode mode;
    AMGX_config_handle config;
    AMGX_resources_handle rsrc;
    AMGX_solver_handle solver;
    AMGX_matrix_handle d_A;
    
    AMGX_vector_handle d_B;
    AMGX_vector_handle d_x;

    bool matrix_created;
    map<int, double> matrix_values;

    //Each of the vectors will be stored host side before we write them to
    // the amgx format
    vector<double> d_B_Host;
    vector<double> d_diagonal_Host;
    vector<double> d_x_Host;
    vector<double> d_t;
    vector<double> d_flux;

    inline bool compare(double num1, double num2)
      {
        double EPSILON=1.e-16;
        
        return (fabs(num1-num2) <= EPSILON);
      };

    
#endif
  };


}
#endif
