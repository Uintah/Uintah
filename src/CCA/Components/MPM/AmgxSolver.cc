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

//#define PETSC_USE_LOG

#include <sci_defs/mpi_defs.h>
#include <sci_defs/amgx_defs.h>

#include <TauProfilerForSCIRun.h>
#include <CCA/Components/MPM/AmgxSolver.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <fstream>

#include <amgx_c.h>

#include <vector>
#include <algorithm>
#include <map>
#include <iostream>


using namespace Uintah;
using namespace std;

#define AMGX_SAFE_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    fprintf(stderr, "AMGX ERROR: file %s line %6d\n", __FILE__, __LINE__); \
    AMGX_get_error_string(err, msg, 4096);\
    fprintf(stderr, "AMGX ERROR: %s\n", msg); \
    AMGX_abort(NULL,1);\
    break; \
  } \
}

void print_callback(const char *msg, int length){
  printf("%s", msg);
} 

MPMAmgxSolver::MPMAmgxSolver(string& amgx_config_string)
{
  mode = AMGX_mode_dDDI;
  AMGX_SAFE_CALL(AMGX_initialize());
  AMGX_SAFE_CALL(AMGX_initialize_plugins());
  AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
  AMGX_SAFE_CALL(AMGX_install_signal_handler());

  AMGX_SAFE_CALL(AMGX_config_create_from_file(&config, amgx_config_string.c_str()));
  //AMGX_SAFE_CALL(AMGX_config_add_parameters(&config, "exception_handling=1"));
  AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, config));
  matrix_created = false;
}

MPMAmgxSolver::~MPMAmgxSolver()
{
  //Safely finalize amgx after this goes out of scope.
  AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
  AMGX_SAFE_CALL(AMGX_config_destroy(config));
  AMGX_SAFE_CALL(AMGX_finalize_plugins());
  AMGX_SAFE_CALL(AMGX_finalize());
}


void MPMAmgxSolver::initialize()
{
  /*
    We do our initialization in the constructor.
   */
  
}
/**************************************************************
 * Creates a mapping from nodal coordinates, IntVector(x,y,z), 
 * to global matrix coordinates.  The matrix is laid out as 
 * follows:
 *
 * Proc 0 patches
 *    patch 0 nodes
 *    patch 1 nodes
 *    ...
 * Proc 1 patches
 *    patch 0 nodes
 *    patch 1 nodes
 *    ...
 * ...
 *
 * Thus the entrance at node xyz provides the global index into the
 * matrix for that nodes entry.  And each processor owns a 
 * consecutive block of those rows.  In order to translate a 
 * nodal position to the processors local position (needed when using 
 * a local array) the global index
 * of the processors first patch must be subtracted from the global
 * index of the node in question.  This will provide a zero-based index 
 * into each processors data.
 *************************************************************/
void 
MPMAmgxSolver::createLocalToGlobalMapping(const ProcessorGroup* d_myworld,
                                           const PatchSet* perproc_patches,
                                           const PatchSubset* patches,
                                           const int DOFsPerNode,
                                           const int n8or27)
{
  int numProcessors = d_myworld->size();
  d_numNodes.resize(numProcessors, 0);
  d_startIndex.resize(numProcessors);
  d_totalNodes = 0;
  //compute the total number of nodes and the global offset for each patch
  for (int p = 0; p < perproc_patches->size(); p++) {
    d_startIndex[p] = d_totalNodes;
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(p);
    for (int ps = 0; ps<patchsub->size(); ps++) {
      const Patch* patch = patchsub->get(ps);
      IntVector plowIndex(0,0,0),phighIndex(0,0,0);
      if(n8or27==8){
        plowIndex = patch->getNodeLowIndex();
        phighIndex = patch->getNodeHighIndex();
      } else if(n8or27==27){
        plowIndex = patch->getExtraNodeLowIndex();
        phighIndex = patch->getExtraNodeHighIndex();
      }

      long nn = (phighIndex[0]-plowIndex[0])*
                (phighIndex[1]-plowIndex[1])*
                (phighIndex[2]-plowIndex[2])*DOFsPerNode;

      d_petscGlobalStart[patch]=d_totalNodes;
      d_totalNodes+=nn;
      mytotal+=nn;
    }
    d_numNodes[p] = mytotal;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);
    IntVector lowIndex,highIndex;
    if(n8or27==8){
        lowIndex = patch->getNodeLowIndex();
        highIndex = patch->getNodeHighIndex() + IntVector(1,1,1);
    } else if(n8or27==27){
        lowIndex = patch->getExtraNodeLowIndex();
        highIndex = patch->getExtraNodeHighIndex() + IntVector(1,1,1);
    }
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();

    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    //For each neighbor and myself
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      IntVector plow(0,0,0),phigh(0,0,0);
      if(n8or27==8){
        plow = neighbor->getNodeLowIndex();
        phigh = neighbor->getNodeHighIndex();
      } else if(n8or27==27){
        plow = neighbor->getExtraNodeLowIndex();
        phigh = neighbor->getExtraNodeHighIndex();
      }
      //intersect my patch with my neighbor patch
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
                                 || ( high.z() < low.z() ) )
         throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
     
      //global start for this neighbor
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;

      //compute the starting index by computing the starting node index and multiplying it by the degrees of freedom per node
      petscglobalIndex += (start.z()*dnodes.x()*dnodes.y()+ start.y()*dnodes.x()+ start.x())*DOFsPerNode; 

      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
        int idx_slab = petscglobalIndex;
        petscglobalIndex += dnodes.x()*dnodes.y()*DOFsPerNode;
        
        for (int colY = low.y(); colY < high.y(); colY ++) {
          int idx = idx_slab;
          idx_slab += dnodes.x()*DOFsPerNode;
          for (int colX = low.x(); colX < high.x(); colX ++) {
            l2g[IntVector(colX, colY, colZ)] = idx;
            idx += DOFsPerNode;
          }
        }
      }
      IntVector d = high-low;
      totalNodes+=d.x()*d.y()*d.z()*DOFsPerNode;
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
  }
  d_DOFsPerNode=DOFsPerNode;

  numlrows = d_numNodes[d_myworld->myrank()];
  numlcolumns = d_numNodes[d_myworld->myrank()];
  d_B_Host.resize(numlrows);
  d_diagonal_Host.resize(numlrows);
}

void MPMAmgxSolver::fromCOOtoCSR(vector<double>& values,
				 vector<int>& row_ptrs,
				 vector<int>& col_inds){
  row_ptrs.push_back(0);
  int col_count = 0;
  for (std::map<int, double>::iterator it = matrix_values.begin();
       it != matrix_values.end();
       ++it){
    int i = it->first / numlrows;
    int j = it->first % numlrows;
    if (i == (int)row_ptrs.size()){
      row_ptrs.push_back(col_count + row_ptrs[row_ptrs.size() - 1]);
      col_count = 0;
    }
    col_count++;
    col_inds.push_back(j);
    values.push_back(it->second);
  }
  row_ptrs.push_back(col_count + row_ptrs[row_ptrs.size() - 1]);
}



void MPMAmgxSolver::solve(vector<double>& guess)
{
  //AMGX_SAFE_CALL(AMGX_vector_upload(d_flux, d_flux_Host.size(), 1, &d_flux_Host[0]));

  //We have been storing our matrix representation in a std::map where an index corresponds to row * n + col
  //we want to convert to a csr format that AMGX can read.
  vector<double> values;
  vector<int> row_ptrs;
  vector<int> col_inds;
  
  fromCOOtoCSR(values, row_ptrs, col_inds);

  ofstream myFile;
  myFile.open("MPMmatrix.dat");
    
  for (std::map<int, double>::iterator it = matrix_values.begin();
       it != matrix_values.end();
       ++it){
    int i = it->first / numlrows;
    int j = it->first % numlrows;
    myFile << i << " " << j << " " << it->second << endl;
  }
  myFile.close();
  
  const vector<int>::iterator mx = max_element(col_inds.begin(), col_inds.end());
  const vector<int>::iterator mn = min_element(col_inds.begin(), col_inds.end());

  
  if ((*mx - *mn + 1) != numlrows) {
    cout << "Matrix is not square" << endl;
    cout << "Matrix dim=" << (*mx - *mn + 1) << ", " << numlrows << endl;
  }
  cout << "Row pointer size: " << row_ptrs.size() << endl;
  cout << "Row pointer end: " << row_ptrs[row_ptrs.size() - 1] << endl;
  cout << "Column inds size: " << col_inds.size() << endl;
  cout << "Values size: " << col_inds.size() << endl;

  AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, config));
  cout << "Matrix Created" << endl;
  AMGX_SAFE_CALL(AMGX_matrix_create(&d_A, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&d_B, rsrc, mode));

  AMGX_SAFE_CALL(AMGX_vector_create(&d_diagonal, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&d_x, rsrc, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&d_t, rsrc, mode));
  
  AMGX_SAFE_CALL(AMGX_vector_create(&d_flux, rsrc, mode));

  //Now we can upload everything to the GPU
  AMGX_SAFE_CALL(AMGX_matrix_upload_all(d_A, numlrows, matrix_values.size(), 1, 1,
					&(row_ptrs[0]), &(col_inds[0]), &(values[0]),
					NULL));
  AMGX_SAFE_CALL(AMGX_vector_upload(d_B, d_B_Host.size(), 1, &d_B_Host[0]));
  //AMGX_SAFE_CALL(AMGX_vector_upload(d_t, d_t_Host.size(), 1, &d_t_Host[0]));

  if (!guess.empty()){
    AMGX_SAFE_CALL(AMGX_vector_upload(d_x, guess.size(), 1, &(guess[0])));
  } else {
    AMGX_SAFE_CALL(AMGX_vector_set_zero(d_x, numlrows, 1));
  }
  
  //Now we can setup the solver and solve the matrix system
  AMGX_SAFE_CALL(AMGX_solver_setup(solver, d_A));
  AMGX_SAFE_CALL(AMGX_solver_solve(solver, d_B, d_x));

  AMGX_SAFE_CALL(AMGX_vector_download(d_x, &d_x_Host[0]));

  AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
  cout << "Matrix Destroyed" << endl;
  AMGX_SAFE_CALL(AMGX_matrix_destroy(d_A));
  AMGX_SAFE_CALL(AMGX_vector_destroy(d_diagonal));
  AMGX_SAFE_CALL(AMGX_vector_destroy(d_x));
  AMGX_SAFE_CALL(AMGX_vector_destroy(d_t));
  AMGX_SAFE_CALL(AMGX_vector_destroy(d_flux));

}
void MPMAmgxSolver::createMatrix(const ProcessorGroup* d_myworld,
                                  const map<int,int>& dof_diag)
{
  //Here we call all of our AMGX_blank_create functions
  //d_B_Host.resize(numlrows);
  d_B_Host.assign(numlrows, 0);
}

void MPMAmgxSolver::destroyMatrix(bool recursion)
{
  if (matrix_created) {
    //We have to destroy the solver first folowed by the matrix and then the vectors.
    //Any other order will cause the program to explode.
    matrix_created = false;
  }
}

void
MPMAmgxSolver::flushMatrix()
{}

void
MPMAmgxSolver::fillMatrix(int numi, int i_indices[], int numj, int j_indices[], double value[]){
  for (int i = 0; i < numi; i++){
    for (int j = 0; j < numj; j++){
      matrix_values.insert(pair<int, double>(i_indices[i] * numlcolumns + j_indices[j], value[i * numj + j]));
    }
  }
}

void
MPMAmgxSolver::fillVector(int i,double v,bool add)
{
  //Write one element to the vector.
  if (add) {
    d_B_Host[i] += v;
  } else {
    d_B_Host[i] = v;
  }
}

void
MPMAmgxSolver::fillTemporaryVector(int i,double v)
{
  d_t_Host[i] = v;
}

void
MPMAmgxSolver::fillFluxVector(int i,double v)
{
  d_flux_Host[i] = v;
}

void
MPMAmgxSolver::assembleVector()
{}

void
MPMAmgxSolver::assembleTemporaryVector()  
{}


void
MPMAmgxSolver::assembleFluxVector()
{}


void matrixMultAdd(vector<double> values,
		   vector<int> col_inds,
		   vector<int> row_ptrs, 
		   vector<double> v1,
		   vector<double> v2,
		   vector<double> output){
  //The matrix is in CSR format 
  //For each row
  int el = 0;
  for (int i = 0; i < (int)row_ptrs.size() - 1; i++){
    double tmp = v2[i];
    //For each column index 
    for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++){
      tmp += values[el] * v1[col_inds[j]];
      el++;
    }
    output[i] = tmp;
  }
}

void
MPMAmgxSolver::applyBCSToRHS()
{
  vector<double> values;
  vector<int> row_ptrs;
  vector<int> col_inds;

  fromCOOtoCSR(values, row_ptrs, col_inds);
  matrixMultAdd(values, col_inds, row_ptrs, d_t_Host, d_B_Host, d_B_Host);
}

void
MPMAmgxSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}


void
MPMAmgxSolver::removeFixedDOF()
{
  //Set boundary elements diagonals to 1 and the corresponding vector elements to 0
  for (set<int>::iterator iter = d_DOFZero.begin(); iter != d_DOFZero.end();
       iter++) {
    int j = *iter;
    d_B_Host[j] = 0;
    matrix_values.insert(pair<int, double>(j*d_B_Host.size() + j, 1.));
  }

  for (int i = 0; i < (int)d_B_Host.size(); i++){
    map<int,double>::iterator diag_el = matrix_values.find(i * d_B_Host.size() + i);
    if (matrix_values.end() ==  diag_el || diag_el->second == 0){
      matrix_values.insert(make_pair(i * d_B_Host.size() + i, 1));
      d_B_Host[i] = 0;
    }
  }

  //Now we zero out row elements
  
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); iter++){

    //Find the first nonzero element in the nodes row 
    map<int, double>::iterator row_index = matrix_values.lower_bound((*iter) * d_B_Host.size());
    
    //Iterate through until we are in the next row setting all the elements to 0 except for the diagonal
    for (; row_index->first / (int)d_B_Host.size() == *iter; row_index++){
      if (row_index->first % (int)d_B_Host.size() != *iter)
	row_index->second = 0;
    }
  }

}


void MPMAmgxSolver::removeFixedDOFHeat()
{
    //Set boundary elements diagonals to 1 and the corresponding vector elements to 0
  for (set<int>::iterator iter = d_DOFZero.begin(); iter != d_DOFZero.end();
       iter++) {
    int j = *iter;

    d_B_Host[j] = 0;
    matrix_values.insert(pair<int, double>(j*d_B_Host.size() + j, 1.));
  }


  if( d_DOF.size() !=0)
  {
    //For each of the boundary nodes we zero out the elements in its column
    // zeroing out the columns
    for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); iter++) {
      const int index = *iter;
      vector<int>& neighbors = d_DOFNeighbors[index];

      for (vector<int>::iterator n = neighbors.begin(); n != neighbors.end();
           n++) {
        // zero out the columns
	matrix_values.insert(pair<int, double>(d_B_Host.size() * (*n) + index, 0));
      }
    }
  }

  //Now we zero out the elements other than the diagonal in the row
  //Iterate over the boundary nodes
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); iter++){

    //Find the first nonzero element in the nodes row 
    map<int, double>::iterator row_index = matrix_values.lower_bound((*iter) * d_B_Host.size());

    //Iterate through until we are in the next row setting all the elements to 0
    for (; row_index->first / (int)d_B_Host.size() == *iter; row_index++){
      if (row_index->first % (int)d_B_Host.size() != *iter)
	row_index->second = 0.0;
    }
  }

  //Now we adjust the right hand side vector by replacing elements with the boundary values from
  //the scaled temporary vector and then adding the flux vector values to specified indices.

  for (int i = 0; i < (int)d_t_Host.size(); i++){
    d_t_Host[i] = -1 * d_t_Host[i];
  }

  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); iter++){
    d_B_Host[*iter] = d_t_Host[*iter];
  }
  
  for (set<int>::iterator iter = d_DOFFlux.begin(); iter != d_DOFFlux.end(); iter++){
    d_B_Host[*iter] += d_flux_Host[*iter];
  }
}

void MPMAmgxSolver::finalizeMatrix()
{
  
}

int MPMAmgxSolver::getSolution(vector<double>& xPetsc)
{
  xPetsc.resize(d_x_Host.size());
  copy(d_x_Host.begin(), d_x_Host.end(), xPetsc.begin());
  return 0;
}

int MPMAmgxSolver::getRHS(vector<double>& QPetsc)
{
  QPetsc.resize(d_B_Host.size());
  std::copy(d_B_Host.begin(), d_B_Host.end(), QPetsc.begin());
  return 0;
}
