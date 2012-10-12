/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/SimpleSolver.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <iostream>
#include <vector>

using namespace Uintah;
using namespace std;

SimpleSolver::SimpleSolver()
{

}

SimpleSolver::~SimpleSolver()
{

}

void SimpleSolver::initialize()
{
  
}

void SimpleSolver::createLocalToGlobalMapping(const ProcessorGroup* d_myworld,
                                              const PatchSet* perproc_patches,
                                              const PatchSubset* patches,
                                              const int DOFsPerNode,
                                              const int n8or27)
{

  int numProcessors = d_myworld->size();
  d_numNodes.resize(numProcessors, 0);
  d_startIndex.resize(numProcessors);
  d_totalNodes = 0;

   for (int p = 0; p < perproc_patches->size(); p++) {
    d_startIndex[p] = d_totalNodes;
    int mytotal = 0;
    const PatchSubset* patchsub = perproc_patches->getSubset(p);
    for (int ps = 0; ps<patchsub->size(); ps++) {
    const Patch* patch = patchsub->get(ps);
    IntVector plowIndex = patch->getNodeLowIndex();
    IntVector phighIndex = patch->getNodeHighIndex();

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
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex() + IntVector(1,1,1);
    Array3<int> l2g(lowIndex, highIndex);
    l2g.initialize(-1234);
    long totalNodes=0;
    const Level* level = patch->getLevel();
    Patch::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      
      IntVector plow = neighbor->getNodeLowIndex();
      IntVector phigh = neighbor->getNodeHighIndex();
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
          || ( high.z() < low.z() ) )
        throw InternalError("Patch doesn't overlap?", __FILE__, __LINE__);
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dnodes.x()*dnodes.y()*DOFsPerNode
        +start.y()*dnodes.x()*(DOFsPerNode-1) + start.x();
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

}

void SimpleSolver::solve(vector<double>& guess)
{
#if 0
  printMatrix();
  printRHS();
#endif

  double Qtot=0.;
  for (int i = 0; i < (int)Q.size(); i++){
    Qtot += fabs(Q[i]);
  }
  cout << "Qtot = " << Qtot << endl;

  d_x.resize(Q.size());

  if (compare(Qtot,0.0)){
    for (int i = 0; i < (int)Q.size(); i++){
      d_x[i] = 0.0;
    }
  }
  else{
    int conflag = 0;
    d_x = cgSolve(KK,Q,conflag);
  }
#if 0
  for (int i=0;i< d_x.size();i++)
    cout << "T[" << i << "]= " << d_x[i] << endl;
#endif
}

void SimpleSolver::createMatrix(const ProcessorGroup* d_myworld,
                                const map<int,int>& diag)
{
  int globalrows = (int)d_totalNodes;
  int globalcolumns = (int)d_totalNodes;
  
  KK.setSize(globalrows,globalcolumns);
  Q.resize(globalrows);
  d_t.resize(globalrows);
  d_flux.resize(globalrows);
}

void SimpleSolver::destroyMatrix(bool recursion)
{
  KK.clear();
  if (recursion == false) {
    d_DOF.clear();
    d_DOFFlux.clear();
    d_DOFZero.clear();
  }
}

void SimpleSolver::fillMatrix(int numi,int i[],int numj,
                              int j[],double value[])
{
   for(int ii=0;ii<numi;ii++){
     for(int jj=0;jj<numj;jj++){
       KK[i[ii]][j[jj]] = KK[i[ii]][j[jj]] + value[ii*numi + jj];
     }
   }
}


void SimpleSolver::flushMatrix()
{

}

void SimpleSolver::fillVector(int i,double v,bool add)
{
  if (add)
    Q[i] += v;
  else
    Q[i] = v;
}

void SimpleSolver::assembleVector()
{
#if 0
  printRHS();
#endif
}

void SimpleSolver::fillTemporaryVector(int i,double v)
{
  d_t[i] = v;
}

void SimpleSolver::fillFluxVector(int i,double v)
{
  d_flux[i] = v;
}
                                                                                
void SimpleSolver::assembleTemporaryVector()
{
                                                                                
}

void SimpleSolver::applyBCSToRHS()
{
  for(int ii=0;ii<d_totalNodes;ii++){
     double rowvecprod=0.;
     for(int jj=0;jj<d_totalNodes;jj++){
        rowvecprod+=KK[ii][jj]*d_t[jj];
     }
     Q[ii]=Q[ii]+rowvecprod;
  }
}

void SimpleSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}


void SimpleSolver::removeFixedDOFHeat()
{

  SparseMatrix<double,int> KKK(KK.Rows(),KK.Columns());
  for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
       itr != KK.end(); itr++) {
    int i = KK.Index1(itr);
    int j = KK.Index2(itr);
    set<int>::iterator find_itr_j = d_DOF.find(j);
    set<int>::iterator find_itr_i = d_DOF.find(i);
    
    if (find_itr_j != d_DOF.end()) {
      Q[j] = 0.;
      if (i == j) {
        KKK[i][j] = 1.;
      }
    }
    else if (find_itr_i != d_DOF.end()) {
      Q[i] = 0.;
      if (i == j) {
        KKK[i][j] = 1.;
      }
    }
    else {
      KKK[i][j] = KK[i][j];
    }
  }
  // Make sure the nodes that are outside of the material have values 
  // assigned and solved for.  The solutions will be 0.
  
  for (set<int>::iterator iter = d_DOFZero.begin(); iter != d_DOFZero.end();
       iter++) {
    int j = *iter;
    KKK[j][j] = 1.;
    Q[j] = 0.;
  }
  KK.clear();
  KK = KKK;
  KKK.clear();

  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
    // Take care of the right hand side
    Q[*iter] = -d_t[*iter];
  }    


  for (set<int>::iterator iter = d_DOFFlux.begin(); iter != d_DOFFlux.end(); 
       iter++) {
    // Take care of the right hand side
    Q[*iter] += d_flux[*iter];
  }    


}

void SimpleSolver::removeFixedDOF()
{
  for (set<int>::iterator iter = d_DOF.begin(); iter != d_DOF.end(); 
       iter++) {
    // Take care of the right hand side
    Q[*iter] = 0;
  }    
  SparseMatrix<double,int> KKK(KK.Rows(),KK.Columns());
  for (SparseMatrix<double,int>::iterator itr = KK.begin(); 
       itr != KK.end(); itr++) {
    int i = KK.Index1(itr);
    int j = KK.Index2(itr);
    set<int>::iterator find_itr_j = d_DOF.find(j);
    set<int>::iterator find_itr_i = d_DOF.find(i);
    
    if (find_itr_j != d_DOF.end()) {
      Q[j] = 0.;
      if (i == j) {
        KKK[i][j] = 1.;
      }
    }
    else if (find_itr_i != d_DOF.end()) {
      Q[i] = 0.;
      if (i == j) {
        KKK[i][j] = 1.;
      }
    }
    else {
      KKK[i][j] = KK[i][j];
    }
  }
  // Make sure the nodes that are outside of the material have values 
  // assigned and solved for.  The solutions will be 0.
  
  for (int j = 0; j < d_totalNodes; j++) {
    if (compare(KK[j][j],0.)) {
      KKK[j][j] = 1.;
      Q[j] = 0.;
    }
  }
  KK.clear();
  KK = KKK;
  KKK.clear();

}

void SimpleSolver::finalizeMatrix()
{
#if 0
  printMatrix();
#endif

}

int SimpleSolver::getSolution(vector<double>& xSimple)
{
  for (int i = 0; i < (int)d_x.size(); i++)
    xSimple.push_back(d_x[i]);

  int begin = 0;
  return begin;
}

int SimpleSolver::getRHS(vector<double>& QSimple)
{
  for (int i = 0; i < (int)Q.size(); i++)
    QSimple.push_back(Q[i]);

  int begin = 0;
  return begin;
}

void SimpleSolver::printMatrix()
{
  for (int i = 0; i < d_totalNodes; i++) {
    cout << "row " << i << ":";
    for (int j = 0; j < d_totalNodes; j++) {
      if (KK[i][j] != 0.)
        cout << " (" << j << ", " << KK[i][j] << ") ";
    }
    cout << endl;
  }

}


void SimpleSolver::printRHS()
{
  for (int i = 0; i < (int)Q.size(); i++)
    cout << Q[i] << endl;;
}
