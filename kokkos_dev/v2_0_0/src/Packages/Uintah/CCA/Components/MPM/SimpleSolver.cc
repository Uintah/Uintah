#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <iostream>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
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
					     const PatchSubset* patches)
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
	(phighIndex[2]-plowIndex[2])*3;

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
    Level::selectType neighbors;
    level->selectPatches(lowIndex, highIndex, neighbors);
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      
      IntVector plow = neighbor->getNodeLowIndex();
      IntVector phigh = neighbor->getNodeHighIndex();
      IntVector low = Max(lowIndex, plow);
      IntVector high= Min(highIndex, phigh);
      
      if( ( high.x() < low.x() ) || ( high.y() < low.y() ) 
	  || ( high.z() < low.z() ) )
	throw InternalError("Patch doesn't overlap?");
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dnodes = phigh-plow;
      IntVector start = low-plow;
      petscglobalIndex += start.z()*dnodes.x()*dnodes.y()*3
	+start.y()*dnodes.x()*2 + start.x();
      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
	int idx_slab = petscglobalIndex;
	petscglobalIndex += dnodes.x()*dnodes.y()*3;
	
	for (int colY = low.y(); colY < high.y(); colY ++) {
	  int idx = idx_slab;
	  idx_slab += dnodes.x()*3;
	  for (int colX = low.x(); colX < high.x(); colX ++) {
	    l2g[IntVector(colX, colY, colZ)] = idx;
	    idx += 3;
	  }
	}
      }
      IntVector d = high-low;
      totalNodes+=d.x()*d.y()*d.z()*3;
    }
    d_petscLocalToGlobal[patch].copyPointer(l2g);
  }

}

void SimpleSolver::solve()
{
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
}

void SimpleSolver::createMatrix(const ProcessorGroup* d_myworld,
				const map<int,int>& diag)
{
  int globalrows = (int)d_totalNodes;
  int globalcolumns = (int)d_totalNodes;
  
  KK.setSize(globalrows,globalcolumns);
  Q.resize(globalrows);
}

void SimpleSolver::destroyMatrix(bool recursion)
{
  KK.clear();
  if (recursion == false)
    d_DOF.clear();
}

void SimpleSolver::fillMatrix(int i,int j,double value)
{
  KK[i][j] = KK[i][j] + value;
}

void SimpleSolver::flushMatrix()
{

}

void SimpleSolver::fillVector(int i,double v)
{
  Q[i] = v;
}

void SimpleSolver::assembleVector()
{

}

void SimpleSolver::copyL2G(Array3<int>& mapping,const Patch* patch)
{
  mapping.copy(d_petscLocalToGlobal[patch]);
}

void SimpleSolver::removeFixedDOF(int num_nodes)
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
  
  for (int j = 0; j < num_nodes; j++) {
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
