/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  BuildFEMatrix.cc:  Build finite element matrix
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001   
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildFEMatrix.h>
#include <iostream>
#include <algorithm>
#include <Core/Util/Timer.h>


namespace BioPSE {

using namespace SCIRun;

//! Constructor
// -- it's private, no occasional object creation
BuildFEMatrix::BuildFEMatrix(TetVolFieldIntHandle hFieldInt,
			     TetVolFieldTensorHandle hFieldTensor,
			     bool index_based,
			     vector<pair<string, Tensor> >& tens,
			     MatrixHandle& hA, 
			     int np, double unitsScale):
  // ---------------------------------------------
  hFieldInt_(hFieldInt),
  hFieldTensor_(hFieldTensor),
  index_based_(index_based),
  hA_(hA),
  np_(np),
  barrier_("BuildFEMatrix barrier"),
  colIdx_(np+1),
  tens_(tens),
  unitsScale_(unitsScale)
{
  if (index_based_) hMesh_ = hFieldInt->get_typed_mesh();
  else hMesh_ = hFieldTensor->get_typed_mesh();

  TetVolMesh::Node::size_type nsize; hMesh_->size(nsize);
  unsigned int nNodes = nsize;
  rows_ = scinew int[nNodes+1];
//  cerr << "unitsScale_ = "<< unitsScale_ << "\n";
}
BuildFEMatrix::~BuildFEMatrix(){}

bool BuildFEMatrix::build_FEMatrix(TetVolFieldIntHandle hFieldInt,
				   TetVolFieldTensorHandle hFieldTensor,
				   bool index_based,
				   vector<pair<string, Tensor> >& tens,
				   MatrixHandle& hA, double unitsScale,
				   int num_procs)
  //------------------------------------------------
{
  int np = Thread::numProcessors();

  if ( np > 2 ) {
    np /= 2;
    if (np>10) {
      np=5;
    }
  }

  if (num_procs > 0) { np = num_procs; }

  hA = 0;

  BuildFEMatrixHandle hMaker =
    new BuildFEMatrix(hFieldInt, hFieldTensor, index_based, tens, 
		      hA, np, unitsScale);
//  cerr << "SetupFEMatrix: number of threads being used = " << np << endl;

  Thread::parallel(Parallel<BuildFEMatrix>(hMaker.get_rep(), 
					   &BuildFEMatrix::parallel),
 		   np, true);
  
  // -- refer to the object one more time not to make it die before
  hMaker = 0;
  
  if (hA.get_rep())
    return true;
  else
    return false;
}

// -- callback routine to execute in parallel
void BuildFEMatrix::parallel(int proc)
{
  //! dividing nodes among processors
  TetVolMesh::Node::size_type nsize; hMesh_->size(nsize);
  int nNodes     = nsize;
  int start_node = nNodes * proc/np_;
  int end_node   = nNodes * (proc+1)/np_;
  int ndof       = end_node - start_node;
  
  int r = start_node;
  int i;
  
  //----------------------------------------------------------------------
  //! Creating sparse matrix structure
  Array1<int> mycols(0, 15*ndof);
  
  if (proc==0){
    hMesh_->synchronize(Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E);
  }

  barrier_.wait(np_);
  
  TetVolMesh::Node::array_type neib_nodes;
  for(i=start_node;i<end_node;i++){
    rows_[r++]=mycols.size();
    neib_nodes.clear();
    
    hMesh_->get_neighbors(neib_nodes, TetVolMesh::Node::index_type(i));
    
    // adding the node itself, sorting and eliminating duplicates
    neib_nodes.push_back(TetVolMesh::Node::index_type(i));
    sort(neib_nodes.begin(), neib_nodes.end());
 
    for (unsigned int jj=0; jj<neib_nodes.size(); jj++){
      mycols.add(neib_nodes[jj]);
    }
  }
  
  colIdx_[proc]=mycols.size();
  
  //! check point
  barrier_.wait(np_);
  
  int st=0;
  if (proc == 0){  
    for(i=0;i<np_;i++){
      int ns=st+colIdx_[i];
      colIdx_[i]=st;
      st=ns;
    }
    
    colIdx_[np_]=st;
    allCols_=scinew int[st];
  }

  //! check point
  barrier_.wait(np_);
  
  //! updating global column by each of the processors
  int s=colIdx_[proc];  
  int n=mycols.size();
  
  for(i=0;i<n;i++){
    allCols_[i+s]=mycols[i];
  }
  
  for(i=start_node;i<end_node;i++){
    rows_[i]+=s;
  }

  //! check point
  barrier_.wait(np_);
  
  //! the main thread makes the matrix
  if(proc == 0){
    rows_[nNodes]=st;
    pA_ = scinew SparseRowMatrix(nNodes, nNodes, rows_, allCols_, st);
    hA_ = pA_;
  }
  
  //! check point
  barrier_.wait(np_);
  
  //! zeroing in parallel
  double* a = pA_->a;
  
  int ns=colIdx_[proc];
  int ne=colIdx_[proc+1];
  
  for(i=ns;i<ne;i++){
    a[i]=0;
  }
  
  //----------------------------------------------------------
  //! Filling the matrix
  TetVolMesh::Cell::iterator ii, iie;
  
  double lcl_matrix[4][4];
   
  TetVolMesh::Node::array_type cell_nodes(4);
  hMesh_->begin(ii); hMesh_->end(iie);
  for (; ii != iie; ++ii){
 
    if (hMesh_->test_nodes_range(*ii, start_node, end_node)){ 
      build_local_matrix(lcl_matrix, *ii);
      add_lcl_gbl(lcl_matrix, *ii, start_node, end_node, cell_nodes);
    }
  }
  barrier_.wait(np_);
}

void
BuildFEMatrix::build_local_matrix( double lcl_a[4][4],
				   TetVolMesh::Cell::index_type c_ind )
{
  Vector grad1, grad2, grad3, grad4;
  double vol = hMesh_->get_gradient_basis(c_ind, grad1, grad2, grad3, grad4);
 
  typedef double onerow[3]; // This 'hack' is necessary to compile under IRIX CC
  onerow *el_cond;

  if (index_based_) el_cond = tens_[hFieldInt_->value(c_ind)].second.mat_;
  else el_cond = hFieldTensor_->value(c_ind).mat_;

  if(fabs(vol) < 1.e-10){
    for(int i = 0; i<4; i++)
      for(int j = 0; j<4; j++)
	lcl_a[i][j]=0;
    return;
  }
  
  double el_coefs[4][3];
  
  // -- this 4x3 array holds the 3 gradients to be used 
  // as coefficients for each of the four nodes of the 
  // element
  
  el_coefs[0][0]=grad1.x();
  el_coefs[0][1]=grad1.y();
  el_coefs[0][2]=grad1.z();
  
  el_coefs[1][0]=grad2.x();
  el_coefs[1][1]=grad2.y();
  el_coefs[1][2]=grad2.z();
  
  el_coefs[2][0]=grad3.x();
  el_coefs[2][1]=grad3.y();
  el_coefs[2][2]=grad3.z();
  
  el_coefs[3][0]=grad4.x();
  el_coefs[3][1]=grad4.y();
  el_coefs[3][2]=grad4.z();
  
  // build the local matrix
  for(int i=0; i<4; i++) {
    for(int j=0; j<4; j++) {

      lcl_a[i][j] = 0.0;

      for (int k=0; k< 3; k++){
	for (int l=0; l<3; l++){
	  lcl_a[i][j] += 
	    (el_cond[k][l]*unitsScale_)*el_coefs[i][k]*el_coefs[j][l];
	}
      }

      lcl_a[i][j] *= fabs(vol);
    }
  }
}

void BuildFEMatrix::add_lcl_gbl(double lcl_a[4][4], TetVolMesh::Cell::index_type c_ind, int s, int e, TetVolMesh::Node::array_type& cell_nodes)
{
  //TetVolMesh::Node::array_type cell_nodes(4);
  hMesh_->get_nodes(cell_nodes, c_ind); 

  for (int i=0; i<4; i++) {
    int ii = cell_nodes[i];
    if (ii>=s && ii<e)          //! the row to update belongs to the process, proceed...
      for (int j=0; j<4; j++) {      
	int jj = cell_nodes[j];
	pA_->get(ii, jj) += lcl_a[i][j];
      }
  }
}

void BuildFEMatrix::io(Piostream&){
}

} // end namespace BioPSE
