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
 *  BuildTriFEMatrix.cc:  Build finite element matrix for TriSurf mesh
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001   
 *  Copyright (C) 2001 SCI Group
 *
 *  Modified (adapted from BuildTetFEMatrix.cc):
 *   Lorena Kreda, Northeastern University, October 2003
 */

#include <Core/Datatypes/Mesh.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildTriFEMatrix.h>
#include <iostream>
#include <algorithm>
#include <Core/Util/Timer.h>


namespace BioPSE {

using namespace SCIRun;

//! Constructor
// -- it's private, no occasional object creation
BuildTriFEMatrix::BuildTriFEMatrix(TriSurfFieldIntHandle hField,
			     vector<pair<string, Tensor> >& tens,
			     MatrixHandle& hA, 
			     int np, double unitsScale):
  // ---------------------------------------------
  hField_(hField),
  hA_(hA),
  np_(np),
  barrier_("BuildTriFEMatrix barrier"),
  colIdx_(np+1),
  tens_(tens),
  unitsScale_(unitsScale)
{
  hMesh_=hField->get_typed_mesh();
  TriSurfMesh::Node::size_type nsize; hMesh_->size(nsize);
  unsigned int nNodes = nsize;
  rows_ = scinew int[nNodes+1];
  cerr << "unitsScale_ = "<< unitsScale_ << "\n";
}
BuildTriFEMatrix::~BuildTriFEMatrix(){}

bool BuildTriFEMatrix::build_FEMatrix(TriSurfFieldIntHandle hField,
				   vector<pair<string, Tensor> >& tens,
				   MatrixHandle& hA, double unitsScale)
  //------------------------------------------------
{
  int np=Thread::numProcessors();

  if ( np > 2 ) {
    np /= 2;
    if (np>10) {
      np=5;
    }
  }

  hA = 0;

  BuildTriFEMatrixHandle hMaker =
    new BuildTriFEMatrix(hField, tens, hA, np, unitsScale);
  cerr << "SetupFEMatrix: number of threads being used = " << np << endl;

  Thread::parallel(Parallel<BuildTriFEMatrix>(hMaker.get_rep(), 
					   &BuildTriFEMatrix::parallel),
 		   np, true);
  
  // -- refer to the object one more time not to make it die before
  hMaker = 0;
  
  if (hA.get_rep()){
    return true;
  }
  else {
    return false;
  }
}

// -- callback routine to execute in parallel
void BuildTriFEMatrix::parallel(int proc)
{
  //! dividing nodes among processors
  TriSurfMesh::Node::size_type nsize; hMesh_->size(nsize);
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
  
  TriSurfMesh::Node::array_type neib_nodes;

  for(i=start_node;i<end_node;i++){
    rows_[r++]=mycols.size();
    neib_nodes.clear();
    hMesh_->get_neighbors(neib_nodes, TriSurfMesh::Node::index_type(i));
    
    // adding the node itself, sorting and eliminating duplicates
    neib_nodes.push_back(TriSurfMesh::Node::index_type(i));
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
  TriSurfMesh::Face::iterator ii, iie;
  
  double lcl_matrix[3][3];
   
  TriSurfMesh::Node::array_type face_nodes(3);
  hMesh_->begin(ii); hMesh_->end(iie);
  int count = 0;
  for (; ii != iie; ++ii){
    if (hMesh_->test_nodes_range(*ii, start_node, end_node)){ 
      build_local_matrix(lcl_matrix, *ii);
      add_lcl_gbl(lcl_matrix, *ii, start_node, end_node, face_nodes);
    }
  }

  barrier_.wait(np_);
}

void BuildTriFEMatrix::build_local_matrix(double lcl_a[3][3], TriSurfMesh::Face::index_type f_ind)
{
  Vector grad1, grad2, grad3;
  double area = hMesh_->get_gradient_basis(f_ind, grad1, grad2, grad3);
 
  int  ind = hField_->value(f_ind);
  double (&el_cond)[3][3] = tens_[ind].second.mat_;
 
  if(fabs(area) < 1.e-10){
    for(int i = 0; i<3; i++)
      for(int j = 0; j<3; j++)
	lcl_a[i][j]=0;
    return;
  }
  
  double el_coefs[3][3];
  
  // -- this 3x3 array holds the 3 gradients to be used 
  // as coefficients for each of the three nodes of the 
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
 
  // build the local matrix
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {

      lcl_a[i][j] = 0.0;
      for (int k=0; k< 3; k++){
	for (int l=0; l<3; l++){
	  //	  cout << "4b ii " << &lcl_a[i][j];
          //cout << " " <<  &el_cond[k][l] << "  " ;
	  // cout << &el_coefs[i][j] << "  ";
          //cout << &el_coefs[j][l] << "  " << k << "  " << l << endl;
	  lcl_a[i][j] += 
	    (el_cond[k][l]*unitsScale_)*el_coefs[i][k]*el_coefs[j][l];
	}
      }
      lcl_a[i][j] *= fabs(area);
    }
  }
}

void BuildTriFEMatrix::add_lcl_gbl(double lcl_a[3][3], TriSurfMesh::Face::index_type f_ind, int s, int e, TriSurfMesh::Node::array_type& face_nodes)
{
  //TriSurfMesh::Node::array_type cell_nodes(4);
  hMesh_->get_nodes(face_nodes, f_ind); 

  for (int i=0; i<3; i++) {
    int ii = face_nodes[i];
    if (ii>=s && ii<e)          //! the row to update belongs to the process, proceed...
      for (int j=0; j<3; j++) {      
	int jj = face_nodes[j];
	pA_->get(ii, jj) += lcl_a[i][j];
      }
  }
}

void BuildTriFEMatrix::io(Piostream&){
}

} // end namespace BioPSE
