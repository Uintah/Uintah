//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : AttributeTrf.cc
//    Author : yesim
//    Date   : Sat Feb  9 11:36:47 2002

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/BioPSE/share/share.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Containers/Array2.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <math.h>


namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE AttributeTrf : public Module {

public:

  // CONSTRUCTOR
  AttributeTrf(GuiContext *context);

  // DESTRUCTOR
  virtual ~AttributeTrf();

  virtual void execute();

  // Other function definitions
  int find_equal(const Array2<int> &tris, int value, Array1<int> &row, Array1<int> &col);
  int find_not_equal(const Array1<int> &mat, int value, Array1<int> &row);
  int matrix_union(Array1<int> &mat_init, Array1<int> &mat_res);
  void matrix_sort(Array1<int> &matrix);
  DenseMatrix *calc_laplacian(Array1<Point> &pts, Array2<int> &tris);
};

DECLARE_MAKER(AttributeTrf)


// CONSTRUCTOR
AttributeTrf::AttributeTrf(GuiContext *context)
  : Module("AttributeTrf", context, Source, "Inverse", "BioPSE")
{
}

// DESTRUCTOR
AttributeTrf::~AttributeTrf(){
}

// --------------------------------------------------------------------------
int AttributeTrf::find_equal(const Array2<int> &tris, int value, Array1<int> &row, Array1<int> &col) {
  // Find row and column indices of elements of *mat that equals "value"
  // so, at the end (*mat)[(*row)[i]][(*col)[i]] = value for i=0:nOccurOfVal-1

  // NOTE: value in this function will be integer, but to make it general, 
  // we may want to use templates!!

  row.resize(0);
  col.resize(0);
  int nRows = tris.dim1();
  int nCols = tris.dim2();

  for (int r=0; r<nRows; r++){
    for (int c=0; c<nCols; c++){
      if (tris(r,c) == value) {
	row.add(r);
	col.add(c);
      }
    }
  }
  return row.size(); // Returns length of row (and col)
}

// --------------------------------------------------------------------------
int AttributeTrf::find_not_equal(const Array1<int> &mat, int value, Array1<int> &row) {
  // Find column indices of elements of *mat that are not equal to "value"

  // NOTE: value in this function will be integer, but to make it general, 
  // we may want to use templates!!

  int nRows = mat.size();
  row.resize(0);
  for (int r=0; r<nRows; r++){
    if (mat[r]!=value) {
      row.add(r);
    }
  }
  return row.size(); // Returns length of row
}

// --------------------------------------------------------------------------
// function to get union
int AttributeTrf::matrix_union(Array1<int> &mat_init, Array1<int> &mat_res) {
  int initLen = mat_init.size();

  // first sort mat_init
  matrix_sort(mat_init); // mat_init is sorted in ascending order
  
  mat_res.resize(0);
  mat_res.add(mat_init[0]);

  for (int r=1; r<initLen; r++){
    if (mat_init[r]!=mat_init[r-1]) {
      mat_res.add(mat_init[r]);
    }
  }
  return mat_res.size();
}

// --------------------------------------------------------------------------
// function to sort column matrix
void AttributeTrf::matrix_sort(Array1<int> &matrix) {
  int len = matrix.size();
  int temp1;

  for (int count=0; count<len-1; count++){
    for (int j=0; j<len-count-1; j++) {
      if (matrix[j]>matrix[j+1]) {
	temp1 = matrix[j];
	matrix[j]=matrix[j+1];
	matrix[j+1]=temp1;
      }
    }
  }
}

// --------------------------------------------------------------------------
// function to compute laplacian matrix from neighbours
DenseMatrix *AttributeTrf::calc_laplacian(Array1<Point> &pts, Array2<int> &tris) {

  DenseMatrix *laplacian = scinew DenseMatrix(pts.size(),pts.size());
  laplacian->zero();

  Array1<int> sub_tri, sub_tri_union, node_nbours;
  Array1<double> dist, odist;

  int nChannels = pts.size();
  int N = tris.dim2();
  Array1<int> rowI, colI, rowI2;
  int node;

  for (node=0; node<nChannels; node++){

    //    (*nbours)[node][0] = node;

    // first find which elements are equal to node #
    int nEqual = find_equal(tris, node, rowI, colI); 

    sub_tri.resize(0);
    for (int i=0; i<nEqual; i++){
      for (int c=0; c<N; c++){
	sub_tri.add(tris(rowI[i],c));
      }
    }
    
    // then delete repetitions, (union of all elements)
    int nNbours = matrix_union(sub_tri, sub_tri_union);
        // index of nbours in sub_tri_union[0...nNbour-1]

    // Delete current node number from the matrix
    nNbours = find_not_equal(sub_tri_union, node, rowI2);

    node_nbours.resize(0);
    for (int i=0; i<nNbours; i++){
      node_nbours.add(sub_tri_union[rowI2[i]]);
    }

    double sum1 = 0;
    double sum2 = 0;

    dist.resize(0);
    odist.resize(0);
    for (int nbrNo=0; nbrNo<nNbours; nbrNo++){
      double d=(pts[node]-pts[node_nbours[nbrNo]]).length(); 
             //Sqrt(_x*_x+_y*_y+_z*_z);
      dist.add(d);
      odist.add(1/d);
      sum1 += d;
      sum2 += 1/d;
    }

    double avg_dist = sum1/nNbours; // hpar
    double avg_odist = sum2/nNbours; // hopar

    (*laplacian)[node][node] = -4.0 * avg_odist / avg_dist;
    for (int nbrNo=0; nbrNo<nNbours; nbrNo++){
      int ind = node_nbours[nbrNo];
      (*laplacian)[node][ind] = 4.0 / (avg_dist*nNbours*(dist[nbrNo]));
    }
    
  }

  return laplacian;

}

///////////////////////////////////////////////
// MODULE EXECUTION
///////////////////////////////////////////////
void AttributeTrf::execute(){

  FieldIPort *iportGeomF = (FieldIPort *)get_iport("InputFld");
  MatrixOPort *oportAttrib = (MatrixOPort *)get_oport("OutputMat");

  if (!iportGeomF) {
    error("Unable to initialize iport 'InputFld'.");
    return;
  }
  if (!oportAttrib) {
    error("Unable to initialize oport 'OutputMat'.");
    return;
  }

  // getting input field
  FieldHandle hFieldGeomF;

  if(!iportGeomF->get(hFieldGeomF)) { 
    error("Couldn't get handle to the Input Field.");
    return;
  }


  // From the input geometry, extract pts and tri data
  MeshHandle mb = hFieldGeomF->mesh();
  TriSurfMesh *tsm = dynamic_cast<TriSurfMesh *>(mb.get_rep());
  
  TriSurfMesh::Node::iterator niter; 
  tsm->begin(niter);
  TriSurfMesh::Node::iterator niter_end; 
  tsm->end(niter_end);
  TriSurfMesh::Node::size_type nsize; 
  tsm->size(nsize);

  Array1<Point> pts;
  while(niter != niter_end) {
    Point p;
    tsm->get_center(p, *niter);
    pts.add(p);
    ++niter;
  }

  TriSurfMesh::Face::size_type nfaces;
  tsm->size(nfaces);
  Array2<int> tris(nfaces, 3);
  TriSurfMesh::Face::iterator fiter;
  tsm->begin(fiter);
  TriSurfMesh::Face::iterator fend;
  tsm->end(fend);
  
  TriSurfMesh::Node::array_type fac_nodes(3);
  int ctr = 0;
  while(fiter != fend) {
    tsm->get_nodes(fac_nodes, *fiter);
    tris(ctr,0)=fac_nodes[0];
    tris(ctr,1)=fac_nodes[1];
    tris(ctr,2)=fac_nodes[2];
    ++fiter;
    ctr++;
  }

  // Using tris and pts get Laplacian matrix
  // DenseMatrix *laplacian of size nNodes x nNodes

  DenseMatrix *laplacian = calc_laplacian(pts, tris);

  ////////////////////////////////////////////////////
  ////////// This part is added to write the 
  ////////// laplacian Matrix in the file
  ////////////////////////////////////////////////////

//  int i, j;
//  FILE *fp;
//  fp = fopen("laplacian.txt","wt");
//  for(i=0; i<64; i++)
//  {  
//	for(j=0; j<64; j++)
//		fprintf(fp,"%f ",(*laplacian)[i][j]);
//    fprintf(fp,"\n");
//  }			
//  fclose(fp);

  ////////////////////////////////////////////////////

  oportAttrib->send(MatrixHandle(laplacian));
}

} // End namespace BioPSE
