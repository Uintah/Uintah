/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

//    File   : SurfaceLaplacian.cc
//    Author : Yesim
//    Date   : Sun Nov 16 23:05:16 MST 2003

#include <Core/Containers/Array2.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Algorithms/Geometry/SurfaceLaplacian.h>

namespace SCIRun {

using namespace SCIRun;

int find_equal(const Array2<int> &tris, int value, Array1<int> &row, Array1<int> &col) {
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
int find_not_equal(const Array1<int> &mat, int value, Array1<int> &row) {
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
// function to sort column matrix
void matrix_sort(Array1<int> &matrix) {
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
// function to get union
int matrix_union(Array1<int> &mat_init, Array1<int> &mat_res) {
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
// function to compute laplacian matrix from neighbours
DenseMatrix *surfaceLaplacian(TriSurfMesh *tsm) {
  //Array1<Point> &pts, Array2<int> &tris) {

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

} // End namespace BioPSE
