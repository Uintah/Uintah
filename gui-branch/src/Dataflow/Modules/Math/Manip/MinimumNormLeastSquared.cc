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

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Util/Assert.h>
#include <iostream>
#include <math.h>

using namespace SCIRun;
using namespace std;

extern "C" {

double det_3x3(double* p1, double* p2, double* p3){
    double D = p1[0]*p2[1]*p3[2]-p1[0]*p2[2]*p3[1]-p1[1]*p2[0]*p3[2]+p2[0]*p1[2]*p3[1]+p3[0]*p1[1]*p2[2]-p1[2]*p2[1]*p3[0];
    return(D);
} 

int solve_3x3(double a[3][3] ,double b[3], double x[3]){
    double D = det_3x3(a[0],a[1],a[2]);
    double D1 = det_3x3(b,a[1],a[2]);
    double D2 = det_3x3(a[0],b,a[2]);
    double D3 = det_3x3(a[0],a[1],b);
    if ( D!=0){ 
        x[0] = D1/D;
        x[1] = D2/D; 
        x[2] = D3/D; 
    } else {
        cerr << "ERROR, DET = 0!"<< endl;
        x[0] = 1;
        x[1] = 0; 
        x[2] = 0;
        return(-1);
    }
    return(0);
}

double error_norm( double* x1, double* x2,int  n){
    double err= 0;
    for (int i=0;i<n;i++)
        err = err + (x1[i]-x2[i])*(x1[i]-x2[i]);
    return(sqrt(err));
}

// This module computes the minimal norm, least squared solution to a
//  nx3 linear system.
// Given four input ColumnMatrices (v0,v1,v2,b),
//  find the three coefficients (w0,w1,w2) that minimize:
//  | (w0v0 + w1v1 + w2v2) - b |.
// If more than one minimum exisits (the system is under-determined),
//  choose the coefficients such that (w0,w1,w2) has minimum norm.
// We output the vector (w0,w1,w2) as a row-matrix,
//  and we ouput the ColumnMatrix (called x), which is: | w0v0 + w1v1 + w2v2 |.

void execute(const vector<MatrixHandle>& in, vector<MatrixHandle>& out)
{
  enum { number_of_columns = 4 };
  int i,j;
  vector<ColumnMatrix *> v(4);
  ASSERT( in.size() == 4 )
  ASSERT( out.size() == 2 )
  for (i = 0; i < in.size(); i++) {
    ASSERT (v[i] = dynamic_cast<ColumnMatrix *>(in[i].get_rep()))
  }
  int size = v[0]->nrows();
  for (i = 1; i < in.size(); i++) {
    ASSERT ( v[i]->nrows() == size )
  }
  ColumnMatrix *b = v[3];
  double *w = new double[3];
  double *x = new double[size];

  double a[3][3];
  double a_b[3];

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++)
      a[i][j]=0;
    a_b[i]=0;
  }

  double dc[3];
  for (i=0; i<3; i++) 
    dc[i] = (*v[i])[0];
  double b_dc=(*b)[0];

  for(i=0; i<size; i++) {
    a[0][0] += ((*v[0])[i]-dc[0])*((*v[0])[i]-dc[0]);
    a[0][1] += ((*v[0])[i]-dc[0])*((*v[1])[i]-dc[1]);
    a[0][2] += ((*v[0])[i]-dc[0])*((*v[2])[i]-dc[2]);
    a[1][1] += ((*v[1])[i]-dc[1])*((*v[1])[i]-dc[1]);
    a[1][2] += ((*v[1])[i]-dc[1])*((*v[2])[i]-dc[2]);
    a[2][2] += ((*v[2])[i]-dc[2])*((*v[2])[i]-dc[2]);
  }
  
  a[1][0] = a[0][1];
  a[2][0] = a[0][2];
  a[2][1] = a[1][2];
   
  for(i=0; i<size; i++) {
    for (j=0; j<3; j++)
      a_b[j] += ((*v[j])[i]-dc[j])*((*b)[i]-b_dc);
  }

  solve_3x3(a, a_b, w);

  for(i=0; i<size; i++) {
    x[i] = 0;
    for (j=0; j<3; j++) 
      x[i] += w[j]*((*v[j])[i]-dc[j]);
  }
   
  ColumnMatrix* w_vec = new ColumnMatrix(4);
  w_vec->put_lhs(w);   
  out[0]=w_vec;

  ColumnMatrix* x_vec = new ColumnMatrix(size);
  x_vec->put_lhs(x);
  out[1]=x_vec;
}

}
