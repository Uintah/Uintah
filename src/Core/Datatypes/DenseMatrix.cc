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
 *  DenseMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <stdio.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
using std::vector;

namespace SCIRun {

static Persistent* maker()
{
  return scinew DenseMatrix;
}

PersistentTypeID DenseMatrix::type_id("DenseMatrix", "Matrix", maker);

DenseMatrix* DenseMatrix::clone(){
  return scinew DenseMatrix(*this);
}

//! constructors
DenseMatrix::DenseMatrix() :
  nc(0), 
  nr(0), 
  data(0),
  dataptr(0)
{
}

DenseMatrix::DenseMatrix(int r, int c)
{
  ASSERT(r>0);
  ASSERT(c>0);
  nr=r;
  nc=c;
  data=scinew double*[nr];
  double* tmp=scinew double[nr*nc];
  dataptr=tmp;
  for(int i=0;i<nr;i++){
    data[i]=tmp;
    tmp+=nc;
  }
}

DenseMatrix::DenseMatrix(const DenseMatrix& m)
{
  nc=m.ncols();
  nr=m.nrows();
  data=scinew double*[nr];
  double* tmp=scinew double[nr*nc];
  dataptr=tmp;
  for(int i=0;i<nr;i++){
    data[i]=tmp;
    double* p=m.data[i];
    for(int j=0;j<nc;j++){
      *tmp++=*p++;
    }
  }
}

DenseMatrix::DenseMatrix(const Transform& t)
{
  nc=nr=4;
  double dummy[16];
  t.get(dummy);
  data=scinew double*[nr];
  double* tmp=scinew double[nr*nc];
  dataptr=tmp;
  double* p=&(dummy[0]);
  for(int i=0;i<nr;i++){
    data[i]=tmp;
    for(int j=0;j<nc;j++){
      *tmp++=*p++;
    }
  }
}

DenseMatrix *DenseMatrix::dense() {
  return this;
}

ColumnMatrix *DenseMatrix::column() {
  ColumnMatrix *cm = scinew ColumnMatrix(nr);
  for(int i=0;i<nr;i++)
    (*cm)[i]=data[i][0];
  return cm;
}

SparseRowMatrix *DenseMatrix::sparse() {
  int nnz = 0;
  int r, c;
  int *rows = scinew int[nr+1];
  for (r=0; r<nr; r++)
    for (c=0; c<nc; c++)
      if (data[r][c] != 0) nnz++;

  int *columns = scinew int[nnz];
  double *a = scinew double[nnz];

  int count=0;
  for (r=0; r<nr; r++) {
    rows[r]=count;
    for (c=0; c<nc; c++)
      if (data[r][c] != 0) {
	columns[count]=c;
	a[count]=data[r][c];
	count++;
      }
  }
  rows[nr]=count;
  return scinew SparseRowMatrix(nr, nc, rows, columns, nnz, a);
}

//! destructor
DenseMatrix::~DenseMatrix()
{
  delete[] dataptr;
  delete[] data;
}

//! assignment operator
DenseMatrix& DenseMatrix::operator=(const DenseMatrix& m)
{
  delete[] dataptr;
  delete[] data;
  nc=m.nc;
  nr=m.nr;
  data=scinew double*[nr];
  double* tmp=scinew double[nr*nc];
  dataptr=tmp;
  for(int i=0;i<nr;i++){
    data[i]=tmp;
    double* p=m.data[i];
    for(int j=0;j<nc;j++){
      *tmp++=*p++;
    }
  }
  return *this;
}

double& DenseMatrix::get(int r, int c) const
{
  ASSERTRANGE(r, 0, nr);
  ASSERTRANGE(c, 0, nc);
  return data[r][c];
}

void DenseMatrix::put(int r, int c, const double& d)
{
  ASSERTRANGE(r, 0, nr);
  ASSERTRANGE(c, 0, nc);
  data[r][c]=d;
}

DenseMatrix *DenseMatrix::transpose() {
  DenseMatrix *m=scinew DenseMatrix(nc,nr);
  double *mptr = &((*m)[0][0]);
  for (int c=0; c<nc; c++)
    for (int r=0; r<nr; r++)
      *mptr++ = data[r][c];
  return m;
}

int DenseMatrix::nrows() const
{
  return nr;
}

int DenseMatrix::ncols() const
{
  return nc;
}

void DenseMatrix::getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val)
{
  idx.resize(nc);
  val.resize(nc);
  int i=0;
  for (int c=0; c<nc; c++) {
    if (data[r][c]!=0.0) {
      idx[i]=c;
      val[i]=data[r][c];
      i++;
    }
  }
}
    
void DenseMatrix::zero()
{
  for(int r=0;r<nr;r++){
    double* row=data[r];
    for(int c=0;c<nc;c++){
      row[c]=0.0;
    }
  }
}

int DenseMatrix::solve(ColumnMatrix& sol, int overwrite)
{
  ColumnMatrix b(sol);
  return solve(b, sol, overwrite);
}

int DenseMatrix::solve(const ColumnMatrix& rhs, ColumnMatrix& lhs, 
		       int overwrite)
{
  ASSERT(nr==nc);
  ASSERT(rhs.nrows()==nc);
  lhs=rhs;

  double **A;
  DenseMatrix *cpy = NULL;
  if (!overwrite) {cpy=clone(); A=cpy->getData2D();}
  else A=data;

  // Gauss-Jordan with partial pivoting
  int i;
  for(i=0;i<nr;i++){
    //	cout << "Solve: " << i << " of " << nr << endl;
    double max=Abs(A[i][i]);
    int row=i;
    int j;
    for(j=i+1;j<nr;j++){
      if(Abs(A[j][i]) > max){
	max=Abs(A[j][i]);
	row=j;
      }
    }
    //	ASSERT(Abs(max) > 1.e-12);
    if (Abs(max) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    if(row != i){
      // Switch rows (actually their pointers)
      double* tmp=A[i];
      A[i]=A[row];
      A[row]=tmp;
      double dtmp=lhs[i];
      lhs[i]=lhs[row];
      lhs[row]=dtmp;
    }
    double denom=1./A[i][i];
    double* r1=A[i];
    double s1=lhs[i];
    for(j=i+1;j<nr;j++){
      double factor=A[j][i]*denom;
      double* r2=A[j];
      for(int k=i;k<nr;k++)
	r2[k]-=factor*r1[k];
      lhs[j]-=factor*s1;
    }
  }

  // Back-substitution
  for(i=1;i<nr;i++){
    //	cout << "Solve: " << i << " of " << nr << endl;
    //	ASSERT(Abs(A[i][i]) > 1.e-12);
    if (Abs(A[i][i]) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    double denom=1./A[i][i];
    double* r1=A[i];
    double s1=lhs[i];
    for(int j=0;j<i;j++){
      double factor=A[j][i]*denom;
      double* r2=A[j];
      for(int k=i;k<nr;k++)
	r2[k]-=factor*r1[k];
      lhs[j]-=factor*s1;
    }
  }

  // Normalize
  for(i=0;i<nr;i++){
    //	cout << "Solve: " << i << " of " << nr << endl;
    //	ASSERT(Abs(A[i][i]) > 1.e-12);
    if (Abs(A[i][i]) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    double factor=1./A[i][i];
    for(int j=0;j<nr;j++)
      A[i][j]*=factor;
    lhs[i]*=factor;
  }
  if (!overwrite) delete cpy;
  return 1;
}

int DenseMatrix::solve(vector<double>& sol, int overwrite)
{
  vector<double> b(sol);
  return solve(b, sol, overwrite);
}

int DenseMatrix::solve(const vector<double>& rhs, vector<double>& lhs, 
		       int overwrite)
{
  ASSERT(nr==nc);
  ASSERT(rhs.size()==(unsigned)nc);
  lhs=rhs;

  double **A;
  DenseMatrix *cpy = NULL;
  if (!overwrite) {cpy=clone(); A=cpy->getData2D();}
  else A=data;

  // Gauss-Jordan with partial pivoting
  int i;
  for(i=0;i<nr;i++){
    //	cout << "Solve: " << i << " of " << nr << endl;
    double max=Abs(A[i][i]);
    int row=i;
    int j;
    for(j=i+1;j<nr;j++){
      if(Abs(A[j][i]) > max){
	max=Abs(A[j][i]);
	row=j;
      }
    }
    //	ASSERT(Abs(max) > 1.e-12);
    if (Abs(max) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    if(row != i){
      // Switch rows (actually their pointers)
      double* tmp=A[i];
      A[i]=A[row];
      A[row]=tmp;
      double dtmp=lhs[i];
      lhs[i]=lhs[row];
      lhs[row]=dtmp;
    }
    double denom=1./A[i][i];
    double* r1=A[i];
    double s1=lhs[i];
    for(j=i+1;j<nr;j++){
      double factor=A[j][i]*denom;
      double* r2=A[j];
      for(int k=i;k<nr;k++)
	r2[k]-=factor*r1[k];
      lhs[j]-=factor*s1;
    }
  }

  // Back-substitution
  for(i=1;i<nr;i++){
    //	cout << "Lhsve: " << i << " of " << nr << endl;
    //	ASSERT(Abs(A[i][i]) > 1.e-12);
    if (Abs(A[i][i]) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    double denom=1./A[i][i];
    double* r1=A[i];
    double s1=lhs[i];
    for(int j=0;j<i;j++){
      double factor=A[j][i]*denom;
      double* r2=A[j];
      for(int k=i;k<nr;k++)
	r2[k]-=factor*r1[k];
      lhs[j]-=factor*s1;
    }
  }

  // Normalize
  for(i=0;i<nr;i++){
    //	cout << "Solve: " << i << " of " << nr << endl;
    //	ASSERT(Abs(A[i][i]) > 1.e-12);
    if (Abs(A[i][i]) < 1.e-12) {
      lhs=rhs;
      if (!overwrite) delete cpy;
      return 0;
    }
    double factor=1./A[i][i];
    for(int j=0;j<nr;j++)
      A[i][j]*=factor;
    lhs[i]*=factor;
  }
  if (!overwrite) delete cpy;
  return 1;
}

void DenseMatrix::mult(const ColumnMatrix& x, ColumnMatrix& b,
		       int& flops, int& memrefs, int beg, int end, 
		       int spVec) const
{
  // Compute A*x=b
  ASSERTEQ(x.nrows(), nc);
  ASSERTEQ(b.nrows(), nr);
  if(beg==-1)beg=0;
  if(end==-1)end=nr;
  int i, j;
  if(!spVec) {
    for(i=beg; i<end; i++){
      double sum=0;
      double* row=data[i];
      for(j=0;j<nc;j++){
	sum+=row[j]*x[j];
      }
      b[i]=sum;
    }
  } else {
    for (i=beg; i<end; i++) b[i]=0;
    for (j=0; j<nc; j++) 
      if (x[j]) 
	for (i=beg; i<end; i++) 
	  b[i]+=data[i][j]*x[j];
  }
  flops+=(end-beg)*nc*2;
  memrefs+=(end-beg)*nc*2*sizeof(double)+(end-beg)*sizeof(double);
}
    
void DenseMatrix::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				 int& flops, int& memrefs, int beg, int end,
				 int spVec) const
{
  // Compute At*x=b
  ASSERT(x.nrows() == nr);
  ASSERT(b.nrows() == nc);
  if(beg==-1)beg=0;
  if(end==-1)end=nc;
  int i, j;
  if (!spVec) {
    for(i=beg; i<end; i++){
      double sum=0;
      for(j=0;j<nr;j++){
	sum+=data[j][i]*x[j];
      }
      b[i]=sum;
    }
  } else {
    for (i=beg; i<end; i++) b[i]=0;
    for (j=0; j<nr; j++)
      if (x[j]) {
	double *row=data[j];
	for (i=beg; i<end; i++)
	  b[i]+=row[i]*x[j];
      }
  }
  flops+=(end-beg)*nr*2;
  memrefs+=(end-beg)*nr*2*sizeof(double)+(end-beg)*sizeof(double);
}

void DenseMatrix::print() const
{
  std::cout << "Dense Matrix: " << nr << " by " << nc << std::endl;
  print(std::cout);
}

void DenseMatrix::print(ostream& ostr) const
{
  for(int i=0;i<nr;i++){
    for(int j=0;j<nc;j++){
      ostr << data[i][j] << "\t";
    }
    ostr << endl;
  }
}

void DenseMatrix::scalar_multiply(double s)
{
  for (int i=0;i<nr;i++)
  {
    for (int j=0;j<nc;j++)
    {
      data[i][j] *= s;
    }
  }
}


#define DENSEMATRIX_VERSION 3

void DenseMatrix::io(Piostream& stream)
{

  int version=stream.begin_class("DenseMatrix", DENSEMATRIX_VERSION);
  // Do the base class first...
  Matrix::io(stream);

  stream.io(nr);
  stream.io(nc);
  if(stream.reading()){
    data=scinew double*[nr];
    double* tmp=scinew double[nr*nc];
    dataptr=tmp;
    for(int i=0;i<nr;i++){
      data[i]=tmp;
      tmp+=nc;
    }
  }
  stream.begin_cheap_delim();

  int split;
  if (stream.reading()) {
    if (version > 2) {
      Pio(stream, separate_raw_);
      if (separate_raw_) {
	Pio(stream, raw_filename_);
	FILE *f=fopen(raw_filename_.c_str(), "r");
	fread(data[0], sizeof(double), nr*nc, f);
	fclose(f);
      }
    } else {
      separate_raw_ = false;
    }
    split = separate_raw_;
  } else {	// writing
    string filename = raw_filename_;
    split = separate_raw_;
    if (split) {
      if (filename == "") {
	if (stream.file_name.c_str()) {
	  char *tmp=strdup(stream.file_name.c_str());
	  char *dot = strrchr( tmp, '.' );
	  if (!dot ) dot = strrchr( tmp, 0);
	  filename = stream.file_name.substr(0,dot-tmp) + ".raw";
	  delete tmp;
	} else split=0;
      }
    }
    Pio(stream, split);
    if (split) {
      Pio(stream, filename);
      FILE *f = fopen(filename.c_str(), "w");
      fwrite(data[0], sizeof(double), nr*nc, f);
      fclose(f);
    }
  }

  if (!split) {
    int idx=0;
    for(int i=0;i<nr;i++)
      for (int j=0; j<nc; j++, idx++)
	stream.io(dataptr[idx]);
  }
  stream.end_cheap_delim();
  stream.end_class();
}

bool
DenseMatrix::invert()
{
  if (nr != nc) return false;
  double** newdata=scinew double*[nr];
  double* tmp=scinew double[nr*nc];
  double* newdataptr=tmp;

  int i;
  for(i=0;i<nr;i++){
    newdata[i]=tmp;
    for(int j=0;j<nr;j++){
      tmp[j]=0;
    }
    tmp[i]=1;
    tmp+=nc;
  }

  // Gauss-Jordan with partial pivoting
  for(i=0;i<nr;i++){
    double max=Abs(data[i][i]);
    int row=i;
    int j;
    for(j=i+1;j<nr;j++){
      if(Abs(data[j][i]) > max){
	max=Abs(data[j][i]);
	row=j;
      }
    }
    if (Abs(max) <= 1.e-12) { 
      delete[] newdataptr;
      delete[] newdata;
      return false; 
    }
    if(row != i){
      // Switch rows (actually their pointers)
      double* tmp=data[i];
      data[i]=data[row];
      data[row]=tmp;
      double* ntmp=newdata[i];
      newdata[i]=newdata[row];
      newdata[row]=ntmp;
    }
    double denom=1./data[i][i];
    double* r1=data[i];
    double* n1=newdata[i];
    for(j=i+1;j<nr;j++){
      double factor=data[j][i]*denom;
      double* r2=data[j];
      double* n2=newdata[j];
      for(int k=0;k<nr;k++){
	r2[k]-=factor*r1[k];
	n2[k]-=factor*n1[k];
      }
    }
  }

  // Back-substitution
  for(i=1;i<nr;i++){
    if (Abs(data[i][i]) <= 1.e-12) {
      delete[] newdataptr;
      delete[] newdata; 
      return false; 
    }
    double denom=1./data[i][i];
    double* r1=data[i];
    double* n1=newdata[i];
    for(int j=0;j<i;j++){
      double factor=data[j][i]*denom;
      double* r2=data[j];
      double* n2=newdata[j];
      for(int k=0;k<nr;k++){
	r2[k]-=factor*r1[k];
	n2[k]-=factor*n1[k];
      }
    }
  }

  // Normalize
  for(i=0;i<nr;i++){
    if (Abs(data[i][i]) <= 1.e-12) { 
      delete[] newdataptr;
      delete[] newdata;
      return false; 
    }
    double factor=1./data[i][i];
    for(int j=0;j<nr;j++){
      data[i][j]*=factor;
      newdata[i][j]*=factor;
    }
  }

  delete[] dataptr;
  delete[] data;    
  dataptr=newdataptr;
  data=newdata;
  return true;
}

void Mult(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.nrows());
  ASSERTEQ(out.nrows(), m1.nrows());
  ASSERTEQ(out.ncols(), m2.ncols());
  int nr=out.nrows();
  int nc=out.ncols();
  int ndot=m1.ncols();
  for(int i=0;i<nr;i++){
    double* row=m1.data[i];
    for(int j=0;j<nc;j++){
      double d=0;
      for(int k=0;k<ndot;k++){
	d+=row[k]*m2.data[k][j];
      }
      out[i][j]=d;
    }
  }
}

void Add(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();

  for(int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j]=m1.data[i][j]+m2.data[i][j];
}

void Sub(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());

  int nr=out.nrows();
  int nc=out.ncols();

  for(int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j]=m1.data[i][j]-m2.data[i][j];
}

void Add(DenseMatrix& out, double f1, const DenseMatrix& m1, double f2, const DenseMatrix& m2){
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m2.nrows());
  
  int nr=out.nrows();
  int nc=out.ncols();
  
  for(int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j]=f1*m1.data[i][j]+f2*m2.data[i][j];
}

void Add(double f1, DenseMatrix& out, double f2, const DenseMatrix& m1){
  ASSERTEQ(out.ncols(), m1.ncols());
  ASSERTEQ(out.nrows(), m1.nrows());
  int nr=out.nrows();
  int nc=out.ncols();
  
  for(int i=0;i<nr;i++)
    for (int j=0; j<nc; j++)
      out[i][j]=f1*out.data[i][j]+f2*m1.data[i][j];
}

void Mult_trans_X(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
  ASSERTEQ(m1.nrows(), m2.nrows());
  ASSERTEQ(out.nrows(), m1.ncols());
  ASSERTEQ(out.ncols(), m2.ncols());
  int nr=out.nrows();
  int nc=out.ncols();
  int ndot=m1.nrows();
  for(int i=0;i<nr;i++){
    for(int j=0;j<nc;j++){
      double d=0;
      for(int k=0;k<ndot;k++){
	d+=m1.data[k][i]*m2.data[k][j];
      }
      out[i][j]=d;
    }
  }
}

void Mult_X_trans(DenseMatrix& out, const DenseMatrix& m1, const DenseMatrix& m2)
{
  ASSERTEQ(m1.ncols(), m2.ncols());
  ASSERTEQ(out.nrows(), m1.nrows());
  ASSERTEQ(out.ncols(), m2.nrows());
  int nr=out.nrows();
  int nc=out.ncols();
  int ndot=m1.ncols();
  for(int i=0;i<nr;i++){
    double* row=m1.data[i];
    for(int j=0;j<nc;j++){
      double d=0;
      for(int k=0;k<ndot;k++){
	d+=row[k]*m2.data[j][k];
      }
      out[i][j]=d;
    }
  }
}

void DenseMatrix::mult(double s)
{
  for(int i=0;i<nr;i++){
    double* p=data[i];
    for(int j=0;j<nc;j++){
      p[j]*=s;
    }
  }
}

double DenseMatrix::sumOfCol(int n){
  ASSERT(n<nc);
  ASSERT(n>=0);

  double sum = 0;
  for (int i=0; i<nr; i++)
    sum+=data[i][n];
  return sum;
}

double DenseMatrix::sumOfRow(int n){
  ASSERT(n<nr);
  ASSERT(n>=0);
  double* rp = data[n];
  double sum = 0;
  int i=0;
  while(i<nc)
    sum+=rp[i++];
  return sum;
}

} // End namespace SCIRun
