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
 *  ColumnMatrix.cc: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/LinAlg.h>
#include <iostream>
using std::endl;
using std::cerr;

namespace SCIRun {

static Persistent* maker()
{
    return scinew ColumnMatrix(0);
}

PersistentTypeID ColumnMatrix::type_id("ColumnMatrix", "Matrix", maker);

ColumnMatrix::ColumnMatrix(int rows)
  : Matrix(Matrix::non_symmetric, Matrix::column), rows(rows)
{
    if(rows)
	data=scinew double[rows];
    else
	data=0;
}

ColumnMatrix::ColumnMatrix(const ColumnMatrix& c)
  : Matrix(Matrix::non_symmetric, Matrix::column), rows(c.rows)
{
    if(rows){
	data=scinew double[rows];
	for(int i=0;i<rows;i++)
	    data[i]=c.data[i];
    } else {
	data=0;
    }
}

ColumnMatrix* ColumnMatrix::clone() {
    return scinew ColumnMatrix(*this);
}

ColumnMatrix& ColumnMatrix::operator=(const ColumnMatrix& c)
{
    if(rows != c.rows){
	if(data)delete[] data;
	rows=c.rows;
	data=scinew double[rows];
    }
    for(int i=0;i<rows;i++)
	data[i]=c.data[i];
    return *this;
}

ColumnMatrix::~ColumnMatrix()
{
    if(data)
	delete[] data;
}

void ColumnMatrix::resize(int new_rows)
{
    if(data)
	delete[] data;
    if(new_rows)
	data=new double[new_rows];
    else
	data=0;
    rows=new_rows;
}

int ColumnMatrix::nrows() const
{
    return rows;
}

void ColumnMatrix::zero()
{
    for(int i=0;i<rows;i++)
	data[i]=0.0;
}

extern "C" double cm_vnorm(int beg, int end, double* data);
extern "C" double dnrm2(int n, double* x, int incx);

double ColumnMatrix::vector_norm()
{
//    double norm=Sqrt(cm_vnorm(0, rows, data));
//    double norm=dnrm2(rows, data, 1);
    double norm=linalg_norm2(rows, data);
    return norm; 
}

double ColumnMatrix::vector_norm(int& flops, int& memrefs)
{
    flops+=rows*2;
    memrefs+=rows*sizeof(double);
    return vector_norm();
}

double ColumnMatrix::vector_norm(int& flops, int& memrefs, int beg, int end)
{
    ASSERTRANGE(end, 0, rows+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*sizeof(double);
    return linalg_norm2((end-beg), data+beg);
}

void ColumnMatrix::print() const
{
    std::cout << "Column Matrix: " << rows << endl;
    print(std::cout);
}

void ColumnMatrix::print(std::ostream& str) const
{
    for(int i=0;i<rows;i++){
	str << data[i] << endl;
    }
}

void ColumnMatrix::print() {
  print(cerr);
}

double& ColumnMatrix::get(int r, int c)
{
    ASSERTRANGE(r, 0, rows);
    ASSERTEQ(c, 0);
    return data[r];
}

double& ColumnMatrix::get(int r)
{
    ASSERTRANGE(r, 0, rows);
    return data[r];
}

void ColumnMatrix::put(int r, int c, const double& d)
{
    ASSERTRANGE(r, 0, rows);
    ASSERTEQ(c, 0);
    data[r]=d;
}

void ColumnMatrix::put(int r, const double& d)
{
    ASSERTRANGE(r, 0, rows);
    data[r]=d;
}

int ColumnMatrix::ncols() const
{
    return 1;
}

double ColumnMatrix::sumOfCol(int c) {
  ASSERTEQ(c, 0);
  double sum = 0;
  for (int i=0; i<rows; i++)
    sum+=data[i];
  return sum;
}

double ColumnMatrix::minValue() {
  double minVal=data[0];
  for (int r=0; r<rows; r++)
    if (data[r] < minVal)
      minVal = data[r];
  return minVal;
}

double ColumnMatrix::maxValue() {
  double maxVal=data[0];
  for (int r=0; r<rows; r++)
    if (data[r] > maxVal)
      maxVal = data[r];
  return maxVal;
}

void ColumnMatrix::getRowNonzeros(int r, Array1<int>& idx, 
				  Array1<double>& val) { 
  idx.resize(1);
  val.resize(1);
  idx[0]=0;
  val[0]=data[r];
}

int ColumnMatrix::solve(ColumnMatrix&) { 
  ASSERTFAIL("Error - called solve on a columnmatrix.\n");
}

int ColumnMatrix::solve(vector<double>&) { 
  ASSERTFAIL("Error - called solve on a columnmatrix.\n");
}

void ColumnMatrix::mult(const ColumnMatrix&, ColumnMatrix&,
			int& , int& , int , int , int) const {
  ASSERTFAIL("Error - called mult on a columnmatrix.\n");
}

void ColumnMatrix::mult_transpose(const ColumnMatrix&, ColumnMatrix&,
				  int&, int&, int, int, int) {
  ASSERTFAIL("Error - called mult_transpose on a columnmatrix.\n");
}

#define COLUMNMATRIX_VERSION 2

void ColumnMatrix::io(Piostream& stream)
{
    int version=stream.begin_class("ColumnMatrix", COLUMNMATRIX_VERSION);
    
    if (version > 1) {
      // New version inherits from Matrix
      Matrix::io(stream);
    }

    stream.io(rows);
    if(stream.reading()){
      data=scinew double[rows];
    }
    int i;
    for(i=0;i<rows;i++)
      stream.io(data[i]);
    stream.end_class();
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_mult(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTRANGE(end, 0, result.rows+1);
    ASSERTRANGE(beg, 0, end);
    linalg_mult(end-beg, result.data+beg, a.data+beg, b.data+beg);
    flops+=(end-beg);
    memrefs+=(end-beg)*3*sizeof(double);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
}

void Sub(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	  int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_sub(result.rows, result.data, a.data, b.data);
    flops+=result.rows;
    memrefs+=result.rows*3*sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, a.data, b.data);
    flops+=result.rows*2;
    memrefs+=result.rows*3*sizeof(double);
}

void ScMult_Add(ColumnMatrix& result, double s, const ColumnMatrix& a,
		const ColumnMatrix& b, int& flops, int& memrefs,
		int beg, int end)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTRANGE(end, 0, result.rows+1);
    ASSERTRANGE(beg, 0, end);
    linalg_smadd(end-beg, result.data+beg, s, a.data+beg, b.data+beg);
    flops+=(end-beg)*2;
    memrefs+=(end-beg)*3*sizeof(double);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(a.rows, b.rows);
    return linalg_dot(a.rows, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs)
{
    ASSERTEQ(a.rows, b.rows);
    flops+=a.rows*2;
    memrefs+=2*sizeof(double)*a.rows;
    return linalg_dot(a.rows, a.data, b.data);
}

double Dot(const ColumnMatrix& a, const ColumnMatrix& b,
	   int& flops, int& memrefs, int beg, int end)
{
    ASSERTEQ(a.rows, b.rows);
    ASSERTRANGE(end, 0, a.rows+1);
    ASSERTRANGE(beg, 0, end);
    flops+=(end-beg)*2;
    memrefs+=2*sizeof(double)*(end-beg);
    return linalg_dot((end-beg), a.data+beg, b.data+beg);
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in)
{
    ASSERTEQ(out.rows, in.rows);
    for(int i=0;i<out.rows;i++)
	out.data[i]=in.data[i];
}

void Copy(ColumnMatrix& out, const ColumnMatrix& in, int&, int& refs,
	  int beg, int end)
{
    ASSERTEQ(out.rows, in.rows);
    ASSERTRANGE(end, 0, out.rows+1);
    ASSERTRANGE(beg, 0, end);
    for(int i=beg;i<end;i++)
	out.data[i]=in.data[i];
    refs+=sizeof(double)*(end-beg);
}

void AddScMult(ColumnMatrix& result, const ColumnMatrix& a,
	       double s, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_smadd(result.rows, result.data, s, b.data, a.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    linalg_add(result.rows, result.data, a.data, b.data);
}

void Add(ColumnMatrix& result, const ColumnMatrix& a, const ColumnMatrix& b,
	 const ColumnMatrix& c)
{
    ASSERTEQ(result.rows, a.rows);
    ASSERTEQ(result.rows, b.rows);
    ASSERTEQ(result.rows, c.rows);
    for(int i=0;i<result.rows;i++)
	result.data[i]=a.data[i]+b.data[i]+c.data[i];
}

void Mult(ColumnMatrix& result, const ColumnMatrix& a, double s)
{
    ASSERTEQ(result.rows, a.rows);
    for(int i=0;i<result.rows;i++)
	result.data[i]=a.data[i]*s;
}

} // End namespace SCIRun


