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


#include <Core/CCA/datawrapper/DenseMatrixWrap.h>

using namespace SSIDL;

DenseMatrixWrap::DenseMatrixWrap(SCIRun::DenseMatrix* mx) : matrix(mx) { } 
DenseMatrixWrap::~DenseMatrixWrap() {if(matrix) delete matrix;}

bool DenseMatrixWrap::is_dense() {return matrix->is_dense();}
bool DenseMatrixWrap::is_sparse() {return matrix->is_sparse();}
bool DenseMatrixWrap::is_column() {return matrix->is_column();}

Matrix::pointer DenseMatrixWrap::transpose() 
{
  return Matrix::pointer(new DenseMatrixWrap(matrix->transpose()));
}

double DenseMatrixWrap::get_val() {double* dbl = matrix->get_val(); if(dbl) return *(dbl); else return 0;}
int DenseMatrixWrap::get_row() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
int DenseMatrixWrap::get_col() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
double DenseMatrixWrap::get(int a, int b) {return matrix->get(a,b);}
void DenseMatrixWrap::put(int r, int c, double val) {matrix->put(r,c,val);}
::std::string DenseMatrixWrap::type_name() {return matrix->type_name();}
void DenseMatrixWrap::zero() {matrix->zero();}
int DenseMatrixWrap::nrows() {return matrix->nrows();}
int DenseMatrixWrap::ncols() {return matrix->ncols();}

/* TODO LATER
   virtual void getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
   ::SSIDL::array1< double>& v) 
   {matrix->getRowNonzeros(r,idx,v);}
*/    

void DenseMatrixWrap::mult(CCALib::SmartPointer<ColumnMatrix >& x, 
		      CCALib::SmartPointer<ColumnMatrix >& b, 
		      int& flops, int& memrefs, int beg, 
		      int end, int spVec) 
{ 
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "DenseMatrixWrap::mult -- Null matrix received\n";
  }
}

int DenseMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs, 
			 double& err, int& niter, int& flops, 
			 int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "DenseMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int DenseMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "DenseMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int DenseMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs, 
			   double& err, int& niter, int& flops, 
			   int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "DenseMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

int DenseMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "DenseMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

void DenseMatrixWrap::mult_transpose(CCALib::SmartPointer<ColumnMatrix >& x,
				CCALib::SmartPointer<ColumnMatrix >& b, 
				int& flops, int& memrefs, int beg, 
				int end, int spVec) 
{
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult_transpose(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "DenseMatrixWrap::mult_transpose -- Null matrix received\n";
  }
}

void DenseMatrixWrap::print() {matrix->print();}
void DenseMatrixWrap::scalar_multiply(double s) {matrix->scalar_multiply(s);}
void DenseMatrixWrap::set_raw(bool v) {matrix->set_raw(v);}
bool DenseMatrixWrap::get_raw() {return matrix->get_raw();}
void DenseMatrixWrap::set_raw_filename(::std::string& f) {matrix->set_raw_filename(f);}
::std::string DenseMatrixWrap::get_raw_filename() {return matrix->get_raw_filename();}
void* DenseMatrixWrap::get_d_object() {return ((void*)matrix);}

//DENSEMATRIX:
double DenseMatrixWrap::sumOfCol(int i) {return matrix->sumOfCol(i);}
double DenseMatrixWrap::sumOfRow(int i) {return matrix->sumOfRow(i);}
int DenseMatrixWrap::solve(ColumnMatrix::pointer& col, int overwrite)
{
  ::SCIRun::ColumnMatrix* col_col = static_cast< ::SCIRun::ColumnMatrix*>(col->get_d_object());
  if(col_col) {
    return matrix->solve(*(col_col),overwrite);
  } else {
    ::std::cerr << "DenseMatrixWrap::solve - Null matrix received\n";
  }
  return 0;
}

int DenseMatrixWrap::solve(ColumnMatrix::pointer& rhs,
			   ColumnMatrix::pointer& lhs, int overwrite)
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    matrix->solve(*(col_rhs),*(col_lhs),overwrite);
  } else {
    ::std::cerr << "DenseMatrixWrap::solve -- Null matrix received\n";
  }
  return 0;
}

bool DenseMatrixWrap::invert() {return matrix->invert();} 
void DenseMatrixWrap::mult(double s) {return matrix->mult(s);}

