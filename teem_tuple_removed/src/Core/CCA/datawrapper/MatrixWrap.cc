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


#include <Core/CCA/datawrapper/MatrixWrap.h>

using namespace SSIDL;

MatrixWrap::MatrixWrap(SCIRun::Matrix* mx) : matrix(mx) { } 
MatrixWrap::~MatrixWrap() {if(matrix) delete matrix;}

bool MatrixWrap::is_dense() {return matrix->is_dense();}
bool MatrixWrap::is_sparse() {return matrix->is_sparse();}
bool MatrixWrap::is_column() {return matrix->is_column();}

Matrix::pointer MatrixWrap::transpose() 
{
  return Matrix::pointer(new MatrixWrap(matrix->transpose()));
}

double MatrixWrap::get_val() {double* dbl = matrix->get_val(); if(dbl) return *(dbl); else return 0;}
int MatrixWrap::get_row() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
int MatrixWrap::get_col() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
double MatrixWrap::get(int a, int b) {return matrix->get(a,b);}
void MatrixWrap::put(int r, int c, double val) {matrix->put(r,c,val);}
::std::string MatrixWrap::type_name() {return matrix->type_name();}
void MatrixWrap::zero() {matrix->zero();}
int MatrixWrap::nrows() {return matrix->nrows();}
int MatrixWrap::ncols() {return matrix->ncols();}

/* TODO LATER
   virtual void getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
   ::SSIDL::array1< double>& v) 
   {matrix->getRowNonzeros(r,idx,v);}
*/    

void MatrixWrap::mult(CCALib::SmartPointer<ColumnMatrix >& x, 
		      CCALib::SmartPointer<ColumnMatrix >& b, 
		      int& flops, int& memrefs, int beg, 
		      int end, int spVec) 
{ 
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "MatrixWrap::mult -- Null matrix received\n";
  }
}

int MatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs, 
			 double& err, int& niter, int& flops, 
			 int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "MatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int MatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "MatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int MatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs, 
			   double& err, int& niter, int& flops, 
			   int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "MatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

int MatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "MatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

void MatrixWrap::mult_transpose(CCALib::SmartPointer<ColumnMatrix >& x,
				CCALib::SmartPointer<ColumnMatrix >& b, 
				int& flops, int& memrefs, int beg, 
				int end, int spVec) 
{
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult_transpose(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "MatrixWrap::mult_transpose -- Null matrix received\n";
  }
}

void MatrixWrap::print() {matrix->print();}
void MatrixWrap::scalar_multiply(double s) {matrix->scalar_multiply(s);}
void MatrixWrap::set_raw(bool v) {matrix->set_raw(v);}
bool MatrixWrap::get_raw() {return matrix->get_raw();}
void MatrixWrap::set_raw_filename(::std::string& f) {matrix->set_raw_filename(f);}
::std::string MatrixWrap::get_raw_filename() {return matrix->get_raw_filename();}
void* MatrixWrap::get_d_object() {return ((void*)matrix);}

