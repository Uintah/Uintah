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



#include <Core/CCA/datawrapper/SparseRowMatrixWrap.h>

using namespace SSIDL;

SparseRowMatrixWrap::SparseRowMatrixWrap(SCIRun::SparseRowMatrix* mx) : matrix(mx) { } 
SparseRowMatrixWrap::~SparseRowMatrixWrap() {if(matrix) delete matrix;}

bool SparseRowMatrixWrap::is_dense() {return matrix->is_dense();}
bool SparseRowMatrixWrap::is_sparse() {return matrix->is_sparse();}
bool SparseRowMatrixWrap::is_column() {return matrix->is_column();}

Matrix::pointer SparseRowMatrixWrap::transpose() 
{
  return Matrix::pointer(new SparseRowMatrixWrap(matrix->transpose()));
}

double SparseRowMatrixWrap::get_val() {double* dbl = matrix->get_val(); if(dbl) return *(dbl); else return 0;}
int SparseRowMatrixWrap::get_row() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
int SparseRowMatrixWrap::get_col() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
double SparseRowMatrixWrap::get(int a, int b) {return matrix->get(a,b);}
void SparseRowMatrixWrap::put(int r, int c, double val) {matrix->put(r,c,val);}
::std::string SparseRowMatrixWrap::type_name() {return matrix->type_name();}
void SparseRowMatrixWrap::zero() {matrix->zero();}
int SparseRowMatrixWrap::nrows() {return matrix->nrows();}
int SparseRowMatrixWrap::ncols() {return matrix->ncols();}

/* TODO LATER
   virtual void getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
   ::SSIDL::array1< double>& v) 
   {matrix->getRowNonzeros(r,idx,v);}
*/    

void SparseRowMatrixWrap::mult(CCALib::SmartPointer<ColumnMatrix >& x, 
		      CCALib::SmartPointer<ColumnMatrix >& b, 
		      int& flops, int& memrefs, int beg, 
		      int end, int spVec) 
{ 
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "SparseRowMatrixWrap::mult -- Null matrix received\n";
  }
}

int SparseRowMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs, 
			 double& err, int& niter, int& flops, 
			 int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "SparseRowMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int SparseRowMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "SparseRowMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int SparseRowMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs, 
			   double& err, int& niter, int& flops, 
			   int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "SparseRowMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

int SparseRowMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "SparseRowMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

void SparseRowMatrixWrap::mult_transpose(CCALib::SmartPointer<ColumnMatrix >& x,
				CCALib::SmartPointer<ColumnMatrix >& b, 
				int& flops, int& memrefs, int beg, 
				int end, int spVec) 
{
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult_transpose(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "SparseRowMatrixWrap::mult_transpose -- Null matrix received\n";
  }
}

void SparseRowMatrixWrap::print() {matrix->print();}
void SparseRowMatrixWrap::scalar_multiply(double s) {matrix->scalar_multiply(s);}
void SparseRowMatrixWrap::set_raw(bool v) {matrix->set_raw(v);}
bool SparseRowMatrixWrap::get_raw() {return matrix->get_raw();}
void SparseRowMatrixWrap::set_raw_filename(::std::string& f) {matrix->set_raw_filename(f);}
::std::string SparseRowMatrixWrap::get_raw_filename() {return matrix->get_raw_filename();}
void* SparseRowMatrixWrap::get_d_object() {return ((void*)matrix);}

//SPARSEROWMATRIX:
void SparseRowMatrixWrap::add(int row, int col, double val) {matrix->add(row,col,val);}
int SparseRowMatrixWrap::getIdx(int i, int j) {return matrix->getIdx(i,j);}
int SparseRowMatrixWrap::get_nnz() {return matrix->get_nnz();}

void SparseRowMatrixWrap::solve(ColumnMatrix::pointer& i) 
{
  ::SCIRun::ColumnMatrix* col_i = static_cast< ::SCIRun::ColumnMatrix*>(i->get_d_object());
  if(col_i) {
    matrix->solve(*(col_i));
  } else {
    ::std::cerr << "SparseRowMatrixWrap::solve -- Null matrix received\n";
  }
}
