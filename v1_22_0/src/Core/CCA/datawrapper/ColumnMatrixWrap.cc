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




#include <Core/CCA/datawrapper/ColumnMatrixWrap.h>

using namespace SSIDL;

ColumnMatrixWrap::ColumnMatrixWrap(SCIRun::ColumnMatrix* mx) : matrix(mx) { }; 
ColumnMatrixWrap::~ColumnMatrixWrap() {if(matrix) delete matrix;}

bool ColumnMatrixWrap::is_dense() {return matrix->is_dense();}
bool ColumnMatrixWrap::is_sparse() {return matrix->is_sparse();}
bool ColumnMatrixWrap::is_column() {return matrix->is_column();}
Matrix::pointer ColumnMatrixWrap::transpose() 
  {return Matrix::pointer(new MatrixWrap(matrix->transpose()));}
double ColumnMatrixWrap::get_val() {double* dbl = matrix->get_val(); if(dbl) return *(dbl); else return 0;}
int ColumnMatrixWrap::get_row() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
int ColumnMatrixWrap::get_col() {int* i = matrix->get_row(); if(i) return *(i); else return 0;}
double ColumnMatrixWrap::get(int a, int b) {return matrix->get(a,b);}
void ColumnMatrixWrap::put(int r, int c, double val) {matrix->put(r,c,val);}
::std::string ColumnMatrixWrap::type_name() {return matrix->type_name();}
void ColumnMatrixWrap::zero() {matrix->zero();}
int ColumnMatrixWrap::nrows() {return matrix->nrows();}
int ColumnMatrixWrap::ncols() {return matrix->ncols();}
void ColumnMatrixWrap::mult(CCALib::SmartPointer<ColumnMatrix >& x,
   		            CCALib::SmartPointer<ColumnMatrix >& b,
		            int& flops, int& memrefs, int beg,
			    int end, int spVec)
{
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "ColumnMatrixWrap::mult -- Null matrix received\n";
  }
}

/* TODO LATER
   void ColumnMatrixWrap::getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
   ::SSIDL::array1< double>& v) 
   {matrix->getRowNonzeros(r,idx,v);}
*/   

int ColumnMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			       CCALib::SmartPointer<ColumnMatrix >& lhs, 
			       double& err, int& niter, int& flops, 
			       int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "ColumnMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int ColumnMatrixWrap::cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			       CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->cg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "ColumnMatrixWrap::cg_solve -- Null matrix received\n";
  }
  return 0;
}

int ColumnMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
				 CCALib::SmartPointer<ColumnMatrix >& lhs, 
				 double& err, int& niter, int& flops, 
				 int& memrefs, double max_error, int toomany) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs),err,niter,flops,memrefs,max_error,toomany);
  } else {
    ::std::cerr << "ColumnMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

int ColumnMatrixWrap::bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
				 CCALib::SmartPointer<ColumnMatrix >& lhs) 
{
  ::SCIRun::ColumnMatrix* col_rhs = static_cast< ::SCIRun::ColumnMatrix*>(rhs->get_d_object());
  ::SCIRun::ColumnMatrix* col_lhs = static_cast< ::SCIRun::ColumnMatrix*>(lhs->get_d_object());
  if((col_rhs)&&(col_lhs)) {
    return matrix->bicg_solve(*(col_lhs),*(col_rhs));
  } else {
    ::std::cerr << "ColumnMatrixWrap::bicg_solve -- Null matrix received\n";
  }
  return 0;
}

void ColumnMatrixWrap::mult_transpose(CCALib::SmartPointer<ColumnMatrix >& x,
				      CCALib::SmartPointer<ColumnMatrix >& b, 
				      int& flops, int& memrefs, int beg, 
				      int end, int spVec) 
{
  ::SCIRun::ColumnMatrix* col_x = static_cast< ::SCIRun::ColumnMatrix*>(x->get_d_object());
  ::SCIRun::ColumnMatrix* col_b = static_cast< ::SCIRun::ColumnMatrix*>(b->get_d_object());
  if((col_x)&&(col_b)) {
    matrix->mult_transpose(*(col_x),*(col_b),flops,memrefs,beg,end,spVec);
  } else {
    ::std::cerr << "ColumnMatrixWrap::mult_transpose -- Null matrix received\n";
  }
}

void ColumnMatrixWrap::print() {matrix->print();}
void ColumnMatrixWrap::scalar_multiply(double s) {matrix->scalar_multiply(s);}
void ColumnMatrixWrap::set_raw(bool v) {matrix->set_raw(v);}
bool ColumnMatrixWrap::get_raw() {return matrix->get_raw();}
void ColumnMatrixWrap::set_raw_filename(::std::string& f) {matrix->set_raw_filename(f);}
::std::string ColumnMatrixWrap::get_raw_filename() {return matrix->get_raw_filename();}

//FROM ColumnMatrix:
double ColumnMatrixWrap::get_data() {double* dbl = matrix->get_data(); if(dbl) return *(dbl); else return 0;}
void ColumnMatrixWrap::set_data(double& d) {matrix->set_data(&d);}
double ColumnMatrixWrap::get(int i) {return matrix->get(i);}
void ColumnMatrixWrap::put(int row, double val) {matrix->put(row,val);}
double ColumnMatrixWrap::vector_norm() {return matrix->vector_norm();}
double ColumnMatrixWrap::vector_norm(int& flops, int& memrefs) {return matrix->vector_norm(flops,memrefs);}

double ColumnMatrixWrap::vector_norm(int& flops, int& memrefs, int beg, int end) 
{
  return matrix->vector_norm(flops,memrefs,beg,end);
}

void ColumnMatrixWrap::resize(int i) {matrix->resize(i);}
void* ColumnMatrixWrap::get_d_object() {return ((void*)matrix);}


