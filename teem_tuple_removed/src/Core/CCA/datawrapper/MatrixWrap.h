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
 */

#ifndef SCIRun_MatrixWrap_h
#define SCIRun_MatrixWrap_h

#include <Core/Datatypes/Matrix.h>

#include <Core/CCA/datawrapper/ColumnMatrixWrap.h>
#include <Core/CCA/datawrapper/Matrix_sidl.h>

namespace SSIDL {
  class MatrixWrap : public ::SSIDL::Matrix {
  public:
    MatrixWrap(SCIRun::Matrix* mx);
    virtual ~MatrixWrap();

    virtual bool is_dense();
    virtual bool is_sparse();
    virtual bool is_column();
    virtual Matrix::pointer transpose(); 
    virtual double get_val();
    virtual int get_row();
    virtual int get_col();
    virtual double get(int a, int b);
    virtual void put(int r, int c, double val);
    virtual ::std::string type_name();
    virtual void zero();
    virtual int nrows();
    virtual int ncols();
    /* TODO LATER
    virtual void getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
				::SSIDL::array1< double>& v); 
    */    

    virtual void mult(CCALib::SmartPointer<ColumnMatrix >& x, 
		      CCALib::SmartPointer<ColumnMatrix >& b, 
		      int& flops, int& memrefs, int beg, 
		      int end, int spVec); 
    virtual int cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs, 
			 double& err, int& niter, int& flops, 
			 int& memrefs, double max_error, int toomany); 
    virtual int cg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			 CCALib::SmartPointer<ColumnMatrix >& lhs);
    virtual int bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs, 
			   double& err, int& niter, int& flops, 
			   int& memrefs, double max_error, int toomany); 
    virtual int bicg_solve(CCALib::SmartPointer<ColumnMatrix >& rhs, 
			   CCALib::SmartPointer<ColumnMatrix >& lhs);
    virtual void mult_transpose(CCALib::SmartPointer<ColumnMatrix >& x,
				CCALib::SmartPointer<ColumnMatrix >& b, 
				int& flops, int& memrefs, int beg, 
				int end, int spVec); 
    virtual void print();
    virtual void scalar_multiply(double s);
    virtual void set_raw(bool v);
    virtual bool get_raw();
    virtual void set_raw_filename(::std::string& f);
    virtual ::std::string get_raw_filename();
   
    void* get_d_object();   
  private:
    ::SCIRun::Matrix* matrix; 
  };
} // End namespace SSIDL 



#endif

