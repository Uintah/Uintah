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

#ifndef SCIRun_ColumnMatrixWrap_h
#define SCIRun_ColumnMatrixWrap_h

#include <Core/Datatypes/ColumnMatrix.h>

#include <Core/CCA/datawrapper/Matrix_sidl.h>
#include <Core/CCA/datawrapper/MatrixWrap.h>

namespace SSIDL {
  class ColumnMatrixWrap : public ::SSIDL::ColumnMatrix {
  public:
    ColumnMatrixWrap(SCIRun::ColumnMatrix* mx); 
    virtual ~ColumnMatrixWrap();

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
    virtual void mult(CCALib::SmartPointer<ColumnMatrix >& x,
	              CCALib::SmartPointer<ColumnMatrix >& b,
	              int& flops, int& memrefs, int beg,
	              int end, int spVec);

    /* TODO LATER
    virtual void getRowNonzeros(int r, ::SSIDL::array1< int>& idx, 
				::SSIDL::array1< double>& v) 
      {matrix->getRowNonzeros(r,idx,v);}
    */

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

    //FROM ColumnMatrix:
    virtual double get_data();
    virtual void set_data(double& d);
    virtual double get(int i);
    virtual void put(int row, double val);
    virtual double vector_norm();
    virtual double vector_norm(int& flops, int& memrefs);
    virtual double vector_norm(int& flops, int& memrefs, int beg, int end);
    virtual void resize(int i);

    void* get_d_object();    
  private:
    ::SCIRun::ColumnMatrix* matrix; 
  };
} // End namespace SSIDL 



#endif

