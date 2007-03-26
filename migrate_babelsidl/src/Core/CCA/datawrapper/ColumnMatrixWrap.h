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

