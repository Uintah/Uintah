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
 *  DenseMatrix.h:  Dense matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DenseMatrix_h
#define SCI_project_DenseMatrix_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

class ColumnMatrix;
class SparseRowMatrix;

class SCICORESHARE DenseMatrix : public Matrix {
  //!private data
  int nc;
  int nr;
//  double   minVal;
//  double   maxVal;
  double** data;
  double*  dataptr;
public:
  //! Constructors
  DenseMatrix();
  DenseMatrix(int r, int c);
  DenseMatrix(const DenseMatrix&);
  DenseMatrix(const Transform &t);

  virtual DenseMatrix *dense();
  virtual SparseRowMatrix *sparse();
  virtual ColumnMatrix *column();

  //! Destructor
  virtual ~DenseMatrix();
  
  //! Public member functions
  DenseMatrix& operator=(const DenseMatrix&);
  
  //! slow setters/getter for polymorphic operations
  virtual double& get(int r, int c) const;
  virtual void    put(int r, int c, double val);
  
  virtual DenseMatrix* transpose();
  
  virtual string type_name() { return "DenseMatrix"; }

  virtual int     nrows() const;
  virtual int     ncols() const;
 
  virtual void    zero();

  virtual double  sumOfCol(int);
  virtual double  sumOfRow(int);
  virtual void    getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
  
  int     solve(ColumnMatrix&, int overwrite=0);
  int     solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		int overwrite=0);
  int     solve(vector<double>& sol, int overwrite=0);
  int     solve(const vector<double>& rhs, vector<double>& lhs,
		int overwrite=0);

  virtual void    mult(const ColumnMatrix& x, ColumnMatrix& b,
		       int& flops, int& memrefs, int beg=-1, int end=-1, 
		       int spVec=0) const;
  virtual void    mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				 int& flops, int& memrefs,
				 int beg=-1, int end=-1, int spVec=0) const;
  virtual void    print() const;
  virtual void    print(ostream&) const;
  
  virtual void scalar_multiply(double s);
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2);

  //! fast accessors
  inline double*  operator[](int r) {
    return data[r];
  };
  inline double const*  operator[](int r) const{
    return data[r];
  };
  
  inline double* getData() { 
    return dataptr;
  }
  
  inline double** getData2D() { 
    return data; 
  }

  //! Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
  //! return false if not invertable.
  bool invert();

  //! throws an assertion if not square
  double determinant();

  void mult(double s);
  virtual DenseMatrix* clone();
  
  void svd(DenseMatrix&, SparseRowMatrix&, DenseMatrix&);
  void eigenvalues(ColumnMatrix&, ColumnMatrix&);
  void eigenvectors(ColumnMatrix&, ColumnMatrix&, DenseMatrix&);

  //! Friend functions
  friend SCICORESHARE void Mult(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Sub(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Add(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Add(DenseMatrix&, double, const DenseMatrix&, double, const DenseMatrix&);
  friend SCICORESHARE void Add(double, DenseMatrix&, double, const DenseMatrix&);
  friend SCICORESHARE void Mult_trans_X(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Mult_X_trans(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
};

} // End namespace SCIRun

#endif
