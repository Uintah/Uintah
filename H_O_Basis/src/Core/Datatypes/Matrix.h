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
 *  Matrix.h: Matrix definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Matrix_h
#define SCI_project_Matrix_h 1

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Transform.h>
#include <Core/Containers/LockingHandle.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using namespace std;


class SparseRowMatrix;
class DenseMatrix;
class ColumnMatrix;
class Matrix;
class MatrixRow;
typedef LockingHandle<Matrix> MatrixHandle;

class SCICORESHARE Matrix : public PropertyManager
{
public:
  Matrix() : separate_raw_(false), raw_filename_("") {}

  //! make a duplicate, needed to support detach from LockingHandle
  virtual ~Matrix();
  virtual Matrix* clone() = 0;

  //! convert this matrix to a DenseMatrix.
  virtual DenseMatrix* dense() = 0;
  //! convert this matrix to a SparseRowMatrix.
  virtual SparseRowMatrix* sparse() = 0;
  //! convert this matrix to a ColumnMatrix.
  virtual ColumnMatrix* column() = 0;

  bool is_dense();
  bool is_sparse();
  bool is_column();

  // No conversion is done.
  // NULL is returned if the matrix is not of the appropriate type.
  DenseMatrix *as_dense();
  SparseRowMatrix *as_sparse();
  ColumnMatrix *as_column();

  virtual Matrix* transpose() = 0;
  virtual double* get_val() { return 0; }
  virtual int* get_row()    { return 0; }
  virtual int* get_col()    { return 0; }

  Transform toTransform();
  
  virtual double& get(int, int) const = 0;
  virtual void    put(int r, int c, double val) = 0;
  inline MatrixRow operator[](int r);

  //friend SCICORESHARE Matrix *Add(Matrix *, Matrix *);
  //friend SCICORESHARE Matrix *Mult(Matrix *, Matrix *);

  virtual string type_name() { return "Matrix"; }

  virtual void zero()=0;
  virtual int nrows() const=0;
  virtual int ncols() const=0;
  virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v)=0;
  virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs,
		    int beg=-1, int end=-1, int spVec=0) const=0;
  DenseMatrix *direct_inverse();
  DenseMatrix *iterative_inverse();
  int cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
	       double &err, int &niter,
	       int& flops, int& memrefs, 
	       double max_error=1.e-6, int toomany=0,
	       int useLhsAsGuess=0) const;
  int cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const;
  int cg_solve(const DenseMatrix& rhs, DenseMatrix& lhs,
	       double &err, int &niter,
	       int& flops, int& memrefs, 
	       double max_error=1.e-6, int toomany=0, 
	       int useLhsAsGuess=0) const;
  int cg_solve(const DenseMatrix& rhs, DenseMatrix& lhs) const;

  int bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		 double &err, int &niter,
		 int& flops, int& memrefs, 
		 double max_error=1.e-6, int toomany=0,
		 int useLhsAsGuess=0) const;
  int bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const;
  int bicg_solve(const DenseMatrix& rhs, DenseMatrix& lhs,
		 double &err, int &niter,
		 int& flops, int& memrefs, 
		 double max_error=1.e-6, int toomany=0,
		 int useLhsAsGuess=0) const;
  int bicg_solve(const DenseMatrix& rhs, DenseMatrix& lhs) const;

  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs,
			      int beg=-1, int end=-1, int spVec=0) const=0;

  virtual void print(ostream&) const {}
  virtual void print() const {}

  virtual void scalar_multiply(double s) = 0;
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2) = 0;

  // Separate raw files.
  void set_raw(bool v) { separate_raw_ = v; }
  bool get_raw() { return separate_raw_; }
  void set_raw_filename( string &f )
  { raw_filename_ = f; separate_raw_ = true; }
  const string get_raw_filename() const { return raw_filename_; }

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  bool     separate_raw_;
  string   raw_filename_;
};


class SCICORESHARE MatrixRow
{
  Matrix* matrix;
  int row;
public:
  inline MatrixRow(Matrix* matrix, int row) : matrix(matrix), row(row) {}
  inline ~MatrixRow() {}

  inline double& operator[](int col) {return matrix->get(row, col);}
};


inline MatrixRow
Matrix::operator[](int row)
{
  return MatrixRow(this, row);
}


void Mult(ColumnMatrix&, const Matrix&, const ColumnMatrix&);

} // End namespace SCIRun

#endif
