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
  //! make a duplicate, needed to support detach from LockingHandle
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
  int cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
	       double &err, int &niter,
	       int& flops, int& memrefs, 
	       double max_error=1.e-6, int toomany=0) const;
  int cg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const;

  int bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		 double &err, int &niter,
		 int& flops, int& memrefs, 
		 double max_error=1.e-6, int toomany=0) const;
  int bicg_solve(const ColumnMatrix& rhs, ColumnMatrix& lhs) const;

  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs,
			      int beg=-1, int end=-1, int spVec=0) const=0;

  virtual void print(ostream&) const {}
  virtual void print() const {}

  virtual void scalar_multiply(double s) = 0;

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
