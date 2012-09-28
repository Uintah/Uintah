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
#include <iosfwd>
#include <Core/Datatypes/share.h>

namespace SCIRun {


class SparseRowMatrix;
class DenseMatrix;
class ColumnMatrix;
class DenseColMajMatrix;
class Matrix;
typedef LockingHandle<Matrix> MatrixHandle;

class SCISHARE Matrix : public PropertyManager
{
public:
  Matrix(int nrows = 0, int ncols = 0) :
    nrows_(nrows), ncols_(ncols), separate_raw_(false), raw_filename_("") {}
  virtual ~Matrix();

  //! Make a duplicate, needed to support detach from LockingHandle
  virtual Matrix* clone() = 0;

  //! Convert this matrix to the specified type.
  virtual DenseMatrix* dense() = 0;
  virtual SparseRowMatrix* sparse() = 0;
  virtual ColumnMatrix* column() = 0;
  virtual DenseColMajMatrix *dense_col_maj() = 0;

  // No conversion is done.
  // NULL is returned if the matrix is not of the appropriate type.
  DenseMatrix *as_dense();
  SparseRowMatrix *as_sparse();
  ColumnMatrix *as_column();
  DenseColMajMatrix *as_dense_col_maj();

  // Test to see if the matrix is this subtype.
  inline bool is_dense() { return as_dense() != NULL; }
  inline bool is_sparse() { return as_sparse() != NULL; }
  inline bool is_column() { return as_column() != NULL; }
  inline bool is_dense_col_maj() { return as_dense_col_maj() != NULL; }

  inline int nrows() const { return nrows_; }
  inline int ncols() const { return ncols_; }

  virtual double* get_val() { return 0; }
  virtual int* get_row()    { return 0; }
  virtual int* get_col()    { return 0; }

  virtual double *get_data_pointer() = 0;
  virtual size_t get_data_size() = 0;

  virtual void zero() = 0;
  virtual double  get(int, int) const = 0;
  virtual void    put(int r, int c, double val) = 0;
  virtual void    add(int r, int c, double val) = 0;

  // getRowNonzerosNocopy returns:
  //   vals = The values.  They are not guaranteed 
  //     to be nonzero, but some of the zeros may be missing.
  //   cols = The columns associated with the vals.  This may be NULL, in
  //     which case the cols are 0-(size-1) (a full row).
  //   size = The number of entries in vals.
  //   stride = The matrix may not be in row order, so this is how far
  //     to walk in vals.  For example vals (and cols) should be
  //     accessed as vals[3*stride] to get the fourth value.  As of this
  //     time all the matrices return 1 for this value.
  //   For example usage see Dataflow/Modules/Fields/ApplyMappingMatrix.h
  virtual void getRowNonzerosNoCopy(int r, int &size, int &stride,
                                    int *&cols, double *&vals) = 0;

  virtual Matrix* transpose() = 0;
  virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs,
		    int beg=-1, int end=-1, int spVec=0) const=0;
  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs,
			      int beg=-1, int end=-1, int spVec=0) const=0;
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2) = 0;

  void scalar_multiply(double s);
  Transform toTransform();
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

  // Separate raw files.
  void set_raw(bool v) { separate_raw_ = v; }
  bool get_raw() { return separate_raw_; }
  void set_raw_filename( std::string &f )
  { raw_filename_ = f; separate_raw_ = true; }
  const std::string get_raw_filename() const { return raw_filename_; }

  virtual void print(std::ostream&) const {}
  virtual void print() const {}

  // Persistent representation.
  virtual std::string type_name() { return "Matrix"; }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  int          nrows_;
  int          ncols_;
  int          separate_raw_;
  std::string  raw_filename_;
};


SCISHARE void Mult(ColumnMatrix&, const Matrix&, const ColumnMatrix&);

} // End namespace SCIRun

#endif
