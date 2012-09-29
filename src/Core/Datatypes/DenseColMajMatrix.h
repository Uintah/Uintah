/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  DenseColMajMatrix.h:  DenseColMaj matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#ifndef SCI_project_DenseColMajMatrix_h
#define SCI_project_DenseColMajMatrix_h 1

#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>
#include <vector>

#include <Core/Datatypes/share.h>

namespace SCIRun {

using std::vector;

class SCISHARE DenseColMajMatrix : public Matrix
{
  double*  dataptr_;

public:
  //! Constructors
  DenseColMajMatrix();
  DenseColMajMatrix(int r, int c);
  DenseColMajMatrix(const DenseColMajMatrix&);

  //! Destructor
  virtual ~DenseColMajMatrix();
  
  //! Public member functions
  virtual DenseColMajMatrix* clone();
  DenseColMajMatrix& operator=(const DenseColMajMatrix&);
  
  virtual DenseMatrix *dense();
  virtual SparseRowMatrix *sparse();
  virtual ColumnMatrix *column();
  virtual DenseColMajMatrix *dense_col_maj();

  virtual double *get_data_pointer();
  virtual size_t get_data_size();

  //! slow setters/getter for polymorphic operations
  virtual void    zero();
  virtual double  get(int r, int c) const;
  virtual void    put(int r, int c, double val);
  virtual void    add(int r, int c, double val);
  virtual void    getRowNonzerosNoCopy(int r, int &size, int &stride,
                                       int *&cols, double *&vals);

  virtual DenseColMajMatrix* transpose();
  virtual void    mult(const ColumnMatrix& x, ColumnMatrix& b,
		       int& flops, int& memrefs, int beg=-1, int end=-1, 
		       int spVec=0) const;
  virtual void    mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				 int& flops, int& memrefs,
				 int beg=-1, int end=-1, int spVec=0) const;
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2);


  double  sumOfCol(int);
  double  sumOfRow(int);
  
#if 0
  bool    solve(ColumnMatrix&, int overwrite=0);
  bool    solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		int overwrite=0);
  bool    solve(vector<double>& sol, int overwrite=0);
  bool    solve(const vector<double>& rhs, vector<double>& lhs,
		int overwrite=0);
#endif

  //! fast accessors
  inline double &iget(int r, int c)
  {
    return dataptr_[c * nrows_ + r];
  };

  //! fast accessors
  inline const double &iget(int r, int c) const
  {
    return dataptr_[c * nrows_ + r];
  };

  //! Return false if not invertable.
  //bool invert();

  //! Throws an assertion if not square
  double determinant();

  //void svd(DenseColMajMatrix&, SparseRowMatrix&, DenseColMajMatrix&);
  //void eigenvalues(ColumnMatrix&, ColumnMatrix&);
  //void eigenvectors(ColumnMatrix&, ColumnMatrix&, DenseColMajMatrix&);

  static DenseColMajMatrix *identity(int size);

  virtual void    print() const;
  virtual void    print(std::ostream&) const;
  
  //! Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};


} // End namespace SCIRun

#endif
