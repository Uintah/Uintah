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
 *  DenseMatrix.h:  Dense matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 */

#ifndef SCI_project_DenseMatrix_h
#define SCI_project_DenseMatrix_h 1

#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MiscMath.h>
#include <vector>

#include <Core/Datatypes/share.h>

namespace SCIRun {

using std::vector;

class SCISHARE DenseMatrix : public Matrix {
  double** data;
  double*  dataptr_;

public:
  //! Constructors
  DenseMatrix();
  DenseMatrix(int r, int c);
  DenseMatrix(const DenseMatrix&);
  DenseMatrix(const Transform &t);
  //! Destructor
  virtual ~DenseMatrix();
  
  //! Public member functions
  virtual DenseMatrix* clone();
  DenseMatrix& operator=(const DenseMatrix&);
  
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
  virtual void    getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
  virtual void    getRowNonzerosNoCopy(int r, int &size, int &stride,
                                       int *&cols, double *&vals);

  virtual DenseMatrix* transpose();
  void gettranspose( DenseMatrix& out);
  virtual void    mult(const ColumnMatrix& x, ColumnMatrix& b,
		       int& flops, int& memrefs, int beg=-1, int end=-1, 
		       int spVec=0) const;
  virtual void    mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				 int& flops, int& memrefs,
				 int beg=-1, int end=-1, int spVec=0) const;
  virtual void scalar_multiply(double s);
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2);

  double  sumOfCol(int);
  double  sumOfRow(int);
  
  int     solve(ColumnMatrix&, int overwrite=0);
  int     solve(const ColumnMatrix& rhs, ColumnMatrix& lhs,
		int overwrite=0);
  int     solve(vector<double>& sol, int overwrite=0);
  int     solve(const vector<double>& rhs, vector<double>& lhs,
		int overwrite=0);

  //! fast accessors
  inline double*  operator[](int r) {
    return data[r];
  };
  inline double const*  operator[](int r) const{
    return data[r];
  };
  
  inline double* getData() {
    return dataptr_;
  }

  //! return false if not invertable.
  bool invert();

  //! throws an assertion if not square
  double determinant();

 
  void mult(double s);
  
  void svd(DenseMatrix&, SparseRowMatrix&, DenseMatrix&);
  void eigenvalues(ColumnMatrix&, ColumnMatrix&);
  void eigenvectors(ColumnMatrix&, ColumnMatrix&, DenseMatrix&);

  static DenseMatrix *identity(int size);

  virtual void    print() const;
  virtual void    print(std::ostream&) const;
  
  //! Persistent representation...
  virtual std::string type_name() { return "DenseMatrix"; }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  //! Friend functions
  

};


//! Friend functions
SCISHARE void Mult(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
SCISHARE void Sub(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
SCISHARE void Add(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
SCISHARE void Add(DenseMatrix&, double, const DenseMatrix&, double, const DenseMatrix&);
SCISHARE void Add(double, DenseMatrix&, double, const DenseMatrix&);
SCISHARE void Mult_trans_X(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
SCISHARE void Mult_X_trans(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
SCISHARE void Concat_rows(DenseMatrix&, const DenseMatrix&, const DenseMatrix&); // Added by Saeed Babaeizadeh, Jan. 2006
SCISHARE void Concat_cols(DenseMatrix&, const DenseMatrix&, const DenseMatrix&); // Added by Saeed Babaeizadeh, Jan. 2006

} // End namespace SCIRun

#endif
