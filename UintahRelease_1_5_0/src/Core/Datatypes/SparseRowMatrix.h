/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  SparseRowMatrix.h:  Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 */


#ifndef SCI_project_SparseRowMatrix_h
#define SCI_project_SparseRowMatrix_h 1

#include <Core/Datatypes/Matrix.h>
#include <Core/Containers/Array1.h>

#include <Core/Datatypes/share.h>

namespace SCIRun {

class SCISHARE SparseRowMatrix : public Matrix {
private:
  SparseRowMatrix(); // This is only used by the maker function.

public:
  //! Public data
  int* rows;
  int* columns;
  int nnz;
  double* a;

  void validate();

  //! Constructors
  // Here's what the arguements for the constructor should be:
  //   r   = number of rows
  //   c   = number of columns
  //   rr  = row accumulation buffer containing r+1 entries where
  //         rr[N+1]-rr[N] is the number of non-zero entries in row N
  //   cc  = column number for each nonzero data entry.  Sorted by
  //         row/col orderand corresponds with the spaces in the rr array.
  //   nnz = number of non zero entries.
  //   d   = non zero data values.
  SparseRowMatrix(int r, int c, int *rr, int *cc, int nnz, double *d = 0);
  SparseRowMatrix(const SparseRowMatrix&);

  virtual SparseRowMatrix* clone();
  SparseRowMatrix& operator=(const SparseRowMatrix&);

  //! Destructor
  virtual ~SparseRowMatrix();

  virtual DenseMatrix *dense();
  virtual SparseRowMatrix *sparse();
  virtual ColumnMatrix *column();
  virtual DenseColMajMatrix *dense_col_maj();

  virtual double *get_data_pointer();
  virtual size_t get_data_size();

  int getIdx(int, int);
  int get_nnz() { return nnz; }
  
  virtual double* get_val() { return a; }
  virtual int*    get_row() { return rows; }
  virtual int*    get_col() { return columns; }

  virtual void zero();
  virtual double get(int, int) const;
  virtual void put(int row, int col, double val);
  virtual void add(int row, int col, double val);
  virtual void getRowNonzerosNoCopy(int r, int &size, int &stride,
                                    int *&cols, double *&vals);

  //! 
  virtual SparseRowMatrix *transpose();
  virtual SparseRowMatrix *transpose() const;
  virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs, int beg=-1, int end=-1,
		    int spVec=0) const;
  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs, int beg=-1, 
			      int end=-1, int spVec=0) const;
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2);

  void sparse_mult(const DenseMatrix& x, DenseMatrix& b) const;
  void sparse_mult_transXB(const DenseMatrix& x, DenseMatrix& b) const;
  MatrixHandle sparse_sparse_mult(const SparseRowMatrix &x) const;
  void solve(ColumnMatrix&);

  static SparseRowMatrix *identity(int size);

  virtual void print() const;
  virtual void print(std::ostream&) const;
 
  //! Persistent representation...
  virtual string type_name() { return "SparseRowMatrix"; }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


  friend SparseRowMatrix *AddSparse(const SparseRowMatrix &a,
						 const SparseRowMatrix &b);
  friend SparseRowMatrix *SubSparse(const SparseRowMatrix &a,
						 const SparseRowMatrix &b);


  static Persistent *maker();
};

} // End namespace SCIRun

#endif
