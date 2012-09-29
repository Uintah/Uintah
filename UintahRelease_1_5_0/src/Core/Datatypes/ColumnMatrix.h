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
 *  ColumnMatrix.h: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 */

#ifndef SCI_project_ColumnMatrix_h
#define SCI_project_ColumnMatrix_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Datatypes/Matrix.h>

#include <iosfwd>  // Forward declarations for KCC C++ I/O routines

#include <Core/Datatypes/share.h>

namespace SCIRun {

class SCISHARE ColumnMatrix : public Matrix {
  double* data;

public:
  ColumnMatrix(int rows = 0);
  ColumnMatrix(const ColumnMatrix&);
  ColumnMatrix& operator=(const ColumnMatrix&);
  virtual ColumnMatrix* clone();
  virtual ~ColumnMatrix();

  virtual DenseMatrix *dense();
  virtual SparseRowMatrix *sparse();
  virtual ColumnMatrix *column();
  virtual DenseColMajMatrix *dense_col_maj();

  virtual double *get_data_pointer();
  virtual size_t get_data_size();

  inline double& operator[](int r) const
  {
    ASSERTRANGE(r, 0, nrows_)
    return data[r];
  }
  double* get_data() const {return data;}
  void set_data(double* d) {data = d;} 
  double  get(int r) const      { ASSERTRANGE(r, 0, nrows_); return data[r]; };
  void    put(int r, double val) { ASSERTRANGE(r, 0, nrows_); data[r] = val; };

  void resize(int);

  virtual void zero();
  virtual double get(int, int) const;
  virtual void put(int row, int col, double val);
  virtual void add(int row, int col, double val);
  virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
  virtual void getRowNonzerosNoCopy(int r, int &size, int &stride,
                                    int *&cols, double *&vals);

  virtual Matrix *transpose();
  virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		    int& flops, int& memrefs, int beg=-1, int end=-1, 
		    int spVec=0) const;
  virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
			      int& flops, int& memrefs,
			      int beg=-1, int end=-1, int spVec=0) const;
  virtual void scalar_multiply(double s);
  virtual MatrixHandle submatrix(int r1, int c1, int r2, int c2);


  int solve(ColumnMatrix&);
  int solve(vector<double>& sol);
  double sumOfCol(int);

  DenseMatrix exterior(const ColumnMatrix &) const;
  double vector_norm() const;
  double vector_norm(int& flops, int& memrefs) const;
  double vector_norm(int& flops, int& memrefs, int beg, int end) const;

  
  virtual void print();
  virtual void print() const;
  virtual void print(std::ostream&) const;
  
  // Persistent representation...
  virtual string type_name() { return "ColumnMatrix"; }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


  SCISHARE friend void Mult(ColumnMatrix&, const ColumnMatrix&, double s);
  SCISHARE friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  SCISHARE friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
				int& flops, int& memrefs);
  SCISHARE friend void Mult(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
				int& flops, int& memrefs, int beg, int end);
  SCISHARE friend void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  SCISHARE friend void Sub(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&,
			       int& flops, int& memrefs);
  SCISHARE friend double Dot(const ColumnMatrix&, const ColumnMatrix&);
  SCISHARE friend double Dot(const ColumnMatrix&, const ColumnMatrix&,
				 int& flops, int& memrefs);
  SCISHARE friend double Dot(const ColumnMatrix&, const ColumnMatrix&,
				 int& flops, int& memrefs, int beg, int end);
  SCISHARE friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&);
  SCISHARE friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&, int& flops, int& memrefs);
  SCISHARE friend void ScMult_Add(ColumnMatrix&, double s, const ColumnMatrix&,
				      const ColumnMatrix&, int& flops, int& memrefs,
				      int beg, int end);
  
  SCISHARE friend void Copy(ColumnMatrix&, const ColumnMatrix&);
  SCISHARE friend void Copy(ColumnMatrix&, const ColumnMatrix&, 
				int& flops, int& refs);
  SCISHARE friend void Copy(ColumnMatrix&, const ColumnMatrix&, 
				int& flops, int& refs,
				int beg, int end);
  SCISHARE friend void AddScMult(ColumnMatrix&, const ColumnMatrix&, 
			double s, const ColumnMatrix&);
  SCISHARE friend void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&);
  SCISHARE friend void Add(ColumnMatrix&, const ColumnMatrix&, const ColumnMatrix&, 
		  const ColumnMatrix&);
};


} // End namespace SCIRun

#endif
