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
  virtual void   put(int r, int c, const double &val);
  
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

  //! fast accessors
  inline double*  operator[](int r){
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

  void mult(double s);
  virtual DenseMatrix* clone();
  
  //! Friend function
  friend SCICORESHARE void Mult(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Add(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Sub(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Add(DenseMatrix&, double, const DenseMatrix&, double, const DenseMatrix&);
  friend SCICORESHARE void Add(double, DenseMatrix&, double, const DenseMatrix&);
  friend SCICORESHARE void Mult_trans_X(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
  friend SCICORESHARE void Mult_X_trans(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
};

} // End namespace SCIRun

#endif
