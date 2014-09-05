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
#include <Core/Math/MiscMath.h>
#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE DenseMatrix : public Matrix {
    int nc;
    int nr;
    double minVal;
    double maxVal;
    double** data;
    double* dataptr;
public:
    DenseMatrix();
    DenseMatrix(int, int);
    virtual ~DenseMatrix();
    DenseMatrix(const DenseMatrix&);
    DenseMatrix& operator=(const DenseMatrix&);
    virtual double& get(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows() const;
    virtual int ncols() const;
    virtual double minValue();
    virtual double maxValue();
    inline double* getData() { return dataptr;}
  inline double** getData2D() { return data; }
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual int solve(ColumnMatrix&);
    virtual int solve(vector<double>& sol);
    virtual void zero();
  virtual double sumOfCol(int);

    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1, 
		      int spVec=0) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs,
				int beg=-1, int end=-1, int spVec=0);
    virtual void print();

    MatrixRow operator[](int r);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    void invert();
    void mult(double s);
    virtual DenseMatrix* clone();

    friend SCICORESHARE void Mult(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
    friend SCICORESHARE void Add(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
    friend SCICORESHARE void Add(DenseMatrix&, double, const DenseMatrix&, double, const DenseMatrix&);
    friend SCICORESHARE void Add(double, DenseMatrix&, double, const DenseMatrix&);
    friend SCICORESHARE void Mult_trans_X(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
    friend SCICORESHARE void Mult_X_trans(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);

};

} // End namespace SCIRun

#endif
