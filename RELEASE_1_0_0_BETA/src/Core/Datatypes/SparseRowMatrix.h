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
 *  SparseRowMatrix.h:  Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_SparseRowMatrix_h
#define SCI_project_SparseRowMatrix_h 1

#include <Core/Datatypes/Matrix.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {


class AddMatrices;
class SCICORESHARE SparseRowMatrix : public Matrix {
    int nnrows;
    int nncols;
    double dummy;
    double minVal;
    double maxVal;
protected:
public:
    int* rows;
    int* columns;
    int nnz;
    double* a;
    SparseRowMatrix();
    SparseRowMatrix(int, int, Array1<int>&, Array1<int>&);
    SparseRowMatrix(int, int, int*, int*, int, double*);
    SparseRowMatrix(int, int, int*, int*, int);
    virtual ~SparseRowMatrix();
    SparseRowMatrix(const SparseRowMatrix&);
    SparseRowMatrix& operator=(const SparseRowMatrix&);

    void transpose( SparseRowMatrix &);
    virtual double& get(int, int);
    int getIdx(int, int);
    virtual void put(int, int, const double&);
    virtual void add(int, int, const double&);
    virtual int nrows() const;
    virtual int ncols() const;
    virtual double minValue();
    virtual double maxValue();
    double density();
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual void solve(ColumnMatrix&);
    virtual void zero();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1,
		      int spVec=0) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, 
				int end=-1, int spVec=0);
    virtual void print();
    MatrixRow operator[](int r);
    friend class AddMatrices;
    virtual double* get_val(){return a;}
    virtual int* get_row(){return rows;}
    virtual int* get_col(){return columns;}
  int get_nnz() { return nnz; }
  
  virtual SparseRowMatrix* clone();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif
