
/*
 *  SymSparseRowMatrix.h:  Symmetric Sparse Row Matrices
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_SymSparseRowMatrix_h
#define SCI_project_SymSparseRowMatrix_h 1

#include <Datatypes/Matrix.h>
#include <Classlib/Array1.h>

class AddMatrices;
class SymSparseRowMatrix : public Matrix {
    int nnrows;
    int nncols;
    int* rows;
    int nnz;
    double dummy;
    double minVal;
    double maxVal;
protected:
    int* columns;
public:
    double* a;
    SymSparseRowMatrix();
    SymSparseRowMatrix(int, int, Array1<int>&, Array1<int>&);
    SymSparseRowMatrix(int, int, int*, int*, int);
    virtual ~SymSparseRowMatrix();
    SymSparseRowMatrix(const SymSparseRowMatrix&);
    SymSparseRowMatrix& operator=(const SymSparseRowMatrix&);
    virtual double& get(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows() const;
    virtual int ncols() const;
    virtual double minValue();
    virtual double maxValue();
    double density();
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual void solve(ColumnMatrix&);
    virtual void zero();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1);
    virtual void print();
    MatrixRow operator[](int r);
    friend class AddMatrices;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
