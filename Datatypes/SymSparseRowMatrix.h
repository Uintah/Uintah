
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

class SymSparseRowMatrix : public Matrix {
    int nnrows;
    int nncols;
    double* a;
    int* columns;
    int* rows;
    int nnz;
    double dummy;
public:
    SymSparseRowMatrix();
    SymSparseRowMatrix(int, int, Array1<int>&, Array1<int>&);
    virtual ~SymSparseRowMatrix();
    SymSparseRowMatrix(const SymSparseRowMatrix&);
    SymSparseRowMatrix& operator=(const SymSparseRowMatrix&);
    virtual double& get(int, int);
    virtual void put(int, int, const double&);
    virtual int nrows();
    virtual int ncols();
    virtual void solve(ColumnMatrix&);
    virtual void zero();
    virtual int mult(ColumnMatrix& product, ColumnMatrix& multiplier,
		     int b=-1, int e=-1);
    virtual int mult_transpose(ColumnMatrix& product, ColumnMatrix& multiplier,
			       int b=-1, int e=-1);
    virtual void print();
    MatrixRow operator[](int r);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
