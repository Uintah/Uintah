
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

#include <Datatypes/Matrix.h>

class DenseMatrix : public Matrix {
    int nc;
    int nr;
    double minVal;
    double maxVal;
    double** data;
    double* dataptr;
public:
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
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& val);
    virtual void solve(ColumnMatrix&);
    virtual void zero();

    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs,
				int beg=-1, int end=-1);
    virtual void print();

    MatrixRow operator[](int r);

    void invert();
    void mult(double s);

    friend void Mult(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
    friend void Mult_trans_X(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);
    friend void Mult_X_trans(DenseMatrix&, const DenseMatrix&, const DenseMatrix&);

};

#endif
