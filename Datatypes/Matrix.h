
/*
 *  Matrix.h: Matrix definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Matrix_h
#define SCI_project_Matrix_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>

class ColumnMatrix;
class Matrix;
class MatrixRow;
typedef LockingHandle<Matrix> MatrixHandle;

class Matrix : public Datatype {
protected:
    enum Sym {
        symmetric,
        non_symmetric,
    };
    Sym sym;
    Matrix(Sym symmetric);
public:
    int is_symmetric();
    virtual ~Matrix();
    virtual Matrix* Matrix::clone();
    virtual double& get(int, int)=0;
    MatrixRow operator[](int r);
    virtual void zero()=0;
    int isolve(ColumnMatrix& lhs, ColumnMatrix& rhs,
	       double error);
    virtual int nrows()=0;
    virtual int ncols()=0;
    virtual void mult(ColumnMatrix& product, ColumnMatrix& multiplier,
		      int b=-1, int e=-1)=0;
    virtual void mult_transpose(ColumnMatrix& product, ColumnMatrix& multiplier,
				int b=-1, int e=-1)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class MatrixRow {
    Matrix* matrix;
    int row;
public:
    MatrixRow(Matrix* matrix, int row);
    ~MatrixRow();

    double& operator[](int);
};

#endif
