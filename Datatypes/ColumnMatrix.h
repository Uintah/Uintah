
/*
 *  ColumnMatrix.h: for RHS and LHS
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColumnMatrix_h
#define SCI_project_ColumnMatrix_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>

class ColumnMatrix;
class ostream;
typedef LockingHandle<ColumnMatrix> ColumnMatrixHandle;

class ColumnMatrix : public Datatype {
    int rows;
    double* data;
public:
    ColumnMatrix(int);
    ~ColumnMatrix();
    ColumnMatrix(const ColumnMatrix&);
    virtual ColumnMatrix* clone() const;
    ColumnMatrix& operator=(const ColumnMatrix&);
    int nrows();
    inline double& operator[](int);
    double vector_norm();

    void zero();
    void print(ostream&);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#include <Classlib/Assert.h>

inline double& ColumnMatrix::operator[](int i)
{
    return data[i];
}

#endif
