
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

#include <Datatypes/Matrix.h>
class ostream;

class ColumnMatrix {
    int rows;
    double* data;
public:
    ColumnMatrix(int);
    ~ColumnMatrix();
    ColumnMatrix(const ColumnMatrix&);
    ColumnMatrix& operator=(const ColumnMatrix&);
    int nrows();
    double& operator[](int);
    double vector_norm();

    void zero();
    void print(ostream&);
};

#include <Classlib/Assert.h>

inline double& ColumnMatrix::operator[](int i)
{
    ASSERTL3(i>=0);
    ASSERTL3(i<rows);
    return data[i];
}

#endif
