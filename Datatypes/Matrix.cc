
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

#include <Datatypes/Matrix.h>
#include <Classlib/NotFinished.h>

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

Matrix::Matrix(Sym sym)
: sym(sym)
{
}

Matrix::~Matrix()
{
}

void Matrix::io(Piostream&)
{
    NOT_FINISHED("Matrix::io");
}

Matrix* Matrix::clone()
{
    return 0;
}

MatrixRow::MatrixRow(Matrix* matrix, int row)
: matrix(matrix), row(row)
{
}

MatrixRow::~MatrixRow()
{
}

double& MatrixRow::operator[](int col)
{
    return matrix->get(row, col);
}

MatrixRow Matrix::operator[](int row)
{
    return MatrixRow(this, row);
}
