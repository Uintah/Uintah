
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
#include <Datatypes/ColumnMatrix.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

Matrix::Matrix(Sym sym)
: sym(sym)
{
}

Matrix::~Matrix()
{
}

Matrix* Matrix::clone()
{
    return 0;
}

#define MATRIX_VERSION 1

void Matrix::io(Piostream& stream)
{
    stream.begin_class("Matrix", MATRIX_VERSION);
    int* tmpsym=(int*)sym;
    stream.io(*tmpsym);
    stream.end_class();
}

#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>

template class LockingHandle<Matrix>;

#endif
