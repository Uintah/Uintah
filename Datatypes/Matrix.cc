
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

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

Matrix::Matrix()
{
}

Matrix::~Matrix()
{
}

void Matrix::io(Piostream&)
{
}

Matrix* Matrix::clone()
{
    return 0;
}
