
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

class Matrix;
typedef LockingHandle<Matrix> MatrixHandle;

class Matrix : public Datatype {
public:
    Matrix();
    virtual ~Matrix();
    virtual Matrix* Matrix::clone();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
