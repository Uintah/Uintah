
/*
 *  ColumnMatrixPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColumnMatrixPort_h
#define SCI_project_ColumnMatrixPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ColumnMatrix.h>

typedef SimpleIPort<ColumnMatrixHandle> ColumnMatrixIPort;
typedef SimpleOPort<ColumnMatrixHandle> ColumnMatrixOPort;

#endif
