
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

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/ColumnMatrix.h>

namespace SCIRun {


typedef SimpleIPort<ColumnMatrixHandle> ColumnMatrixIPort;
typedef SimpleOPort<ColumnMatrixHandle> ColumnMatrixOPort;

} // End namespace SCIRun


#endif

