
/*
 *  MatrixPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MatrixPort_h
#define SCI_project_MatrixPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {


typedef SimpleIPort<MatrixHandle> MatrixIPort;
typedef SimpleOPort<MatrixHandle> MatrixOPort;

} // End namespace SCIRun


#endif
