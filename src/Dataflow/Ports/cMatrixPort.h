
/*
 *  cMatrixPort.h
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_cMatrixPort_h
#define SCI_project_cMatrixPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/cMatrix.h>

namespace SCIRun {


typedef SimpleIPort<cMatrixHandle> cMatrixIPort;
typedef SimpleOPort<cMatrixHandle> cMatrixOPort;

} // End namespace SCIRun


#endif
