
/*
 *  ImagePort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ImagePort_h
#define SCI_project_ImagePort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Image.h>

namespace SCIRun {


typedef SimpleIPort<ImageHandle> ImageIPort;
typedef SimpleOPort<ImageHandle> ImageOPort;

} // End namespace SCIRun


#endif
