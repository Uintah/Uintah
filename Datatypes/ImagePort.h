
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

#include <Datatypes/SimplePort.h>
#include <Datatypes/Image.h>

typedef Mailbox<SimplePortComm<ImageHandle>*> _cfront_bug_Image_;
typedef SimpleIPort<ImageHandle> ImageIPort;
typedef SimpleOPort<ImageHandle> ImageOPort;

#endif
