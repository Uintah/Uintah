
/*
 *  ColormapPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColormapPort_h
#define SCI_project_ColormapPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/Colormap.h>

typedef Mailbox<SimplePortComm<ColormapHandle>*> _cfront_bug_Colormap_;
typedef SimpleIPort<ColormapHandle> ColormapIPort;
typedef SimpleOPort<ColormapHandle> ColormapOPort;

#endif
