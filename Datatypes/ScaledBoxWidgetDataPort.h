
/*
 *  ScaledBoxWidgetDataPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ScaledBoxWidgetDataHandlePort_h
#define SCI_project_ScaledBoxWidgetDataHandlePort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ScaledBoxWidgetData.h>

typedef Mailbox<SimplePortComm<ScaledBoxWidgetDataHandle>*> _cfront_bug_ScaledBoxWidgetData_;
typedef SimpleIPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataIPort;
typedef SimpleOPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataOPort;

#endif
