
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

#include <Dataflow/Ports/SimplePort.h>
#include <Dataflow/Ports/ScaledBoxWidgetData.h>

namespace SCIRun {

typedef SimpleIPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataIPort;
typedef SimpleOPort<ScaledBoxWidgetDataHandle> ScaledBoxWidgetDataOPort;

} // End namespace SCIRun


#endif
