
/*
 *  IntervalPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_IntervalPort_h
#define SCI_project_IntervalPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/Interval.h>

typedef SimpleIPort<IntervalHandle> IntervalIPort;
typedef SimpleOPort<IntervalHandle> IntervalOPort;

#endif
