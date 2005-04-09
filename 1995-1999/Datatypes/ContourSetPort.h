
/*
 *  ContourSetPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ContourSetPort_h
#define SCI_project_ContourSetPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ContourSet.h>

typedef SimpleIPort<ContourSetHandle> ContourSetIPort;
typedef SimpleOPort<ContourSetHandle> ContourSetOPort;

#endif
