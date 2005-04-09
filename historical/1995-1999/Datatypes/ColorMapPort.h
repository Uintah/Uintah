
/*
 *  ColorMapPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColorMapPort_h
#define SCI_project_ColorMapPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/ColorMap.h>

typedef SimpleIPort<ColorMapHandle> ColorMapIPort;
typedef SimpleOPort<ColorMapHandle> ColorMapOPort;

#endif
