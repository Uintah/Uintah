/*
 *  ContourSet.cc: The ContourSet Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <ContourSet.h>
#include <Surface.h>
#include <Geom.h>

#define Sqr(x) ((x)*(x))

ContourSet::ContourSet()
: axis(z)
{
};

ContourSet::ContourSet(const ContourSet &copy)
: space(copy.space)
{
}

ContourSet::~ContourSet() {
}

void ContourSet::io(Piostream& stream) 
{
}
