
/*
 *  Lighting.cc:  The light sources in a scene
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Lighting.h>

Lighting::Lighting()
: amblight(Color(0,0,0))
{
}

Lighting::~Lighting()
{
}

#ifdef __GNUG__
#include <Classlib/Array1.cc>
template class Array1<Light*>;

#endif
