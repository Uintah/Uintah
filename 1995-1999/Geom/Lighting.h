
/*
 *  Lighting.h:  The light sources in a scene
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Lighting_h
#define SCI_Geom_Lighting_h 1

#include <Classlib/Array1.h>
#include <Geom/Color.h>
class Light;

struct Lighting {
    Array1<Light*> lights;
    Color amblight;

    Lighting();
    ~Lighting();

    friend void Pio(Piostream&, Lighting&);
};

#endif /* SCI_Geom_Lighting_h */
