
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

#include <Core/share/share.h>

#include <Core/Geom/Light.h>
#include <Core/Containers/Array1.h>
#include <Core/Geom/Color.h>

namespace SCIRun {


class SCICORESHARE Lighting {
public:
    Lighting();
    ~Lighting();

  // Dd: Lighting was a struct... don't know why the following
  //     were made private... things don't compile that way...
  // private:
    Array1<Light*> lights;
    Color amblight;

    friend SCICORESHARE void Pio( Piostream&, Lighting& );
};

} // End namespace SCIRun


#endif /* SCI_Geom_Lighting_h */

