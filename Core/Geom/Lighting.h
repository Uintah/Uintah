
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

#include <Geom/Light.h>
#include <Containers/Array1.h>
#include <Geom/Color.h>

namespace SCICore {

namespace GeomSpace {
  struct Lighting;
}

namespace PersistentSpace {
  class Piostream;
  void Pio( Piostream &, GeomSpace::Lighting & );
}

namespace GeomSpace {

using SCICore::Containers::Array1;

struct Lighting {
    Array1<Light*> lights;
    Color amblight;

    Lighting();
    ~Lighting();

    friend void PersistentSpace::Pio(Piostream&, Lighting&);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:11  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:09  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_Lighting_h */

