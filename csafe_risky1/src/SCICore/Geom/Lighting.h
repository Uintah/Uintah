
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

#include <SCICore/share/share.h>

#include <SCICore/Geom/Light.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geom/Color.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::Array1;

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

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:20  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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

